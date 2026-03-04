# Frontend Implementation: Round History API + Scripted Level 5

The Python inference server now accepts a `round_history` field and supports `scripted_5`. This doc describes the corresponding changes needed in **spaces-game-node**.

## Files to modify

| File | Change |
|------|--------|
| `src/utils/board-encoding.ts` | Add `buildFogBoard()` |
| `src/utils/ai-agent-inference.ts` | Build and send `round_history` payload |
| `src/types/opponent.ts` | Add `'scripted_5'` to `AiAgentSkillLevel` |
| `src/constants/game-rules.ts` | Add level 5 entry to `AI_AGENT_SKILL_LEVELS` |

---

## 1. `src/utils/board-encoding.ts` — add `buildFogBoard()`

New exported function that filters a board down to what the AI agent could see during simulation.

```ts
export function buildFogBoard(
  board: Board,
  lastStep: number,
  hitTrap: boolean,
  trapPosition?: { row: number; col: number }
): Board {
  // Filter sequence: include piece/final moves where move.order <= lastStep + 1
  const filtered = board.sequence.filter((move) => {
    if (move.type === 'piece' || move.type === 'final') {
      return move.order <= lastStep + 1;
    }
    // Include trap only if the agent hit it at that position
    if (move.type === 'trap' && hitTrap && trapPosition) {
      return (
        move.position.row === trapPosition.row &&
        move.position.col === trapPosition.col
      );
    }
    return false;
  });

  const grid = deriveGridFromSequence(filtered, board.boardSize);

  return {
    ...board,
    sequence: filtered,
    grid,
  };
}
```

**Why**: The AI agent only sees the opponent's piece moves up to the step it reached, plus the trap if it hit one. This mirrors the fog-of-war visibility rules already used in training.

---

## 2. `src/utils/ai-agent-inference.ts` — build `round_history`

Inside `requestAiAgentBoard()`, construct the payload from the existing `roundHistory` parameter (already of type `RoundResult[]`).

### Perspective flipping

The game tracks everything from the **player's** perspective. The inference server expects the **AI agent's** perspective. So:

| Game field | Maps to |
|---|---|
| `r.opponentBoard` | `agent_board` (the AI built this board) |
| `r.playerBoard` | opponent board (the player built this) |
| `r.opponentPoints` | `agent_score` |
| `r.playerPoints` | `opponent_score` |
| `r.simulationDetails.opponentLastStep` | `agent_last_step` (how far the AI got) |
| `r.simulationDetails.playerLastStep` | `opponent_last_step` |
| `r.simulationDetails.opponentHitTrap` | `agent_hit_trap` |
| `r.simulationDetails.playerHitTrap` | `opponent_hit_trap` |

### Code

Add this before the fetch call, after the existing `round_scores` construction:

```ts
import { buildFogBoard, encodeMinimalBoard } from './board-encoding';

// Build rich round history for the agent
const roundHistoryPayload = (roundHistory ?? []).map((r) => {
  // AI's board is opponentBoard from game perspective
  const agentBoard = encodeMinimalBoard(r.opponentBoard);

  // Fog view of the player's board: what the AI could see
  const fogBoard = buildFogBoard(
    r.playerBoard,
    r.simulationDetails?.opponentLastStep ?? -1,
    r.simulationDetails?.opponentHitTrap ?? false,
    r.simulationDetails?.opponentTrapPosition
  );
  const opponentBoardFog = encodeMinimalBoard(fogBoard);

  return {
    agent_score: r.opponentPoints ?? 0,
    opponent_score: r.playerPoints ?? 0,
    agent_board: agentBoard,
    opponent_board_fog: opponentBoardFog,
    agent_last_step: r.simulationDetails?.opponentLastStep ?? -1,
    opponent_last_step: r.simulationDetails?.playerLastStep ?? -1,
    agent_hit_trap: r.simulationDetails?.opponentHitTrap ?? false,
    opponent_hit_trap: r.simulationDetails?.playerHitTrap ?? false,
    collision: r.collision ?? false,
  };
});
```

Add `round_history: roundHistoryPayload` to the request body alongside the existing `round_scores`.

**Keep `round_scores`** — scripted agents 1-4 still use it.

---

## 3. `src/types/opponent.ts` — add `scripted_5`

```ts
type AiAgentSkillLevel =
  | 'beginner' | 'beginner_plus'
  | 'intermediate' | 'intermediate_plus'
  | 'advanced' | 'advanced_plus'
  | 'test_fail'
  | 'scripted_1' | 'scripted_2' | 'scripted_3' | 'scripted_4'
  | 'scripted_5';   // <-- add this
```

---

## 4. `src/constants/game-rules.ts` — add level 5 entry

```ts
scripted_5: {
  emoji: '🍄',
  defaultName: 'Myco',
  color: '#1B5E20',
  label: 'Supermove',
},
```

Add this after the `scripted_4` entry in the `AI_AGENT_SKILL_LEVELS` object.

---

## Backward compatibility

- `round_history` defaults to `[]` on the server — old clients without this field continue to work
- `round_scores` is unchanged — scripted agents 1-4 ignore `round_history`
- `opponent_history` (used for RL model obs encoding) is unchanged

## Verification

1. `npx tsc --noEmit` compiles cleanly
2. Unit test for `buildFogBoard()`:
   - Piece-only filtering up to `lastStep`
   - Trap included only when `hitTrap` is true and position matches
   - Grid derived correctly from filtered sequence
3. Play a game against `scripted_5` — round 0 should produce a trap board, two consecutive ties should trigger a supermove board
