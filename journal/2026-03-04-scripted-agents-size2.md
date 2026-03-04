# Training Journal: Scripted Agents for Size 2

**Date:** March 4, 2026
**Author:** Claude (with Ryan's direction)
**Stage:** Size 2 difficulty progression — scripted agents

---

## The Problem with Size 2

Back on Feb 25, we confirmed that size 2 self-play is a coin flip. A 2x2 board has 4 cells and at most 1 trap — there simply isn't enough strategic depth for an RL agent to learn meaningful round-over-round adaptation. The self-play eval win rate sat at 50% for the entire run.

Ryan's solution: don't train a model at all. Instead, write scripted agents that implement a deliberate difficulty progression. Four levels, each adding one new capability, giving players a clear ramp from "trivially easy" to "you need to think about this."

---

## What We Built

### Four Scripted Difficulty Levels

| Level | Name | What It Does |
|-------|------|-------------|
| `scripted_1` — Simple | Straight-path board, no traps. Switches column once at a random point in the game. The tutorial-grade opponent. |
| `scripted_2` — Reactive | Still no traps, but now tracks round scores. If it lost last round, it switches columns. The player's first taste of an opponent that responds to them. |
| `scripted_3` — Trapper | Same reactive column logic, but now places a trap adjacent to the starting piece. The player has to deal with traps for the first time while the opponent also adapts. |
| `scripted_4` — Adaptive | Always switches columns between rounds (fully predictable pattern, but keeps the player honest). Uses traps by default, but has an escape hatch: after losing 2 rounds in a row, it drops the trap and goes simple. A small strategic wrinkle — the player can "earn" an easier board by winning consecutively, or get punished for losing. |

### Board Templates

Both board types are size-agnostic — they work for any board size, not just 2. The piece walks straight up one column:

- **Simple board**: piece at every row in column C, final at (-1, C)
- **Trap board**: same, but with a trap placed adjacent to the starting piece at (N-1, 1-C)

### API Changes

Added `round_scores` to `ConstructBoardRequest` — a list of `{agent, opponent}` dicts giving per-round point totals. The existing API only sent cumulative scores, which isn't enough for the reactive column-switching logic. The field is optional and backward-compatible (defaults to empty list).

The scripted agents route through an early-exit block in `main.py` before any model loading happens. No registry, no opponent pools, no RL inference — just board construction and validation.

---

## Files Changed

### Python (spaces-game-python)
- `inference_server/models.py` — `scripted_1` through `scripted_4` in `SkillLevel` enum, `round_scores` field on request
- `inference_server/scripted_agents.py` — new module with board builders and level dispatch
- `inference_server/main.py` — early-exit routing for scripted skill levels
- `tests/test_scripted_agents.py` — 27 tests covering all levels, column logic, escape hatch, size-agnostic behavior
- `DEPLOYMENT.md` — documented scripted agents and `round_scores` format

### Node frontend (spaces-game-node)
- `src/types/opponent.ts` — added scripted skill levels to `AiAgentSkillLevel` type
- `src/constants/game-rules.ts` — 4 new entries with plant-themed emoji and green color gradient
- `src/utils/ai-agent-inference.ts` — `roundHistory` parameter, converts to `round_scores` in request body
- `src/App.tsx` — both `requestAiAgentBoard` call sites now pass `state.roundHistory`

---

## Design Decisions

**Why scripted instead of RL?** The self-play experiment proved size 2 can't produce adaptive behavior through learning. Scripted agents let us control the difficulty curve precisely, and they're trivially cheap to run — no model files, no GPU inference, just a few lines of logic.

**Why plant emoji?** Ryan's existing skill levels use animal emoji for the RL agents (chick, fox, owl, wolf, dragon). The plant theme gives scripted agents a distinct visual identity — you can tell at a glance whether you're playing against a scripted opponent or an RL model.

**Why `round_scores` instead of deriving from cumulative?** The cumulative `agent_score` and `opponent_score` don't tell you who won each individual round. The reactive logic needs "did I lose last round?" which requires per-round granularity. Adding a new field is cleaner than trying to reconstruct round results from cumulative totals.

---

## Verification

All 27 scripted agent tests pass. All 33 existing inference server tests still pass (60 total). TypeScript compiles cleanly with the new types.

Next step: deploy the Python server, wire up the UI to show scripted levels in the opponent picker, and play some games against each level to confirm the feel matches the intent.

---

## What's Next

These scripted agents are the size 2 story — done. The pattern could generalize to other sizes as a "practice mode" before players face RL opponents, but that's a future consideration. The immediate priority remains size 5+ training and fog of war UI work.
