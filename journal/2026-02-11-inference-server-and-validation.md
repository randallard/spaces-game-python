# Training Journal: Inference Server, Difficulty Levels, and Board Validation

**Date:** February 11, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Inference serving, difficulty-level training, validation hardening

---

## Starting Point

The last journal entry (Feb 9) left us with the scoring fix and no-revisit masking in place, but models not yet retrained. The "What's Next" list was: retrain size 3, design difficulty levels, and start fog of war. Ryan had also been building the Node.js frontend (`spaces-game-node`) with AI agent opponent support, so the focus shifted toward getting the trained models serving in production.

---

## Difficulty-Level Training Infrastructure

**Commit:** `a14a955` — Add difficulty-level training checkpoints and update docs

Ryan wanted players to choose difficulty when playing against the AI — beginner, intermediate, expert — rather than a single best model. We added automatic checkpoint saving at opponent phase milestones during Stage 3 training:

| Difficulty | Saved after | What it knows |
|------------|-------------|---------------|
| Beginner | Phase 0 complete | Builds valid boards, beats simple straight-path opponents |
| Intermediate | Phase 2 complete | Uses traps, handles mixed simple + one-trap opponents |
| Expert | Training end | Full strategy against all opponent types |

The `--min-phase-steps` flag (default 10k, recommend 100k for separation) ensures each phase gets deep training so the difficulty levels are meaningfully different.

`play_against_agent.py` got a `--difficulty` flag and interactive difficulty selection when checkpoint files exist. Docs across README, TRAINING_PLAN, and examples README were updated to reflect the current state.

---

## Size 2 Retraining

Ryan kicked off a size 2 retrain with the scoring fix, no-revisit masking, and difficulty checkpoints:

```
python examples/train_simultaneous.py \
    --size 2 --timesteps 1000000 --min-phase-steps 100000 \
    --resume models/size2/stage3/best/best_model.zip
```

Results at 1M steps: opponent phase 2 (intermediate), 90% win rate, 100% valid boards. It produced beginner and intermediate checkpoints. Expert was saved at training end but is functionally intermediate-level since the model didn't advance past phase 2 within the timestep budget. More timesteps would push it further.

---

## FastAPI Inference Server

**Commit:** `7cf6d50` — Add FastAPI inference server for AI agent board construction

This was a big piece — a standalone inference server that the Node frontend calls to get agent-constructed boards. The architecture:

- **`inference_server/main.py`** — FastAPI app with `/construct-board` and `/health` endpoints
- **`inference_server/inference.py`** — Core board construction logic (extracted from `play_against_agent.py`)
- **`inference_server/model_registry.py`** — Convention-based model discovery mapping skill levels to checkpoint types (early/mid/advanced), with deterministic vs stochastic sampling per level
- **`inference_server/models.py`** — Pydantic request/response schemas

The Node frontend sends `skill_level` (beginner through advanced_plus), the server maps it to a model checkpoint and sampling mode. Six levels from two knobs: three checkpoint tiers x deterministic/stochastic.

---

## The Bugs Ryan Found

Ryan deployed the inference server on his laptop, pulled the updated code to the Node frontend, and played a few games. Two problems surfaced immediately:

### Bug 1: Beginner (Pip) — One-Step Board

Playing against Pip (beginner, stochastic), Ryan got a board with a single piece placed at row 0 and a final move. One step total. This "board" passed `is_board_playable()` because technically the piece was at row 0, the final aligned — all the existing rules were satisfied. But it's not a real board. There's no path, no strategy, nothing for the opponent to navigate.

### Bug 2: Expert (Ember) — No Valid Board in 5 Tries

Playing against Ember (advanced, stochastic), the agent couldn't produce a valid board in all 5 retry attempts. The frontend got `valid: false` and showed a dead-end alert with no way to continue the game.

The root cause for Ember turned out to be the same `piece_visited_positions` bug from the Feb 9 journal entry. We'd fixed it in `play_against_agent.py` but the inference server had its own copy of the construction loop — and it was missing the tracking. Action masks during inference didn't match training, so the model was effectively flying blind.

Ryan's prediction from Feb 9 came true: *"any code that drives agent construction needs to track visited positions."*

---

## Validation Hardening

**Commit:** `f67f6a6` — Require full-path boards and add retry/forfeit for inference failures

Two new rules added to `is_board_playable()`:

1. **Piece must visit every row** (0 through board_size-1). A real board has a path from bottom to top. No more single-piece-at-row-0 shortcuts.
2. **Sequence must contain a final move.** The board must actually reach the goal.

Before adding these, we verified all 7,257 boards across every pool (data/, boards/, curated) already pass both rules. The change only catches degenerate agent-produced boards, not legitimate human or generated boards.

This is a shift from the Feb 9 decision where we kept no-revisit as an agent-only rule. The difference: no-revisit restricts *strategy* (humans should discover it's wasteful), but "piece must reach every row" defines *what a board is*. A board without a path from bottom to top isn't a board at all.

---

## Inference Server Fix

Same commit fixed the inference server's construction loop to track `piece_visited_positions`, matching the `play_against_agent.py` version. Also updated `build_board_for_round()` to return `(Board, attempts_used)` so the API response includes how many attempts the agent needed.

---

## Frontend Retry/Forfeit

**Commit (Node):** `c9d4c80` — Handle AI agent board construction failures with retry/forfeit

Changed `requestAiAgentBoard()` to return a structured `AiAgentBoardResult` with `{ board, failed, attemptsUsed }` instead of `Board | null`. When the agent fails:

1. Player sees: *"Pip couldn't decide on a board (5 attempts)."*
2. Two choices: **OK** = give them more time (fires another request), **Cancel** = they forfeit the round
3. If retry also fails, auto-forfeit with a message
4. Forfeit scores as player win (1-0) and the game continues

Added `forfeit?: boolean` to `RoundResult` type so the UI can eventually distinguish forfeited rounds visually.

---

## Files Changed

### Python (spaces-game-python)
- `spaces_game/validation.py` — Full-path + final-move requirements
- `inference_server/inference.py` — `piece_visited_positions` tracking, return attempts_used
- `inference_server/main.py` — Pass attempts_used to response
- `inference_server/models.py` — `attempts_used` field in response schema
- `tests/test_validation.py` — 4 new tests, updated supermove test
- `examples/train_simultaneous.py` — Difficulty checkpoints, `--min-phase-steps`
- `examples/play_against_agent.py` — `--difficulty` flag, interactive selection
- `README.md`, `TRAINING.md`, `TRAINING_PLAN.md`, `examples/README.md` — Updated docs
- `VERIFY.md` — Verification checklist for all changes

### Node (spaces-game-node)
- `src/utils/ai-agent-inference.ts` — Structured return type with attempts
- `src/utils/ai-agent-inference.test.ts` — Updated tests
- `src/App.tsx` — Retry/forfeit flow for AI agent failures
- `src/types/game-state.ts` — `forfeit` field on RoundResult

---

## Decision: Fresh Retraining Required

After implementing the full-path validation, Ryan asked whether the currently-running size 2 training (already at 1.5M steps, opponent phase 2, 90% win rate) would pick up the new rules. The answer: no. The running process loaded the old `is_board_playable()` at startup. Its `valid_rate: 100%` is against the old rules that would pass a 1-step board.

The validation chain during training:
1. Agent signals "done" → `simultaneous_play_env.py` calls `is_board_playable(agent_board)`
2. Returns `valid_board` in info dict → training callback reads it for `valid_rate`
3. Same function, loaded once at process start

Since the rules changed, old models may have learned to build boards that don't visit every row. Resuming from those weights would start with broken habits. Fresh training from scratch is the right call.

**Commands:**
```bash
# Size 2 - 5M steps, board library, no resume
python examples/train_simultaneous.py \
    --size 2 --board-library new_boards_2.json \
    --timesteps 5000000 --min-phase-steps 100000

# Size 3 - 5M steps, board library, no resume
python examples/train_simultaneous.py \
    --size 3 --board-library new_boards_3.json \
    --timesteps 5000000 --min-phase-steps 100000
```

5M steps gives enough budget for all 5 opponent phases with 100k minimum per phase. The board libraries provide construction scaffolding so the agent learns from curated examples.

---

## What's Next

- Monitor fresh retraining runs for both sizes
- Verify difficulty separation: play-test beginner vs intermediate vs expert
- Deploy retrained models to inference server, verify via Node frontend
- Implement fog of war in `SimultaneousPlayEnv` (Stage 4)
- Consider a proper UI component for the retry/forfeit prompt (currently `confirm()` dialog)

---

*A board without a path from bottom to top isn't a board — it's a shortcut the agent found that we hadn't thought to block.*
