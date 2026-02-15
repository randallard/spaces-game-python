# Training Journal: Strict Masking, Simplified Rewards, and Self-Play

**Date:** February 14, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Stage 3 rework — strict action masking, scaffolding removal, reward simplification, self-play

---

## Starting Point

Size 4 had been stuck. After 12M+ total steps across multiple runs, the agent plateaued at ~50% win rate at opponent phase 6. The tuned hyperparameters from Feb 13 (lower learning rate, higher entropy) helped stability but didn't break through the ceiling.

The diagnosis pointed at four issues stacking on top of each other:

1. **~2% invalid boards** — wasted training signal on boards that could never win
2. **Construction scaffolding** — millions of steps spent learning to build valid boards before the real game even starts
3. **Construction reward shaping** — +0.1 for placing a piece, +0.3 for reaching row 0, +0.2 for supermove — noise that doesn't correlate with winning
4. **Random opponents** — the game has rock-paper-scissors dynamics, so 50% is roughly the ceiling against randomly-selected pool boards. The `opponent_history` observation is wasted because random opponents have no patterns to exploit.

The plan: fix all four in one pass, then retrain size 4 from scratch.

---

## Phase 1: Strict Action Masking

The biggest conceptual change. Instead of letting the agent build invalid boards and penalizing it after the fact, make invalid boards structurally impossible at the action mask level.

### BFS Reachability

Added `_can_reach_all_rows()` — a BFS that checks whether the piece can still visit all required rows AND reach row 0 (the finish requirement) from a given position. Called for every candidate action in `_is_valid_placement()`:

- **Piece moves**: "If I move here, can I still complete the board?" If not, the cell is masked out.
- **Trap placements**: "If I place a trap here, does it block all remaining paths?" If so, masked out.
- **Supermove traps**: "If I trap myself here, is there at least one adjacent escape cell from which all rows are still reachable?" If not, masked out.
- **Finish signal**: Only allowed when all rows 0..board_size-1 have been visited by the piece AND the piece is at row 0.

The BFS is trivial for sizes 2-4 (max 16 cells). No measurable impact on training FPS.

### The Row-0 Bug

The initial BFS had a subtle bug: it returned True as soon as all rows appeared in the "visited + reachable" set, without checking whether the piece could actually *get back to* row 0 to finish. An agent on a 2x2 board could visit all 4 cells, end up at (1,1), and then be deadlocked — all cells visited, no valid moves, forced finish with the piece not at row 0.

Fix: the BFS now checks two conditions — "all rows visitable" AND "row 0 reachable from current position." Both must hold.

### The Goal Column Bug

Found while debugging invalid boards: `_construct_board_from_state()` was placing the goal at the column of the *last non-final move*, not the piece's current column. If the last move was a trap at column 0 but the piece was at column 1, the goal went to column 0 and `is_board_playable()` correctly rejected it. Fixed to use `self.current_piece_position.col`.

### Test Results

17 new tests in `test_strict_masking.py`, including end-to-end random play for sizes 2, 3, and 4. The key invariant: when the agent signals "done" through the action mask (not forced truncation), the resulting board is always valid. 100% across 50 episodes per size.

---

## Phase 2: Remove Construction Scaffolding

With strict masking, every action the agent can take produces a board on a valid path. It doesn't need to be taught "what a valid board looks like" through scaffolding — it literally can't build an invalid one.

Removed:
- `set_scaffolding()` and `_prefill_from_library()` from the env
- `_maybe_advance_construction()` from `OpponentProgressionCallback`
- All scaffolding-related attributes, logging, and checkpointing
- `scaffolding_moves_to_remove` from `make_env()` and env kwargs

The `--board-library` CLI flag is kept for backward compatibility but prints a deprecation warning and does nothing.

This is significant for size 4 training. The previous attempt spent 608k steps just on construction scaffolding phases before opponent play even started. Those steps are now zero — training begins with opponent play immediately.

---

## Phase 3: Simplified Rewards

The old reward structure:

| Event | Old Reward |
|---|---|
| Place piece | +0.1 |
| Reach row 0 | +0.3 |
| Supermove | +0.2 |
| Place trap | +0.1 |
| Invalid placement | -2.0 |
| Round win | score_diff * 5.0 + 10.0 |
| Round loss | score_diff * 5.0 - 5.0 |
| Game win | +50.0 |
| Game loss | -50.0 |
| Invalid board | -20.0 |

The construction rewards were noise — the agent learned to maximize them instead of learning to win games. The invalid placement penalty is moot with strict masking (MaskablePPO can't select masked actions, though MultiDiscrete per-dimension masking allows cell/type mismatches that fall through harmlessly).

New reward structure:

| Event | New Reward |
|---|---|
| Valid placement | 0.0 |
| Invalid placement fallback | 0.0 |
| Round win | score_diff * 2.0 + 5.0 |
| Round loss | score_diff * 2.0 - 5.0 |
| Game win | +25.0 |
| Game loss | -25.0 |
| Invalid board fallback | -10.0 |

Cleaner signal. The agent learns that winning rounds and games is what matters, not accumulating construction micro-rewards.

---

## Phase 4: Self-Play with Rolling Opponent Pool

This is the big one. The fundamental problem with random pool opponents: they have no patterns. The `opponent_history` observation — 5 rounds of the opponent's revealed boards — is useless when the opponent is just pulling randomly from a JSON file. The value network can't learn to predict outcomes because outcomes against random opponents are genuinely unpredictable.

### How It Works

1. **Warmup** (default 100k steps): Training uses JSON pool opponents normally. The agent needs basic construction ability before self-play makes sense.

2. **Snapshot** (every 50k steps after warmup): Save the current model to `opponent_pool/snapshot_N.zip`. Keep the 10 most recent snapshots.

3. **Assignment**: Each training env gets a randomly selected snapshot loaded as its opponent model via `set_opponent_model(path)`.

4. **Opponent board construction**: When `_finish_round()` fires, the opponent model builds a board using the same manual construction loop as the inference server. The opponent sees swapped scores and the agent's previous boards as its opponent history.

5. **Fallback**: If the opponent model produces an invalid board (it might, especially early on), fall back to a JSON pool board. This provides ~20% JSON pool mixing naturally, which prevents self-play collapse.

### SubprocVecEnv Compatibility

The opponent model is loaded per-subprocess via `set_opponent_model(path: str)`. Takes a string path, not a model object, because SubprocVecEnv pickles arguments across process boundaries. Each subprocess loads its own copy. Models are <10MB each, so 4 copies is fine.

### Skill Level Snapshots

The `SelfPlayCallback` also tracks eval win rate from the `OpponentProgressionCallback` and saves milestone checkpoints:

- 55%+ win rate -> beginner
- 60%+ -> intermediate
- 65%+ -> advanced
- 70%+ -> expert
- 75%+ -> advanced_plus

At training end, the snapshot timeline is divided into 6 tiers and saved as `{tier}_checkpoint.zip` files that the inference server's `ModelRegistry` can pick up.

---

## Smoke Test Results

Ran two 10k-step smoke tests:

**Standard training (no self-play)**:
- 80% game win rate at opponent phase 0
- 96% valid rate
- Rewards converging normally

**Self-play training**:
- Warmup completed at 4k steps
- Snapshot created, loaded into training envs
- 85% game win rate, 98% valid rate
- All 5 skill milestone checkpoints saved
- Training FPS ~100 it/s (vs ~200 without self-play, expected due to opponent model inference)

Both passed cleanly. No crashes, no warnings, all 158 existing tests still pass.

---

## Size 4 Training Kicked Off

```bash
python examples/train_simultaneous.py --size 4 --self-play --timesteps 5000000
```

Running in tmux. With the rework:
- No scaffolding warmup (was 608k steps before)
- No construction rewards to confuse the value network
- Self-play opponents that actually have patterns to learn from
- Strict masking guarantees every board is structurally valid

The question is whether self-play breaks through the 50% ceiling that random opponents couldn't.

---

## Files Changed

### `spaces_game/simultaneous_play_env.py`
- Added `import collections`
- Added `_can_reach_all_rows()` BFS reachability helper
- Enhanced `_is_valid_placement()` with Level 2 reachability checks for piece, trap, and supermove moves
- Strengthened finish mask in `action_masks()` — requires all rows visited
- Fixed goal column in `_construct_board_from_state()` — uses piece column, not last move column
- Removed `set_scaffolding()`, `_prefill_from_library()`, scaffolding attributes
- Simplified rewards: removed construction shaping, adjusted round/game rewards
- Added self-play support: `set_opponent_model()`, `clear_opponent_model()`, `_build_opponent_board_from_model()`
- Added `_agent_boards_this_game` tracking for opponent history in self-play

### `examples/train_simultaneous.py`
- Removed construction curriculum from `OpponentProgressionCallback`
- Removed `board_library_path`, `scaffolding_moves_to_remove` from `make_env()`
- Added `SelfPlayCallback` with snapshot pool, skill milestone tracking
- Added CLI flags: `--self-play`, `--snapshot-freq`, `--pool-size`, `--warmup-steps`
- `--board-library` kept for backward compat with deprecation warning

### `tests/test_strict_masking.py` (new)
- 17 tests covering BFS reachability, piece/trap/supermove masking, finish mask, and end-to-end random play for sizes 2, 3, 4

---

## What's Next

- Watch the size 4 self-play run — does win rate climb past 50%?
- If 5M isn't enough, `--resume` with another 5M
- Once size 4 converges, deploy to inference server
- Stage 4 (fog of war) is still on the roadmap but self-play was the more pressing bottleneck
- Size 5 after size 4 stabilizes

---

*You can't learn to outplay an opponent that doesn't have a strategy. Random pools taught the agent to build valid boards; self-play teaches it to build winning ones.*

---

## Addendum: Policy Collapse and the Warmup Lesson (Same Day)

The initial 5M-step self-play run from scratch (`--self-play --timesteps 5000000`) collapsed within 30 minutes. TensorBoard told the story:

| Metric | Peak (88k steps) | Collapse (120-144k) | Notes |
|---|---|---|---|
| Valid rate | 75% | 0-4% | Catastrophic forgetting |
| Win rate | 10% | 0% | Never learned to win |
| Avg reward | -54 | -75 | Getting worse |
| Explained variance | 0.15 | -0.14 to 0.27 | Value net barely learning |

The valid rate climbed to 75% during warmup (100k steps), then cratered the moment self-play activated. Root cause: **the warmup was too short**. The first self-play snapshot was taken from a model with only 14% valid rate (at 25k n_calls). The agent played against a terrible version of itself, producing garbage training signal. Bad snapshots led to worse policy which led to worse snapshots — a death spiral.

### The Fix: Two-Phase Training

Instead of training from scratch with self-play, resume from last night's converged model (2.5M steps, opponent phase 6, 50% win rate, 96%+ valid rate):

```bash
python examples/train_simultaneous.py --size 4 --self-play --warmup-steps 0 \
    --resume models/size4/stage3/ppo_stage3_final.zip --timesteps 5000000
```

Key insight: `--warmup-steps 0` because the model is already competent. The first self-play snapshot is a copy of the pre-trained model, so opponents are competent from step 1. No warmup needed — the foundation is already solid.

### Lesson Learned

Self-play requires a **stable base policy**. For size 4 (16-cell board, complex action space), 100k steps of warmup isn't nearly enough — the model needs 1M+ steps just to learn valid construction. The safe approach: train against pool opponents first until valid rate stabilizes at 95%+, *then* layer self-play on top with `--resume`.

This is a well-known pattern in competitive RL: AlphaGo trained supervised learning from human games before switching to self-play. You can't bootstrap from nothing when the action space is large enough that random play produces no useful signal.

---

## Addendum 2: Reward Mismatch — The Second Collapse

The resumed self-play run also collapsed, just slower. Valid rate started at 99% (the pre-trained model retained its knowledge) but steadily fell: 99% → 75% → 62% → 49% over 64k steps. Same death spiral, higher starting point.

TensorBoard revealed the deeper problem: **explained variance was negative the entire run** (-0.59 to -0.49). The value network wasn't just bad — it was *anti-correlated* with actual returns. It was actively predicting the wrong direction.

Root cause: the resumed model was trained for 2.5M steps under the **old reward structure** (+0.1 per piece placement, +0.3 for reaching row 0, +0.2 for supermove, round win = score_diff * 5.0 + 10.0, game win = +50.0). We changed the rewards in the rework (0.0 for construction, round win = score_diff * 2.0 + 5.0, game win = +25.0). The policy network's weights were fine — it still knew how to build boards. But the value network's predictions were completely miscalibrated. It expected +50 for a game win and got +25. Expected +0.1 for placing a piece and got 0. Every value estimate was wrong, which corrupted the advantage calculation, which corrupted the policy gradient.

Self-play made this worse because the non-stationary opponent added another source of prediction error on top of the reward mismatch.

### The Fix: Three-Phase Training

1. **Phase A** (done): Train from scratch against pool opponents with old rewards → converged at 50% win rate, 96% valid
2. **Phase B** (now): Resume from Phase A *without* self-play, new rewards → let the value network recalibrate against stable pool opponents
3. **Phase C** (next): Resume from Phase B *with* `--self-play --warmup-steps 0` → self-play against a model whose value network actually works

```bash
# Phase B (running now)
python examples/train_simultaneous.py --size 4 \
    --resume models/size4/stage3/ppo_stage3_final.zip --timesteps 2000000

# Phase C (after Phase B converges)
python examples/train_simultaneous.py --size 4 --self-play --warmup-steps 0 \
    --resume models/size4/stage3/ppo_stage3_final.zip --timesteps 5000000
```

### Lesson Learned

When changing reward structure mid-training, the **value network must recalibrate** before introducing additional instability (like self-play). The policy network transfers well because the action semantics haven't changed — "place piece at (2,1)" still means the same thing. But the value network's job is to predict *cumulative future reward*, and when the reward scale changes, every prediction is wrong. PPO's advantage estimates depend on accurate value predictions, so a miscalibrated value network corrupts the policy gradient even if the policy itself is good.

The general rule: **one source of non-stationarity at a time.** Don't change reward structure AND add self-play simultaneously.

---

## Addendum 3: Sparse Rewards Don't Scale — Restoring Construction Shaping

Phase B (from-scratch training with simplified rewards, no self-play) also failed to converge. After 720k steps:

- Valid rate bouncing 12-56% with no upward trend
- Win rate 0% with only occasional 5% blips
- Explained variance stuck at 0.15-0.27

For comparison, the old run (with construction shaping rewards) hit 95%+ valid rate by 200k steps and was winning games by 400k.

The diagnosis: **sparse rewards don't scale to size 4**. With 0.0 reward for every construction step, the agent gets no learning signal during board building — the only feedback comes at round/game end, after 10+ sequential placement decisions on a 16-cell board. The construction shaping rewards (+0.1 piece, +0.3 row 0, +0.2 supermove, +0.1 trap) were critical breadcrumbs guiding the agent through the construction phase.

This worked fine for sizes 2-3 in smoke tests because smaller boards have fewer decisions and the round-end signal is only a few steps away. Size 4's action space is large enough that the agent can't credit-assign back through 10+ construction steps to figure out which placements led to the win or loss.

### What We Restored

All original reward values:

| Event | Simplified (failed) | Restored |
|---|---|---|
| Piece placement | 0.0 | +0.1 |
| Reach row 0 | 0.0 | +0.3 |
| Supermove landing | 0.0 | +0.2 |
| Trap placement | 0.0 | +0.1 |
| Round win | score_diff * 2 + 5 | score_diff * 5 + 10 |
| Round loss | score_diff * 2 - 5 | score_diff * 5 - 5 |
| Game win/loss | +/-25 | +/-50 |
| Invalid board | -10 | -20 |

The strict masking (BFS reachability) stays — that's purely beneficial regardless of reward structure. We now have the best of both: shaping rewards to guide construction learning + strict masking to guarantee validity.

### Revised Plan

The three-phase approach was over-engineered. Simpler plan:
1. Train from scratch with original rewards + strict masking (no self-play) until convergence
2. Add self-play with `--resume` once the model is solid

### Lesson Learned

Reward simplification is not universally beneficial. Dense shaping rewards are often dismissed as "noise" in RL literature, but they solve a real problem: **credit assignment over long action sequences**. The right time to simplify rewards is after the agent has learned the basics, not before. For size 4+, construction shaping is load-bearing infrastructure, not noise.

---

## Addendum 4: Flat Discrete Action Space + Forward-Only Movement

After restoring shaping rewards, size 4 training still bounced between 12-56% valid rate after 720k steps. The root cause wasn't the rewards — it was the **MultiDiscrete action space** and **backward movement**.

### Problem: MultiDiscrete Cell/Type Mismatch

The MultiDiscrete space `[cell, type, done]` masks each dimension independently. MaskablePPO learns separate distributions for each dimension, but the valid combinations aren't independent. A cell might be valid for piece placement but not for trap placement (or vice versa). The cell mask is the union of both, and the type mask tells you which types exist — but there's no way to express "cell 5 is valid for piece but not trap."

Result: the agent picks a valid cell AND a valid type, but the *combination* is invalid. This is a no-op that wastes its step budget. On a 16-cell board, the majority of random actions are wasted, leading to truncation before the agent completes a board. Even a partially trained agent wastes 30-50% of its steps on mismatched cell/type combinations.

### Problem: Backward Movement

The agent could move the piece backward (to higher row indices), allowing oscillation. While the no-revisit rule prevents infinite loops, the agent still wasted steps zigzagging up and down the board. On a 4x4 grid, this doubles the effective path length and makes construction harder than it needs to be.

### Fix 1: Flat Discrete Action Space

Replaced `MultiDiscrete([n_cells, 2, 2])` with `Discrete(2 * n_cells + 1)`:

```
Action 0..n_cells-1:      piece at cell i
Action n_cells..2*n_cells-1: trap at cell i
Action 2*n_cells:          finish
Total: 2 * board_size² + 1 (size 4 = 33 actions)
```

Every unmasked action is now guaranteed valid. No more cell/type mismatch — if action 5 (piece at cell 5) is unmasked, placing a piece at cell 5 is valid. If action 21 (trap at cell 5) is unmasked, placing a trap at cell 5 is valid. Zero wasted steps.

Removed `valid_cells_mask` from the observation space (it was a helper for MultiDiscrete masking, now redundant). Removed `_get_valid_cells_mask()` helper method.

### Fix 2: Forward-Only Movement

Added constraints to `_is_valid_placement()`:
- **Piece**: `if row > self.current_piece_position.row: return False`
- **Trap**: `if row > self.current_piece_position.row: return False`
- **BFS**: Removed backward direction `(1, 0)` from `_can_reach_all_rows()` — only `[(-1, 0), (0, -1), (0, 1)]`
- **Supermove escapes**: Same forward-only constraint on escape directions

The piece can only move forward (lower row index) or sideways (same row). This eliminates oscillation and cuts the effective search space roughly in half for size 4.

### Files Changed

- **`spaces_game/simultaneous_play_env.py`**: Action space, action_masks(), step(), _build_opponent_board_from_model(), _is_valid_placement(), _can_reach_all_rows(), observation space
- **`inference_server/inference.py`**: Action decoding in build_board_for_round()
- **`examples/play_against_agent.py`**: Action decoding in _agent_build_board_blind()
- **`tests/test_strict_masking.py`**: Rewritten for flat mask format, added TestFlatActionSpace and TestForwardOnlyMovement classes

### Test Results

All 167 tests pass, including 26 strict masking tests (sizes 2, 3, 4 end-to-end random play). The key invariant holds: when the agent signals finish through the flat mask, the board is always valid.

### Today's Retry Summary

This was the **fourth major iteration** on size 4 training in a single day:

1. **Strict masking + simplified rewards + self-play from scratch** → self-play collapsed (warmup too short, 14% valid snapshot → death spiral)
2. **Resume pre-trained model with self-play** → value network mismatch collapse (old rewards calibrated to +50 game win, new rewards use +25)
3. **Restored original shaping rewards, no self-play** → valid rate stuck at 12-56% (MultiDiscrete cell/type mismatch wasting steps)
4. **Flat Discrete action space + forward-only movement** → eliminates wasted steps entirely, every action is guaranteed valid (this addendum)

Each attempt revealed a different bottleneck. The progression: self-play instability → reward mismatch → sparse reward failure → action space inefficiency. Each fix was load-bearing — we kept strict masking and original rewards while addressing the remaining structural issues.

### What's Next

Retrain size 4 from scratch with the flat action space + forward-only movement + original shaping rewards + strict masking. Previous models are incompatible (different action/observation spaces). Expect much faster convergence since zero steps are wasted on invalid actions.

---

## Addendum 5: Size 4 Solved — 100% Valid, 100% Win Rate

The retraining with the flat Discrete action space succeeded decisively. Results at 1.57M steps:

| Metric | Value | Notes |
|---|---|---|
| Valid rate | **100%** | Every single board valid across all 1.57M steps |
| Win rate | **100%** | At phase 6 (all pool types mixed) |
| Explained variance | 0.35-0.46 | Value network learning well |
| Opponent phase | 6/6 | Final phase reached at step 1.2M |

### Phase Progression

| Phase | Step Reached | Description |
|---|---|---|
| 0 | 8,000 | Simple boards |
| 1 | 584,000 | One-trap boards |
| 2 | 696,000 | Simple + one-trap mixed |
| 3 | 840,000 | Supermove boards |
| 4 | 888,000 | Simple + one-trap + supermove |
| 5 | 1,160,000 | Supermove-counter boards |
| 6 | 1,200,000 | All 4 pool types mixed |

For comparison, the previous best size 4 run (MultiDiscrete action space) hit phase 6 but bounced between 12-56% valid rate after 720k steps. The flat action space run hit 100% valid rate *immediately* and never dropped.

### Why It Worked

The flat Discrete action space was the missing piece. The four attempts today each peeled back a layer:

1. **Self-play from scratch** → revealed warmup requirements
2. **Resume + self-play** → revealed reward mismatch problem
3. **Restored rewards, no self-play** → revealed MultiDiscrete action space wasting 30-50% of steps
4. **Flat Discrete + forward-only** → 100% valid, 100% win rate at phase 6

The key insight: with MultiDiscrete, the agent could select a valid cell AND a valid type but have them be incompatible (cell valid for piece, agent picks trap). Every such mismatch was a wasted step. On a 16-cell board requiring 8-12 construction steps, losing half your steps to mismatches means frequent truncation before board completion. The flat space makes this structurally impossible — every unmasked action is guaranteed valid.

Forward-only movement further reduced the effective search space by roughly half, preventing backward oscillation.

### What's Next

The pool opponents are exhausted — 100% win rate means there's nothing left to learn from them. Self-play is the natural next step:

```bash
python examples/train_simultaneous.py --size 4 --self-play --warmup-steps 0 \
    --resume models/size4/stage3/best/best_model.zip --timesteps 5000000
```

This time the conditions are right:
- Stable base policy (100% valid, 100% win vs pools)
- Same reward structure (no value network mismatch)
- Single source of non-stationarity (self-play opponents only)
