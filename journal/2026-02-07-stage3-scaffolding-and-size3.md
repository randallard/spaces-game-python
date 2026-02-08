# Training Journal: Stage 3 Scaffolding and Size 3 Pipeline

**Date:** February 7, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Size 2 pipeline validation, Size 3 Stage 3 with construction scaffolding

---

## Starting Point

We had a complete size 2 pipeline (Stage 2 reverse curriculum + Stage 3 simultaneous play) and a size 3 agent that had completed Stage 1 and Stage 2. The goal was to validate the size 2 pipeline end-to-end, then apply lessons to size 3.

---

## Size 2 Pipeline Results

Ran the full pipeline (`train_size2_pipeline.py`). First had to fix a `ModuleNotFoundError` -- the script used `from examples.train_reverse_curriculum import train` which doesn't work when running from within the `examples/` directory. Fixed with `sys.path` insertion.

### Stage 2 (Reverse Curriculum): Obsolete with Trap Limit

Stage 2 ran but never advanced past Phase 0. Valid rate was 100% but win rate stuck at 16-34%, well below the 75% threshold to advance. This isn't a bug -- it's a consequence of the trap limit rule (`max_traps = board_size - 1`). With balanced boards, blind play against random opponents tops out around 40-65% win rate. You can't reliably beat an opponent you can't see.

**Conclusion:** Stage 2 is no longer useful as a standalone stage. The trap limit makes its advancement criteria unreachable.

### Stage 3 (Simultaneous Play): Success

Stage 3 advanced through all 5 opponent phases, reaching Phase 4 (all opponent types mixed) by 54k steps. Final win rate settled at 40-65%, which Ryan confirmed makes sense -- in 5 rounds of blind play against mixed opponents, there's no way to consistently predict what's coming.

Playing against the final agent confirmed it builds valid, strategic boards and adapts across rounds.

---

## Applying Lessons to Size 3

### Cleaning the Board Library

`new_boards_3.json` had 14 boards, but 3 had 3 traps -- violating the trap limit of 2 for size 3. Removed them, leaving 11 clean boards.

### Creating Size 3 Opponent Pools

Size 2 had well-organized opponent pools in `boards/size2/`. Size 3 had none. Created four pools:

| Pool | Boards | Description |
|------|--------|-------------|
| `simple.json` | 3 | Straight paths (col 0/1/2), 0 traps |
| `one_trap.json` | 3 | Straight path + 1 adjacent trap |
| `super_move.json` | 3 | Supermove boards (trap on piece = move again) |
| `super_move_counter.json` | 3 | 2 traps with crossover patterns |

All validated via `is_board_playable()`.

---

## Size 3 Construction: The Problem

Ran Stage 3 for size 3 without scaffolding. Valid rate: **0% after 17k steps**, climbing to just **8% at 27k steps**. For comparison, size 2 hit 100% valid rate by 8k steps.

The 3x3 grid (9 cells, sequences up to 7 moves) is too large for random exploration to discover valid boards. The agent needs help.

---

## The Solution: Construction Scaffolding

Instead of a separate Stage 2, we built reverse-curriculum scaffolding directly into `SimultaneousPlayEnv`. One training run now handles both construction learning and opponent curriculum:

### Two-Phase Curriculum

**Construction phases** (advance on valid_rate >= 95%):
- Phase C0: Pre-fill all but goal -- agent just signals "done"
- Phase C1: Pre-fill all but last piece + goal
- Phase C2: Pre-fill all but last 2 pieces + goal
- ...
- Phase CN: No pre-fill -- agent builds from scratch

**Opponent phases** (advance on game_win_rate >= 70%):
- Phase 0-4: Progressive opponent difficulty (existing logic)

Construction runs first. Once the agent builds valid boards from scratch, opponent phases begin.

### Implementation

Added to `SimultaneousPlayEnv`:
- `board_library_path` parameter to load library boards
- `scaffolding_moves_to_remove` state (-1=disabled, 0=easiest, N=harder)
- `_prefill_from_library()` -- picks random board, pre-fills partial sequence into construction state
- `set_scaffolding()` -- callable via `SubprocVecEnv.env_method()`

Extended `OpponentProgressionCallback`:
- Tracks `construction_phase` and `in_construction_mode`
- Evaluates valid_rate for construction, game_win_rate for opponents
- Logs both phases to TensorBoard

Added `--board-library` CLI arg to `train_simultaneous.py`.

---

## The Bug Hunt

What should have been a straightforward feature addition turned into a debugging session.

### Bug 1: Round Observation Overflow

`round` observation used `Discrete(5)` (values 0-4), but after the 5th round `current_round` becomes 5. Terminal observation sent value 5 to SB3's `F.one_hot()`which expects values < 5.

**Fix:** `min(self.current_round, self.ROUNDS_PER_GAME - 1)` in `_get_observation()`.

### Bug 2: Eval Env Space Mismatch

The callback's eval env used default `max_construction_steps=20` while training envs used `board_size * 10 = 30`. Different `Discrete` bounds between envs caused SB3 preprocessing failures.

**Fix:** Pass `max_construction_steps = board_size * 10` to callback's eval env.

### Bug 3: Construction Step Overflow (The Real One)

After fixing bugs 1 and 2, still getting `RuntimeError: Class values must be smaller than num_classes`. Narrowed it down:

| Configuration | Result |
|--------------|--------|
| DummyVecEnv(1 env), 100 steps | OK |
| DummyVecEnv(2 envs), 500 steps | FAIL |
| SubprocVecEnv(1 env), 500 steps | FAIL |
| Manual stepping, 500 steps | OK |

Spent a while adding debug instrumentation before Ryan said: *"take a second and check what's different between this and the size 2 version? I didn't think it would be this hard."*

That reframe led to the answer. The only difference is scaffolding. Scaffolding pre-fills `construction_step` to a non-zero value (e.g., 6 for a 7-move board). The agent can then take up to `max_construction_steps` (30) additional actions, each valid placement incrementing `construction_step`. So `construction_step` can reach 6 + 30 = 36, but `Discrete(31)` only allows values 0-30.

Without scaffolding this can't happen -- `construction_step` starts at 0 and maxes at 30, just fitting in `Discrete(31)`.

**Fix:** `min(self.construction_step, self.max_construction_steps)` in `_get_observation()`. One line, same pattern as the round clamp.

---

## Key Takeaways

1. **Stage 2 is obsolete** -- with the trap limit, balanced boards can't achieve 75% win rate in blind play. Construction scaffolding within Stage 3 replaces it entirely.

2. **Size matters exponentially** -- size 2 (4 cells) discovers valid boards through random exploration in ~8k steps. Size 3 (9 cells) barely reaches 8% at 27k. Scaffolding is essential for larger boards.

3. **Discrete observation overflow is subtle** -- `F.one_hot` crashes when values exceed space bounds. This only shows up during `model.learn()` (not manual stepping) because gymnasium doesn't validate observations on `step()`, but SB3's preprocessing does. Scaffolding shifted the baseline of `construction_step`, making overflow possible where it wasn't before.

4. **"What's different from the working version?"** -- Ryan's question cut through an hour of debugging. When a parameterized system breaks for new parameters, compare against the working configuration rather than adding instrumentation.

---

## Files Changed

- `spaces_game/simultaneous_play_env.py` -- Added scaffolding (`board_library_path`, `set_scaffolding()`, `_prefill_from_library()`), clamped `round` and `construction_step` observations
- `examples/train_simultaneous.py` -- Extended callback with construction + opponent dual curriculum, added `--board-library` CLI arg
- `examples/train_size2_pipeline.py` -- Fixed import paths
- `new_boards_3.json` -- Cleaned from 14 to 11 boards (removed 3-trap violations)
- `boards/size3/simple.json` -- Created (3 boards)
- `boards/size3/one_trap.json` -- Created (3 boards)
- `boards/size3/super_move.json` -- Created (3 boards)
- `boards/size3/super_move_counter.json` -- Created (3 boards)
- Old `models/size2/stage3/` checkpoints -- Cleaned up 71 pre-trap-limit files

---

## What's Next

Size 3 training is running:
```bash
python examples/train_simultaneous.py --size 3 --board-library new_boards_3.json --timesteps 500000
```

This single command handles both construction scaffolding (replacing Stage 2) and opponent curriculum (Stage 3). Watch for:
- Construction phases advancing on valid_rate >= 95%
- Transition to opponent phases once building from scratch
- Opponent phases advancing on game_win_rate >= 70%

Monitor with: `tensorboard --logdir logs/size3_stage3/`

---

*"What's different from the version that works?" -- the question that saves hours.*
