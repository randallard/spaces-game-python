# Plan: Refactor Training Infrastructure + Self-Play Curriculum

**Date:** February 17, 2026
**Status:** Approved

## Context

The training script (`train_simultaneous.py`) has grown to 1,053 lines with two large callbacks, 28 function parameters, and 30 CLI flags. We need to:
1. Clean it up so adding size 5, 6, etc. is trivial
2. Make it easy to re-run previous sizes with different parameters
3. Replace binary self-play/recovery mode-switching with a progressive self-play curriculum that backtracks on failure instead of abandoning self-play entirely

## New Directory Structure

```
spaces_game/
  callbacks/
    __init__.py                 # Exports all callbacks + utils
    pool_utils.py               # discover_pools, build_phase_map, constants
    opponent_progression.py     # OpponentProgressionCallback (extracted as-is)
    self_play.py                # SelfPlayCurriculumCallback (rewritten)
```

## Self-Play Curriculum Algorithm

Replaces the current binary block scheduling (self-play block OR pool recovery) with graduated difficulty:

- **Window level** (0..N): controls how many snapshots are active as opponents
  - Level 0: seed model only
  - Level k: seed + snapshots[0..k-1]
- **Advance**: pool win rate >= `advance_threshold` (default 0.70) for `min_steps_per_level` (default 50k) → level + 1
- **Backtrack**: pool win rate < `backtrack_threshold` (default 0.55) → level - 1 (one level at a time)
- **Pool recovery**: level 0 AND win rate < `backtrack_threshold` → switch to pool opponents (ratio=0.0) until win rate >= `recovery_win_rate` (default 0.70), then resume self-play at level 0
- **Snapshot quality gate**: only snapshot when pool win rate >= `snapshot_win_rate` (default: midpoint of backtrack and advance thresholds)
- **Seed model**: resumed model permanently in pool, never pruned

TensorBoard `self_play/` panel: `window_level`, `max_level`, `in_recovery`, `pool_snapshots`, `pool_win_rate`

## CLI Changes

Remove (with deprecation warnings): `--self-play-block-steps`, `--pool-recovery-steps`, `--min-pool-win-rate`, `--self-play-ratio`

Add:
- `--advance-threshold` (default 0.70) — win rate to advance window level
- `--backtrack-threshold` (default 0.55) — win rate to back up a level
- `--min-steps-per-level` (default 50,000) — minimum steps before advancing

Keep unchanged: `--self-play`, `--snapshot-freq`, `--pool-size`, `--warmup-steps`, `--recovery-win-rate`, `--snapshot-win-rate`, `--fog`, `--size`, all hyperparameter flags

## Implementation Steps

### Step 1: Extract pool_utils.py
- Move `discover_pools()`, `build_phase_map()`, `LEGACY_POOL_ORDER`, `DIFFICULTY_CHECKPOINTS` to `spaces_game/callbacks/pool_utils.py`
- Create `spaces_game/callbacks/__init__.py`
- Update imports in `train_simultaneous.py`
- Add `tests/test_pool_utils.py`

### Step 2: Extract OpponentProgressionCallback
- Move class to `spaces_game/callbacks/opponent_progression.py` (no behavior changes)
- Update imports in `train_simultaneous.py`

### Step 3: Rewrite SelfPlayCallback → SelfPlayCurriculumCallback
- Create `spaces_game/callbacks/self_play.py` with progressive window algorithm
- Keep seed model, quality gate, TensorBoard logging from current implementation
- Add window level tracking, advancement, backtracking, pool recovery fallback
- Add `tests/test_callbacks.py` with mocked phase_callback

### Step 4: Slim down train_simultaneous.py
- Remove extracted code
- Update CLI flags (deprecation warnings for removed, add new ones)
- Target: ~250-300 lines (arg parsing + env/model/callback wiring + learn call)

### Step 5: Update exports
- Add callbacks to `spaces_game/__init__.py`

## Files Modified
- `examples/train_simultaneous.py` — extract callbacks, slim down
- `spaces_game/__init__.py` — add callback exports

## Files Created
- `spaces_game/callbacks/__init__.py`
- `spaces_game/callbacks/pool_utils.py`
- `spaces_game/callbacks/opponent_progression.py`
- `spaces_game/callbacks/self_play.py`
- `tests/test_pool_utils.py`
- `tests/test_callbacks.py`

## Files Unchanged
- `spaces_game/simultaneous_play_env.py` — self-play methods stay in env (they need env internals)
- `examples/play_against_agent.py` — no changes needed
- All existing test files — no changes needed

## Verification
1. `pytest tests/` — all 179 existing tests + new tests pass
2. Pool-only: `python examples/train_simultaneous.py --size 2 --timesteps 10000` works
3. Self-play: `python examples/train_simultaneous.py --size 2 --self-play --timesteps 50000` uses new curriculum
4. Fog: `python examples/train_simultaneous.py --size 3 --fog --timesteps 10000` works
5. TensorBoard shows `self_play/window_level` metric during self-play runs
6. Old CLI flags (`--self-play-block-steps` etc.) print deprecation warnings but don't crash
