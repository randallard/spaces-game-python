# Training Journal: Self-Play Quality Controls

**Date:** February 17, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Stage 4 — fog of war + self-play (size 3)

---

## How We Got Here

The Feb 16 fog training against pool opponents was a clean success: 100% valid rate, ~82% win rate by 256k steps, all 6 opponent phases cleared. The model learned to play well under partial observability without ever seeing full opponent boards.

Then we added self-play.

### First Fog + Self-Play Attempt (Feb 16-17)

```bash
python examples/train_simultaneous.py --size 3 --fog --self-play \
    --warmup-steps 0 \
    --resume models/size3/stage4/best/best_model.zip \
    --timesteps 5000000 \
    --self-play-block-steps 200000 \
    --pool-recovery-steps 100000 \
    --min-pool-win-rate 0.60
```

This used the block scheduling we designed on Feb 16. By ~1.83M steps, the picture was clear:

| Metric | Value | Expected |
|--------|-------|----------|
| Valid rate | 100% | 100% |
| Phase progression | 6/6 by 600k | Similar |
| Win rate | ~46% avg (volatile, 0.20-0.80 range) | >70% |
| Explained variance | 0.03-0.35, declining | >0.30 stable |
| Loss | 137-396, erratic | Stable |

The model reached phase 6 and maintained perfect construction — but game performance was at coin-flip levels. The 82% win rate from pool-only training had evaporated.

### Diagnosing the Death Spiral

Two structural problems with the original self-play setup:

**1. Snapshot quality degradation.** Snapshots were taken every 50k steps unconditionally. The initial snapshot was the good 82% model, but every subsequent snapshot came from the increasingly confused model training during self-play. By ~500k steps, the pool had rotated out the original strong model entirely and was filled with 10 mediocre snapshots. The agent trained against confused opponents, became more confused, got snapshotted, and the pool degraded further.

**2. Recovery didn't actually fix anything.** When pool win rate dropped below 60% and recovery kicked in, it ran for exactly 100k steps against pool opponents, then immediately switched back to self-play — regardless of whether the model had actually recovered. The self-play snapshot pool still held the same degraded models. Nothing prevented the next spiral.

The core issue: **nothing prevented low-quality snapshots from poisoning the opponent pool, and nothing ensured the model was ready before resuming self-play.**

---

## Three Fixes

### Fix 1: Recovery Win Rate Threshold

**Before**: Recovery ran for a fixed number of steps, then returned to self-play unconditionally.

**After**: Recovery requires both minimum steps AND a win rate threshold against pool opponents. The model must prove it can beat pool opponents at the target rate before self-play resumes.

New CLI flag: `--recovery-win-rate` (default 0.70)

The flow is now:
1. Self-play block (N steps)
2. Check pool win rate → if < `min-pool-win-rate`, enter recovery
3. Recovery runs for at least `pool-recovery-steps`, but won't exit until pool win rate >= `recovery-win-rate`
4. Back to self-play

If the model is stuck in the low-to-mid range, it stays in recovery until it's actually ready. No more premature returns to self-play.

### Fix 2: Snapshot Quality Gate

**Before**: Every `snapshot-freq` steps, save a snapshot unconditionally.

**After**: Only save snapshots when pool win rate is above a quality threshold. Bad models never enter the pool.

New CLI flag: `--snapshot-win-rate` (default: midpoint of `min-pool-win-rate` and `recovery-win-rate`)

With defaults of 60% and 70%, the quality gate is 65%. This filters out degraded models during self-play while still allowing snapshots when the model is adapting but hasn't regressed. The threshold can be set explicitly if the derived midpoint isn't right for a given run.

When a snapshot is skipped, the callback still reassigns existing pool opponents to training envs so self-play continues — it just doesn't add a bad model to the mix.

### Fix 3: Seed Model (Permanent Pool Member)

**Before**: When `--resume` was used with `--self-play`, the resumed model's weights initialized the policy but nothing kept a copy in the opponent pool. After enough snapshots, the original strong model was pruned out.

**After**: The resumed model is copied into the opponent pool as `seed_model.zip` and is never pruned. It's always available as an opponent candidate.

This guarantees the pool always has at least one known-good model. Even if every subsequent snapshot fails the quality gate, the agent still has a competent sparring partner in the seed model.

---

## Training Run: Take 2

```bash
python examples/train_simultaneous.py --size 3 --fog --self-play \
    --warmup-steps 0 \
    --resume models/size3/stage4/best/best_model.zip \
    --timesteps 7500000 \
    --self-play-block-steps 500000 \
    --pool-recovery-steps 100000 \
    --min-pool-win-rate 0.60 \
    --recovery-win-rate 0.80 \
    --snapshot-win-rate 0.65
```

Key parameter choices:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block steps | 500k (up from 200k) | Longer blocks give the model more time to develop coherent counter-strategies before checking pool performance |
| Recovery win rate | 80% | Close to the pool-only convergence of 82% — ensures the model is nearly fully recovered before self-play resumes |
| Snapshot win rate | 65% | Explicitly set rather than derived. Above the recovery floor (60%) but allows snapshots during active adaptation |
| Timesteps | 7.5M | More runway for the longer blocks and potentially extended recovery periods |
| Seed model | best_model.zip | The 82% pool-only model stays in the pool permanently |

### What to Watch For

1. **Recovery duration**: With 80% required, recovery might take a while if self-play destabilizes significantly. That's fine — better to recover fully than rush back. If stuck above 70% but below 80% for millions of steps, consider nudging down to 75%.

2. **Snapshot frequency**: With the quality gate, early self-play blocks may produce zero snapshots if the model dips below 65%. The pool will be small (seed + whatever passed the gate) but high quality.

3. **Pool composition over time**: Monitor how many snapshots are in the pool. If it's just the seed model for a long time, the agent is essentially doing self-play against its pre-fog-self-play self, which is still useful but limited.

4. **Win rate trajectory**: The key question is whether the model can maintain 80%+ pool performance while also improving against self-play opponents. If it oscillates between recovery and self-play without net progress, the block scheduling needs further tuning.

---

## Design Principles Reinforced

Three lessons keep recurring across this project:

1. **One source of non-stationarity at a time.** Fog already makes the environment harder to predict. Adding self-play on top doubles the non-stationarity. The quality controls don't eliminate this, but they provide guardrails: the model must demonstrate it hasn't lost its footing before facing the next challenge.

2. **Agents will find shortcuts.** If there's a way for bad models to pollute the training signal, they will. Unconditional snapshotting is the self-play equivalent of the validation bug from Feb 5 — the system assumed quality that wasn't there.

3. **Learning benefits from consistency.** The longer self-play blocks (500k) are the same principle as block scheduling itself: sustained exposure to one type of challenge, then sustained recovery, rather than rapid switching that prevents learning in either direction.

---

## TensorBoard Observability for Self-Play

One thing missing from the block scheduling implementation was visibility. The `SelfPlayCallback` printed transitions to stdout, but TensorBoard — the primary monitoring tool — had no idea self-play existed. The `curriculum/` metrics showed win rate and phase, but there was no way to correlate performance changes with self-play mode transitions without scrolling through terminal output.

### New `self_play/` TensorBoard Panel

Added `_log_metrics()` to `SelfPlayCallback` that records six metrics every step:

| Metric | What it shows |
|--------|--------------|
| `self_play/mode` | 1.0 during self-play blocks, 0.0 during recovery — square wave showing block transitions |
| `self_play/pool_win_rate` | Win rate against pool opponents (same data as `curriculum/game_win_rate` but always in the self-play panel) |
| `self_play/block_count` | Cumulative self-play blocks completed |
| `self_play/recovery_count` | Cumulative recovery periods triggered |
| `self_play/pool_snapshots` | Number of snapshots currently in the opponent pool |
| `self_play/steps_in_block` | Progress through current block/recovery period (resets at transitions) |

The `mode` chart is the most useful — overlay it with `pool_win_rate` and you can immediately see: "win rate dropped during self-play block 2, recovered during pool recovery, held steady through self-play block 3." No more grepping terminal logs.

TensorBoard groups metrics by prefix, so these automatically appear in their own panel separate from `curriculum/`, `train/`, and `eval/`.
