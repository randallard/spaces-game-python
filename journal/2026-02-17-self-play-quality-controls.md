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

---

## Take 2 Results: Still Destabilizing

The quality-controlled run reached ~880k steps with an encouraging recovery (35% → 65% → 70% win rate). But then the model entered its second self-play block and destabilized again:

| Step | Pool Win Rate | Explained Variance |
|------|--------------|-------------------|
| 880k | 70% | 0.55 |
| 1.648M | 40% | 0.26 |
| 1.656M | 45% | 0.13 |
| 1.664M | 30% | 0.19 |
| 1.672M | 55% | 0.34 |

The explained variance briefly went **negative** (-0.003 at 1.646M) — the value network completely lost predictive power. Episode reward crashed from ~135 to near zero. The quality controls prevented a full death spiral (win rate didn't go to single digits like the first attempt), but the model is oscillating around 40-55% instead of improving.

**Root cause**: the binary block scheduling is too coarse. 500k steps of pure self-play is enough to significantly destabilize the model, and 100k steps of pool recovery (even with the 80% threshold) isn't enough structural change to the approach.

---

## Rethinking: Self-Play as a Curriculum

The insight came from asking: what if we don't need to switch modes at all?

The snapshot pool is naturally ordered by difficulty — chronologically later snapshots come from more trained models. Instead of binary "all self-play" vs "all pool", we should treat the snapshot pool as its own curriculum:

1. Start with just the seed model (easiest self-play opponent)
2. Gradually include newer snapshots as the model proves it can handle the current set
3. If performance drops, back up one level (fewer snapshots) instead of abandoning self-play entirely
4. Only fall back to pool opponents as a last resort — when the model can't even beat the seed

This mirrors how the opponent phase progression already works (simple → one_trap → mixed → super_move → all), but applied to self-play. Each level is only incrementally harder.

### Why Binary Mode-Switching Fails

The current approach has two sudden transitions:
- **Into self-play**: ratio jumps from 0.0 to 1.0, opponent changes from JSON pool to random snapshots
- **Into recovery**: ratio jumps from 1.0 to 0.0, opponent changes back to JSON pool

Each transition is a distribution shift. The value network was calibrated for one opponent type and suddenly faces another. The progressive curriculum eliminates these sharp transitions — the opponent distribution changes gradually.

---

## Refactoring Plan

The training script has grown to 1,053 lines. Before implementing the self-play curriculum, we're extracting the callbacks into a proper module structure. This also makes it trivial to add new board sizes — just create `boards/sizeN/` with pool files and run `--size N`.

Full plan: [plans/2026-02-17-refactor-training-self-play-curriculum.md](../plans/2026-02-17-refactor-training-self-play-curriculum.md)

Key changes:
- Extract `OpponentProgressionCallback` → `spaces_game/callbacks/opponent_progression.py`
- Extract pool utilities → `spaces_game/callbacks/pool_utils.py`
- Rewrite `SelfPlayCallback` → `SelfPlayCurriculumCallback` with progressive window algorithm
- Slim `train_simultaneous.py` from 1,053 to ~250-300 lines

### Self-Play Curriculum Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--advance-threshold` | 0.70 | Pool win rate to advance window level |
| `--backtrack-threshold` | 0.55 | Pool win rate to back up one level |
| `--min-steps-per-level` | 50,000 | Minimum steps before level advancement |
| `--recovery-win-rate` | 0.70 | Win rate to exit pool recovery (last resort) |
| `--snapshot-win-rate` | (derived) | Quality gate for snapshot creation |

---

## Take 3: Progressive Window Curriculum (Feb 17)

After the refactoring, take 3 launched with the new `SelfPlayCurriculumCallback`:

```bash
python examples/train_simultaneous.py --size 3 --fog --self-play \
    --warmup-steps 0 \
    --resume models/size3/stage4/best/best_model.zip \
    --timesteps 7.5M \
    --advance-threshold 0.70 \
    --backtrack-threshold 0.55 \
    --min-steps-per-level 50k
```

### Results: Level Oscillation

The progressive curriculum was a clear improvement over binary block scheduling — no death spiral, no recovery mode entered. But a new pattern emerged: **level oscillation**.

| Step | Window Level | Win Rate | Event |
|------|-------------|----------|-------|
| 57K | 0 → 1 | 70%+ | Advanced |
| 74K | 1 → 0 | <55% | Backtracked |
| 131K | 0 → 1 | 70%+ | Advanced |
| 201K | 1 → 2 | 70%+ | Advanced (max level!) |
| 209K | 2 → 0 | <55% | Backtracked twice |
| 352K | 0 → 1 | 70%+ | Advanced |
| 360K | 1 → 0 | <55% | Backtracked |
| 498K | 0 → 1 | 70%+ | Advanced |
| 553K | 1 → 2 | 70%+ | Advanced (max level again!) |
| 586K | 2 → 0 | <55% | Backtracked twice |

The agent reached level 2 three times and got knocked back to 0 each time. Stopped at 660K steps with 85% pool win rate but level 0.

### Root Cause: No Min-Step Guard on Backtracking

The `min_steps_per_level` of 50K only applied to **advancing**. Backtracking was immediate — a single bad eval below 0.55 triggered it. With fog's noisy evaluations (win rates swing 45-85% between evals), one unlucky eval right after advancing caused instant retreat.

The level 2 → 0 cascades were particularly destructive: one bad eval at level 2 → backtrack to 1 → still in the same eval window → another bad eval → backtrack to 0. Two levels lost from what might have been transient noise.

---

## Take 4: Tuned Thresholds + Backtrack Guard (Feb 17)

Three changes for take 4:

1. **Wider threshold gap**: advance at 0.75, backtrack at 0.45 (was 0.70/0.55). The 0.45 backtrack is very hard to hit from noise alone — if the agent is at 0.45 against pool opponents, it's genuinely struggling.

2. **Min-step guard on backtracking**: Same `min_steps_per_level` (50K) now required before backtracking too. Prevents knee-jerk retreats from a single bad eval.

3. **Discord notifications**: New `DiscordNotifierCallback` for remote monitoring. Sends milestone alerts (phase advance, level changes, recovery) and periodic check-ins with trend analysis.

```bash
python examples/train_simultaneous.py --size 3 --fog --self-play \
    --resume models/size3/stage4/ppo_stage3_660000_steps.zip \
    --timesteps 10M \
    --warmup-steps 0 \
    --advance-threshold 0.75 \
    --backtrack-threshold 0.45 \
    --min-steps-per-level 50k \
    --discord-webhook "$DISCORD_WEBHOOK" \
    --discord-check-in 60
```

### Early Results (650K steps)

The run started with a value network recalibration problem. Resuming from take 3's checkpoint (trained at phase 6) but with curriculum reset to phase 0 caused a reward distribution mismatch:

| Step Range | Win Rate | Explained Variance | Phase |
|-----------|----------|-------------------|-------|
| 0-100K | 0-25% | -0.35 → 0.23 | 0-1 |
| 100-230K | 75-100% | 0.19-0.42 | 1-4 |
| 230-270K | 50-80% | 0.18-0.27 | 4-5 |
| 272K | **15%** | 0.17 | Backtrack 2→1→0 |
| 272-450K | 5-40% | 0.31-0.58 | Grinding at level 0 |
| 450-570K | 35-70% | 0.33-0.63 | Slow recovery |
| 560K | 65% | 0.33 | Phase 6 cleared |
| 570-650K | 30-85% | 0.17-0.39 | Volatile |

The backtrack at 272K with the min-step guard means the agent was below 0.45 for a sustained period — not noise. The value network (explained variance 0.17-0.20 recently) is struggling to predict returns accurately.

### Assessment

Take 4 hasn't collapsed like takes 1-2, and the backtrack guard prevented oscillation. But the core challenge remains: **the value network can't keep up with the non-stationarity of fog + self-play**. At 650K/10M steps (6.5%), explained variance is low and win rate is volatile.

### Discussion: What Could Improve Learning?

Several approaches under consideration:

**1. Don't reset curriculum on resume.** The biggest early destabilizer was forcing a phase-6-trained model back through phases 0-5. The model's value estimates were calibrated for hard opponents, then it faced trivial ones — reward distribution shift. Using `--start-opponent-phase 6` on resume would skip re-clearing phases.

**2. Lower learning rate for resumed self-play.** The value network from take 3 was well-calibrated (0.83-0.91 explained variance). Take 4's 3e-4 learning rate may be too aggressive — it's overwriting good value estimates too fast. Try 1e-4 to give the network more time to adapt incrementally.

**3. Larger rollout buffer.** With `--n-steps 2048` across 4 envs (512 per env), each update is based on relatively few episodes. Fog makes individual episodes noisy, so more data per update would reduce variance. Try `--n-steps 4096` or `--n-steps 8192`.

**4. Separate pool eval from self-play eval.** Currently the same `game_win_rate` metric drives both phase advancement and self-play level decisions. But the eval always runs against the current phase's pool opponents. A dedicated self-play evaluation (against the current window of snapshots) would give better signal for level transitions.

**5. Entropy coefficient tuning.** Currently 0.05. With fog + self-play, the optimal strategy space is wider — the agent needs to explore more diverse board constructions to handle varied opponents. Try 0.1 for more exploration.

---

## Take 5: All Five Improvements (Feb 17)

Implementing all five changes simultaneously:

### Changes Made

1. **Skip curriculum on resume** (`--start-opponent-phase 6`): The model already cleared all 6 phases. Forcing it back to phase 0 caused reward distribution mismatch and value network recalibration. Now starts at the final phase.

2. **Lower learning rate** (`--learning-rate 1e-4`): Take 3 had explained variance 0.83-0.91. Take 4's 3e-4 overwrote those calibrated estimates too aggressively. 1e-4 gives the network time to adapt incrementally.

3. **Larger rollout buffer** (`--n-steps 4096`): Doubles the data per PPO update (1024 per env instead of 512). With fog's noisy evaluations, more data per update reduces gradient variance.

4. **Separate self-play evaluation**: New `_evaluate_against_snapshots()` method in `SelfPlayCurriculumCallback`. Runs games against the current snapshot window every `eval_freq` steps. Level advance/backtrack decisions now use this self-play win rate instead of pool win rate. Pool win rate still drives phase progression and recovery decisions. Logged as `self_play/sp_eval_win_rate` in TensorBoard.

5. **Higher entropy** (`--ent-coef 0.1`): Doubles exploration. With fog + self-play, the agent needs to discover more diverse strategies to handle varied opponents.

### Additional fixes from earlier in the session

- **Min-step guard on backtracking**: Both advance and backtrack now require `min_steps_per_level` before transitioning. Prevents knee-jerk retreats from noisy evals.
- **Wider threshold gap**: Advance at 0.75, backtrack at 0.45 (was 0.70/0.55).
- **Console self-play status**: Eval printout now shows current level, max level, steps at level, and snapshot count.
- **Discord notifications**: Milestone alerts + periodic check-ins with trend analysis.

```bash
python examples/train_simultaneous.py \
    --size 3 --fog --self-play \
    --resume models/size3/stage4/ppo_stage3_660000_steps.zip \
    --start-opponent-phase 6 \
    --timesteps 10M \
    --warmup-steps 0 \
    --learning-rate 1e-4 \
    --ent-coef 0.1 \
    --n-steps 4096 \
    --advance-threshold 0.75 \
    --backtrack-threshold 0.45 \
    --min-steps-per-level 50k \
    --discord-webhook "$DISCORD_WEBHOOK" \
    --discord-check-in 60
```

### What to Watch

- **Explained variance**: Should stay higher than take 4's 0.17 thanks to lower LR and skipped curriculum reset. Target >0.50 sustained.
- **`self_play/sp_eval_win_rate`**: New TensorBoard metric showing direct performance against snapshot opponents. This drives level transitions now.
- **Level stability**: With wider thresholds + min-step guard + separate eval signal, levels should be much more stable. If the agent reaches level 2 it should stay there or progress.
- **Discord**: Watch for startup confirmation line: `DISCORD: Sent 'Training Started...'`. If it doesn't appear, the webhook URL wasn't passed.

---

## Consistent Pool Opponent Style Per Game (Feb 17)

**Problem**: `_select_opponent_board()` picked a random pool file and random board each round. Within a single 5-round game, the opponent might play simple in round 1, super_move_counter in round 2, etc. No real opponent plays like that — they have a consistent style.

**Fix**: At game reset, lock one pool file for the entire game. All 5 rounds draw boards from the same pool. The agent now faces a "simple player" or a "super_move player" — not a random mix.

This makes pool evaluation meaningful again: the agent must learn to read an opponent's style from early rounds and adapt. Previously, there was no pattern to detect.

**TODO (verify)**: Confirm this change improves pool eval signal quality. Compare pool win rates before and after in the next run that uses pool opponents (either pool-only training or recovery mode). If the agent's pool win rate becomes less volatile and more predictive of actual play quality, the fix is validated.
