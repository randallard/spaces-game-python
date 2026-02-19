# Training Journal: Size 3 Fog Plateau, Size 4 Fog Start

**Date:** February 18, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Stage 4 — fog of war

---

## Size 3 Fog + Self-Play Take 5: Final Assessment

Take 5 ran from Feb 17-18 with all five improvements (skip curriculum reset, LR 1e-4, n-steps 4096, ent-coef 0.1, separate self-play eval, Discord notifications). Stopped at 1.73M/10M steps.

### What Went Right

- **Reached level 4** — best of any take. Previous takes oscillated at levels 0-2.
- **100% valid rate throughout** — strict masking rock solid as always.
- **Phase 6 maintained** — `--start-opponent-phase 6` eliminated the value network recalibration problem from take 4.
- **Discord notifications working** — level advances, check-ins, and milestone alerts all delivered.
- **No death spiral** — min-step guard + wider thresholds prevented the oscillation that killed takes 1-3.

### What Went Wrong

The agent plateaued. Win rate by 200K windows tells the story:

| Window | Avg WR | Phase |
|--------|--------|-------|
| 0-200K | 74% | Strong start, climbing levels |
| 200-400K | 35% | Crater from rapid level advances |
| 400-600K | 29% | Bottom — recovery period |
| 600-800K | 65% | Recovery, productive learning |
| 800-1M | 62% | Peak sustained performance |
| 1-1.2M | 53% | Gradual decline begins |
| 1.2-1.4M | 52% | Flat |
| 1.4-1.6M | 40% | Another dip (level 4 advance) |
| 1.6-1.73M | 52% | Partial recovery, still volatile |

The explained variance collapsed in the final window (0.24 avg, touching near 0). The value network can't predict returns well enough for the agent to improve. It's been churning at 40-55% for 700K+ steps with no upward trend.

### Root Cause

Level 4 = 5 opponents in the self-play window (seed + 4 snapshots). This is too much opponent diversity for the current MLP architecture to handle simultaneously. The agent can beat any individual opponent but can't generalize across all five. Each PPO update tries to satisfy conflicting strategies, and the value function averages them into mush.

### Possible Next Steps for Size 3 Fog Self-Play

1. **Cap max window level at 3** — let the agent converge deeper with fewer opponents
2. **Larger network** — more capacity to represent diverse strategies
3. **Longer evaluation windows** — 20+ episodes to reduce noise in level decisions
4. **Fine-tune from best model with lower LR** — extract more from the existing learned features

Decision: **Pause size 3 fog self-play** and move to size 4 fog pool-only. The size 3 fog best model is good enough for deployment testing.

---

## Size 4 Fog: Starting Fresh

### Why From Scratch

Two reasons we can't resume from the existing size 4 stage 3 model:

1. **Obs space mismatch** — `--fog` adds `fog_outcomes` to the observation space. The stage 3 model has a different network architecture (no fog input neurons). Can't resume across that boundary.
2. **No non-fog base needed** — Size 3 fog from scratch converged at 82% in just 256K steps. No evidence that pre-training without fog helps — the model never develops full-information dependencies to unlearn.

### Training Command

```bash
python examples/train_simultaneous.py \
    --size 4 --fog \
    --timesteps 10M \
    --discord-webhook "$DISCORD_WEBHOOK" \
    --discord-check-in 30
```

Using default hyperparameters (LR 3e-4, ent-coef 0.05, n-steps 2048). Size 3 fog converged with defaults, so starting there for size 4. Will tune if needed.

### What to Watch

- **Phase progression speed** — size 4 has 16 cells (vs 9 for size 3), so construction is harder. Expect slower phase advancement.
- **Valid rate** — should be 100% with strict masking. If it drops, something is wrong.
- **Win rate at phase 6** — target 70%+ sustained before considering self-play.
- **Discord check-ins** — 30-minute intervals for closer monitoring of the new size.

### Plan After Pool Convergence

Once all phases cleared with stable win rate:
1. Stop pool-only training
2. Resume with `--self-play` from the converged fog model
3. Apply lessons from size 3: wider thresholds, min-step guards, maybe cap max level

---

## Deployment Plan

The size 3 fog best model is ready for deployment testing:
- Path: `models/size3/stage4/best/best_model.zip`
- Trained through level 4 self-play, ~65% peak pool win rate
- 100% valid construction rate

Plan to deploy size 3 fog models to the inference server and test against real human play. This will give us ground truth on how the fog model performs beyond pool opponents and self-play metrics.

Size 4 fog models will follow once pool training converges.

---

## Inference Server: Indexed Model Selection & Temperature Control

With level_advancement models saved at each self-play level transition, the existing skill_level system (beginner/intermediate/advanced) was too rigid for experimentation. Two changes make all trained checkpoints accessible through the inference server.

### Indexed Model Selection

Added a `GET /models` endpoint that returns a flat indexed list of every discovered model — the standard skill-level checkpoints plus all `level_advancement/*.zip` files. Each entry includes board size, stage, label (from filename), path, and whether it uses fog.

A new optional `model_index` field on `/construct-board` selects a specific model by index, bypassing the skill_level/agent_type lookup entirely. Board size is validated against the model's metadata. The existing skill_level flow is untouched when model_index is omitted.

### Temperature-Controlled Stochasticity

Previously, stochasticity was binary: deterministic (argmax) or not (sample from policy distribution). Added an optional `temperature` field (0.0-2.0) that gives fine-grained control over action sampling.

The implementation reaches into SB3's policy internals — gets the action distribution via `policy.get_distribution()`, scales the logits by `1/temperature`, then samples via `torch.multinomial`. Action masks are preserved through the scaling (masked logits at -1e8 stay negligible at any temperature).

- `temperature=0.0` or omitted: deterministic
- `temperature=0.5`: sharper, more confident moves
- `temperature=1.0`: standard stochastic (matches training distribution)
- `temperature=1.5`: more exploratory, occasionally surprising boards

This enables testing how different models behave across the confidence spectrum — useful for finding the sweet spot between predictable play and variety for human opponents.

---

## February 19 Update: SP Win Rate Gate, Size 5 Fog-First

### Size 3 Fog + Self-Play Take 6: Success

Switched the snapshot quality gate and skill milestones from pool win rate to SP eval win rate. This was the missing piece — takes 1-5 all used pool WR as the benchmark, but during self-play the agent trains against snapshots, not pool opponents. Pool WR is a different distribution.

Results: reached level 10 (pool_size cap) by 2M steps, sustained ~90% SP eval WR through 7.5M steps. Pool WR sat at 40-50% — expected and no longer the benchmark. No plateau, no collapse, no recovery episodes. The model handles 11 opponents (seed + 10 snapshots) comfortably.

Stopped at ~7.5M steps — the model was no longer learning. SP WR oscillating around 90% with no upward trend. Explained variance bouncing 0.27-0.65. Converged at the pool_size=10 ceiling.

### Size 4 Fog Pool-Only: Converged

Size 4 fog pool-only training completed. All phases cleared, production models (beginner/intermediate/expert) deployed to `models/size4/stage4/`. Level advancement snapshots saved with datetime prefixes. Ready for self-play when revisited.

### Level Advancement Auto-Save

Added `_save_level_advancement()` to the self-play callback. Saves a model snapshot at every level transition (advance before/after, backtrack, recovery enter/exit) with datetime-prefixed filenames so models accumulate across runs without overwriting.

### Size 5: Skipping Stage 3, Going Fog-First

Generated size 5 pool boards (`boards/size5/` — simple, mixed_traps, super_move, super_move_counter). Instead of training Stage 3 (full reveal) first, going straight to Stage 4 (fog) to test whether fog-only training is sufficient. If it works, we can skip Stage 3 entirely for new sizes — the agent never develops full-information dependencies to unlearn.

```bash
python examples/train_simultaneous.py \
    --size 5 \
    --fog \
    --timesteps 10000000 \
    --learning-rate 1e-4 \
    --ent-coef 0.1 \
    --n-steps 4096
```

Rationale: size 3 fog from scratch converged at 82% in 256K steps without any Stage 3 pretraining. If size 5 fog converges similarly, Stage 3 becomes unnecessary overhead for the training pipeline. Focus shifts to UI enhancements while sizes 4 (self-play) and 5 (pool) train.

### Size 5 Fog + Self-Play: Converged

Size 5 fog pool-only converged quickly (all 7 phases by 400K steps, 72-77% WR). Resumed with self-play — reached level 10 by 2M steps, ~90% SP WR. Same convergence pattern as size 3. Production models deployed to `models/size5/stage4/`.

### Size 6: Pool Boards and Fog Training Started

Generated size 6 pool boards (`boards/size6/` — 4 files, 4 boards each). Same structure as size 5: simple, mixed_traps (2-4 traps), super_move (1 supermove each), super_move_counter (4 traps with redirects). Max traps = 5 (board_size - 1).

Started size 6 fog pool-only training:

```bash
python examples/train_simultaneous.py \
    --size 6 --fog \
    --timesteps 10000000 \
    --learning-rate 1e-4 \
    --ent-coef 0.1 \
    --n-steps 4096 \
    --discord-webhook "$DISCORD_WEBHOOK" \
    --discord-check-in 30
```

Size 6 has 36 cells — biggest board yet. Expect slower phase progression and longer convergence than size 5 (25 cells).
