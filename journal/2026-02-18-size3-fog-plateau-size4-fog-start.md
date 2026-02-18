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
