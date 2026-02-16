# Retraining Plan: Size 2 and Size 3 with Current Architecture

**Created:** February 15, 2026

## Why Retrain?

The current size 2 and size 3 deployed models were trained with the **old MultiDiscrete action space** (Feb 12). The codebase has since been reworked (Feb 14) with:

1. **Flat Discrete action space** — `Discrete(2 * n_cells + 1)` instead of `MultiDiscrete([n_cells, 2, 2])`. Every unmasked action is guaranteed valid. Zero wasted steps.
2. **Forward-only movement** — piece can only move forward or sideways, eliminating backward oscillation.
3. **Strict BFS masking** — invalid boards are structurally impossible at the action mask level.
4. **Self-play with pool mixing** — `--self-play-ratio 0.5` prevents specialization collapse.

The old models are **incompatible** with the current env (different action/observation spaces). They still work in the inference server because the server loads them with their saved architecture, but they represent an older, less capable training approach.

Retraining brings sizes 2 and 3 up to the same standard as size 4: flat action space, strict masking, self-play mixing, and proper difficulty tier snapshots.

## Training Steps

### Size 2 — COMPLETE

Size 2 is small (4 cells, `Discrete(9)` action space). Convergence is fast.

**Phase 1: Pool opponents** ✅
```bash
python examples/train_simultaneous.py --size 2 --timesteps 500000
```
- All 7 phases cleared by 98k steps. 100% valid rate throughout.

**Phase 2: Self-play with pool mixing** ✅
```bash
python examples/train_simultaneous.py --size 2 --self-play --self-play-ratio 0.5 \
    --warmup-steps 0 --resume models/size2/stage3/best/best_model.zip \
    --win-rate-threshold 0.55 --timesteps 1000000
```
- **Note**: Size 2 requires `--win-rate-threshold 0.55` (default 0.70 is unreachable due to high variance on 4-cell boards).
- Phase 6 reached by 288k steps. Asymptotic at ~50% win rate (theoretical equilibrium for size 2).
- All 5 difficulty tiers saved. Early-stopped at 624k steps.

**Deploy** ✅ — Models copied and committed.

### Size 3

Size 3 is moderate (9 cells, `Discrete(19)` action space).

**Phase 1: Pool opponents**
```bash
python examples/train_simultaneous.py --size 3 --timesteps 2000000
```
- Expected: 100% valid rate, all opponent phases cleared within 1.5M steps
- Previous training (old action space) cleared all phases in 1.84M steps; flat action space should be faster

**Phase 2: Self-play with pool mixing**
```bash
python examples/train_simultaneous.py --size 3 --self-play --self-play-ratio 0.5 \
    --warmup-steps 0 --resume models/size3/stage3/best/best_model.zip --timesteps 2000000
```
- Expected: asymptotic within 1M steps
- Early-stop if plateau detected

**Deploy: Copy difficulty tiers**
```bash
cp models/size3/stage3/difficulty/beginner.zip models/size3/stage3/beginner.zip
cp models/size3/stage3/difficulty/intermediate.zip models/size3/stage3/intermediate.zip
cp models/size3/stage3/difficulty/expert.zip models/size3/stage3/expert.zip
```

## Execution Order

Train sequentially (hardware can only handle one run at a time):

1. Size 2 Phase 1 (~30 min)
2. Size 2 Phase 2 (~1 hr, early-stop if asymptotic)
3. Size 3 Phase 1 (~2-3 hrs)
4. Size 3 Phase 2 (~2-3 hrs, early-stop if asymptotic)

**Total estimate**: ~6-8 hours

## Deployment

After each size completes:
1. Copy difficulty `.zip` files to `models/size{N}/stage3/`
2. `git add` and commit the 3 model files
3. `git push` to trigger Railway redeployment

## Success Criteria

- 100% valid rate throughout training
- All opponent phases cleared (phase 4 for sizes 2/3)
- Self-play asymptotic win rate ≥60% against pool eval
- All 5 difficulty tiers saved (beginner through advanced_plus)
- Inference server loads new models and serves boards correctly

## Rollback

If retraining produces worse models, the old ones are still in git history. Revert the model file commits to restore previous versions.
