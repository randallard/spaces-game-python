# Training Journal: Retraining Results, Deterministic Retries, and Hardware Limits

**Date:** February 12, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Size 2 & 3 retraining complete, planning size 4 & 5

---

## Starting Point

The previous session (Feb 11) ended with a decision to retrain both size 2 and size 3 from scratch. Three fixes had landed — first-visit scoring, no-revisit action masking, and full-path board validation — but the running training process had loaded the old `is_board_playable()` at startup. Its 100% valid rate was against rules that would pass a 1-step board. Fresh training was the right call.

Commands kicked off:
```bash
# Size 2
python examples/train_simultaneous.py \
    --size 2 --board-library new_boards_2.json \
    --timesteps 5000000 --min-phase-steps 100000

# Size 3
python examples/train_simultaneous.py \
    --size 3 --board-library new_boards_3.json \
    --timesteps 5000000 --min-phase-steps 100000
```

---

## Size 2 Results — All Phases Cleared

Size 2 finished first. Full curriculum completed in 5M steps:

| Phase | Type | Reached at |
|---|---|---|
| C0→C3 | Construction scaffolding | 8k → 1.21M |
| Construction done | | 1.61M |
| Opponent 0 → 1 | simple → one_trap | 2.01M |
| Opponent 1 → 2 | → simple + one_trap mix | 2.48M |
| Opponent 2 → 3 | → super_move | 2.89M |
| Opponent 3 → 4 | → all types | 3.29M |
| Opponent 4 → 5 | Graduated | 3.82M |

**Valid rate: 100%** throughout. Rock solid — the new full-path validation (piece visits every row, legal finish) never dipped.

**Win rate at phase 5: ~45-60%**, fluctuating around 50%. Not surprising for size 2 — the boards are small enough that there isn't much room for strategic advantage once both players know all opponent types. The important thing is it cleared every phase's 80% threshold to advance.

All three difficulty checkpoints saved: beginner (phase 0), intermediate (phase 2), expert (training end).

---

## Size 3 Results — All Phases Cleared

Size 3 needed more of its budget for construction (7 phases vs 4 for size 2 — bigger boards need more scaffolding) but still cleared the full opponent curriculum:

| Phase | Type | Reached at |
|---|---|---|
| C0→C6 | Construction scaffolding | 8k → 2.41M |
| Construction done | | 2.81M |
| Opponent 0 → 1 | simple → one_trap | 3.21M |
| Opponent 1 → 2 | → simple + one_trap mix | 3.61M |
| Opponent 2 → 3 | → super_move | 4.01M |
| Opponent 3 → 4 | → all types | 4.41M |
| Opponent 4 → 5 | Graduated | 4.81M |

**Valid rate: 95-100%**, averaging ~98%. Not the rock-solid 100% that size 2 achieved — with a bigger action space, the agent occasionally produces a board that misses a row. It wobbles but never collapses.

**Win rate at phase 5: ~60-85%**, averaging around 70%. Stronger strategic separation than size 2, which makes sense — more board real estate means more room for trap placement and path complexity to matter.

All three difficulty checkpoints saved. Phase 5 was reached at step 4.81M, just 190k steps before the budget ran out — tight but it made it.

---

## The Deterministic Retry Problem

While reviewing how the difficulty levels work in the inference server, Ryan spotted something: retries are worthless in deterministic mode.

The inference server retries up to 5 times when the agent produces an invalid board. But in deterministic mode, the model always picks its highest-probability action. Same inputs + same model + deterministic = same output. Retrying just does the same thing 5 times.

This matters because three of the six skill levels use deterministic sampling (beginner_plus, intermediate_plus, advanced_plus).

**Decision:** On retry, fall back to stochastic sampling. First attempt uses whatever mode the skill level calls for (deterministic for the `_plus` levels). If that produces an invalid board, subsequent attempts sample from the policy distribution, giving them a real chance of finding a different — and hopefully valid — board. Ryan will implement this on the inference server machine.

---

## Hardware Limits — No Parallel Training

Ryan asked whether the training machine (his laptop) could handle two training runs simultaneously for the upcoming size 4 and size 5 work.

The answer: no.

```
CPU: Intel Core i5-3470 @ 3.20GHz — 4 cores, no hyperthreading
RAM: 7.7GB total, 3.0GB available
Current training process: 1.8GB resident, 99% CPU
SubprocVecEnv: 4 worker processes per training run
```

One training run already saturates a core and uses ~1.8GB. A second run would need another ~1.8GB (more for larger board sizes) plus 4 more worker processes competing for 4 physical cores. With only 3GB available and 1GB of swap already in use, the machine would be deep into swap thrashing. Both runs would slow to maybe 30-40% of normal speed — taking longer than sequential.

Size 4 and 5 will also have bigger observation/action spaces, so memory usage per run will be *higher* than the 1.8GB seen for size 3. Training them sequentially is the practical path on this hardware. Reducing the number of parallel envs (`n_envs=2` instead of 4) might help keep size 5 memory in check.

---

## Current State

Both size 2 and size 3 models are retrained with all three validation fixes and ready for deployment to the inference server. The model files:

```
models/size2/stage3/difficulty/beginner.zip
models/size2/stage3/difficulty/intermediate.zip
models/size2/stage3/difficulty/expert.zip

models/size3/stage3/difficulty/beginner.zip
models/size3/stage3/difficulty/intermediate.zip
models/size3/stage3/difficulty/expert.zip
```

---

## What's Next

- Deploy retrained models to inference server, verify via Node frontend
- Implement stochastic fallback on retry for deterministic skill levels
- Play-test difficulty separation: beginner vs intermediate vs expert
- Begin size 4 training (sequential on this machine)
- Stage 4 (fog of war) design — still on the roadmap but training larger board sizes comes first

---

*Four cores, no hyperthreading, and 8 gigs of RAM — you work with what you've got.*
