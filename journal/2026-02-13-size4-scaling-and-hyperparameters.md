# Training Journal: Size 4 Scaling, Hyperparameter Tuning, and Looking Ahead

**Date:** February 13, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Size 4 first attempt, hyperparameter tuning, pool simplification

---

## Starting Point

Size 2 and size 3 retraining (from the Feb 11-12 session) finished successfully. Both cleared all opponent phases with the new full-path validation in place:

- **Size 2**: All phases in 3.82M of 5M steps. 100% valid rate throughout. Win rate ~50% at the final phase (expected for small boards).
- **Size 3**: All phases in 4.81M of 5M steps. 95-100% valid rate. Win rate ~70% at final phase.

All six difficulty checkpoints saved (beginner/intermediate/expert for each size). The deterministic retry problem was identified — retries do nothing when the model always picks the same action — and the plan is to fall back to stochastic sampling on retry. Ryan will implement that on the inference server machine.

With those confirmed working, Ryan shifted focus to size 4.

---

## Board Curation and the `--save-to` Flag

Ryan started building size 4 boards using the interactive builder (`spaces-game test --interactive --size 4`). The existing flow prompted for a save path every time, so we added a `--save-to` flag that auto-appends to a specified file without confirmation prompts:

```bash
spaces-game test --interactive --size 4 --save-to new_boards_4.json
spaces-game test --interactive --size 4 --save-to boards/size4/00_simple.json
```

The duplicate detection and JSON formatting still work — it just skips the "Would you like to save?" and "File exists, append?" prompts.

Ryan built 20 demonstration boards for the scaffolding library (`new_boards_4.json`) and created 5 opponent pools: simple (4 boards), one_trap (18 boards), multi_trap (6 boards), super_move (12 boards), and super_move_counter (6 boards).

---

## Size 4 First Training Attempt — Stalled

Ryan kicked off the first size 4 run with the same parameters that worked for sizes 2 and 3:

```bash
python examples/train_simultaneous.py \
    --size 4 --board-library new_boards_4.json --timesteps 8000000
```

The dynamic pool discovery (`discover_pools`) found 5 pools and `build_phase_map` generated 9 opponent phases (up from 5 for sizes 2/3). The solo-then-cumulative-mix pattern means each new pool adds 2 phases.

**Construction completed quickly** — phases 0 through 8 in 608k steps. The agent learned to build valid size-4 boards without trouble.

**Then it stalled.** Opponent phase 0 — just 4 straight-path simple boards — for 5.5 million steps. Win rate bounced between 0% and 65% with no upward trend. The 70% threshold to advance was never reached.

Worse, the training was destabilizing:

| Steps | Avg Reward |
|---|---|
| 1M | +11.7 |
| 3M | +2.1 |
| 5M | -48.2 |
| 6M | -179.5 |

The reward got *worse* over time, and the valid rate started wobbling (dropping to 67% before recovering). The model was forgetting how to build valid boards while failing to learn how to win.

---

## Root Cause: Hyperparameters Don't Scale

The hyperparameters that worked for size 2 (4 cells) and size 3 (9 cells) don't scale to size 4 (16 cells):

- **`learning_rate=3e-4`** — too aggressive for the larger action space, causing oscillation instead of convergence
- **`ent_coef=0.05`** — too little exploration; the agent needs to try more diverse board layouts to find winning strategies
- **`n_steps=2048`** (512 per env) — too short for longer episodes on 4x4; gradient estimates are too noisy
- **9 opponent phases** — the budget can't cover this many phases even if the agent could advance

---

## Fix 1: CLI Hyperparameter Flags

Added `--learning-rate`, `--ent-coef`, `--n-steps`, and `--batch-size` CLI arguments to `train_simultaneous.py`. All default to the values that work for size 2/3, so existing training is unaffected. For size 4+, you override:

```bash
python examples/train_simultaneous.py --size 4 --board-library new_boards_4.json \
    --timesteps 10000000 --min-phase-steps 100000 \
    --learning-rate 1e-4 --ent-coef 0.1 --n-steps 4096
```

The banner now prints the active hyperparameters so you can verify what's running.

---

## Fix 2: Simplified Opponent Pools

Merged `01_one_trap.json` (18 boards) and `02_multi_trap.json` (6 boards) into a single `01_mixed_traps.json` (4 boards). Trimmed all pools to 4 boards each. The agent doesn't need 18 variations of one-trap boards to learn that traps exist — 3-4 well-chosen examples per concept is what worked for sizes 2 and 3.

New pool structure:

```
boards/size4/
  00_simple.json             (4 boards - straight paths, one per column)
  01_mixed_traps.json        (4 boards - 1-3 traps, various placements)
  02_super_move.json         (4 boards - trap on own cell, different columns)
  03_super_move_counter.json (4 boards - redirect traps, left and right)
```

4 pools = 7 opponent phases (down from 9). With tuned hyperparameters and a 10M step budget, this should be manageable.

---

## The Bigger Picture: Scaling and Self-Play

Ryan raised an important point during the size 4 discussion: as boards get bigger, hand-curating demonstration boards gets harder. The combinatorial space explodes, and no reasonable set of 15-20 boards can cover all the strategies a 4x4 or 5x5 board allows.

The scaffolding boards serve a specific purpose — they teach the agent what a valid board *looks like*: pieces in every row, legal trap placement, super moves, goal alignment. They're demonstrations, not exhaustive coverage. A modest set covering all *move types* is enough; the agent has to generalize from there.

But there's a gap in the current pipeline. After scaffolding (imitation) and opponent curriculum (optimization against fixed opponents), the agent may converge on a narrow set of winning templates rather than discovering novel strategies. Ryan asked: *"I wonder if we need another phase where the agent is encouraged to train itself on its own boards, be more creative?"*

This is essentially **self-play** — the standard industry term. The agent builds a board, then plays against it (or a copy of itself plays against it). If the board is too easy to beat, it gets a low reward. This pushes toward genuinely tricky, diverse boards instead of templates that happen to beat the canned opponent pools.

We discussed two approaches:
1. **Full self-play**: Agent builds and plays both sides. Significant architecture change.
2. **Self-curriculum**: Agent's own boards get fed back into the opponent pool. Lighter — no new env needed, just a new pool populated from training output.

Both are future work. Step 1 is getting the baseline training converging for size 4 with tuned hyperparameters. Self-play comes after.

---

## Hardware Reality

Ryan also asked if the training machine could handle two training runs in parallel (size 4 + size 5). The answer: no. The machine has 4 CPU cores (Intel i5-3470, no hyperthreading), 7.7GB RAM, and one training run already saturates a core at 99% while using 1.8GB resident memory. Two runs would thrash into swap and both would slow to 30-40% of normal speed. Sequential training is the practical path. Reducing the number of parallel envs (`--envs 2`) may help keep memory in check for size 5.

---

## Files Changed

- `examples/train_simultaneous.py` — Added `--learning-rate`, `--ent-coef`, `--n-steps`, `--batch-size` CLI flags; pass through to `train()` and `MaskablePPO`
- `spaces_game/cli.py` — Added `--save-to` flag for interactive board builder
- `boards/size4/01_mixed_traps.json` — New: merged from one_trap + multi_trap, 4 boards
- `boards/size4/02_super_move.json` — Trimmed to 4 boards
- `boards/size4/03_super_move_counter.json` — Trimmed to 4 boards
- Removed: `boards/size4/01_one_trap.json`, `boards/size4/02_multi_trap.json`, `boards/size4/03_super_move.json`, `boards/size4/04_super_move_counter.json`
- `TRAINING_PLAN.md` — Added hyperparameter tuning docs, size 4 pools, updated status
- `README.md` — Updated training progress for size 4

---

## What's Next

- Run size 4 training with tuned hyperparameters (lr=1e-4, ent=0.1, n_steps=4096, 10M steps)
- If the tuned run converges, begin size 5 board curation
- Implement stochastic fallback on retry for deterministic inference skill levels
- Begin fog of war implementation for size 2/3 (Stage 4)
- Design self-play or self-curriculum phase for post-opponent-curriculum training

---

*The same hyperparameters that cruise through a 2x2 grid will spin their wheels on a 4x4 — bigger boards need more patience, more exploration, and longer rollouts.*
