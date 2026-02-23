# Training Plan: How to Train a New Board Size

This document is the practical guide for training an RL agent from scratch for any board size. It covers the complete flow from creating opponent pools through Stage 3 (full reveal), Stage 4 (fog of war), and self-play.

For hardware details and constraints, see [TRAINING_ARCHITECTURE.md](TRAINING_ARCHITECTURE.md).
For fog of war experiments and research, see [EXPERIMENTS.md](EXPERIMENTS.md).
For deployment to the inference server, see [DEPLOYMENT.md](DEPLOYMENT.md).
For Discord training notifications, see [DISCORD_SETUP.md](DISCORD_SETUP.md).

---

## Quick Reference: What Exists

| Size | Stage 3 (Full Reveal) | Stage 4 (Fog) | Self-Play | Model Path |
|------|----------------------|---------------|-----------|------------|
| 2 | Complete (retiring) | Fog+SP complete | Complete | `models/size2/stage4/` |
| 3 | Complete | Fog+SP complete (~90% SP WR, level 10) | Complete | `models/size3/stage4/best/` |
| 4 | Complete (~78% WR) | Fog+SP complete (80% SP WR, level 10) | Complete | `models/size4/stage3/best/`, `models/size4/stage4/` |
| 5 | Skipped (fog-first) | Fog+SP complete (~90% SP WR, level 10) | Complete | `models/size5/stage4/` |
| 6 | Skipped (fog-first) | Fog+SP complete (100% SP WR, level 10) | Complete | `models/size6/stage4/` |
| 7 | Skipped (fog-first) | Fog+SP complete (100% SP WR, level 10) | Complete | `models/size7/stage4/` |
| 8 | Skipped (fog-first) | Fog+SP complete (100% SP WR, level 10) | Complete | `models/size8/stage4/` |
| 9 | Skipped (fog-first) | Pipeline running | In progress | `models/size9/stage4/` |
| 10 | Skipped (fog-first) | Board pools ready, training not started | Pending | `boards/size10/` |

---

## Training a New Size: Step by Step

This is the complete checklist for training size N from scratch. Example uses size 5.

### Step 1: Create Opponent Board Pools

The agent trains against opponent boards organized by difficulty. You need 3-4 pool files in `boards/sizeN/`, with 4 boards each.

**Pool categories (in curriculum order):**

| File | Description | Example |
|------|-------------|---------|
| `00_simple.json` | Straight paths, no traps | Piece walks column 0 top to bottom |
| `01_mixed_traps.json` | 1 to N-2 traps, various placements | Traps on edges, center, blocking paths |
| `02_super_move.json` | Supermove (trap on own piece's cell) | Piece lands on trap to skip a row |
| `03_super_move_counter.json` | Redirect traps forcing column changes | Crossover patterns, multi-trap setups |

**How to create pools:**

Option A: Interactive board builder (recommended for first pools)
```bash
# Build boards one at a time, append to file
spaces-game test --interactive --size 5 --save-to boards/size5/00_simple.json

# Repeat for each pool category
spaces-game test --interactive --size 5 --save-to boards/size5/01_mixed_traps.json
spaces-game test --interactive --size 5 --save-to boards/size5/02_super_move.json
spaces-game test --interactive --size 5 --save-to boards/size5/03_super_move_counter.json
```

Option B: Generate with TypeScript engine (bulk generation, then manually sort)
```bash
# Generate many boards, then curate into categories
spaces-game generate-boards --size 5 --limit 100 --output raw_size5.json

# Validate
spaces-game validate raw_size5.json
```

**Board format** (JSON array of board objects):
```json
[
  {
    "boardSize": 5,
    "grid": [["piece", "empty", ...], ...],
    "moves": [
      {"type": "piece", "position": {"row": 0, "col": 0}, "order": 1},
      {"type": "piece", "position": {"row": 1, "col": 0}, "order": 2},
      {"type": "trap", "position": {"row": 2, "col": 1}, "order": 3},
      ...
      {"type": "final", "position": {"row": -1, "col": 0}, "order": 6}
    ]
  }
]
```

**Rules to follow:**
- Max traps per board = `board_size - 1` (size 5 = max 4 traps)
- Piece must visit every row (0 through N-1) before the final/goal move
- Traps must be adjacent to the piece path (orthogonal)
- Supermove trap is placed on the piece's own cell
- 4 boards per pool is the standard (matches sizes 2-4)

**Verify pools load correctly:**
```bash
spaces-game validate boards/size5/00_simple.json
spaces-game stats boards/size5/00_simple.json
```

### Step 2: Train Stage 3 (Full Reveal, Pool Opponents)

The agent learns blind board construction and opponent adaptation with full opponent board reveal after each round.

```bash
python examples/train_simultaneous.py \
    --size 5 \
    --timesteps 5000000 \
    --envs 4 \
    --learning-rate 1e-4 \
    --ent-coef 0.1 \
    --n-steps 4096

# Monitor
tensorboard --logdir logs/size5_stage3/
```

**Hyperparameter guidance by size:**

| Size | Learning Rate | Entropy Coef | N Steps | Batch Size | Notes |
|------|--------------|-------------|---------|------------|-------|
| 2 | 3e-4 | 0.05 | 2048 | 64 | Defaults work fine |
| 3 | 3e-4 | 0.05 | 2048 | 64 | Defaults work fine |
| 4 | 1e-4 | 0.1 | 4096 | 64 | Lower LR, more exploration |
| 5+ | 1e-4 | 0.1 | 4096-8192 | 128 | Longer rollouts, bigger batches |

For size 5+, consider `--envs 2` if memory is tight (see [TRAINING_ARCHITECTURE.md](TRAINING_ARCHITECTURE.md)).

**What to watch in TensorBoard:**

- `curriculum/valid_rate` — should hit 100% quickly (strict masking prevents invalid boards)
- `curriculum/game_win_rate` — must reach 70% + valid rate 90% to advance phases
- `curriculum/opponent_phase` — should steadily climb through phases
- `rollout/ep_rew_mean` — overall training reward trend

**How phases work:**

With 4 pool files, `build_phase_map(4)` creates 7 phases:
```
Phase 0: simple (solo)
Phase 1: mixed_traps (solo)
Phase 2: simple + mixed_traps (mixed)
Phase 3: super_move (solo)
Phase 4: simple + mixed_traps + super_move (mixed)
Phase 5: super_move_counter (solo)
Phase 6: all pools (mixed)
```

The agent auto-advances when game win rate >= 70% and valid rate >= 90% at each phase.

**When it's done:**

Pool-only training is "done" when the agent clears the final phase (all pools mixed) with a stable win rate. For sizes 2-3 this happens in 200K-2M steps. For size 4, ~2M steps. Size 5 will likely need 5-10M.

Output:
```
models/size5/stage3/best/best_model.zip          # Best eval model
models/size5/stage3/difficulty/beginner.zip       # Phase 0 checkpoint
models/size5/stage3/difficulty/intermediate.zip   # Phase 2 checkpoint
models/size5/stage3/difficulty/expert.zip         # Training end
models/size5/stage3/phase_history.json            # Full progression log
```

### Step 3: Add Self-Play (Optional but Recommended for Size 4+)

Pool opponents are static — the agent eventually memorizes counter-strategies. Self-play adds adaptive opponents that force the agent to generalize.

**Important**: Train pool-only first until converged, then resume with self-play. Starting self-play from scratch leads to policy collapse (weak snapshots create a death spiral).

```bash
# Resume from converged pool model with progressive window self-play
python examples/train_simultaneous.py \
    --size 5 \
    --self-play \
    --warmup-steps 0 \
    --resume models/size5/stage3/best/best_model.zip \
    --timesteps 10000000 \
    --learning-rate 1e-4 \
    --ent-coef 0.1 \
    --n-steps 4096 \
    --advance-threshold 0.70 \
    --backtrack-threshold 0.55 \
    --min-steps-per-level 50000
```

**How progressive window self-play works:**

Instead of binary self-play/recovery switching, the callback manages a **window** of active opponent snapshots:

- **Level 0**: Seed model only (the `--resume` model stays permanently in pool)
- **Level 1**: Seed + 1st snapshot
- **Level 2**: Seed + 1st + 2nd snapshot
- **Level k**: Seed + first k snapshots

**Transitions:**
- **Advance** (level + 1): Pool win rate >= `--advance-threshold` (0.70) sustained for `--min-steps-per-level` (50k) steps
- **Backtrack** (level - 1): Pool win rate drops below `--backtrack-threshold` (0.55)
- **Pool recovery**: At level 0 and still failing → switches to pure pool opponents (ratio=0.0) until win rate recovers to `--recovery-win-rate` (0.70)
- **Snapshot quality gate**: Only saves a new snapshot when pool win rate >= `--snapshot-win-rate` (default: midpoint of backtrack and advance thresholds)

**TensorBoard metrics** (`self_play/` panel):
- `window_level` — current difficulty level
- `max_level` — highest level ever reached
- `in_recovery` — 1.0 if in pool recovery, 0.0 otherwise
- `pool_snapshots` — total snapshots in pool
- `pool_win_rate` — latest eval win rate against pool opponents

**Self-play is "done" when** the window level stabilizes and pool win rate plateaus. For size 4, this was ~78% at 2M steps.

### Step 4: Train Stage 4 (Fog of War)

Same as Stage 3 but with partial observability. After simulation, the agent only sees opponent piece moves up to the step where the agent's round ended. Traps are hidden unless the agent hit one.

```bash
# Fog from scratch (recommended — never develops full-info dependency)
python examples/train_simultaneous.py \
    --size 5 \
    --fog \
    --timesteps 10000000 \
    --learning-rate 1e-4 \
    --ent-coef 0.1 \
    --n-steps 4096
```

The `--fog` flag:
- Adds `fog_outcomes` to the observation space (per-round signals: trap hits, collisions, visibility)
- Filters `opponent_history` to only show moves up to `playerLastStep`
- Hides all opponent traps except the one the agent landed on (the "sprung" trap)

Construction and scoring are unchanged — fog only affects what the agent sees of the opponent after simulation.

Output goes to `models/sizeN/stage4/` and `logs/sizeN_stage4/`.

### Step 5: Fog + Self-Play

Same resume pattern as Step 3, but with `--fog`:

```bash
python examples/train_simultaneous.py \
    --size 5 \
    --fog \
    --self-play \
    --warmup-steps 0 \
    --resume models/size5/stage4/best/best_model.zip \
    --timesteps 10000000 \
    --advance-threshold 0.70 \
    --backtrack-threshold 0.55
```

---

## Play Against the Agent

```bash
# Difficulty selection
python examples/play_against_agent.py --size 5 --difficulty beginner
python examples/play_against_agent.py --size 5 --difficulty expert

# With fog of war display
python examples/play_against_agent.py --size 5 --rounds 5 --fog

# Stochastic mode (agent samples policy for varied play)
python examples/play_against_agent.py --size 5 --difficulty beginner --stochastic
```

---

## CLI Flag Reference

### Core Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--size N` | 2 | Board size (NxN grid) |
| `--timesteps N` | 200,000 | Total training steps |
| `--envs N` | 4 | Parallel environments |
| `--eval-freq N` | 2,000 | Steps between evaluations |
| `--save-freq N` | 10,000 | Steps between checkpoints |
| `--resume PATH` | None | Resume from saved model |
| `--output-dir PATH` | Auto | Override output directory |
| `--fog` | Off | Enable fog of war (Stage 4) |

### Hyperparameters

| Flag | Default | Guidance |
|------|---------|----------|
| `--learning-rate` | 3e-4 | Try 1e-4 for size 4+ |
| `--ent-coef` | 0.05 | Try 0.1 for size 4+ |
| `--n-steps` | 2048 | Try 4096-8192 for size 4+ |
| `--batch-size` | 64 | Try 128 for size 5+ |

### Curriculum

| Flag | Default | Description |
|------|---------|-------------|
| `--win-rate-threshold` | 0.70 | Win rate to advance opponent phase |
| `--min-phase-steps` | 10,000 | Min steps before advancing phase |
| `--board-pools` | Auto-discover | Override pool file paths (comma-separated) |
| `--start-opponent-phase` | 0 | Skip to a specific phase (use with `--resume`) |

### Self-Play

| Flag | Default | Description |
|------|---------|-------------|
| `--self-play` | Off | Enable self-play curriculum |
| `--advance-threshold` | 0.70 | Win rate to advance window level |
| `--backtrack-threshold` | 0.55 | Win rate to backtrack a level |
| `--min-steps-per-level` | 50,000 | Min steps before advancing level |
| `--recovery-win-rate` | 0.70 | Win rate to exit pool recovery |
| `--snapshot-win-rate` | Auto | Quality gate for saving snapshots |
| `--snapshot-freq` | 50,000 | Steps between snapshots |
| `--pool-size` | 10 | Max snapshots to keep |
| `--warmup-steps` | 100,000 | Steps before self-play activates |

### Discord Notifications

| Flag | Default | Description |
|------|---------|-------------|
| `--discord-webhook URL` | None | Discord webhook URL for training notifications |
| `--discord-check-in N` | 30 | Minutes between periodic check-in messages |

To create a webhook: Discord Server Settings > Integrations > Webhooks > New Webhook. Copy the webhook URL and pass it with `--discord-webhook`.

When enabled, sends:
- **Milestone alerts**: phase advances, self-play level changes, recovery enter/exit, training complete
- **Periodic check-ins**: progress summary with win rate trends and commentary (every N minutes)

No notifications are sent if `--discord-webhook` is not provided.

### Deprecated (accepted but ignored)

| Flag | Replaced By |
|------|-------------|
| `--self-play-block-steps` | `--min-steps-per-level` |
| `--pool-recovery-steps` | Automatic recovery |
| `--min-pool-win-rate` | `--backtrack-threshold` |
| `--self-play-ratio` | Progressive window levels |
| `--board-library` | Strict masking (no scaffolding needed) |

---

## Architecture Overview

### Training Pipeline

```
boards/sizeN/*.json          Pool files (opponent boards by difficulty)
        |
        v
train_simultaneous.py        CLI entry point, env/model wiring
        |
        +-- spaces_game/callbacks/pool_utils.py
        |     discover_pools(), build_phase_map()
        |
        +-- spaces_game/callbacks/opponent_progression.py
        |     OpponentProgressionCallback (phase advancement)
        |
        +-- spaces_game/callbacks/self_play.py
        |     SelfPlayCurriculumCallback (window levels, snapshots)
        |
        +-- spaces_game/callbacks/discord_notifier.py
        |     DiscordNotifierCallback (webhook alerts, check-ins)
        |
        v
spaces_game/simultaneous_play_env.py
        SimultaneousPlayEnv (Gymnasium env)
        - Board construction (strict BFS masking)
        - 5-round game simulation
        - Opponent board loading (pool or self-play model)
        - Fog of war filtering (when use_fog=True)
        - Scoring (first-visit forward movement)
```

### Key Design Decisions

**Strict action masking**: BFS reachability checks make invalid boards structurally impossible. Every action the agent can take leads to a completable board. No scaffolding needed.

**Flat Discrete action space**: `Discrete(2 * n_cells + 1)` — piece placements, trap placements, and finish. Every unmasked action is guaranteed valid. Previous MultiDiscrete space wasted 30-50% of steps on invalid combinations.

**Forward-only movement**: Piece can only move to equal or lower row indices. BFS checks forward + sideways only. Prevents path oscillation.

**Consistent opponent style per game**: At game reset, one pool file is locked for all 5 rounds. The opponent plays a consistent style (e.g., all simple or all super_move), matching realistic play patterns. TODO (verify): confirm this improves pool eval signal quality in the next pool-training run.

**First-visit scoring**: Points only for reaching a new best row. No oscillation farming.

**Trap limit**: Max traps = `board_size - 1`. Forces strategic trade-offs between supermove, regular trap, and speed.

**No-revisit masking**: Agent's piece can't revisit cells. This is an agent optimization (not a game rule) — revisiting wastes moves.

---

## Completed Training History

### Size 2 (Feb 2026)
- Stage 3: 200K steps, all phases cleared, 40-65% win rate
- Self-play: Not needed (pool opponents sufficient)

### Size 3 (Feb 2026)
- Stage 3: 2M steps, all phases cleared, 65-100% win rate (avg ~75%)
- Stage 3 + self-play: Complete, difficulty tiers saved
- Stage 4 fog pool-only: Converged at ~82% win rate by 256K steps
- Stage 4 fog + self-play takes 1-5: Various failures (block scheduling, quality controls, progressive window, tuned thresholds). Take 5 reached level 4 but plateaued at ~50% WR.
- Stage 4 fog + self-play take 6 (SP win rate gate): **Complete** — switched snapshot quality gate and skill milestones to use SP eval win rate instead of pool WR. Reached level 10 (pool_size cap) by 2M steps, sustained ~90% SP eval WR through 7.5M steps. Pool WR ~40-50% (expected — model optimized for self-play, not pool). Level advancement snapshots saved automatically at each transition. Converged at level 10 cap.

### Size 4 (Feb 2026)
- Stage 3 pool-only: 2M steps, all 7 phases cleared, 100% valid rate
- Stage 3 + self-play: ~78% win rate at 2.05M steps with pool mixing
- 5 difficulty tiers saved in `models/size4/stage3/difficulty/`
- Stage 4 fog pool-only: Converged, all phases cleared. Models deployed to `models/size4/stage4/`
- Stage 4 fog + self-play: Level 10 by 1.8M steps, 80% SP WR at 5.25M steps. 20 level advancement snapshots saved.

### Size 5 (Feb 2026)
- Stage 3 skipped — went fog-first to test whether Stage 3 pretraining is necessary
- Stage 4 fog pool-only: All 7 phases cleared by 400K steps, 72-77% WR. Fog-first validated.
- Stage 4 fog + self-play: Reached level 10 by 2M steps, ~90% SP WR. Same convergence as size 3. Production models deployed.

### Size 6 (Feb 2026)
- Stage 3 skipped (fog-first confirmed unnecessary)
- Stage 4 fog pool-only: All phases cleared by ~1.5M steps. WR oscillated 55-80% at phase 6, avg ~65%. Cut at 4.1M steps.
- Stage 4 fog + self-play: Reached level 10 by 1.9M steps, 100% SP WR at 1.95M. 21 level advancement snapshots saved. Production models deployed.

### Size 7 (Feb 2026)
- Stage 3 skipped (fog-first pipeline)
- Stage 4 fog pool-only: All phases cleared by 1.6M steps
- Stage 4 fog + self-play: Level 10 at 2.12M steps, 100% SP WR. Fully automated pipeline. Production models deployed.

### Size 8 (Feb 2026)
- Stage 3 skipped (fog-first pipeline)
- Stage 4 fog pool-only: All phases cleared by 3.01M steps. Beginner/easy/medium captured during pool training.
- Stage 4 fog + self-play: Level 10 at 2.73M steps, 100% SP WR. 22 level advancement snapshots saved. Production models deployed.

### Size 9 (Feb 2026, in progress)
- Stage 3 skipped (fog-first pipeline)
- Stage 4 fog pool-only: Phase 0 took 0.70M steps (3.5x slower than size 8). Phase 1 took 1.88M steps. Phase 2 entered at 1.88M steps — still training at 21.7M steps. Mean reward improved from -110 to ~0. EV 0.786. Beginner/intermediate snapshots saved. Pool convergence (phase 6) unlikely within 30M step budget.
- Stage 4 fog + self-play: Not yet started — waiting for pool training to complete or exhaust budget.

### Next Steps

**Fog-only deployment**: All production models are fog-trained. Sizes 2-7 complete. Stage 3 models for sizes 2-4 can be retired.

**Verify difficulty snapshots for sizes 2-7**: The beginner/easy/medium snapshots were captured from pool-only training checkpoints. They need gameplay verification to confirm appropriate challenge levels:
- Play 5-10 games against each difficulty tier per size
- Beginner should lose most games but still make legal moves
- Easy should be competitive but beatable by a casual player
- Medium should require some strategy to beat
- If any tier is too strong or too weak, re-capture from a different phase checkpoint

**Size 8**: Complete. Pool converged at 3.01M steps, self-play at 2.73M steps (level 10, 100% SP WR). All difficulty tiers deployed.

**Size 9**: Pipeline running. Pool training at ~21.7M/30M steps, still on phase 2. Beginner and intermediate snapshots saved. Phase 2 at 21.7M steps suggests pool convergence (phase 6) unlikely within 30M budget — may need manual self-play launch from best pool model. The 81-cell board (163 actions/step) is a significant scaling wall vs size 8 (3M steps to converge).

**Size 10**: Board pools created and validated (16 boards across 4 categories). Ready to start training pipeline.

**Pipeline automation for sizes 8-10**: Productionize the training pipeline script so it can run unattended for new sizes. Requirements:
- Single command: `python scripts/train_pipeline.py --size N --discord-webhook URL`
- Auto-discovers pool boards from `boards/sizeN/`
- Runs fog pool-only until phase 6 converged, then auto-transitions to self-play
- **Save pool-only best model** before starting self-play (`pool_best.zip`) — pool snapshots are overwritten by self-play and can't be recovered
- Stops at level 10 + 100% SP WR (or 10M step cap)
- Deploys production models (beginner/intermediate/expert) on completion
- Hourly Discord updates with level, SP WR, EV, and milestone alerts
- Subprocess stdout to file (not pipe) to avoid blocking
- Stall detection with alerts if no progress for 30+ minutes

**Difficulty tiers — lower half needs pool-only models**: Self-play snapshots (even level 0) are too strong for beginner/easy difficulty. The agent has already mastered the full pool curriculum before self-play starts. For truly easy opponents, we capture weak models mid-pool-training:

Difficulty mapping:
  - **Beginner**: Pool phase 1 (mixed_traps solo) at 65% WR — barely competent
  - **Easy**: Pool phase 2 (simple+mixed combined) at 65% WR
  - **Medium**: Entering pool phase 3 (super_move solo) — weakest at new challenge
  - **Hard**: Self-play level 3-5 snapshots (from level_advancement/)
  - **Expert**: Self-play level 8-10 / best model

Scripts:
  - `scripts/train_pipeline.py` — full pipeline (pool + self-play), saves all 5 tiers during training
  - `scripts/train_pool_difficulty.py` — pool-only, captures beginner/easy/medium for existing sizes
    - Usage: `python scripts/train_pool_difficulty.py 2-7 [WEBHOOK_URL]`
    - Stops training as soon as all 3 snapshots are captured (no need to converge)
    - Saves to `models/sizeN/stage4/difficulty/{beginner,easy,medium}.zip`
    - Runs each size sequentially; skips sizes where all 3 already exist

**Other**:
- **Pool size increase**: Consider `--pool-size 20` for future self-play runs to push past level 10 ceiling
- **UI enhancements**: Focus on frontend improvements — all sizes 2-7 trained and deployed

### Deployment Plan
- Size 3 fog model: `models/size3/stage4/best/best_model.zip` — level 10 self-play, ~90% SP WR
- Size 4 fog model: `models/size4/stage4/beginner.zip`, `intermediate.zip`, `expert.zip` — level 10 self-play, 80% SP WR
- Size 5 fog model: `models/size5/stage4/beginner.zip`, `intermediate.zip`, `expert.zip`
- Size 6 fog model: `models/size6/stage4/beginner.zip`, `intermediate.zip`, `expert.zip` — level 10 self-play, 100% SP WR
- Level advancement models available via indexed selection (`GET /models`)

---

## Troubleshooting

**Agent not advancing phases**: Check `curriculum/valid_rate` — must be >= 90%. If valid rate is low despite strict masking, the agent may be hitting truncation (`max_construction_steps`). Try increasing timesteps.

**Self-play collapse (win rate craters)**: The snapshot pool has degraded. Stop training, check which snapshots exist in `models/sizeN/stage3/opponent_pool/`. Restart from the best pool-only model with `--resume`.

**Memory issues**: Drop to `--envs 2`. Close browsers. Check `free -h` before starting.

**Fog training slower than expected**: Normal — the learning signal is noisier with partial observability. Expect 2-3x more steps than the full-reveal equivalent.

**Deprecated flag warnings**: Old flags (`--self-play-block-steps`, etc.) are accepted but ignored. Update your commands to use the new flags.
