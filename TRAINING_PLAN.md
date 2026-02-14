# RL Training Plan: Progressive Curriculum for Spaces Game

This document outlines the complete training progression from simple deck selection to advanced board construction with fog of war.

## 🎯 End Goal

Train an agent that can play the **full Spaces Game**:
- **5 rounds** of simultaneous board construction
- **Fog of war**: Partial observability (only see opponent moves until collision/trap/goal)
- **Meta-game**: Infer opponent strategy from partial information across rounds
- **Competitive**: Beat human players and discover emergent strategies

## 📊 Training Stages Overview

```
Stage 0: Deck Selection (Board Evaluation)              ✅ COMPLETE
   ↓
Stage 1: Perfect Counter-Play (Board Construction)      ✅ COMPLETE
   ↓
Stage 2: Reverse Curriculum Construction                 ⚠️  OBSOLETE (replaced by Stage 3 scaffolding)
   ↓
Stage 3: Simultaneous 5-Round Play (Full Reveal)        ✅ SIZE 2 + SIZE 3 DONE, SIZE 4 IN PROGRESS (self-play)
   ↓
Stage 4: 5-Round Play with Fog of War                   ⏳ NEXT UP
   ↓
Stage 5: Self-Play (Meta-Game)                           ✅ MERGED INTO STAGE 3 (--self-play flag)
   ↓
FINAL BOSS: 5-Round Fog of War vs Humans
```

### Trap Limit Rule

**Max traps per board = board_size - 1** (size 2 = 1 trap max).

Without this, the agent discovers a single dominant strategy (supermove + extra trap)
and never adapts. With the limit, the agent must choose between:
- **Supermove** (trap on own cell) — gets past the cell but uses your only trap
- **Regular trap** (on adjacent cell) — blocks opponent but no supermove
- **No trap** — fastest path, no defense

Enforced via `_is_valid_placement()` in both `ReverseCurriculumBuilderEnv` and
`SimultaneousPlayEnv`. Action masks flow through automatically.

### Why Stages 0 and 1 Don't Need Retraining

Stages 2 and 3 train **from scratch** (fresh random weights) — no weight transfer
from earlier stages. The Stage 1 model is used during Stage 2 to select base boards
from the library, but since `new_boards_2.json` has been cleaned to only contain
legal boards (≤1 trap), the Stage 1 model can only pick from compliant boards.
Stage 3 (`SimultaneousPlayEnv`) doesn't use Stage 0 or 1 models at all.

### Scoring Rule: First-Visit Forward Movement

Points are awarded only the **first time** a piece reaches a new forward row.
Oscillating back and forth does NOT earn repeated points. This prevents agents
from farming points through path oscillation. Validated against TypeScript
reference implementation (52 parity tests pass).

### Agent No-Revisit Rule (Action Masking, NOT a Game Rule)

The agent's action masks prevent piece moves to previously visited cells
(`piece_visited_positions`). This is an **agent optimization** — revisiting
wastes moves while the opponent advances. It is NOT a game rule: human players
are allowed to revisit cells (they'll learn it's a bad strategy on their own).

**App integration note:** If driving agent construction outside of `env.step()`
(e.g., in a web app), you MUST track `piece_visited_positions` and pass it
through `_is_valid_placement()` for correct action masking. Without it,
MaskablePPO's policy will happily revisit cells — masking is enforcement,
not learned behavior. See `play_against_agent.py::_agent_build_board_blind()`
for the reference implementation.

### Strict Action Masking (Feb 14, 2026)

BFS reachability checks in `_is_valid_placement()` make invalid boards structurally
impossible. Every action the agent can take keeps the board on a completable path.
This replaces construction scaffolding entirely — the agent doesn't need to be taught
what a valid board looks like because it literally can't build an invalid one.

The BFS checks two conditions for every candidate move:
1. All rows 0..board_size-1 remain visitable (via already-visited + reachable cells)
2. Row 0 is reachable from the current/hypothetical position (for the finish condition)

Cost is trivial: max 16 cells for size 4.

### Training Progression Per Board Size

For each board size (2, 3, ...):
1. Implement trap limit (`board_size - 1` max traps)
2. Train Stage 3 with opponent curriculum (strict masking handles construction)
3. Optionally add `--self-play` for sizes where random opponents cap out
4. Train Stage 4 with fog of war
5. Scale to next board size

Each stage builds on the previous one's learned skills.

---

## Stage 0: Deck Selection (COMPLETE)

**Purpose**: Learn board evaluation and game mechanics without construction complexity.

### What Agent Learns:
- Scoring dynamics (how boards match up against each other)
- Resource management (timing of when to play which boards)
- Basic strategy (save strong boards for critical rounds)
- Game flow (who picks first, score accumulation)

### Implementation Status:
- ✅ Environment: `SpacesGameEnv` with deck selection
- ✅ Training script: `train_basic.py`
- ✅ Evaluation: `evaluate_agent.py`
- ✅ Validation: `test_board_selection.py`

### Board Pools for Training:

**Stage 0 uses curated board sets for controlled learning:**

- `new_boards_2.json` ⭐ - 8 carefully designed size-2 boards for validation
  - Used for controlled testing and optimal selection validation
  - Covers key board patterns: traps, no-traps, left/right columns

- `new_boards_3.json` ✅ - 11 curated size-3 boards (cleaned: removed 3-trap violations)

- `data/boards_size_2.json` - 16 auto-generated size-2 boards (for comparison)
- `data/boards_size_3.json` - 500 auto-generated size-3 boards
- `data/boards_size_3_large.json` - 1,704 size-3 boards (exhaustive)

### Training Commands:

#### Phase 0a: Partial Observability (Baseline)
```bash
# Size 2 boards - Learn basics with curated set
python examples/train_basic.py \
    --board-pool new_boards_2.json \
    --opponent greedy \
    --timesteps 500000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000
```

**Expected Results**:
- Win rate: 45-55% vs greedy (limited by hidden info)
- Learns basic board selection patterns

#### Phase 0b: Perfect Information
```bash
# Size 2 with perfect info (curated boards)
python examples/train_basic.py \
    --board-pool new_boards_2.json \
    --perfect-info \
    --opponent greedy \
    --timesteps 500000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000
```

**Expected Results**:
- Win rate: 70-85% vs greedy (agent can see opponent's deck)
- Learns board matchup fundamentals

#### Phase 0c: Scale to Size 3

**⚠️ PREREQUISITE**: Create `new_boards_3.json` with curated size-3 boards first!

Recommended: 12-16 boards covering:
- Various trap placements (corner, center, edge)
- Different path lengths (3-7 steps)
- Mobility vs trap-heavy strategies

```bash
# Size 3 with perfect info (curated boards - AFTER creating new_boards_3.json)
python examples/train_basic.py \
    --board-pool new_boards_3.json \
    --perfect-info \
    --opponent greedy \
    --timesteps 1000000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000

# Alternative: Use auto-generated boards for more variety
python examples/train_basic.py \
    --board-pool data/boards_size_3_large.json \
    --perfect-info \
    --opponent greedy \
    --timesteps 1000000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000
```

**Expected Results**:
- Win rate: 75-90% vs greedy
- More strategic depth, discovers counter-strategies

### Validation Gate:

Before moving to Stage 1, agent must demonstrate board understanding:

```bash
# Test optimal selection accuracy on size 2
python examples/test_board_selection.py \
    models/ppo_spacegame_final.zip \
    --board-pool new_boards_2.json \
    --perfect-info
```

**Success Criteria**: ≥80% optimal board selection accuracy

**If agent fails (<80%)**: More training needed, or hyperparameter tuning required.

**Note**: This uses `new_boards_2.json` - the curated 8-board validation set designed specifically for testing optimal selection.

---

## Stage 1: Perfect Counter-Play (COMPLETE)

**Purpose**: Learn to BUILD boards that counter known opponent boards.

### What Agent Learns:
- Board construction mechanics (how to place pieces and traps)
- Spatial reasoning (avoid opponent traps, maximize own path)
- Optimization (minimize opponent score, maximize own score)
- Validity constraints (boards must have valid path to goal)

### Implementation Status:
- ✅ Environment: `BoardConstructionEnv` with action masking
- ✅ Training script: `train_construction.py`
- ✅ 100% optimal play on 8 curated size-2 boards
- ✅ Model: `models/construction/best/best_model.zip`

### Game Flow:
```
1. Opponent plays fixed/random board (FULL VISIBILITY - no fog yet)
2. Agent sees opponent's board completely
3. Agent constructs counter-board
4. Simulate round (sequential, agent goes second)
5. Reward: score differential
```

### Action Space Design:

**Option A: Parameterized Action (Recommended)**
```python
Action = {
    "piece_path": List[(row, col, order)],  # Piece movement sequence
    "trap": Optional[(row, col, order)],    # Optional trap placement
}
```

**Option B: Sequential Building**
```python
# Multi-step action (10+ substeps per board)
Actions:
- place_piece(row, col, order)
- place_trap(row, col, order)
- finish_board()
```

### Training Command (Future):
```bash
# Train to build counter-boards on size 2
python examples/train_construction.py \
    --board-pool data/boards_size_2.json \
    --mode sequential \
    --timesteps 5000000 \
    --n-envs 4
```

### Success Criteria:
- 90%+ win rate against fixed opponent boards
- Agent avoids opponent traps consistently (95%+ trap avoidance)
- Agent maximizes path length (reaches goal >98% of time)

### Example Success:
```
Opponent board: Trap at (1,0), 5 steps
Agent builds:   Right column path (avoids trap), 6 steps
Result:         Agent scores 6, Opponent scores 0 (trapped by agent)
Win rate:       95%
```

---

## Stage 2: Reverse Curriculum Construction (OBSOLETE)

**Purpose**: Was intended to teach board construction from scratch using a standalone reverse curriculum.

### Status: Replaced by Stage 3 Construction Scaffolding

With the trap limit rule (`max_traps = board_size - 1`), Stage 2 can no longer advance past Phase 0. Boards are balanced enough that blind play against random opponents tops out at 16-34% win rate, well below the 75% threshold to advance phases.

**The solution**: Construction scaffolding was built directly into Stage 3's `SimultaneousPlayEnv`. A single training run now handles both construction learning (reverse curriculum) and opponent curriculum. See Stage 3 for details.

### Historical Results (pre-obsolescence):
- Size 2: `models/stage2_optimized/ppo_stage2_final.zip`
- Size 3: `models/size3/stage2/ppo_stage2_final.zip`

---

## Stage 3: Simultaneous 5-Round Play with Full Reveal

**Purpose**: Learn to construct boards blindly (no peeking at opponent) and adapt across 5 rounds based on what the opponent played in previous rounds.

**Status**:
- ✅ **Size 2**: Complete. All opponent phases, 40-65% win rate. Model: `models/size2/stage3/best/best_model.zip`
- ✅ **Size 3**: Complete. All construction + opponent phases, 65-100% win rate (avg ~75%). Model: `models/size3/stage3/best/best_model.zip`

### What Agent Learns:
- Build competitive boards without seeing opponent's current board
- Recognize opponent patterns from revealed previous-round boards
- Adapt strategy across 5 rounds (counter what opponent tends to play)
- The 50/50 column choice as fundamental strategic uncertainty
- **Strategic trade-offs**: supermove vs regular trap vs no trap (max `board_size - 1` traps)

### Implementation:
- ✅ Environment: `SimultaneousPlayEnv` in `spaces_game/simultaneous_play_env.py`
- ✅ Training script: `examples/train_simultaneous.py`
- ✅ Multi-round play script: `examples/play_against_agent.py --rounds 5`
- ✅ Trap limit enforcement in `_is_valid_placement()` + action masks
- ✅ Strict action masking: BFS reachability prevents invalid boards at mask level
- ✅ Self-play: `--self-play` with rolling opponent pool and skill snapshots
- ✅ CLI hyperparameter tuning: `--learning-rate`, `--ent-coef`, `--n-steps`, `--batch-size`
- ✅ Dynamic pool discovery from `boards/sizeN/` with numeric prefix ordering
- ✅ Dynamic phase map generation (`build_phase_map`) based on number of pools
- ✅ Curated opponent board pools in `boards/size2/` and `boards/size3/`:
  - `simple.json` - straight paths, no traps
  - `one_trap.json` - straight path + 1 trap
  - `super_move.json` - supermove (trap on own cell)
  - `super_move_counter.json` - 2 traps with crossover patterns
- ✅ Size 4 opponent pools in `boards/size4/` (4 boards each):
  - `00_simple.json` - straight paths, no traps
  - `01_mixed_traps.json` - 1-3 traps, various placements
  - `02_super_move.json` - supermove (trap on own cell)
  - `03_super_move_counter.json` - redirect traps forcing column changes

### Construction Scaffolding (REMOVED — Feb 14, 2026):

Construction scaffolding (`--board-library`) has been replaced by strict action masking.
BFS reachability checks make invalid boards structurally impossible, so the agent doesn't
need scaffolding to learn what a valid board looks like. The `--board-library` flag is
accepted for backward compatibility but prints a deprecation warning and does nothing.

### Progressive Opponent Curriculum:
- Phase 0: Simple boards only
- Phase 1: One-trap boards
- Phase 2: Simple + one-trap mixed
- Phase 3: Supermove boards
- Phase 4: All board types mixed
- Auto-advances when game win rate >= 70% and valid rate >= 90%

### Game Flow (per round):
```
1. Agent constructs board (blind - can't see opponent)
2. Opponent picks from their archetype pool
3. Simulation runs
4. Opponent's FULL board revealed in opponent_history
5. Scores update, next round begins
```

### Observation Space:
```python
{
    "building_board":    (size, size, 2),    # current construction state
    "construction_step": Discrete,
    "valid_cells_mask":  MultiBinary,
    "round":             Discrete(5),        # 0-4
    "score_diff":        float,
    "agent_score":       float,
    "opponent_score":    float,
    "opponent_history":  (5, size, size, 2), # full reveal of past rounds
}
```

### Training Commands:
```bash
# Size 2
python examples/train_simultaneous.py --size 2 --timesteps 200000

# Size 3
python examples/train_simultaneous.py --size 3 --timesteps 2000000

# Size 4 with self-play (recommended for sizes where random opponents cap out)
python examples/train_simultaneous.py --size 4 --self-play --timesteps 5000000

# Size 4 with tuned hyperparameters (if self-play alone isn't enough)
python examples/train_simultaneous.py --size 4 --self-play \
    --timesteps 10000000 --min-phase-steps 100000 \
    --learning-rate 1e-4 --ent-coef 0.1 --n-steps 4096

# Monitor
tensorboard --logdir logs/size4_stage3/
```

### Hyperparameter Tuning:

Default hyperparameters work for size 2-3. Larger boards need adjustment:

```bash
--learning-rate  # Default: 3e-4. Try 1e-4 for size 4+ (reduces oscillation)
--ent-coef       # Default: 0.05. Try 0.1 for size 4+ (more exploration)
--n-steps        # Default: 2048. Try 4096-8192 for size 4+ (longer rollouts)
--batch-size     # Default: 64. Try 128 for larger boards
```

Size 2-3 training is unaffected — defaults match what worked previously.

### Size 3 Results (2M timesteps = 500k n_calls with 4 envs):

| Phase | Steps | Valid Rate | Win Rate |
|-------|-------|-----------|----------|
| Construction C0-C6 | 0-410k | 27% -> 96% | 0% -> 70% |
| Opponent O0-O5 | 410-460k | 98-100% | 70-85% |
| Final (all mixed) | 460-500k | 96-100% | 65-100% (avg ~75%) |

Construction phases accelerate as knowledge compounds: C0 took 164k steps, C3-C6 took only 44k combined.

### Key findings:
- Size 2 learns construction from scratch (0% -> 100% valid rate in ~8k steps)
- Size 3 cannot learn construction from scratch (8% valid rate at 27k steps) -- scaffolding essential
- Size 3 with scaffolding completes full curriculum in ~460k n_calls (1.84M timesteps)
- Scoring fix (first-visit forward movement only) was critical -- agent exploited oscillation for free points
- `ent_coef=0.05` works well

---

## Stage 4: 5-Round Play with Fog of War (NEXT UP)

**Purpose**: Same as Stage 3 but with partial observability. After simulation, agent only sees opponent's moves up to the point they were stopped (trap/collision/goal), plus outcome info. Must infer opponent's full strategy from limited data.

**Prerequisites**: Stage 3 complete for the target board size.

### What Agent Learns:
- Inference from partial data (opponent stopped early -> had a trap/was trapped)
- Pattern recognition from incomplete information across rounds
- Risk assessment (opponent's hidden traps vs revealed traps)
- Adaptive construction under uncertainty

### Game Flow (per round):
```
1. Agent constructs board (blind)
2. Opponent picks from pool
3. Simulation runs
4. Agent sees PARTIAL opponent board (moves up to opponentLastStep)
5. Agent sees outcome: opponent_hit_trap, collision, opponent_reached_goal
6. Scores update, next round begins
```

### Fog of War Visibility Rules (from TypeScript reference):

Three tiers of information under fog:

**1. Fully visible:**
- Opponent piece moves up to their last executed step (`opponentLastStep`)
- Opponent traps that the agent actually stepped on (sprung traps — position revealed)
- Round outcome: who won, scores, how it ended (goal/trap/collision)

**2. Partially visible (existence only, no position):**
- When opponent places a trap during an executed step, the explanation reveals
  "a trap was set" — but NOT where. The agent knows the opponent has a trap
  somewhere, but not its location. This is a key strategic signal.

**3. Completely hidden:**
- Opponent moves after `opponentLastStep` (if round ended early)
- Opponent traps that were never triggered AND placed after the last executed step
- The full board layout / unexecuted portions of the opponent's sequence

### Observation Space Changes from Stage 3:
- `opponent_history` shows only revealed moves (sequence[:opponentLastStep+1])
- Opponent trap channel: only filled for sprung traps (agent hit them)
- Positions rotated to agent's frame via _rotate_position()
- `fog_outcomes` per round: [opponent_hit_trap, collision, opponent_reached_goal, opponent_placed_trap_visible]
  - `opponent_placed_trap_visible`: 1 if a trap was placed during an executed step (existence signal, no position)
- When opponent hits agent's trap early, their later moves (especially traps) remain hidden

### Current State:
- `play_against_agent.py --fog` already implements fog for the **human player's view** (display-only)
- The **agent's training env** does NOT have fog yet -- `opponent_history` always shows full boards
- Need to add fog to `SimultaneousPlayEnv` so the agent trains under partial observability

### Implementation Plan:
- Add `fog_of_war` flag to `SimultaneousPlayEnv`
- Modify `_encode_opponent_board()` to use `simulationDetails.opponentLastStep` for partial encoding
- Add `fog_outcomes` to observation space: per-round [opponent_hit_trap, collision, opponent_reached_goal]
- Option A: Add fog as a `--fog` flag for a separate training run (resume from Stage 3 weights)
- Option B: Add fog as later opponent curriculum phases (auto-transition after full-reveal phases)

### Training Progression:
```bash
# Size 3: fog of war (resume from Stage 3 model)
python examples/train_simultaneous.py --size 3 --fog --timesteps 2000000 \
    --resume models/size3/stage3/best/best_model.zip

# Or from scratch with scaffolding + fog
python examples/train_simultaneous.py --size 3 --fog \
    --board-library new_boards_3.json --timesteps 3000000
```

### Open Questions:
- Should fog be a separate run or added as later curriculum phases?
- Can we resume from Stage 3 weights? (obs space changes if fog_outcomes added)
- What win rate threshold makes sense under fog? (inherently noisier than full reveal)

---

## Stage 5: Self-Play (MERGED INTO STAGE 3 — Feb 14, 2026)

Self-play is now built into Stage 3 via the `--self-play` flag. No separate training
script or environment needed.

### How It Works:
1. **Warmup** (default 100k steps): JSON pool opponents, agent learns basic construction
2. **Snapshot** (every 50k steps): Freeze current model, add to rolling pool of 10 snapshots
3. **Opponent assignment**: Each training env gets a random snapshot as its opponent
4. **Fallback**: If opponent model produces an invalid board, falls back to JSON pool (~20% mixing prevents collapse)

### Training Command:
```bash
python examples/train_simultaneous.py --size 4 --self-play --timesteps 5000000

# With custom self-play parameters:
python examples/train_simultaneous.py --size 4 --self-play \
    --snapshot-freq 50000 --pool-size 10 --warmup-steps 100000 \
    --timesteps 10000000
```

### Skill Level Snapshots:
Milestone checkpoints are saved when eval win rate crosses thresholds:
- 55% -> beginner, 60% -> intermediate, 65% -> advanced, 70% -> expert, 75% -> advanced_plus
- Training end: snapshot timeline divided into 6 tiers for inference server

### Why Self-Play Matters:
Random pool opponents cap strategic learning at ~50% win rate (rock-paper-scissors dynamics).
Self-play opponents have actual patterns — the `opponent_history` observation becomes useful
because the agent can learn to exploit its own tendencies and adapt across rounds.

---

## 🎮 Final Boss: Human vs Agent

Once all stages complete, the agent should be ready to play the real game:

```
5-Round Fog of War Game:
- Both players build boards simultaneously each round
- Fog of war (partial observability)
- Meta-game (infer opponent strategy from partial observations)
```

### Agent Capabilities at End:
- ✅ Understand board evaluation (Stage 0)
- ✅ Construct optimal boards (Stage 1-2)
- ✅ Adapt across rounds with full info (Stage 3)
- ✅ Infer from partial info / fog of war (Stage 4)
- ✅ Counter-adapt via self-play (Stage 5)

### Exhibition Match Format:
```bash
python examples/play_vs_agent.py \
    --agent models/final_league_agent.zip \
    --human-name "Ryan" \
    --rounds 10
```

---

## 📈 Training Timeline Estimates

### With Current Hardware (8GB RAM, RTX 3060):

| Stage | Board Size | Timesteps | Parallel Envs | Est. Time |
|-------|-----------|-----------|---------------|-----------|
| 0a: Partial obs | 2 | 500K | 4 | ~30 mins |
| 0b: Perfect info | 2 | 500K | 4 | ~30 mins |
| 0c: Size 3 | 3 | 1M | 4 | ~2 hours |
| 1: Construction | 2 | 5M | 4 | ~10 hours |
| 2: Fog of war | 3 | 10M | 4 | ~20 hours |
| 3: Self-play | 3 | 50M | 4 | ~100 hours |

**Total**: ~130 hours (~5-6 days of training)

### With Upgraded Hardware (32GB RAM, RTX 3060):

| Stage | Board Size | Timesteps | Parallel Envs | Est. Time |
|-------|-----------|-----------|---------------|-----------|
| 0a-0c | 2-3 | 1M | 16 | ~1 hour |
| 1: Construction | 3 | 5M | 16 | ~2.5 hours |
| 2: Fog of war | 3-4 | 10M | 16 | ~5 hours |
| 3: Self-play | 4 | 50M | 16 | ~25 hours |

**Total**: ~35 hours (~1.5 days of training)

---

## ⚡ Quick Start: Current Status

### Training Commands:

```bash
# Size 2 (strict masking, no scaffolding needed)
python examples/train_simultaneous.py --size 2 --timesteps 200000

# Size 3
python examples/train_simultaneous.py --size 3 --timesteps 2000000

# Size 4 with self-play
python examples/train_simultaneous.py --size 4 --self-play --timesteps 5000000

# Monitor
tensorboard --logdir logs/size4_stage3/

# Produces: models/size{N}/stage3/difficulty/{beginner,intermediate,...,expert}.zip
```

### Play Against the Agent:
```bash
# Difficulty selection (after training)
python examples/play_against_agent.py --size 3 --difficulty beginner
python examples/play_against_agent.py --size 3 --difficulty intermediate
python examples/play_against_agent.py --size 3 --difficulty expert

# Stochastic + beginner = easiest; deterministic + expert = hardest
python examples/play_against_agent.py --size 3 --difficulty beginner --stochastic
python examples/play_against_agent.py --size 3 --difficulty expert

# Interactive model selection (auto-discovers difficulty models)
python examples/play_against_agent.py --size 3 --board-library new_boards_3.json

# With fog of war (display-only, not yet in training)
python examples/play_against_agent.py --size 3 --rounds 5 --fog
```

### Difficulty Levels:

Training saves named checkpoints at opponent phase milestones:

| Difficulty | Saved after | What it knows |
|------------|-------------|---------------|
| `beginner` | O0 complete | Builds valid boards, beats simple straight-path opponents |
| `intermediate` | O2 complete | Uses traps, handles mixed simple + one-trap opponents |
| `expert` | Training end | Full strategy against all opponent types |

Use `--min-phase-steps 100000` (vs default 10000) to ensure each phase gets deep training and difficulty levels are well-separated.

### What's Next:

**Immediate**:
- ✅ Size 2 + 3 retrained with all fixes (Feb 12) — all phases cleared
- ✅ Strict masking + self-play rework (Feb 14) — scaffolding removed, rewards simplified
- Size 4 self-play training in progress (5M steps)

**Short-term**:
- Evaluate size 4 self-play results — does win rate break past 50%?
- Deploy size 4 model to inference server
- Implement fog of war in `SimultaneousPlayEnv` (Stage 4)

**Long-term**:
- Scale to size 5+
- Human vs agent exhibition matches

---

## 🔍 Monitoring & Debugging

### TensorBoard:
```bash
tensorboard --logdir logs/
# Monitor: ep_rew_mean, eval/mean_reward, train/loss
# Stage 3 specific: curriculum/opponent_phase, curriculum/valid_rate, curriculum/game_win_rate
```

### Key Metrics by Stage:

**Stage 0**: Win rate vs greedy, optimal selection accuracy
**Stage 1**: Trap avoidance, path completion, win rate vs fixed boards
**Stage 2**: ~~Valid board rate, win rate vs library boards~~ (obsolete)
**Stage 3**: Game win rate (5-round), valid rate, construction phase progression, opponent phase progression
**Stage 4**: Game win rate under fog, inference quality (adaptation across rounds)
**Stage 5**: Win rate vs historical self, strategy diversity, Elo progression

---

## 📚 Additional Resources

- `examples/README.md` - Detailed tool documentation
- `TRAINING.md` - Training machine setup and tips
- `README.md` - Project overview and installation
- `journal/` - Training journals with detailed analysis per session

---

**This is a living document** - Will be updated as each stage is implemented and results are analyzed.

Current Status: **Size 2 + 3 complete (Feb 12). Size 4 self-play training in progress (Feb 14).** Stage 3 reworked: strict action masking (BFS reachability) makes invalid boards impossible, construction scaffolding removed, rewards simplified, self-play with rolling opponent pool added. Size 4 running with `--self-play --timesteps 5000000`.
