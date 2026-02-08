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
Stage 2: Reverse Curriculum Construction (Size 2 & 3)   ✅ COMPLETE
   ↓
Stage 3: Simultaneous 5-Round Play (Full Reveal)        ✅ Size 2 COMPLETE
   ↓
Stage 4: 5-Round Play with Fog of War                   ⏳ IN PROGRESS
   ↓
Stage 5: Self-Play (Meta-Game)                           ⏳ TODO
   ↓
FINAL BOSS: 5-Round Fog of War vs Humans
```

### Training Progression Per Board Size

For each board size (2, 3, ...):
1. Train simultaneous 5-round play with full reveal (Stage 3)
2. Train 5-round play with fog of war (Stage 4)
3. Scale to next board size

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

- `new_boards_3.json` ✅ - 14 curated size-3 boards with diverse strategies

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

## Stage 2: Reverse Curriculum Construction (COMPLETE)

**Purpose**: Learn to construct valid, competitive boards from scratch using a reverse curriculum that progressively removes scaffolding.

### What Agent Learns:
- Board construction from scratch (place pieces, traps, goal)
- Valid board generation (path to goal, legal move sequences)
- Counter-strategy (build boards that beat opponent boards)

### Results:
- ✅ **Size 2**: Solved. Model: `models/stage2_optimized/ppo_stage2_final.zip`
- ✅ **Size 3**: Solved. Model: `models/size3/stage2/ppo_stage2_final.zip`
- Agents win or tie against most library boards; intermittent invalid boards handled by retry logic in play script

---

## Stage 3: Simultaneous 5-Round Play with Full Reveal (Size 2 COMPLETE)

**Purpose**: Learn to construct boards blindly (no peeking at opponent) and adapt across 5 rounds based on what the opponent played in previous rounds.

### What Agent Learns:
- Build competitive boards without seeing opponent's current board
- Recognize opponent patterns from revealed previous-round boards
- Adapt strategy across 5 rounds (counter what opponent tends to play)
- The 50/50 column choice as fundamental strategic uncertainty

### Implementation:
- ✅ Environment: `SimultaneousPlayEnv` in `spaces_game/simultaneous_play_env.py`
- ✅ Training script: `examples/train_simultaneous.py`
- ✅ Curated opponent board pools in `boards/size2/`:
  - `simple.json` - straight paths, no traps (col 0 and col 1 variants)
  - `one_trap.json` - straight path + trap on opposite column
  - `super_move.json` - supermove (trap on own cell) + straight path
  - `super_move_counter.json` - cross-column path that beats supermove

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

### Results (Size 2):
- ✅ 91% game win rate against all board types mixed (phase 5)
- ✅ 99% valid board rate
- ✅ Reached phase 5 in ~770k steps (~24 min)
- Model: `models/size2/stage3/ppo_stage3_final.zip`
- Key finding: `ent_coef=0.1` needed for sufficient exploration against harder opponents

### Training Command:
```bash
# Size 2 (solved)
python examples/train_simultaneous.py --size 2 --timesteps 1000000

# Size 3 (TODO - create boards/size3/ opponent pools first)
python examples/train_simultaneous.py --size 3 --timesteps 2000000
```

---

## Stage 4: 5-Round Play with Fog of War (IN PROGRESS)

**Purpose**: Same as Stage 3 but with partial observability. After simulation, agent only sees opponent's moves up to the point they were stopped (trap/collision/goal), plus outcome info. Must infer opponent's full strategy from limited data.

### What Agent Learns:
- Inference from partial data (opponent stopped early → had a trap/was trapped)
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

### Key Difference from Stage 3:
- `opponent_history` shows only revealed moves (sequence[:opponentLastStep+1])
- Positions rotated to agent's frame via _rotate_position()
- `fog_outcomes` per round: [opponent_hit_trap, collision, opponent_reached_goal]
- When opponent hits agent's trap early, their later moves (especially traps) remain hidden

### Implementation Plan:
- Add `fog_of_war` flag to `SimultaneousPlayEnv`
- Add `fog_outcomes` to observation space
- When fog on, partial encode opponent board using simulation result
- Extend phase map: phases 5+ enable fog of war
- Update `play_against_agent.py` for multi-round play to verify learning

### Training Progression:
```bash
# Size 2: fog of war (after Stage 3 size 2 is solved)
python examples/train_simultaneous.py --size 2 --fog --timesteps 2000000

# Size 3+: simultaneous full reveal first, then fog
python examples/train_simultaneous.py --size 3 --timesteps 2000000
python examples/train_simultaneous.py --size 3 --fog --timesteps 2000000
```

---

## Stage 5 (future): Self-Play (Meta-Game)

**Purpose**: Discover emergent strategies through adversarial co-evolution.

### What Agent Learns:
- Counter-counter-strategies (if opponent learns to avoid traps, agent learns deception)
- Deception (early boards mislead opponent about later strategy)
- Adaptation (adjust strategy based on opponent's adaptation)
- Meta-meta-game (infinite strategic depth)

### Self-Play Variants:

#### Option A: Classic Self-Play
```python
# Agent plays against copy of itself
opponent_policy = copy.deepcopy(agent_policy)  # Update every 10K steps
```

**Pros**: Discovers counter-strategies
**Cons**: Can collapse to local optima

#### Option B: League Training (Recommended)
```python
# Agent plays against:
# - Latest self
# - Historical snapshots
# - Specialist exploiter agents
```

**Pros**: More robust, diverse strategies
**Cons**: Requires more compute

### Training Command (Future):
```bash
python examples/train_selfplay.py \
    --board-pool data/boards_size_3.json \
    --mode league \
    --timesteps 50000000 \
    --n-envs 8 \
    --opponent-update-freq 20000
```

### Success Criteria:
- Wins >60% vs frozen snapshot from 1M steps ago (shows improvement)
- Demonstrates strategic diversity (uses different board styles across games)
- Discovers novel strategies not seen in fixed-opponent training

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

### Play Against the Agent:
```bash
# Single-round play (Stage 2 agent)
python examples/play_against_agent.py --size 2

# Size 3
python examples/play_against_agent.py --size 3 --board-library new_boards_3.json
```

### Train Next Stage:
```bash
# Stage 3: Simultaneous 5-round (size 2 solved, size 3 next)
python examples/train_simultaneous.py --size 3 --timesteps 2000000

# Stage 4: Fog of war (after implementing fog flag)
python examples/train_simultaneous.py --size 2 --fog --timesteps 2000000
```

### What's Next:

**Immediate**:
- Implement fog of war in `SimultaneousPlayEnv`
- Update `play_against_agent.py` for multi-round play verification
- Create `boards/size3/` opponent pools for size 3 training

**Short-term**:
- Train Stage 4 (fog of war) on size 2
- Train Stage 3 + 4 on size 3
- Scale to size 4+

**Long-term**:
- Self-play training (Stage 5)
- Human vs agent exhibition matches

---

## 🔍 Monitoring & Debugging

### TensorBoard:
```bash
tensorboard --logdir logs/
# Monitor: ep_rew_mean, eval/mean_reward, train/loss
```

### Key Metrics by Stage:

**Stage 0**: Win rate vs greedy, optimal selection accuracy
**Stage 1**: Trap avoidance, path completion, win rate vs fixed boards
**Stage 2**: Valid board rate, win rate vs library boards
**Stage 3**: Game win rate (5-round), valid rate, opponent phase progression
**Stage 4**: Game win rate under fog, inference quality (adaptation across rounds)
**Stage 5**: Win rate vs historical self, strategy diversity, Elo progression

---

## 📚 Additional Resources

- `examples/README.md` - Detailed tool documentation
- `TRAINING.md` - Training machine setup and tips
- `README.md` - Project overview and installation

---

**This is a living document** - Will be updated as each stage is implemented and results are analyzed.

Current Status: **Stages 0-3 Complete (Size 2)** - Simultaneous 5-round play solved at 91% win rate. Next: Fog of war (Stage 4), then scale to size 3+.
