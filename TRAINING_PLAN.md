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
Stage 0: Deck Selection (Board Evaluation)
   ↓
Stage 1: Perfect Counter-Play (Board Construction Basics)
   ↓
Stage 2: Construction + Fog of War (Inference)
   ↓
Stage 3: Self-Play (Meta-Game)
   ↓
FINAL BOSS: 5-Round Fog of War vs Humans
```

Each stage builds on the previous one's learned skills.

---

## Stage 0: Deck Selection (CURRENT - IN PROGRESS)

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

### Training Commands:

#### Phase 0a: Partial Observability (Baseline)
```bash
# Size 2 boards - Learn basics
python examples/train_basic.py \
    --board-pool data/boards_size_2.json \
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
# Size 2 with perfect info
python examples/train_basic.py \
    --board-pool data/boards_size_2.json \
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
```bash
# Size 3 with perfect info (more complexity)
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
# Test optimal selection accuracy
python examples/test_board_selection.py \
    models/ppo_spacegame_final.zip \
    --board-pool new_boards_2.json \
    --perfect-info
```

**Success Criteria**: ≥80% optimal board selection accuracy

**If agent fails (<80%)**: More training needed, or hyperparameter tuning required.

---

## Stage 1: Perfect Counter-Play (Board Construction Basics)

**Purpose**: Learn to BUILD boards that counter known opponent boards.

### What Agent Learns:
- Board construction mechanics (how to place pieces and traps)
- Spatial reasoning (avoid opponent traps, maximize own path)
- Optimization (minimize opponent score, maximize own score)
- Validity constraints (boards must have valid path to goal)

### Implementation Status:
- ⏳ Environment: `CounterPlayEnv` (to be implemented)
- ⏳ Action space: Parameterized board construction
- ⏳ Training script: `train_construction.py` (to be implemented)

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

## Stage 2: Construction + Fog of War

**Purpose**: Learn to infer opponent's full board from partial observations.

### What Agent Learns:
- Inference from partial data (opponent stopped at step 3 → likely has trap there)
- Probability estimation (opponent probably has more traps based on R1 behavior)
- Adaptive construction (build counters based on inferred opponent strategy)

### Game Flow:
```
Round 1:
  - Both build boards simultaneously (no info yet)
  - Simulate with FOG OF WAR
  - Agent sees: opponent's moves until trap/collision/goal
  - Agent doesn't see: rest of opponent's board

Round 2:
  - Agent uses R1 partial info to infer opponent strategy
  - Builds counter-board based on inference
  - Simulate with fog of war
  - Accumulate more partial info

... Rounds 3-5 continue building inference ...
```

### Observation Space:
```python
{
    "round": 1-5,
    "score_diff": current differential,
    "opponent_visible_history": [
        # Round 1: What we saw
        {
            "visible_moves": [(row, col, order), ...],
            "stopped_at_step": 3,
            "reason": "trap" | "collision" | "goal",
        },
        # Round 2: What we saw
        ...
    ],
}
```

### Training Command (Future):
```bash
python examples/train_fog_of_war.py \
    --board-pool data/boards_size_3.json \
    --opponent pattern_based \  # Has predictable style
    --timesteps 10000000 \
    --n-envs 4
```

### Success Criteria:
- 70%+ win rate vs pattern-based opponents
- Agent demonstrates inference (different choices based on fog observations)
- Beats "always trap center" opponent >80% of time

---

## Stage 3: Self-Play (Meta-Game)

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
- ✅ Construct optimal boards (Stage 1)
- ✅ Infer from partial info (Stage 2)
- ✅ Adapt to opponent (Stage 3)

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

### What You Can Do Right Now:

1. **Train Stage 0** (deck selection):
   ```bash
   python examples/train_basic.py --perfect-info --timesteps 500000
   ```

2. **Evaluate training**:
   ```bash
   python examples/evaluate_agent.py models/ppo_spacegame_final.zip --perfect-info
   ```

3. **Validate learning**:
   ```bash
   python examples/test_board_selection.py models/ppo_spacegame_final.zip
   ```

### What's Next:

**Immediate** (After Stage 0 completes):
- Analyze results: Did perfect info help?
- Validate: Does agent understand board matchups (80%+ accuracy)?

**Short-term** (Next 1-2 weeks):
- Implement Stage 1 (board construction environment)
- Design and test action space for construction
- Train first construction agent

**Medium-term** (Next month):
- Implement fog of war mechanics
- Train inference-based agents (Stage 2)
- Begin self-play experiments (Stage 3)

**Long-term** (2-3 months):
- Mature self-play training
- Human vs agent exhibition matches
- Publish results/strategies discovered

---

## 🔍 Monitoring & Debugging

### TensorBoard:
```bash
tensorboard --logdir logs/
# Monitor: ep_rew_mean, eval/mean_reward, train/loss
```

### Key Metrics by Stage:

**Stage 0**:
- Win rate vs greedy opponent
- Score differential
- Optimal selection accuracy

**Stage 1**:
- Trap avoidance rate
- Path completion rate
- Win rate vs fixed boards

**Stage 2**:
- Inference accuracy (predicted vs actual opponent boards)
- Win rate vs pattern-based opponents

**Stage 3**:
- Win rate vs historical self
- Strategy diversity metrics
- Elo rating progression

---

## 📚 Additional Resources

- `examples/README.md` - Detailed tool documentation
- `TRAINING.md` - Training machine setup and tips
- `README.md` - Project overview and installation

---

**This is a living document** - Will be updated as each stage is implemented and results are analyzed.

Current Status: **Stage 0 - In Progress** ✅
