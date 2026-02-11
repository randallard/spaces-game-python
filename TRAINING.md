# RL Training Guide for Spaces Game (Stage 0 Era)

> **Note:** This guide covers Stage 0 (deck selection) training with `train_basic.py`. For current training workflows (Stage 3 simultaneous play, construction scaffolding, difficulty levels), see [TRAINING_PLAN.md](TRAINING_PLAN.md). For the play script and interactive model selection, see the [play commands in TRAINING_PLAN.md](TRAINING_PLAN.md#-quick-start-current-status).

Guide for training RL agents on the **tenx-rltec** training machine.

## Training Machine Specs

- **GPU**: RTX 3060 12GB
- **RAM**: 8GB DDR3 (constraint to watch)
- **OS**: Omarchy (Arch Linux)
- **PyTorch**: 2.1.0 + CUDA

## Setup on Training Machine

### 1. Clone and Install

```bash
# Clone the repo (if not already done)
git clone <repo-url> spaces-game-python
cd spaces-game-python

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies including RL libraries
pip install -e ".[rl]"
```

**Note**: The `[rl]` extra installs Stable Baselines3 and TensorBoard for training.

### 2. Verify Installation

```bash
# Check environment is valid
python examples/check_env.py

# Run basic self-play test
python -c "
from spaces_game import SpacesGameEnv
env = SpacesGameEnv()
obs, _ = env.reset(seed=42)
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f'Reward: {reward}, Score: {info[\"agent_total_score\"]}-{info[\"opponent_total_score\"]}')
"

# Run tests
pytest tests/test_gym_env.py -v
```

## Training Workflow

### Step 1: Start with Size 2 Boards (Simplest, Fastest)

```bash
# Basic training run (100K timesteps, ~15-20 minutes)
python examples/train_basic.py \
    --timesteps 100000 \
    --n-envs 4 \
    --save-freq 10000 \
    --eval-freq 5000

# Monitor training progress
tensorboard --logdir logs/
```

**Why Size 2 First?**:
- Smallest game space (easiest to learn)
- Only 16 boards (basically zero memory)
- Fastest training iterations
- Validates your training pipeline works

**Settings**:
- `--n-envs 4`: 4 parallel environments (conservative for 8GB RAM)
- `--timesteps 100000`: Start small to test
- Board pool: `boards_size_2.json` (16 boards, 15KB) - default!

### Step 2: Evaluate Your Agent

```bash
# Evaluate trained model vs random opponent
python examples/evaluate_agent.py \
    models/ppo_spacegame_final.zip \
    --episodes 1000 \
    --baseline

# This will show:
# - Win rate vs random opponent
# - Average score differential
# - Comparison to random baseline
```

### Step 3: Scale Up to Size 3 Boards

Once size 2 training works:

```bash
# Train on size 3 boards (500 boards, more variety)
python examples/train_basic.py \
    --board-pool data/boards_size_3.json \
    --timesteps 500000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000
```

### Step 4: Longer Training Runs

Once you're comfortable:

```bash
# Full training run (1M timesteps, ~2-3 hours on size 2)
python examples/train_basic.py \
    --timesteps 1000000 \
    --n-envs 4 \
    --save-freq 50000 \
    --eval-freq 10000
```

### Step 5: Scale Up (After Hardware Upgrade)

With 32GB DDR5 RAM:

```bash
# More parallel environments
python examples/train_basic.py \
    --timesteps 5000000 \
    --n-envs 16 \
    --use-subprocess \
    --save-freq 100000

# Larger board pool (size 4 boards)
python examples/train_basic.py \
    --board-pool data/boards_size_4.json \
    --timesteps 5000000 \
    --n-envs 8
```

## Command Reference

### Training Options

```bash
python examples/train_basic.py --help
```

Key parameters:
- `--board-pool`: Board pool JSON file
- `--opponent`: Opponent strategy (`random` or `greedy`)
- `--timesteps`: Total training timesteps
- `--n-envs`: Number of parallel environments
- `--save-dir`: Where to save models (default: `models/`)
- `--save-freq`: Save checkpoint every N steps
- `--eval-freq`: Evaluate every N steps
- `--seed`: Random seed for reproducibility
- `--load-model`: Continue training from saved model
- `--use-subprocess`: Use SubprocVecEnv (more parallel, more RAM)

### Evaluation Options

```bash
python examples/evaluate_agent.py --help
```

Key parameters:
- `model_path`: Path to trained model (required)
- `--episodes`: Number of evaluation episodes
- `--opponent`: Opponent to evaluate against
- `--baseline`: Also test random baseline for comparison

## Memory Management Tips

With 8GB RAM, watch for:

1. **Board Pool Size**: Start with `boards_size_2.json` (16 boards), then `boards_size_3.json` (~500 boards)
2. **Parallel Environments**: Use `--n-envs 4` (not 16+)
3. **Batch Size**: Default 64 is fine, don't increase
4. **SubprocVecEnv**: Avoid `--use-subprocess` until RAM upgrade

## Training Progression Path

**Recommended learning curve**:
1. Size 2 boards (16 boards) - Learn the basics, ~1-2 hours training
2. Size 3 boards (500 boards) - More variety, ~3-5 hours training
3. Size 4 boards (5000 boards) - Significant complexity, requires RAM upgrade
4. Size 5 boards (50K+ boards) - Advanced, full game complexity

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/
# Open browser to http://localhost:6006
```

Metrics to watch:
- `rollout/ep_rew_mean`: Average episode reward (higher = better)
- `eval/mean_reward`: Evaluation reward
- `train/loss`: Training loss (should decrease)

### Checkpoints

Models saved to `models/`:
- `ppo_spacegame_10000_steps.zip`: Checkpoint at 10K steps
- `ppo_spacegame_20000_steps.zip`: Checkpoint at 20K steps
- `ppo_spacegame_final.zip`: Final trained model
- `best/best_model.zip`: Best model based on evaluation

### Resume Training

```bash
# Continue from checkpoint
python examples/train_basic.py \
    --load-model models/ppo_spacegame_50000_steps.zip \
    --timesteps 100000  # Train 100K more steps
```

## Expected Results

### Random Baseline
- Win rate vs random opponent: ~50%
- This is your baseline to beat

### After 100K Steps
- Win rate: 55-65% (modest improvement)
- Agent learning basic strategies

### After 1M Steps
- Win rate: 70-85% (significant improvement)
- Agent has strong strategies

### After 5M+ Steps
- Win rate: 85-95% (expert level)
- Consistent strong play

## Next Steps After Initial Training

1. **Experiment with Opponents**:
   ```bash
   # Train against greedy opponent
   python examples/train_basic.py --opponent greedy --timesteps 500000
   ```

2. **Hyperparameter Tuning**:
   - Adjust learning rate (`learning_rate` in code)
   - Try different batch sizes
   - Experiment with entropy coefficient

3. **Self-Play Training**:
   - Train agent against itself (advanced)
   - Requires implementing opponent that uses trained model

4. **Board Construction**:
   - Move from board selection to board building (harder problem)
   - Requires new observation/action space

## Troubleshooting

### Out of Memory
```bash
# Reduce parallel environments
python examples/train_basic.py --n-envs 2

# Or use smaller board pool
# (boards_size_3.json is smallest)
```

### Training Too Slow
```bash
# Check GPU utilization (should be low for this problem)
nvidia-smi

# This is normal - RL training is CPU-bound for this problem
# GPU has plenty of headroom
```

### Model Not Learning
- Check TensorBoard for `rollout/ep_rew_mean` - should increase over time
- If flat, try:
  - Longer training (more timesteps)
  - Different random seed
  - Adjust learning rate

## Hardware Upgrade Impact

**Current (8GB DDR3)**:
- 4 parallel envs
- Size 3 boards only
- ~50K steps/minute

**After Upgrade (32GB DDR5)**:
- 16+ parallel envs
- Size 4-5 boards
- ~200K steps/minute
- 4x faster training

The upgrade will significantly speed up iteration time, but you can get meaningful results on current hardware with size 3 boards.
