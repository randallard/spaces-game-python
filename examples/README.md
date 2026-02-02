# Spaces Game Examples

This directory contains example scripts demonstrating how to use the Spaces Game Gymnasium environment.

## Available Examples

### 1. Random Agent (`random_agent.py`)

A simple random agent that selects boards uniformly at random from available boards.

**Usage:**
```bash
# Run 100 episodes with default settings
python examples/random_agent.py

# Run 10 episodes with rendering enabled
python examples/random_agent.py --episodes 10 --render

# Run against greedy opponent
python examples/random_agent.py --opponent greedy --episodes 100

# Use larger board pool (size 4)
python examples/random_agent.py --board-pool data/boards_size_4.json --episodes 50

# Set random seed for reproducibility
python examples/random_agent.py --seed 42 --episodes 10
```

**Options:**
- `--episodes N`: Number of episodes to run (default: 100)
- `--board-pool PATH`: Path to board pool JSON file (default: `data/boards_size_3.json`)
- `--deck-size N`: Number of boards in each deck (default: 10)
- `--opponent STRATEGY`: Opponent strategy - `random` or `greedy` (default: `random`)
- `--render`: Render each episode to console
- `--seed N`: Random seed for reproducibility

**Example Output:**
```
============================================================
Random Agent vs Random Opponent
Board Pool: data/boards_size_3.json
Deck Size: 10
Episodes: 100
Seed: 42
============================================================

Episode   1/100: WIN  | Score:   1 -   6 | Reward:  -105.0
Episode   2/100: WIN  | Score:   4 -   1 | Reward:  +103.0
...

============================================================
FINAL STATISTICS
============================================================
Total Episodes: 100
Wins:           52 (52.0%)
Losses:         45 (45.0%)
Ties:           3 (3.0%)

Avg Agent Score:    3.8
Avg Opponent Score: 3.9
Avg Score Diff:     -0.1

Avg Episode Reward: +3.2 ± 98.7
============================================================
```

### 2. Environment Check (`check_env.py`)

Verifies that the SpacesGameEnv passes all Gymnasium API compatibility checks.

**Usage:**
```bash
# Check with default settings
python examples/check_env.py

# Check with different board pool
python examples/check_env.py --board-pool data/boards_size_4.json

# Check with greedy opponent
python examples/check_env.py --opponent greedy
```

**Options:**
- `--board-pool PATH`: Path to board pool JSON file (default: `data/boards_size_3.json`)
- `--deck-size N`: Number of boards in each deck (default: 10)
- `--opponent STRATEGY`: Opponent strategy - `random` or `greedy` (default: `random`)

## Using the Environment in Your Code

### Basic Usage

```python
from spaces_game import SpacesGameEnv
import numpy as np

# Create environment
env = SpacesGameEnv(
    board_pool_path="data/boards_size_3.json",
    deck_size=10,
    opponent_strategy="random",
    render_mode=None,  # or "human" or "ansi"
)

# Reset environment
obs, info = env.reset(seed=42)

# Play episode
terminated = False
total_reward = 0.0

while not terminated:
    # Select action (board from deck)
    action = np.random.randint(0, env.action_space.n)

    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # Optional: render
    if env.render_mode:
        env.render()

print(f"Episode finished. Total reward: {total_reward}")
print(f"Final score: Agent {info['agent_total_score']} - {info['opponent_total_score']} Opponent")

env.close()
```

### Observation Space

The observation is a dictionary with the following keys:

- `round` (int): Current round number (1-5)
- `score_diff` (np.ndarray): Score differential (agent - opponent), shape (1,)
- `agent_score` (np.ndarray): Agent's total score, shape (1,)
- `opponent_score` (np.ndarray): Opponent's total score, shape (1,)
- `first_picker` (int): Who picks first this round (0=agent, 1=opponent)
- `agent_history` (np.ndarray): Agent's board selections (indices), shape (5,), -1 = not played
- `opponent_history` (np.ndarray): Opponent's board selections (indices), shape (5,), -1 = not played

**Note:** The agent only sees the opponent's board selection **after** each round is played. This creates partial observability.

### Action Space

- Discrete(10): Select one board from the deck (indices 0-9)

### Rewards

- **Per round:** Score differential (agent_score - opponent_score)
- **Episode end bonus:** +100 for win, -100 for loss, 0 for tie

### Episode Structure

- 5 rounds per episode
- Each round: both players simultaneously select a board
- Boards are revealed and simulated
- Episode terminates after round 5

## Board Pools

Available board pools in `data/`:

- `boards_size_2.json`: 16 boards (15KB) - Very small boards, for testing
- `boards_size_3.json`: 500 boards (811KB) - Default, good for quick training
- `boards_size_3_large.json`: 1,704 boards (3.0MB) - Exhaustive size 3 boards
- `boards_size_4.json`: 5,000 boards (11MB) - Medium complexity
- `boards_size_4_large.json`: 25,000 boards (61MB) - High diversity
- `boards_size_5.json`: 50,000 boards (152MB) - Large boards, complex strategies

## Creating Custom Agents

To create your own agent, implement a policy that selects actions based on observations:

```python
def my_agent_policy(observation):
    """
    Custom agent policy.

    Args:
        observation: Dict with keys ['round', 'score_diff', 'agent_score',
                     'opponent_score', 'first_picker', 'agent_history',
                     'opponent_history']

    Returns:
        action: Integer in [0, 9] representing board index
    """
    # Example: Select board based on round number
    round_num = observation['round']

    # Get available boards (not yet played)
    agent_history = observation['agent_history']
    used_boards = set(agent_history[agent_history >= 0])
    available_boards = [i for i in range(10) if i not in used_boards]

    # Simple strategy: cycle through boards
    return available_boards[0] if available_boards else 0

# Use your policy
env = SpacesGameEnv()
obs, info = env.reset()

for _ in range(5):
    action = my_agent_policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

env.close()
```

## Next Steps: RL Training

The environment is ready for use with any RL library that supports Gymnasium:

### Stable Baselines3 (example)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from spaces_game import SpacesGameEnv

# Create vectorized environment
env = make_vec_env(lambda: SpacesGameEnv(), n_envs=4)

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("spaces_game_ppo")

# Load and evaluate
model = PPO.load("spaces_game_ppo")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
```

## Testing

Run the environment tests:
```bash
pytest tests/test_gym_env.py -v
```

Run all tests with coverage:
```bash
pytest --cov=spaces_game --cov-report=term-missing
```
