# Spaces Game - Python Implementation

Python port of the Spaces Game engine for ML/RL training with native Gymnasium integration.

## Overview

This is a Python implementation of the Spaces Game simulation engine, designed for reinforcement learning and machine learning research. The Python version maintains 100% parity with the TypeScript implementation.

## Features

- ✅ **Identical Results**: Matches TypeScript engine exactly (validated with 52+ test cases)
- 🎯 **Gymnasium Integration**: Standard RL environment interface
- 🚀 **High Performance**: Optimized for training loops
- 🔒 **Type Safe**: Full type hints with mypy strict mode
- 📊 **Pre-generated Boards**: Fast opponent sampling from validated board pools
- 🧪 **Comprehensive Tests**: 90%+ coverage with property-based testing

## Installation

```bash
# Clone repository
git clone <repo-url>
cd spaces-game-python

# Install dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Simulation

```python
from spaces_game import simulate_round, load_boards

# Load pre-generated boards
player_board = load_boards('my-boards.json', index=0)
opponent_board = load_boards('data/boards_size_3.json', index=42)

# Run simulation
result = simulate_round(1, player_board, opponent_board)

print(f"Winner: {result.winner}")
print(f"Player: {result.player_points} points")
print(f"Opponent: {result.opponent_points} points")
```

### Gymnasium Environment

```python
import gymnasium as gym
from spaces_game.gym_env import SpacesGameEnv

# Create environment
env = SpacesGameEnv(board_size=3, opponent_pool='data/boards_size_3.json')

# Training loop
obs, info = env.reset()
for _ in range(5):  # 5 rounds
    action = env.action_space.sample()  # Your agent here
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break

print(f"Final score: {info['score']}")
```

## Project Structure

```
spaces-game-python/
├── spaces_game/          # Main package
│   ├── __init__.py
│   ├── simulation.py     # Core game engine
│   ├── validation.py     # Board validation
│   ├── types.py          # Data structures (frozen dataclasses)
│   ├── board_loader.py   # Load pre-generated boards
│   └── gym_env.py        # Gymnasium environment
├── data/                 # Pre-generated board pools
│   ├── boards_size_2.json
│   ├── boards_size_3.json
│   ├── boards_size_4.json
│   └── boards_size_5.json
├── tests/                # Test suite
│   ├── test_simulation.py
│   ├── test_validation.py
│   ├── test_parity.py    # Cross-validation with TypeScript
│   └── test_gym_env.py
└── tools/                # Utilities
    └── generate_boards.sh
```

## Board Pools

Pre-generated opponent boards are included for sizes 2-5:

- **Size 2**: ~16 boards (complete exhaustive)
- **Size 3**: ~500 boards (training baseline)
- **Size 4**: ~5,000 boards (intermediate)
- **Size 5**: ~50,000 boards (advanced)

Boards are generated using the TypeScript CLI and validated for:
- Legal move sequences
- No backward movement
- No duplicate traps
- Proper trap adjacency
- Goal completion

## Development

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=spaces_game --cov-report=html

# Parity tests only (validate against TypeScript)
pytest tests/test_parity.py -v

# Property-based tests
pytest tests/test_simulation.py::test_deterministic_results
```

### Type Checking

```bash
mypy spaces_game --strict
```

### Generating New Board Pools

```bash
# From TypeScript engine
cd ../spaces-game-engine
npm run cli -- generate-boards --size 3 --limit 500 --output ../spaces-game-python/data/boards_size_3.json
```

## Training Curriculum

Recommended progression for RL training:

1. **Size 3** (500 boards): Learn fundamentals
   - Basic trap placement
   - Forward movement optimization
   - Opponent anticipation

2. **Size 4** (5K boards): Intermediate complexity
   - Multi-step planning
   - Trap combination strategies
   - Column selection importance

3. **Size 5** (50K boards): Advanced play
   - Long-term planning
   - Complex trap networks
   - Meta-game patterns

4. **Size 6+**: Transfer learning (sampled)

## Gymnasium Environment Details

### Observation Space

The environment uses **partial observability** (opponent's current board is hidden):

```python
observation = {
    'round': int,              # Current round (1-5)
    'score_diff': float,       # Score differential
    'my_board_history': [int], # Indices of boards I used
    'opp_board_history': [int],# Indices opponent used (revealed after round)
    'who_picks_first': int,    # 0=me, 1=opponent
}
```

### Action Space

```python
action = int  # Select board index from deck (0-9)
```

### Reward Shaping

- **Win**: +10
- **Loss**: -10
- **Tie**: 0
- **Score differential**: ±1 per point difference

## Parity with TypeScript

This implementation maintains 100% parity with the TypeScript version:

- ✅ All 52 test cases from `session-2026-01-30T13-28-47-534Z.json` pass
- ✅ 1000+ random boards produce identical results
- ✅ Property-based tests verify determinism
- ✅ Same scoring, collision detection, trap mechanics

See `tests/test_parity.py` for validation details.

## Design Principles

1. **Immutability**: Board definitions use frozen dataclasses
2. **Type Safety**: Strict mypy checking throughout
3. **Performance**: Optimized for training loops
4. **Testability**: 90%+ code coverage
5. **Parity**: Identical to TypeScript implementation

## Resources

- **Game Rules**: See `GAME_DESCRIPTION.md` in TypeScript repo
- **TypeScript Implementation**: `../spaces-game-engine/`
- **Web Version**: https://spaces-game.vercel.app
- **Port Plan**: See `PYTHON_PORT_PLAN.md` in TypeScript repo

## License

MIT
