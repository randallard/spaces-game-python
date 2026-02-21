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

### Command Line Interface

The package includes a CLI for common operations:

```bash
# Verify parity with TypeScript (52 test cases)
spaces-game test-parity

# Validate board files
spaces-game validate data/boards_size_3.json

# Play a game
spaces-game play data/boards_size_3.json --seed 42

# Show board pool statistics
spaces-game stats data/boards_size_3.json

# Generate boards (wraps TypeScript CLI)
spaces-game generate-boards --size 3 --limit 1000 --output my_boards.json
```

See [CLI.md](CLI.md) for complete CLI documentation.

### Basic Simulation

```python
from spaces_game import simulate_round, load_board_by_index

# Load pre-generated boards
player_board = load_board_by_index('data/boards_size_3.json', 0)
opponent_board = load_board_by_index('data/boards_size_3.json', 42)

# Run simulation
result = simulate_round(1, player_board, opponent_board)

print(f"Winner: {result.winner}")
print(f"Player: {result.playerPoints} points")
print(f"Opponent: {result.opponentPoints} points")
```

### Gymnasium Environment

```python
from spaces_game import SpacesGameEnv

# Create environment
env = SpacesGameEnv(
    board_pool_path='data/boards_size_3.json',
    deck_size=10,
    opponent_strategy='random'
)

# Training loop
obs, info = env.reset(seed=42)
terminated = False

while not terminated:
    action = env.action_space.sample()  # Your agent here
    obs, reward, terminated, truncated, info = env.step(action)

print(f"Final score: {info['agent_total_score']} - {info['opponent_total_score']}")
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
├── mcp_server/           # MCP server for Claude Desktop
│   ├── main.py           # FastMCP server (resource + tools)
│   ├── config.py         # Knowledge base path config
│   └── board_helpers.py  # JSON ↔ dataclass conversion
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

## MCP Server

An MCP server exposes game knowledge, board validation, and round simulation as tools for Claude Desktop. See [MCP_SERVER.md](../MCP_SERVER.md) for setup and usage.

## Training Progress

See [TRAINING_PLAN.md](TRAINING_PLAN.md) for full details, curriculum design, and training commands. See [EXPERIMENTS.md](EXPERIMENTS.md) for fog of war experiments and LLM vs RL comparisons. See [DEPLOYMENT.md](DEPLOYMENT.md) for Railway deployment and inference server setup.

### Completed Stages

- **Stage 0 - Deck Selection**: Board evaluation and matchup strategy
- **Stage 1 - Board Construction**: 100% optimal counter-play on 8 curated size-2 boards
- **Stage 2 - Reverse Curriculum**: Obsolete (replaced by strict masking in Stage 3)
- **Stage 3 - Simultaneous 5-Round Play (Full Reveal)**: Blind board construction + opponent adaptation
  - Size 2 + 3: Complete (Feb 12, 2026). All opponent phases cleared.
  - Size 4: Complete (Feb 15, 2026). 100% valid rate, ~78% win rate with self-play opponent mixing.
  - Feb 14 rework: strict action masking (invalid boards impossible), flat Discrete action space, forward-only movement, self-play added.
  - Feb 15: Added `--self-play-ratio` for pool opponent mixing during self-play, preventing specialization collapse.
- **Stage 4 - Fog of War**: In progress (Feb 16-17, 2026). Partial opponent board reveal after simulation.
  - `--fog` flag on `train_simultaneous.py` enables fog-filtered encoding + `fog_outcomes` observation
  - Agent only sees opponent moves up to `playerLastStep`; traps hidden except sprung trap
  - Size 3 fog pool-only: converged at ~82% win rate by 256k steps, all 6 phases cleared
  - Size 3 fog + self-play: in progress with quality controls (snapshot gate, recovery threshold, seed model)

### Current Status: Stage 4 Fog + Self-Play (Size 3)

Stage 3 (full reveal) solved for sizes 2-4. Stage 4 fog pool-only training converged quickly (~82% win rate, 256k steps). Initial fog + self-play attempt showed the opponent snapshot pool can degrade into a quality death spiral — weak snapshots produce weaker training signal.

Three quality controls added (Feb 17):
- **Recovery win rate** (`--recovery-win-rate`): model must hit target win rate against pool opponents before self-play resumes
- **Snapshot quality gate** (`--snapshot-win-rate`): only snapshot when pool win rate is above threshold, preventing bad models from entering the pool
- **Seed model**: resumed model is kept as a permanent pool member, never pruned

```bash
# Stage 3: Full reveal (sizes 2-4 already solved)
python examples/train_simultaneous.py --size 4 --self-play --timesteps 5000000

# Stage 4: Fog + self-play with quality controls
python examples/train_simultaneous.py --size 3 --fog --self-play \
    --warmup-steps 0 \
    --resume models/size3/stage4/best/best_model.zip \
    --timesteps 7500000 \
    --self-play-block-steps 500000 \
    --pool-recovery-steps 100000 \
    --min-pool-win-rate 0.60 \
    --recovery-win-rate 0.80 \
    --snapshot-win-rate 0.65
```

### Play Against the Agent

```bash
# With difficulty selection (after training produces checkpoints)
python examples/play_against_agent.py --size 3 --difficulty beginner
python examples/play_against_agent.py --size 3 --difficulty expert

# Interactive model selection
python examples/play_against_agent.py --size 3 --board-library new_boards_3.json

# With fog of war display (human sees partial opponent boards)
python examples/play_against_agent.py --size 3 --rounds 5 --fog

# Stochastic mode (agent samples from policy for varied play)
python examples/play_against_agent.py --size 3 --difficulty beginner --stochastic
```

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
