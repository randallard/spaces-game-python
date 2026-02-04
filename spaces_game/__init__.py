"""
Spaces Game - Python Implementation

A two-player abstract strategy game engine for ML/RL research.
"""

__version__ = "0.1.0"

# Core types
from .types import (
    Board,
    BoardMove,
    Position,
    RoundResult,
    SimulationDetails,
    GameResult,
)

# Board loading
from .board_loader import (
    load_boards_from_json,
    load_board_by_index,
    BoardPool,
)

# Validation
from .validation import (
    validate_board,
    validate_board_or_throw,
    is_board_playable,
    ValidationResult,
)

# Simulation
from .simulation import (
    simulate_round,
    simulate_multiple_rounds,
)

# Gymnasium Environments
from .gym_env import SpacesGameEnv
from .construction_env import BoardConstructionEnv
from .builder_env import BoardBuilderEnv

__all__ = [
    "__version__",
    # Types
    "Board",
    "BoardMove",
    "Position",
    "RoundResult",
    "SimulationDetails",
    "GameResult",
    # Board loading
    "load_boards_from_json",
    "load_board_by_index",
    "BoardPool",
    # Validation
    "validate_board",
    "validate_board_or_throw",
    "is_board_playable",
    "ValidationResult",
    # Simulation
    "simulate_round",
    "simulate_multiple_rounds",
    # Gymnasium
    "SpacesGameEnv",
    "BoardConstructionEnv",
    "BoardBuilderEnv",
]
