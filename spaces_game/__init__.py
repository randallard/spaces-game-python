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

# Will be added as we port more modules
# from .simulation import simulate_round
# from .validation import validate_board, validate_board_or_throw

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
    # Coming soon
    # "simulate_round",
    # "validate_board",
    # "validate_board_or_throw",
]
