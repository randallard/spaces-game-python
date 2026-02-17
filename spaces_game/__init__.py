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
from .reverse_builder_env import ReverseCurriculumBuilderEnv
from .simultaneous_play_env import SimultaneousPlayEnv

# Training callbacks
from .callbacks import (
    discover_pools,
    build_phase_map,
    LEGACY_POOL_ORDER,
    DIFFICULTY_CHECKPOINTS,
    OpponentProgressionCallback,
    SelfPlayCurriculumCallback,
)

# Optional: BoardModifierEnv (Stage 1.5 - experimental)
try:
    from .modifier_env import BoardModifierEnv
    _has_modifier = True
except ImportError:
    _has_modifier = False

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
    "ReverseCurriculumBuilderEnv",
    "SimultaneousPlayEnv",
    # Callbacks
    "discover_pools",
    "build_phase_map",
    "LEGACY_POOL_ORDER",
    "DIFFICULTY_CHECKPOINTS",
    "OpponentProgressionCallback",
    "SelfPlayCurriculumCallback",
]

# Add BoardModifierEnv to exports if available
if _has_modifier:
    __all__.append("BoardModifierEnv")
