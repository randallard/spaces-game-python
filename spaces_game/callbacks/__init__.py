"""Training callbacks for the Spaces Game RL pipeline."""

from .pool_utils import (
    discover_pools,
    build_phase_map,
    LEGACY_POOL_ORDER,
    DIFFICULTY_CHECKPOINTS,
)
from .opponent_progression import OpponentProgressionCallback
from .self_play import SelfPlayCurriculumCallback

__all__ = [
    "discover_pools",
    "build_phase_map",
    "LEGACY_POOL_ORDER",
    "DIFFICULTY_CHECKPOINTS",
    "OpponentProgressionCallback",
    "SelfPlayCurriculumCallback",
]
