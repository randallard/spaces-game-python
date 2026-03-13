"""
Pydantic request/response models for the inference server API.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Agent type selection: standard (full reveal) or fog (fog of war)."""
    standard = "standard"
    fog = "fog"


class SkillLevel(str, Enum):
    """Available skill levels for the AI agent."""
    beginner = "beginner"
    beginner_plus = "beginner_plus"
    intermediate = "intermediate"
    intermediate_plus = "intermediate_plus"
    advanced = "advanced"
    advanced_plus = "advanced_plus"
    test_fail = "test_fail"
    scripted_1 = "scripted_1"
    scripted_2 = "scripted_2"
    scripted_3 = "scripted_3"
    scripted_4 = "scripted_4"
    scripted_5 = "scripted_5"


class RoundHistoryEntry(BaseModel):
    """A single completed round from the AI agent's perspective."""
    agent_score: float = 0.0
    opponent_score: float = 0.0
    agent_board: str = ""
    opponent_board_fog: str = ""
    agent_last_step: int = -1
    opponent_last_step: int = -1
    agent_hit_trap: bool = False
    opponent_hit_trap: bool = False
    collision: bool = False


class MoveDict(BaseModel):
    """A single move in a board sequence."""
    row: int
    col: int
    type: str
    order: int


class OpponentBoardHistory(BaseModel):
    """A board from a previous round, as sent by the frontend."""
    sequence: list[MoveDict] = Field(default_factory=list)


class ConstructBoardRequest(BaseModel):
    """Request to construct a board for a given round."""
    board_size: int = Field(..., ge=2, le=99, description="Board grid dimension (NxN)")
    round_num: int = Field(..., ge=0, le=4, description="Current round number (0-indexed)")
    agent_score: float = Field(default=0.0, description="Agent's cumulative score")
    opponent_score: float = Field(default=0.0, description="Opponent's cumulative score")
    opponent_history: list[OpponentBoardHistory] = Field(
        default_factory=list,
        description="Opponent's boards from previous rounds",
    )
    skill_level: SkillLevel = Field(
        default=SkillLevel.intermediate,
        description="AI skill level",
    )
    round_scores: list[dict] = Field(
        default_factory=list,
        description="Per-round scores: [{agent: float, opponent: float}, ...]",
    )
    round_history: list[RoundHistoryEntry] = Field(
        default_factory=list,
        description="Rich per-round history from the AI agent's perspective",
    )
    agent_type: AgentType = Field(
        default=AgentType.fog,
        description="Agent type: 'standard' (full reveal) or 'fog' (fog of war)",
    )
    model_id: Optional[str] = Field(
        None,
        description="Stable model ID (overrides model_index/skill_level/agent_type). See GET /models for available IDs.",
    )
    model_index: Optional[int] = Field(
        None,
        description="Direct model index (overrides skill_level/agent_type). See GET /models for available indices.",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. None = use skill_level default, 0.0 = deterministic, 1.0 = standard stochastic, >1.0 = more random.",
    )


class MovePosition(BaseModel):
    """Position on the board."""
    row: int
    col: int


class ResponseMove(BaseModel):
    """A move in the response board sequence."""
    position: MovePosition
    type: str
    order: int


class ConstructBoardResponse(BaseModel):
    """Response containing the constructed board."""
    board: Optional[dict] = Field(
        None,
        description="Board with sequence, boardSize, and grid",
    )
    valid: bool = Field(default=False, description="Whether the board is playable")
    attempts_used: int = Field(default=1, description="Number of construction attempts used")
    model_info: Optional[dict] = Field(
        None,
        description="Optional info about the model used",
    )
    error: Optional[str] = Field(None, description="Error message if request failed")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"


class InfoResponse(BaseModel):
    """Server info response."""
    loaded_models: dict = Field(default_factory=dict)
    supported_board_sizes: list[int] = Field(default_factory=list)
