"""
Pydantic request/response models for the inference server API.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SkillLevel(str, Enum):
    """Available skill levels for the AI agent."""
    beginner = "beginner"
    beginner_plus = "beginner_plus"
    intermediate = "intermediate"
    intermediate_plus = "intermediate_plus"
    advanced = "advanced"
    advanced_plus = "advanced_plus"


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
