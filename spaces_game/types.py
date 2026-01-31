"""
Core data structures for Spaces Game engine.

All board-related types use frozen dataclasses for immutability.
This prevents accidental mutations during simulation and makes the code more predictable.
"""

from dataclasses import dataclass
from typing import Literal, Optional

# Type aliases
CellContent = Literal['empty', 'piece', 'trap', 'final']
MoveType = Literal['piece', 'trap', 'final']
Winner = Literal['player', 'opponent', 'tie']

# Board size constraints
MIN_BOARD_SIZE = 2
MAX_BOARD_SIZE = 99


def is_valid_board_size(size: int) -> bool:
    """Validate board size is within acceptable range."""
    return isinstance(size, int) and MIN_BOARD_SIZE <= size <= MAX_BOARD_SIZE


@dataclass(frozen=True)
class Position:
    """
    Position on the board.

    Coordinate system:
    - (0, 0) is top-left
    - row increases downward
    - col increases rightward
    - row -1 is the goal (off the top of the board)
    """
    row: int
    col: int

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"


@dataclass(frozen=True)
class BoardMove:
    """
    A single move in the board sequence.

    Attributes:
        position: Where the move occurs
        type: Type of move (piece movement, trap placement, or goal)
        order: Execution order in sequence (1, 2, 3, ...)
    """
    position: Position
    type: MoveType
    order: int

    def __str__(self) -> str:
        return f"{self.order}. {self.type} at {self.position}"


@dataclass(frozen=True)
class Board:
    """
    Complete board definition.

    A board represents one player's complete strategy - a sequence of moves
    from start to goal, including trap placements.

    Attributes:
        boardSize: Grid dimension (N for an NxN grid)
        grid: Immutable 2D grid showing final piece/trap positions
        sequence: Ordered list of moves defining the strategy

    Note: grid and sequence are stored as tuples for immutability.
    When loading from JSON, convert lists to tuples.
    """
    boardSize: int
    grid: tuple[tuple[CellContent, ...], ...]
    sequence: tuple[BoardMove, ...]

    def __post_init__(self) -> None:
        """Validate board structure after creation."""
        if not is_valid_board_size(self.boardSize):
            raise ValueError(f"Invalid board size: {self.boardSize}")

        if len(self.grid) != self.boardSize:
            raise ValueError(f"Grid rows don't match board size: {len(self.grid)} != {self.boardSize}")

        for i, row in enumerate(self.grid):
            if len(row) != self.boardSize:
                raise ValueError(f"Grid row {i} length doesn't match board size: {len(row)} != {self.boardSize}")


@dataclass(frozen=True)
class SimulationDetails:
    """
    Detailed information about a round simulation.

    Tracks all the low-level details of how the round played out,
    useful for debugging and analysis.
    """
    playerMoves: int
    opponentMoves: int
    playerHitTrap: bool
    opponentHitTrap: bool
    playerLastStep: int  # Last sequence step executed (-1 if none)
    opponentLastStep: int  # Last sequence step executed (-1 if none)
    playerTrapPosition: Optional[Position] = None
    opponentTrapPosition: Optional[Position] = None


@dataclass(frozen=True)
class RoundResult:
    """
    Result of a single round simulation.

    Contains everything needed to understand what happened in the round,
    including final positions, scores, and detailed simulation info.
    """
    round: int
    winner: Winner
    playerBoard: Board
    opponentBoard: Board
    playerFinalPosition: Position
    opponentFinalPosition: Position
    playerPoints: int
    opponentPoints: int
    collision: bool
    simulationDetails: SimulationDetails


@dataclass(frozen=True)
class GameResult:
    """
    Complete game result (multiple rounds).

    For round-by-round mode: 5 rounds
    For deck mode: 10 rounds (not yet implemented)
    """
    rounds: tuple[RoundResult, ...]
    finalPlayerScore: int
    finalOpponentScore: int
    winner: Winner


# Optional: Named board for human-readable output
@dataclass(frozen=True)
class NamedBoard:
    """
    Board with optional metadata for human-readable logs.
    Not used in core simulation, only for CLI/visualization.
    """
    board: Board
    name: Optional[str] = None

    def __str__(self) -> str:
        return self.name if self.name else f"Board(size={self.board.boardSize})"
