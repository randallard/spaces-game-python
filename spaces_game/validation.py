"""
Board validation utilities.

Validates boards using the same logic as the TypeScript engine.
Critical for ensuring Python and TypeScript implementations match exactly.
"""

from dataclasses import dataclass
from typing import Optional

from .types import Board, Position


@dataclass
class ValidationResult:
    """Validation result with user-friendly error messages."""

    valid: bool
    errors: list[str]


def _position_key(row: int, col: int) -> str:
    """Generate unique key for position."""
    return f"{row},{col}"


def _is_adjacent_orthogonal(from_pos: Position, to_pos: Position) -> bool:
    """
    Check if two positions are adjacent orthogonally (up, down, left, right).
    Not diagonally - only one direction, exactly 1 square away.
    """
    row_diff = abs(to_pos.row - from_pos.row)
    col_diff = abs(to_pos.col - from_pos.col)

    # Must move exactly 1 square in one direction (orthogonal)
    return (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)


def is_board_playable(board: Board) -> bool:
    """
    Validate a board using core game rules.

    This is the authoritative validation function - all other validation
    should defer to this function.

    Args:
        board: Board to validate

    Returns:
        True if board is playable, False otherwise

    Validation rules:
    - Must have at least one move in sequence
    - All positions must be in bounds
    - Piece moves must be orthogonally adjacent (no diagonals/jumps)
    - Piece cannot move into a trap
    - Traps must be placed adjacent to current piece position (or at current position for supermove)
    - Supermove (trap at current position) requires piece to move immediately after
    - Final move must be at row -1
    - Sequence must contain a final move (piece must reach the goal)
    - Piece must visit every row (0 through board_size-1)
    """
    # Must have at least one move in sequence
    if len(board.sequence) == 0:
        return False

    size = len(board.grid)
    max_index = size - 1

    # Track current piece position and trap locations
    current_position: Optional[Position] = None
    trap_positions: set[str] = set()
    supermove_position: Optional[Position] = None  # Track if last move was supermove
    rows_with_piece: set[int] = set()  # Track rows visited by piece moves
    has_final: bool = False

    # Validate each move in sequence
    for i, move in enumerate(board.sequence):
        row = move.position.row
        col = move.position.col

        # Skip validation for final moves (they're off the board at row -1)
        if move.type == 'final':
            if row != -1:
                return False  # Final moves must be at row -1
            # Supermove constraint: piece must move before reaching goal
            if supermove_position is not None:
                return False  # Cannot reach goal right after supermove without moving
            # Piece must have reached the top row (row 0) to cross the goal line
            if current_position is None:
                return False  # No piece placed before goal
            if current_position.row != 0:
                return False  # Piece must be at row 0 to reach goal at row -1
            # Goal column must match piece column (adjacent move up)
            if current_position.col != col:
                return False  # Goal must be directly above piece
            has_final = True
            continue  # Don't check grid content for final moves

        # Check bounds (dynamic grid size)
        if row < 0 or row > max_index or col < 0 or col > max_index:
            return False

        # Process piece moves
        if move.type == 'piece':
            # Piece position must have 'piece' or 'trap' in grid (trap overrides piece waypoint)
            content = board.grid[row][col]
            if content != 'piece' and content != 'trap':
                return False  # Piece move must correspond to piece or trap in grid

            # Check supermove constraint: piece MUST move after placing trap at current position
            if supermove_position is not None:
                # Piece must move to a different position
                if move.position.row == supermove_position.row and move.position.col == supermove_position.col:
                    return False  # Piece must move away from supermove position
                supermove_position = None  # Supermove constraint satisfied

            if current_position is not None:
                # Check adjacency for piece moves (orthogonal only)
                if not _is_adjacent_orthogonal(current_position, move.position):
                    return False  # Invalid: diagonal or jump move

                # Check that piece is not moving into a trap
                target_key = _position_key(move.position.row, move.position.col)
                if target_key in trap_positions:
                    return False  # Invalid: piece cannot move into trap

            # Update current position
            current_position = move.position
            rows_with_piece.add(row)

        # Process trap moves
        if move.type == 'trap':
            # Check that grid has trap at this position
            content = board.grid[row][col]
            if content != 'trap':
                return False  # Trap in sequence must match grid

            if current_position is None:
                return False  # Cannot place trap before piece is on board

            # Check if trap is at current position (supermove)
            same_position = (
                move.position.row == current_position.row and
                move.position.col == current_position.col
            )

            if same_position:
                # Supermove: trap at current position
                # Mark that piece must move on next step
                supermove_position = move.position
            else:
                # Trap at different position - must be adjacent to current piece position
                if not _is_adjacent_orthogonal(current_position, move.position):
                    return False  # Invalid: trap not adjacent to piece

                # Supermove constraint: cannot place adjacent trap right after supermove
                if supermove_position is not None:
                    return False  # Must move piece first before placing another trap

            # Track trap position
            trap_positions.add(_position_key(move.position.row, move.position.col))

    # Final check: if sequence ends with supermove without moving, that's invalid
    # (unless goal is reached, but that's already handled above)
    if supermove_position is not None:
        return False  # Sequence cannot end with supermove without piece moving

    # Must reach the goal (sequence must contain a final move)
    if not has_final:
        return False

    # Piece must visit every row (path from bottom to top)
    required_rows = set(range(size))
    if rows_with_piece != required_rows:
        return False

    return True


def validate_board(board: Board) -> ValidationResult:
    """
    Validate a board using the engine's validation logic.

    This is a thin wrapper around is_board_playable() that provides
    user-friendly error messages for CLI display.

    IMPORTANT: Uses exact same validation as RL/ML training.
    DO NOT add additional validation logic here.

    Args:
        board: Board to validate

    Returns:
        Validation result with error messages
    """
    is_valid = is_board_playable(board)

    if is_valid:
        return ValidationResult(
            valid=True,
            errors=[],
        )

    # Board is invalid - provide user-friendly error message
    # Note: is_board_playable() doesn't return specific error reasons,
    # so we provide a generic message pointing to common issues
    return ValidationResult(
        valid=False,
        errors=[
            'Board validation failed. Common issues:',
            '  • Diagonal or jump moves (only orthogonal moves allowed)',
            '  • Piece moving into a trap',
            '  • Trap placed non-adjacent to piece position',
            '  • Supermove without immediate piece movement',
            '  • Sequence pointing to empty cells',
            '  • Invalid board size or out-of-bounds positions',
            '',
            'Run with verbose mode to see full board details.',
        ],
    )


def validate_board_or_throw(board: Board) -> None:
    """
    Validate a board and raise an error if invalid.

    Args:
        board: Board to validate

    Raises:
        ValueError: If board is invalid
    """
    result = validate_board(board)

    if not result.valid:
        raise ValueError('\n'.join(result.errors))


def is_adjacent_orthogonal(
    from_pos: Position,
    to_pos: Position
) -> bool:
    """
    Check if a move is orthogonally adjacent to current position.

    Used for real-time validation during interactive board building.

    Args:
        from_pos: Current position
        to_pos: Target position

    Returns:
        True if move is valid (1 square orthogonally)
    """
    return _is_adjacent_orthogonal(from_pos, to_pos)


def is_position_in_bounds(
    position: Position,
    board_size: int
) -> bool:
    """
    Check if a position is within board bounds.

    Args:
        position: Position to check
        board_size: Board size

    Returns:
        True if position is valid
    """
    # Allow row -1 for final/goal position
    return (
        position.row >= -1 and
        position.row < board_size and
        position.col >= 0 and
        position.col < board_size
    )


def validate_interactive_move(
    current_position: Optional[Position],
    next_position: Position,
    move_type: str,  # 'piece' or 'trap'
    board_size: int,
    existing_traps: set[str]
) -> ValidationResult:
    """
    Validate a move during interactive building.

    Provides immediate feedback before adding to sequence.

    Args:
        current_position: Current piece position (None if first move)
        next_position: Proposed next position
        move_type: Type of move ('piece' or 'trap')
        board_size: Board size
        existing_traps: Set of existing trap positions (as "row,col" strings)

    Returns:
        Validation result
    """
    errors: list[str] = []

    # Check bounds
    if not is_position_in_bounds(next_position, board_size):
        errors.append('Position is out of bounds')

    # Check move validity
    if current_position and move_type == 'piece':
        # Piece move must be adjacent and orthogonal
        if not is_adjacent_orthogonal(current_position, next_position):
            errors.append('Piece must move exactly 1 square orthogonally (up/down/left/right)')

        # Piece cannot move into trap
        trap_key = f"{next_position.row},{next_position.col}"
        if trap_key in existing_traps:
            errors.append('Piece cannot move into a trap')

    if current_position and move_type == 'trap':
        # Trap must be adjacent OR at current position (supermove)
        same_position = (
            next_position.row == current_position.row and
            next_position.col == current_position.col
        )

        if not same_position and not is_adjacent_orthogonal(current_position, next_position):
            errors.append('Trap must be placed adjacent to current position or at current position (supermove)')

        # Warn about supermove
        if same_position:
            errors.append('⚠️  SUPERMOVE: Piece must move out of this space on the very next step')

    return ValidationResult(
        valid=len([e for e in errors if not e.startswith('⚠️')]) == 0,
        errors=errors,
    )
