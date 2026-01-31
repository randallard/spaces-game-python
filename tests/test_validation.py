"""
Tests for board validation utilities.
"""

import pytest

from spaces_game.validation import (
    is_board_playable,
    validate_board,
    validate_board_or_throw,
    is_adjacent_orthogonal,
    is_position_in_bounds,
    validate_interactive_move,
    ValidationResult,
)
from spaces_game.types import Board, BoardMove, Position
from spaces_game.board_loader import load_boards_from_json


class TestIsBoardPlayable:
    """Test core board validation logic."""

    def test_valid_boards_from_pool(self):
        """All pre-generated boards should be valid."""
        boards = load_boards_from_json('data/boards_size_2.json')

        for board in boards:
            assert is_board_playable(board), f"Board should be valid: {board}"

    def test_empty_sequence(self):
        """Board with empty sequence is invalid."""
        board = Board(
            boardSize=2,
            grid=(('empty', 'empty'), ('empty', 'empty')),
            sequence=(),
        )

        assert not is_board_playable(board)

    def test_final_move_wrong_row(self):
        """Final move must be at row -1."""
        board = Board(
            boardSize=2,
            grid=(
                ('piece', 'empty'),
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(1, 0), 'final', 1),  # Wrong: should be row -1, not row 1
            ),
        )

        assert not is_board_playable(board)

    def test_piece_move_into_trap(self):
        """Piece cannot move into a trap."""
        board = Board(
            boardSize=2,
            grid=(
                ('piece', 'trap'),
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 1), 'trap', 1),
                BoardMove(Position(0, 1), 'piece', 2),  # Invalid: moving into trap
            ),
        )

        assert not is_board_playable(board)

    def test_diagonal_move(self):
        """Piece cannot move diagonally."""
        board = Board(
            boardSize=2,
            grid=(
                ('piece', 'empty'),
                ('empty', 'piece')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(1, 1), 'piece', 1),  # Invalid: diagonal
            ),
        )

        assert not is_board_playable(board)

    def test_jump_move(self):
        """Piece cannot jump (must move exactly 1 square)."""
        board = Board(
            boardSize=3,
            grid=(
                ('piece', 'empty', 'piece'),
                ('empty', 'empty', 'empty'),
                ('empty', 'empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 2), 'piece', 1),  # Invalid: jumping 2 squares
            ),
        )

        assert not is_board_playable(board)

    def test_trap_not_adjacent(self):
        """Trap must be adjacent to current piece position."""
        board = Board(
            boardSize=3,
            grid=(
                ('piece', 'empty', 'trap'),
                ('empty', 'empty', 'empty'),
                ('empty', 'empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 2), 'trap', 1),  # Invalid: trap not adjacent
            ),
        )

        assert not is_board_playable(board)

    def test_trap_before_piece(self):
        """Cannot place trap before piece is on board."""
        board = Board(
            boardSize=2,
            grid=(
                ('trap', 'piece'),
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'trap', 0),  # Invalid: no piece yet
                BoardMove(Position(0, 1), 'piece', 1),
            ),
        )

        assert not is_board_playable(board)

    def test_supermove_without_moving(self):
        """Supermove requires piece to move immediately after."""
        board = Board(
            boardSize=2,
            grid=(
                ('trap', 'trap'),
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 0), 'trap', 1),  # Supermove at current position
                BoardMove(Position(0, 1), 'trap', 2),  # Invalid: must move piece first
            ),
        )

        assert not is_board_playable(board)

    def test_supermove_reaching_goal(self):
        """Cannot reach goal immediately after supermove without moving."""
        board = Board(
            boardSize=2,
            grid=(
                ('trap', 'empty'),
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 0), 'trap', 1),  # Supermove
                BoardMove(Position(-1, 0), 'final', 2),  # Invalid: must move piece first
            ),
        )

        assert not is_board_playable(board)

    def test_valid_supermove(self):
        """Valid supermove followed by piece movement."""
        # Grid shows 'trap' at (0,0) because trap overrides piece waypoint display
        board = Board(
            boardSize=2,
            grid=(
                ('trap', 'piece'),  # (0,0) shows trap (piece goes here first, then trap placed)
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),  # Piece at (0,0) - grid can show 'piece' or 'trap'
                BoardMove(Position(0, 0), 'trap', 1),   # Supermove: trap at current position
                BoardMove(Position(0, 1), 'piece', 2),  # Valid: piece moves out
                BoardMove(Position(-1, 1), 'final', 3),  # Final position at row -1
            ),
        )

        assert is_board_playable(board)

    def test_sequence_mismatch_grid(self):
        """Sequence must match grid content."""
        board = Board(
            boardSize=2,
            grid=(
                ('piece', 'empty'),  # Grid says empty
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 1), 'trap', 1),  # Sequence says trap, but grid is empty
            ),
        )

        assert not is_board_playable(board)

    def test_out_of_bounds(self):
        """Position must be within board bounds."""
        board = Board(
            boardSize=2,
            grid=(
                ('piece', 'empty'),
                ('empty', 'empty')
            ),
            sequence=(
                BoardMove(Position(0, 0), 'piece', 0),
                BoardMove(Position(0, 5), 'piece', 1),  # Out of bounds
            ),
        )

        assert not is_board_playable(board)


class TestValidateBoard:
    """Test user-facing validation function."""

    def test_valid_board(self):
        """Valid board returns success result."""
        boards = load_boards_from_json('data/boards_size_2.json')
        board = boards[0]

        result = validate_board(board)

        assert result.valid
        assert len(result.errors) == 0

    def test_invalid_board(self):
        """Invalid board returns error messages."""
        board = Board(
            boardSize=2,
            grid=(('empty', 'empty'), ('empty', 'empty')),
            sequence=(),  # Invalid: empty sequence
        )

        result = validate_board(board)

        assert not result.valid
        assert len(result.errors) > 0
        assert any('validation failed' in err.lower() for err in result.errors)


class TestValidateBoardOrThrow:
    """Test exception-throwing validation."""

    def test_valid_board_no_exception(self):
        """Valid board does not raise exception."""
        boards = load_boards_from_json('data/boards_size_2.json')
        board = boards[0]

        # Should not raise
        validate_board_or_throw(board)

    def test_invalid_board_raises(self):
        """Invalid board raises ValueError."""
        board = Board(
            boardSize=2,
            grid=(('empty', 'empty'), ('empty', 'empty')),
            sequence=(),
        )

        with pytest.raises(ValueError) as exc_info:
            validate_board_or_throw(board)

        assert 'validation failed' in str(exc_info.value).lower()


class TestIsAdjacentOrthogonal:
    """Test adjacency checking."""

    def test_adjacent_right(self):
        """Moving right 1 square is valid."""
        assert is_adjacent_orthogonal(Position(0, 0), Position(0, 1))

    def test_adjacent_left(self):
        """Moving left 1 square is valid."""
        assert is_adjacent_orthogonal(Position(0, 1), Position(0, 0))

    def test_adjacent_down(self):
        """Moving down 1 square is valid."""
        assert is_adjacent_orthogonal(Position(0, 0), Position(1, 0))

    def test_adjacent_up(self):
        """Moving up 1 square is valid."""
        assert is_adjacent_orthogonal(Position(1, 0), Position(0, 0))

    def test_diagonal_invalid(self):
        """Diagonal moves are invalid."""
        assert not is_adjacent_orthogonal(Position(0, 0), Position(1, 1))

    def test_jump_invalid(self):
        """Jumping 2+ squares is invalid."""
        assert not is_adjacent_orthogonal(Position(0, 0), Position(0, 2))

    def test_same_position_invalid(self):
        """Same position is not adjacent."""
        assert not is_adjacent_orthogonal(Position(0, 0), Position(0, 0))


class TestIsPositionInBounds:
    """Test bounds checking."""

    def test_valid_position(self):
        """Normal position is in bounds."""
        assert is_position_in_bounds(Position(0, 0), 3)
        assert is_position_in_bounds(Position(2, 2), 3)

    def test_final_position(self):
        """Final position (row -1) is allowed."""
        assert is_position_in_bounds(Position(-1, 0), 3)

    def test_negative_col_invalid(self):
        """Negative column is out of bounds."""
        assert not is_position_in_bounds(Position(0, -1), 3)

    def test_row_too_large(self):
        """Row >= board_size is out of bounds."""
        assert not is_position_in_bounds(Position(3, 0), 3)

    def test_col_too_large(self):
        """Col >= board_size is out of bounds."""
        assert not is_position_in_bounds(Position(0, 3), 3)


class TestValidateInteractiveMove:
    """Test interactive move validation."""

    def test_valid_piece_move(self):
        """Valid piece move."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(0, 1),
            move_type='piece',
            board_size=3,
            existing_traps=set()
        )

        assert result.valid
        assert len([e for e in result.errors if not e.startswith('⚠️')]) == 0

    def test_piece_move_into_trap(self):
        """Piece cannot move into trap."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(0, 1),
            move_type='piece',
            board_size=3,
            existing_traps={'0,1'}
        )

        assert not result.valid
        assert any('trap' in err.lower() for err in result.errors)

    def test_piece_diagonal_move(self):
        """Piece cannot move diagonally."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(1, 1),
            move_type='piece',
            board_size=3,
            existing_traps=set()
        )

        assert not result.valid
        assert any('orthogonal' in err.lower() for err in result.errors)

    def test_valid_trap_adjacent(self):
        """Valid trap placement adjacent to piece."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(0, 1),
            move_type='trap',
            board_size=3,
            existing_traps=set()
        )

        assert result.valid

    def test_valid_trap_supermove(self):
        """Valid supermove (trap at current position)."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(0, 0),
            move_type='trap',
            board_size=3,
            existing_traps=set()
        )

        # Valid but with warning
        assert result.valid
        assert any('SUPERMOVE' in err for err in result.errors)

    def test_trap_not_adjacent(self):
        """Trap must be adjacent to piece."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(0, 2),
            move_type='trap',
            board_size=3,
            existing_traps=set()
        )

        assert not result.valid
        assert any('adjacent' in err.lower() for err in result.errors)

    def test_out_of_bounds(self):
        """Position out of bounds."""
        result = validate_interactive_move(
            current_position=Position(0, 0),
            next_position=Position(0, 5),
            move_type='piece',
            board_size=3,
            existing_traps=set()
        )

        assert not result.valid
        assert any('bounds' in err.lower() for err in result.errors)


class TestValidationParity:
    """
    Test validation against pre-generated boards.

    All boards in our pools should pass validation.
    """

    def test_all_size_2_boards_valid(self):
        """All size 2 boards should be valid."""
        boards = load_boards_from_json('data/boards_size_2.json')

        for i, board in enumerate(boards):
            result = validate_board(board)
            assert result.valid, f"Board {i} should be valid but got errors: {result.errors}"

    def test_all_size_3_boards_valid(self):
        """All size 3 boards should be valid."""
        boards = load_boards_from_json('data/boards_size_3.json')

        for i, board in enumerate(boards):
            result = validate_board(board)
            assert result.valid, f"Board {i} should be valid but got errors: {result.errors}"
