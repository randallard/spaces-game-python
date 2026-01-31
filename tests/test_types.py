"""
Tests for core data structures.
"""

import pytest
from spaces_game.types import (
    Position,
    BoardMove,
    Board,
    SimulationDetails,
    RoundResult,
    is_valid_board_size,
    MIN_BOARD_SIZE,
    MAX_BOARD_SIZE,
)


class TestPosition:
    """Test Position dataclass."""

    def test_create_position(self):
        pos = Position(row=1, col=2)
        assert pos.row == 1
        assert pos.col == 2

    def test_position_immutable(self):
        pos = Position(row=1, col=2)
        with pytest.raises(AttributeError):
            pos.row = 3  # type: ignore

    def test_position_str(self):
        pos = Position(row=1, col=2)
        assert str(pos) == "(1, 2)"

    def test_position_equality(self):
        pos1 = Position(row=1, col=2)
        pos2 = Position(row=1, col=2)
        pos3 = Position(row=2, col=1)

        assert pos1 == pos2
        assert pos1 != pos3


class TestBoardMove:
    """Test BoardMove dataclass."""

    def test_create_board_move(self):
        pos = Position(row=2, col=1)
        move = BoardMove(position=pos, type='piece', order=1)

        assert move.position == pos
        assert move.type == 'piece'
        assert move.order == 1

    def test_board_move_immutable(self):
        pos = Position(row=2, col=1)
        move = BoardMove(position=pos, type='piece', order=1)

        with pytest.raises(AttributeError):
            move.order = 2  # type: ignore

    def test_board_move_str(self):
        pos = Position(row=2, col=1)
        move = BoardMove(position=pos, type='piece', order=1)

        assert str(move) == "1. piece at (2, 1)"


class TestBoard:
    """Test Board dataclass."""

    def test_create_board(self):
        grid = (
            ('empty', 'piece'),
            ('piece', 'empty'),
        )
        sequence = (
            BoardMove(Position(1, 0), 'piece', 1),
            BoardMove(Position(0, 1), 'piece', 2),
            BoardMove(Position(-1, 1), 'final', 3),
        )

        board = Board(boardSize=2, grid=grid, sequence=sequence)

        assert board.boardSize == 2
        assert board.grid == grid
        assert board.sequence == sequence

    def test_board_immutable(self):
        grid = (
            ('empty', 'piece'),
            ('piece', 'empty'),
        )
        sequence = (
            BoardMove(Position(1, 0), 'piece', 1),
        )

        board = Board(boardSize=2, grid=grid, sequence=sequence)

        with pytest.raises(AttributeError):
            board.boardSize = 3  # type: ignore

    def test_board_validation_invalid_size(self):
        """Board with invalid size should raise ValueError."""
        grid = (('empty',),)
        sequence = ()

        with pytest.raises(ValueError, match="Invalid board size"):
            Board(boardSize=1, grid=grid, sequence=sequence)

        with pytest.raises(ValueError, match="Invalid board size"):
            Board(boardSize=100, grid=grid, sequence=sequence)

    def test_board_validation_grid_size_mismatch(self):
        """Grid dimensions must match boardSize."""
        # Wrong number of rows
        grid = (('empty',),)  # 1 row instead of 2
        sequence = ()

        with pytest.raises(ValueError, match="Grid rows don't match board size"):
            Board(boardSize=2, grid=grid, sequence=sequence)

        # Wrong number of columns
        grid = (
            ('empty', 'empty'),
            ('empty',),  # 1 col instead of 2
        )

        with pytest.raises(ValueError, match="Grid row .* length doesn't match"):
            Board(boardSize=2, grid=grid, sequence=sequence)


class TestBoardSize:
    """Test board size validation."""

    def test_valid_board_sizes(self):
        assert is_valid_board_size(2) is True
        assert is_valid_board_size(3) is True
        assert is_valid_board_size(50) is True
        assert is_valid_board_size(99) is True

    def test_invalid_board_sizes(self):
        assert is_valid_board_size(1) is False
        assert is_valid_board_size(100) is False
        assert is_valid_board_size(0) is False
        assert is_valid_board_size(-1) is False

    def test_board_size_constants(self):
        assert MIN_BOARD_SIZE == 2
        assert MAX_BOARD_SIZE == 99
