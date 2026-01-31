"""
Tests for board loading utilities.
"""

import json
from pathlib import Path
import pytest

from spaces_game.board_loader import (
    load_boards_from_json,
    load_board_by_index,
    BoardPool,
)
from spaces_game.types import Board


class TestLoadBoards:
    """Test loading boards from JSON files."""

    def test_load_size_2_boards(self):
        """Test loading pre-generated size 2 boards."""
        boards = load_boards_from_json('data/boards_size_2.json')

        assert len(boards) > 0
        assert all(isinstance(board, Board) for board in boards)
        assert all(board.boardSize == 2 for board in boards)

    def test_load_size_3_boards(self):
        """Test loading pre-generated size 3 boards."""
        boards = load_boards_from_json('data/boards_size_3.json')

        assert len(boards) == 500
        assert all(isinstance(board, Board) for board in boards)
        assert all(board.boardSize == 3 for board in boards)

    def test_boards_are_immutable(self):
        """Verify loaded boards are frozen."""
        boards = load_boards_from_json('data/boards_size_2.json')
        board = boards[0]

        with pytest.raises(AttributeError):
            board.boardSize = 5  # type: ignore

    def test_load_board_by_index(self):
        """Test loading specific board by index."""
        board = load_board_by_index('data/boards_size_3.json', 0)

        assert isinstance(board, Board)
        assert board.boardSize == 3

    def test_load_board_by_index_out_of_range(self):
        """Test loading with invalid index."""
        with pytest.raises(IndexError):
            load_board_by_index('data/boards_size_3.json', 10000)


class TestBoardPool:
    """Test BoardPool for efficient board sampling."""

    def test_create_pool(self):
        """Test creating a board pool."""
        pool = BoardPool('data/boards_size_3.json', cache=False)

        assert len(pool) == 500
        assert all(isinstance(board, Board) for board in pool)

    def test_sample_one(self):
        """Test sampling a single board."""
        pool = BoardPool('data/boards_size_3.json', cache=False)
        board = pool.sample_one()

        assert isinstance(board, Board)
        assert board.boardSize == 3

    def test_sample_multiple(self):
        """Test sampling multiple boards."""
        pool = BoardPool('data/boards_size_3.json', cache=False)
        boards = pool.sample(10)

        assert len(boards) == 10
        assert all(isinstance(board, Board) for board in boards)

    def test_get_by_index(self):
        """Test getting board by index."""
        pool = BoardPool('data/boards_size_3.json', cache=False)
        board = pool.get(0)

        assert isinstance(board, Board)

    def test_indexing(self):
        """Test pool supports indexing."""
        pool = BoardPool('data/boards_size_3.json', cache=False)
        board = pool[0]

        assert isinstance(board, Board)

    def test_iteration(self):
        """Test pool supports iteration."""
        pool = BoardPool('data/boards_size_3.json', cache=False)
        boards = list(pool)

        assert len(boards) == 500

    def test_board_structure(self):
        """Verify loaded boards have correct structure."""
        pool = BoardPool('data/boards_size_3.json', cache=False)
        board = pool[0]

        # Check grid is tuple of tuples
        assert isinstance(board.grid, tuple)
        assert all(isinstance(row, tuple) for row in board.grid)

        # Check sequence is tuple
        assert isinstance(board.sequence, tuple)

        # Check all moves in sequence
        assert len(board.sequence) > 0
        assert board.sequence[-1].type == 'final'  # Last move should be goal
