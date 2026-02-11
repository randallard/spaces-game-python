"""
Unit tests for inference module functions.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spaces_game.types import Board, BoardMove, Position

from inference_server.inference import (
    encode_board_for_agent,
    discover_opponent_pools,
    load_agent,
    get_model_board_size,
    is_stage3_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_board(board_size: int = 2) -> Board:
    """Create a simple valid board for testing."""
    sequence = (
        BoardMove(Position(row=1, col=0), "piece", 1),
        BoardMove(Position(row=0, col=0), "piece", 2),
        BoardMove(Position(row=-1, col=0), "final", 3),
    )
    grid = (
        ("piece", "empty"),
        ("piece", "empty"),
    )
    return Board(boardSize=board_size, grid=grid, sequence=sequence)


def _make_board_with_trap(board_size: int = 2) -> Board:
    """Create a board with a trap for testing."""
    sequence = (
        BoardMove(Position(row=1, col=0), "piece", 1),
        BoardMove(Position(row=1, col=0), "trap", 2),
        BoardMove(Position(row=0, col=0), "piece", 3),
        BoardMove(Position(row=-1, col=0), "final", 4),
    )
    grid = (
        ("piece", "empty"),
        ("trap", "empty"),
    )
    return Board(boardSize=board_size, grid=grid, sequence=sequence)


# ---------------------------------------------------------------------------
# encode_board_for_agent tests
# ---------------------------------------------------------------------------

class TestEncodeBoardForAgent:

    def test_basic_encoding(self):
        """Test that encoding produces correct shape and non-zero values."""
        board = _make_board(board_size=2)
        result = encode_board_for_agent(board, board_size=2)

        assert result.shape == (2, 2, 2)
        assert result.dtype == np.int32
        # Should have non-zero values for piece positions
        assert result.sum() > 0

    def test_180_rotation(self):
        """Test that positions are rotated 180 degrees."""
        board = _make_board(board_size=2)
        result = encode_board_for_agent(board, board_size=2)

        # Original: piece at (1,0) order=1, piece at (0,0) order=2
        # Rotated 180 on 2x2: (1,0) -> (0,1), (0,0) -> (1,1)
        # Channel 0 = pieces
        assert result[0, 1, 0] == 1  # rotated (1,0) with order 1
        assert result[1, 1, 0] == 2  # rotated (0,0) with order 2

    def test_trap_encoding(self):
        """Test that traps go in channel 1."""
        board = _make_board_with_trap(board_size=2)
        result = encode_board_for_agent(board, board_size=2)

        # Channel 1 = traps
        # Trap at (1,0), rotated 180 on 2x2 -> (0,1)
        assert result[0, 1, 1] == 2  # trap order=2

    def test_final_move_ignored(self):
        """Test that final moves are not encoded."""
        board = _make_board(board_size=2)
        result = encode_board_for_agent(board, board_size=2)

        # No final move should appear in the grid
        # Final is at (-1, 0) which is off-grid, so it should be ignored
        total_values = result.sum()
        # Only piece orders should be present: 1 + 2 = 3
        assert total_values == 3

    def test_larger_board(self):
        """Test encoding with a 3x3 board."""
        sequence = (
            BoardMove(Position(row=2, col=1), "piece", 1),
            BoardMove(Position(row=1, col=1), "piece", 2),
            BoardMove(Position(row=0, col=1), "piece", 3),
            BoardMove(Position(row=-1, col=1), "final", 4),
        )
        grid = (
            ("empty", "piece", "empty"),
            ("empty", "piece", "empty"),
            ("empty", "piece", "empty"),
        )
        board = Board(boardSize=3, grid=grid, sequence=sequence)

        result = encode_board_for_agent(board, board_size=3)
        assert result.shape == (3, 3, 2)

        # Rotated 180 on 3x3: (r,c) -> (2-r, 2-c)
        # (2,1) -> (0,1), (1,1) -> (1,1), (0,1) -> (2,1)
        assert result[0, 1, 0] == 1
        assert result[1, 1, 0] == 2
        assert result[2, 1, 0] == 3

    def test_empty_board_encoding(self):
        """Test encoding a board with minimal sequence."""
        grid = (
            ("empty", "empty"),
            ("empty", "empty"),
        )
        sequence = (
            BoardMove(Position(row=-1, col=0), "final", 1),
        )
        board = Board(boardSize=2, grid=grid, sequence=sequence)
        result = encode_board_for_agent(board, board_size=2)

        assert result.shape == (2, 2, 2)
        assert result.sum() == 0  # No non-final moves


# ---------------------------------------------------------------------------
# discover_opponent_pools tests
# ---------------------------------------------------------------------------

class TestDiscoverOpponentPools:

    def test_discovers_json_files(self):
        """Test that JSON files are discovered in the correct directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size_dir = Path(tmpdir) / "size2"
            size_dir.mkdir()

            # Create some JSON files
            (size_dir / "simple.json").write_text("[]")
            (size_dir / "one_trap.json").write_text("[]")
            (size_dir / "super_move.json").write_text("[]")

            pools = discover_opponent_pools(board_size=2, boards_dir=tmpdir)

            assert len(pools) == 3
            assert any("simple.json" in p for p in pools)
            assert any("one_trap.json" in p for p in pools)
            assert any("super_move.json" in p for p in pools)

    def test_returns_sorted(self):
        """Test that results are sorted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size_dir = Path(tmpdir) / "size3"
            size_dir.mkdir()

            (size_dir / "c_pool.json").write_text("[]")
            (size_dir / "a_pool.json").write_text("[]")
            (size_dir / "b_pool.json").write_text("[]")

            pools = discover_opponent_pools(board_size=3, boards_dir=tmpdir)

            assert len(pools) == 3
            filenames = [Path(p).name for p in pools]
            assert filenames == sorted(filenames)

    def test_empty_when_no_directory(self):
        """Test that empty list is returned when directory doesn't exist."""
        pools = discover_opponent_pools(board_size=99, boards_dir="/nonexistent")
        assert pools == []

    def test_empty_when_no_json_files(self):
        """Test that empty list is returned when no JSON files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size_dir = Path(tmpdir) / "size2"
            size_dir.mkdir()
            # Create a non-JSON file
            (size_dir / "data.txt").write_text("not json")

            pools = discover_opponent_pools(board_size=2, boards_dir=tmpdir)
            assert pools == []

    def test_ignores_other_size_directories(self):
        """Test that only the matching size directory is searched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size2_dir = Path(tmpdir) / "size2"
            size3_dir = Path(tmpdir) / "size3"
            size2_dir.mkdir()
            size3_dir.mkdir()

            (size2_dir / "pool_a.json").write_text("[]")
            (size3_dir / "pool_b.json").write_text("[]")

            pools = discover_opponent_pools(board_size=2, boards_dir=tmpdir)
            assert len(pools) == 1
            assert "pool_a.json" in pools[0]


# ---------------------------------------------------------------------------
# load_agent tests
# ---------------------------------------------------------------------------

class TestLoadAgent:

    def test_load_maskable_ppo(self):
        """Test loading a MaskablePPO model."""
        mock_model = MagicMock()

        with patch("inference_server.inference.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = True

            with patch.dict("sys.modules", {"sb3_contrib": MagicMock()}):
                import sys
                sys.modules["sb3_contrib"].MaskablePPO.load.return_value = mock_model

                model, uses_masks = load_agent("fake_model.zip")

                assert uses_masks is True
                assert model is mock_model

    def test_load_ppo_fallback(self):
        """Test falling back to PPO when MaskablePPO fails."""
        mock_model = MagicMock()

        with patch("inference_server.inference.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = True

            # Make MaskablePPO fail
            mock_sb3_contrib = MagicMock()
            mock_sb3_contrib.MaskablePPO.load.side_effect = Exception("No MaskablePPO")

            mock_sb3 = MagicMock()
            mock_sb3.PPO.load.return_value = mock_model

            with patch.dict("sys.modules", {
                "sb3_contrib": mock_sb3_contrib,
                "stable_baselines3": mock_sb3,
            }):
                model, uses_masks = load_agent("fake_model.zip")

                assert uses_masks is False
                assert model is mock_model

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_agent("/nonexistent/model.zip")


# ---------------------------------------------------------------------------
# get_model_board_size tests
# ---------------------------------------------------------------------------

class TestGetModelBoardSize:

    def test_dict_observation_space(self):
        """Test extracting board size from Dict observation space."""
        mock_model = MagicMock()
        mock_model.observation_space.spaces = {
            "building_board": MagicMock(shape=(3, 3, 2)),
        }

        size = get_model_board_size(mock_model)
        assert size == 3

    def test_box_observation_space(self):
        """Test extracting board size from Box observation space."""
        mock_model = MagicMock()
        mock_model.observation_space.spaces = {}  # No building_board
        del mock_model.observation_space.spaces  # Remove spaces attr
        mock_model.observation_space.shape = (4, 4, 2)

        size = get_model_board_size(mock_model)
        assert size == 4

    def test_unknown_observation_space(self):
        """Test returning None when obs space is unrecognizable."""
        mock_model = MagicMock()
        mock_model.observation_space = MagicMock(spec=[])  # No spaces or shape

        size = get_model_board_size(mock_model)
        assert size is None


# ---------------------------------------------------------------------------
# is_stage3_model tests
# ---------------------------------------------------------------------------

class TestIsStage3Model:

    def test_stage3_model(self):
        """Test detecting a Stage 3 model with opponent_history."""
        mock_model = MagicMock()
        mock_model.observation_space.spaces = {
            "building_board": MagicMock(),
            "opponent_history": MagicMock(),
        }

        assert is_stage3_model(mock_model) is True

    def test_non_stage3_model(self):
        """Test detecting a non-Stage 3 model without opponent_history."""
        mock_model = MagicMock()
        mock_model.observation_space.spaces = {
            "building_board": MagicMock(),
        }

        assert is_stage3_model(mock_model) is False

    def test_model_without_spaces(self):
        """Test handling model with no dict observation space."""
        mock_model = MagicMock()
        mock_model.observation_space = MagicMock(spec=[])  # No spaces attr

        assert is_stage3_model(mock_model) is False
