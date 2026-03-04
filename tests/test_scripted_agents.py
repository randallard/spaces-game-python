"""
Tests for scripted AI agents (size 2 difficulty levels 1-4).
"""

import pytest

from inference_server.scripted_agents import (
    build_simple_board,
    build_trap_board,
    pick_column_level1,
    pick_column_level2,
    scripted_board,
)
from spaces_game.types import Board, BoardMove, Position
from spaces_game.validation import is_board_playable


def _board_dict_to_board(board_dict: dict) -> Board:
    """Convert a board dict to a Board object for validation."""
    seq = []
    for m in board_dict["sequence"]:
        seq.append(BoardMove(
            position=Position(row=m["position"]["row"], col=m["position"]["col"]),
            type=m["type"],
            order=m["order"],
        ))
    grid = tuple(tuple(r) for r in board_dict["grid"])
    return Board(boardSize=board_dict["boardSize"], grid=grid, sequence=tuple(seq))


# ---------------------------------------------------------------------------
# Board builders
# ---------------------------------------------------------------------------

class TestBuildSimpleBoard:
    def test_size2_column0(self):
        board = build_simple_board(2, 0)
        assert board["boardSize"] == 2
        assert len(board["sequence"]) == 3  # 2 pieces + 1 final
        assert board["sequence"][-1]["type"] == "final"
        assert is_board_playable(_board_dict_to_board(board))

    def test_size2_column1(self):
        board = build_simple_board(2, 1)
        assert board["sequence"][0]["position"]["col"] == 1
        assert is_board_playable(_board_dict_to_board(board))

    def test_size3(self):
        board = build_simple_board(3, 0)
        assert board["boardSize"] == 3
        assert len(board["sequence"]) == 4  # 3 pieces + 1 final
        assert is_board_playable(_board_dict_to_board(board))

    def test_no_traps(self):
        board = build_simple_board(2, 0)
        for move in board["sequence"]:
            assert move["type"] != "trap"


class TestBuildTrapBoard:
    def test_size2_column0_has_trap(self):
        board = build_trap_board(2, 0)
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1
        # Trap at row 1 (bottom), col 1 (opposite of column 0)
        assert trap_moves[0]["position"]["row"] == 1
        assert trap_moves[0]["position"]["col"] == 1
        assert is_board_playable(_board_dict_to_board(board))

    def test_size2_column1_has_trap(self):
        board = build_trap_board(2, 1)
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1
        assert trap_moves[0]["position"]["col"] == 0  # opposite column
        assert is_board_playable(_board_dict_to_board(board))

    def test_size3_valid(self):
        board = build_trap_board(3, 0)
        assert is_board_playable(_board_dict_to_board(board))
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1


# ---------------------------------------------------------------------------
# Column picking
# ---------------------------------------------------------------------------

class TestPickColumnLevel2:
    def test_round0_always_col0(self):
        assert pick_column_level2(2, 0, []) == 0

    def test_switch_after_loss(self):
        scores = [{"agent": 0, "opponent": 1}]  # lost round 0
        assert pick_column_level2(2, 1, scores) == 1

    def test_stay_after_win(self):
        scores = [{"agent": 1, "opponent": 0}]  # won round 0
        assert pick_column_level2(2, 1, scores) == 0

    def test_switch_back_after_two_losses(self):
        scores = [
            {"agent": 0, "opponent": 1},  # lost → switch to 1
            {"agent": 0, "opponent": 1},  # lost → switch to 0
        ]
        assert pick_column_level2(2, 2, scores) == 0

    def test_stay_after_win_then_loss(self):
        scores = [
            {"agent": 1, "opponent": 0},  # won → stay at 0
            {"agent": 0, "opponent": 1},  # lost → switch to 1
        ]
        assert pick_column_level2(2, 2, scores) == 1


# ---------------------------------------------------------------------------
# Scripted board dispatcher
# ---------------------------------------------------------------------------

class TestScriptedBoard:
    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_all_levels_produce_valid_boards(self, level):
        """Every level should produce a playable board."""
        for round_num in range(5):
            scores = [{"agent": 1, "opponent": 0}] * round_num
            board = scripted_board(level, 2, round_num, scores)
            assert is_board_playable(_board_dict_to_board(board)), \
                f"Level {level}, round {round_num} produced invalid board"

    def test_level1_no_traps(self):
        for round_num in range(5):
            board = scripted_board(1, 2, round_num, [])
            for m in board["sequence"]:
                assert m["type"] != "trap"

    def test_level2_no_traps(self):
        scores = [{"agent": 0, "opponent": 1}] * 4
        for round_num in range(5):
            board = scripted_board(2, 2, round_num, scores[:round_num])
            for m in board["sequence"]:
                assert m["type"] != "trap"

    def test_level3_has_traps(self):
        board = scripted_board(3, 2, 0, [])
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1

    def test_level4_escape_hatch(self):
        """After losing 2 in a row, level 4 drops the trap."""
        scores = [
            {"agent": 0, "opponent": 1},
            {"agent": 0, "opponent": 1},
        ]
        board = scripted_board(4, 2, 2, scores)
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 0, "Level 4 should drop trap after 2 consecutive losses"

    def test_level4_keeps_trap_normally(self):
        """Level 4 uses traps when not on a losing streak."""
        scores = [{"agent": 1, "opponent": 0}]
        board = scripted_board(4, 2, 1, scores)
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1

    def test_level4_always_switches_columns(self):
        """Level 4 alternates columns every round."""
        scores = [{"agent": 1, "opponent": 0}] * 4
        cols = []
        for round_num in range(5):
            board = scripted_board(4, 2, round_num, scores[:round_num])
            # Get the piece column from first move
            first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
            cols.append(first_piece["position"]["col"])
        # Should alternate: 0, 1, 0, 1, 0
        for i in range(1, len(cols)):
            assert cols[i] != cols[i - 1], f"Round {i} should switch column"

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            scripted_board(5, 2, 0, [])

    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_size3_valid(self, level):
        """Scripted agents should work for any board size."""
        board = scripted_board(level, 3, 0, [])
        assert is_board_playable(_board_dict_to_board(board))
