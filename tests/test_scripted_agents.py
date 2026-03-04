"""
Tests for scripted AI agents (size 2 difficulty levels 1-5).
"""

import pytest

from inference_server.models import RoundHistoryEntry
from inference_server.scripted_agents import (
    build_simple_board,
    build_supermove_board,
    build_trap_board,
    decode_starting_column,
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
            scripted_board(99, 2, 0, [])

    @pytest.mark.parametrize("level", [1, 2, 3, 4])
    def test_size3_valid(self, level):
        """Scripted agents should work for any board size."""
        board = scripted_board(level, 3, 0, [])
        assert is_board_playable(_board_dict_to_board(board))


# ---------------------------------------------------------------------------
# Compact board decoder
# ---------------------------------------------------------------------------

class TestDecodeStartingColumn:
    def test_column0_size2(self):
        # Cell 2 on size-2 board = row 1, col 0
        assert decode_starting_column("2|2p3t0pG0f") == 0

    def test_column1_size2(self):
        # Cell 3 on size-2 board = row 1, col 1
        assert decode_starting_column("2|3p2t1pG1f") == 1

    def test_column0_size3(self):
        # Cell 6 on size-3 board = row 2, col 0
        assert decode_starting_column("3|6p7t3p0pG0f") == 0

    def test_column2_size3(self):
        # Cell 8 on size-3 board = row 2, col 2
        assert decode_starting_column("3|8p6t5p2pG2f") == 2

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            decode_starting_column("invalid")

    def test_no_piece(self):
        with pytest.raises(ValueError):
            decode_starting_column("2|3t0f")


# ---------------------------------------------------------------------------
# Supermove board builder
# ---------------------------------------------------------------------------

class TestBuildSupermoveBoard:
    def test_size2_column0_valid(self):
        board = build_supermove_board(2, 0)
        assert is_board_playable(_board_dict_to_board(board))

    def test_size2_column1_valid(self):
        board = build_supermove_board(2, 1)
        assert is_board_playable(_board_dict_to_board(board))

    def test_size3_valid(self):
        board = build_supermove_board(3, 0)
        assert is_board_playable(_board_dict_to_board(board))

    def test_has_trap_on_starting_cell(self):
        board = build_supermove_board(2, 0)
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1
        assert trap_moves[0]["position"]["row"] == 1  # bottom row
        assert trap_moves[0]["position"]["col"] == 0

    def test_trap_same_cell_as_first_piece(self):
        board = build_supermove_board(2, 1)
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        trap = next(m for m in board["sequence"] if m["type"] == "trap")
        assert first_piece["position"] == trap["position"]

    def test_grid_shows_trap_on_start(self):
        board = build_supermove_board(2, 0)
        # Grid at (1, 0) should be trap (overwrites piece)
        assert board["grid"][1][0] == "trap"
        # Grid at (0, 0) should be piece
        assert board["grid"][0][0] == "piece"


# ---------------------------------------------------------------------------
# Level 5 scripted agent
# ---------------------------------------------------------------------------

def _make_round_history_entry(
    agent_score=0.0, opponent_score=0.0, agent_board="", **kwargs
) -> RoundHistoryEntry:
    return RoundHistoryEntry(
        agent_score=agent_score,
        opponent_score=opponent_score,
        agent_board=agent_board,
        **kwargs,
    )


class TestLevel5:
    def test_round0_produces_trap_board(self):
        """Round 0 should produce a trap board (not supermove)."""
        board = scripted_board(5, 2, 0, [])
        trap_moves = [m for m in board["sequence"] if m["type"] == "trap"]
        assert len(trap_moves) == 1
        # Trap should be on a different cell than the first piece
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        assert trap_moves[0]["position"] != first_piece["position"]
        assert is_board_playable(_board_dict_to_board(board))

    def test_round0_valid(self):
        board = scripted_board(5, 2, 0, [])
        assert is_board_playable(_board_dict_to_board(board))

    def test_all_rounds_valid_no_ties(self):
        """All rounds produce valid boards when there are no ties."""
        # Simulate: agent wins every round, started in column 0
        rh = []
        for r in range(5):
            board = scripted_board(5, 2, r, [], round_history=rh)
            assert is_board_playable(_board_dict_to_board(board)), \
                f"Round {r} produced invalid board"
            # Simulate agent winning
            rh.append(_make_round_history_entry(
                agent_score=2.0, opponent_score=0.0,
                agent_board="2|2p3t0pG0f",  # column 0 trap board
            ))

    def test_supermove_triggers_on_two_ties(self):
        """Two consecutive ties should trigger a supermove board."""
        rh = [
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
        ]
        board = scripted_board(5, 2, 2, [], round_history=rh)
        # Should be a supermove board: trap on same cell as first piece
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        trap = next(m for m in board["sequence"] if m["type"] == "trap")
        assert first_piece["position"] == trap["position"], "Should be a supermove board"
        assert is_board_playable(_board_dict_to_board(board))

    def test_no_supermove_on_single_tie(self):
        """A single tie should not trigger supermove."""
        rh = [
            _make_round_history_entry(agent_score=2.0, opponent_score=0.0, agent_board="2|2p3t0pG0f"),
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
        ]
        board = scripted_board(5, 2, 2, [], round_history=rh)
        # Should be a trap board, not supermove
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        trap = next(m for m in board["sequence"] if m["type"] == "trap")
        assert first_piece["position"] != trap["position"], "Should NOT be a supermove board"

    def test_column_switch_on_supermove_when_scored_zero(self):
        """When agent scored 0 two rounds ago, supermove switches column."""
        # Agent started col 0, scored 0 in both tie rounds
        rh = [
            _make_round_history_entry(agent_score=0.0, opponent_score=0.0, agent_board="2|2p3t0pG0f"),
            _make_round_history_entry(agent_score=0.0, opponent_score=0.0, agent_board="2|2p3t0pG0f"),
        ]
        board = scripted_board(5, 2, 2, [], round_history=rh)
        # Should have switched to column 1
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        assert first_piece["position"]["col"] == 1

    def test_same_column_on_supermove_when_scored_points(self):
        """When agent scored points two rounds ago, supermove stays same column."""
        rh = [
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
        ]
        board = scripted_board(5, 2, 2, [], round_history=rh)
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        assert first_piece["position"]["col"] == 0  # same as starting column

    def test_after_supermove_switches_to_opposite_trap(self):
        """After playing supermove, next round should be trap from opposite column."""
        # Round 0-1: ties → round 2: supermove from col 0 → round 3: trap from col 1
        rh = [
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="2|2p3t0pG0f"),
            _make_round_history_entry(agent_score=2.0, opponent_score=0.0, agent_board="2|2p2t0pG0f"),  # supermove col 0
        ]
        board = scripted_board(5, 2, 3, [], round_history=rh)
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        trap = next(m for m in board["sequence"] if m["type"] == "trap")
        # Should be trap board (not supermove) from column 1
        assert first_piece["position"] != trap["position"], "Should be trap board, not supermove"
        assert first_piece["position"]["col"] == 1, "Should switch to opposite column"

    def test_derives_column_from_round_history(self):
        """Level 5 derives its starting column from round_history[0].agent_board."""
        # Round 0 used column 1 (cell 3 on size-2 = row 1, col 1)
        rh = [
            _make_round_history_entry(agent_score=2.0, opponent_score=0.0, agent_board="2|3p2t1pG1f"),
        ]
        board = scripted_board(5, 2, 1, [], round_history=rh)
        first_piece = next(m for m in board["sequence"] if m["type"] == "piece")
        assert first_piece["position"]["col"] == 1  # same as round 0

    def test_size3_supermove_valid(self):
        """Supermove boards should be valid for size 3."""
        rh = [
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="3|6p7t3p0pG0f"),
            _make_round_history_entry(agent_score=1.0, opponent_score=1.0, agent_board="3|6p7t3p0pG0f"),
        ]
        board = scripted_board(5, 3, 2, [], round_history=rh)
        assert is_board_playable(_board_dict_to_board(board))
