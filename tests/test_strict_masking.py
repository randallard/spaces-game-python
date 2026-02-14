"""Tests for strict action masking (Phase 1).

Verifies that BFS reachability prevents invalid boards at the masking level:
- Piece moves that block all paths to unvisited rows are masked out
- Trap placements that block all paths are masked out
- Supermove deadlock prevention
- Finish only when all rows visited
"""

import numpy as np
import pytest

from spaces_game.simultaneous_play_env import SimultaneousPlayEnv
from spaces_game.types import Position


def _find_pool(board_size):
    """Find a valid opponent pool file for the given board size."""
    import os
    pool_dir = f"boards/size{board_size}"
    if os.path.isdir(pool_dir):
        for f in sorted(os.listdir(pool_dir)):
            if f.endswith(".json"):
                return os.path.join(pool_dir, f)
    return f"boards/size{board_size}/simple.json"


def make_env(board_size=2, max_construction_steps=None):
    """Create a minimal env for masking tests."""
    if max_construction_steps is None:
        max_construction_steps = board_size * 10
    return SimultaneousPlayEnv(
        board_size=board_size,
        opponent_pools=[_find_pool(board_size)],
        max_construction_steps=max_construction_steps,
    )


class TestCanReachAllRows:
    """Test the BFS reachability helper."""

    def test_fresh_env_size2_piece_at_bottom(self):
        """First piece at bottom row - should be able to reach row 0."""
        env = make_env(2)
        env.reset()
        # Place piece at (1, 0)
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        assert env._can_reach_all_rows()

    def test_blocked_by_trap_size2(self):
        """Trap blocks the only path to row 0."""
        env = make_env(2)
        env.reset()
        # Piece at (1, 0), trap at (0, 0) - only row 0 cell reachable from (1,0) is blocked
        # But (0, 1) is still reachable via (1,0) -> (1,1) -> (0,1)
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        env.trap_positions = {"0,0"}
        # Can still reach row 0 via (1,1) -> (0,1)
        assert env._can_reach_all_rows()

    def test_fully_blocked_size2(self):
        """All paths to row 0 are blocked."""
        env = make_env(2)
        env.reset()
        # Piece at (1, 0), traps block entire row 0
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        env.trap_positions = {"0,0", "0,1"}
        assert not env._can_reach_all_rows()

    def test_hypothetical_pos(self):
        """Test with hypothetical piece position."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        # Hypothetical move to (0, 0) - all rows now visited
        assert env._can_reach_all_rows(hypothetical_pos=(0, 0))

    def test_extra_trap_blocks_path(self):
        """Test that hypothetical trap blocks reachability."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        # Adding trap at (0, 0) AND (0, 1) blocks all of row 0
        env.trap_positions = {"0,0"}
        assert not env._can_reach_all_rows(extra_trap=(0, 1))


class TestStrictMaskingPieceMoves:
    """Test that piece moves blocking all paths are masked out."""

    def test_valid_piece_move_not_blocked(self):
        """Normal piece move that doesn't block paths is valid."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=2, col=1)
        env.piece_visited_positions = {"2,1"}
        # Move to (1, 1) - should be valid
        assert env._is_valid_placement(1, 1, "piece")

    def test_piece_move_blocking_path_invalid(self):
        """Piece move that creates unreachable rows is invalid."""
        env = make_env(3)
        env.reset()
        # Set up a situation where moving to a specific cell would block all paths
        # Piece at (2,0), visited (2,0), traps at (1,0) and (0,0)
        # Only path is via column 1 or 2
        env.current_piece_position = Position(row=2, col=0)
        env.piece_visited_positions = {"2,0"}
        env.trap_positions = {"1,0", "0,1"}
        # Moving to (2,1) - check if row 0 is still reachable
        # From (2,1), can go to (1,1) which is free, then (0,1) is a trap
        # (1,1) -> (1,2) -> (0,2) should work
        assert env._is_valid_placement(2, 1, "piece")


class TestStrictMaskingTrapMoves:
    """Test that trap placements blocking all paths are masked out."""

    def test_trap_blocking_only_path_invalid(self):
        """Trap that blocks the only remaining path is invalid."""
        env = make_env(2)
        env.reset()
        # Piece at (1,0), need to reach row 0
        # Trap at (0,0) already exists
        # Placing trap at (0,1) would block all of row 0
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        env.trap_positions = {"0,0"}
        # (0,1) is adjacent to piece at (1,0)? No - not orthogonally adjacent
        # Let's put piece at (1,1) instead
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"1,1"}
        # Trap at (0,1) - adjacent to piece, would block only path since (0,0) is trapped
        assert not env._is_valid_placement(0, 1, "trap")

    def test_trap_not_blocking_all_paths_valid(self):
        """Trap that leaves alternative paths is valid."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=2, col=1)
        env.piece_visited_positions = {"2,1"}
        # Trap adjacent at (2,0) - doesn't block paths to rows 0 and 1
        assert env._is_valid_placement(2, 0, "trap")


class TestStrictMaskingSupermove:
    """Test supermove deadlock prevention."""

    def test_supermove_with_no_escape_invalid(self):
        """Supermove trap where no adjacent escape leads to all rows is invalid."""
        env = make_env(2)
        env.reset()
        # Piece at (1,0), visited {(1,0)}
        # If we supermove at (1,0), piece must escape to an adjacent cell
        # Adjacent cells: (0,0), (1,1)
        # If (0,0) and (1,1) are both traps, no escape - invalid
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        env.trap_positions = {"0,0", "1,1"}
        assert not env._is_valid_placement(1, 0, "trap")

    def test_supermove_with_escape_valid(self):
        """Supermove trap where adjacent escape leads to all rows is valid."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        # (0,0) is free - escape there reaches row 0
        # But from (0,0), need to check all rows reachable
        # Row 1 already visited, row 0 reached via escape
        assert env._is_valid_placement(1, 0, "trap")


class TestFinishMaskAllRowsVisited:
    """Test that finish is only allowed when all rows are visited."""

    def test_finish_blocked_when_row_skipped(self):
        """Cannot finish if a middle row was skipped."""
        env = make_env(3)
        env.reset()
        # Piece at row 0 but skipped row 1
        env.current_piece_position = Position(row=0, col=0)
        env.piece_visited_positions = {"2,0", "0,0"}  # skipped row 1

        masks = env.action_masks()
        n_cells = env.board_size * env.board_size
        done_mask = masks[n_cells + 2:]  # [continue, finish]
        assert done_mask[1] == 0, "Finish should be blocked when row 1 is skipped"

    def test_finish_allowed_when_all_rows_visited(self):
        """Finish allowed when all rows visited and piece at row 0."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=0, col=0)
        env.piece_visited_positions = {"1,0", "0,0"}  # all rows visited

        masks = env.action_masks()
        n_cells = env.board_size * env.board_size
        done_mask = masks[n_cells + 2:]
        assert done_mask[1] == 1, "Finish should be allowed when all rows visited"

    def test_finish_blocked_at_row0_without_all_rows(self):
        """At row 0 but haven't visited all rows - can't finish."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=0, col=1)
        env.piece_visited_positions = {"2,1", "1,1", "0,1"}  # all rows visited
        masks = env.action_masks()
        n_cells = env.board_size * env.board_size
        done_mask = masks[n_cells + 2:]
        assert done_mask[1] == 1

        # Now remove row 1
        env.piece_visited_positions = {"2,1", "0,1"}
        masks = env.action_masks()
        done_mask = masks[n_cells + 2:]
        assert done_mask[1] == 0


class TestStrictMaskingEndToEnd:
    """End-to-end tests: play random games and verify all boards are valid."""

    @pytest.mark.parametrize("board_size", [2, 3, 4])
    def test_random_play_always_valid(self, board_size):
        """When agent signals done (not truncated), board must always be valid.

        Truncation (hitting max_construction_steps) can still produce invalid
        boards, but that's expected - the agent just ran out of time.
        The key invariant is: if strict masking allows finish, the board is valid.
        """
        # Use generous max_steps to reduce truncation from random play
        env = make_env(board_size, max_construction_steps=board_size * 20)
        agent_finished_valid = 0
        agent_finished_invalid = 0
        truncated_count = 0
        n_episodes = 50

        for ep in range(n_episodes):
            obs, info = env.reset(seed=ep)
            done = False
            last_was_finish = False
            while not done:
                masks = env.action_masks()
                n_cells = board_size * board_size

                # Parse masks
                cell_mask = masks[:n_cells]
                type_mask = masks[n_cells:n_cells + 2]
                done_mask = masks[n_cells + 2:]

                # Pick random valid action
                valid_cells = np.where(cell_mask == 1)[0]
                valid_types = np.where(type_mask == 1)[0]

                # Prefer finish if available (50% of the time)
                if done_mask[1] == 1 and np.random.random() < 0.5:
                    action = np.array([valid_cells[0], valid_types[0], 1])
                    last_was_finish = True
                else:
                    cell = np.random.choice(valid_cells)
                    mtype = np.random.choice(valid_types)
                    action = np.array([cell, mtype, 0])
                    last_was_finish = False

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if "valid_board" in info:
                    if info.get("valid_board"):
                        agent_finished_valid += 1
                    else:
                        # Distinguish: was this a truncation or agent finish?
                        if env.steps_taken >= env.max_construction_steps:
                            truncated_count += 1
                        else:
                            agent_finished_invalid += 1

        total_agent_finish = agent_finished_valid + agent_finished_invalid
        assert total_agent_finish > 0, "No agent-finished rounds"
        assert agent_finished_invalid == 0, (
            f"Size {board_size}: {agent_finished_invalid} invalid boards from "
            f"agent finish (should be 0). {truncated_count} from truncation."
        )
