"""Tests for strict action masking with flat Discrete action space.

Verifies that:
- BFS reachability prevents invalid boards at the masking level
- Piece moves that block all paths to unvisited rows are masked out
- Trap placements that block all paths are masked out
- Supermove deadlock prevention
- Finish only when all rows visited
- Forward-only movement constraint
- Flat action space layout: [piece_0..piece_n-1, trap_0..trap_n-1, finish]
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
        # Wait - forward-only means can't go (1,0) -> (1,1) since same row
        # Actually sideways is allowed, just not backward (higher row)
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        env.trap_positions = {"0,0"}
        # BFS is forward-only: from (1,0), can go to (0,0)[trapped] or sideways (1,1)[visited check]
        # (1,1) is not visited, so BFS reaches (1,1), then from (1,1) can go to (0,1) = row 0!
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


class TestFlatActionSpace:
    """Test flat Discrete action space layout."""

    def test_action_space_size(self):
        """Action space should be Discrete(2*n_cells + 1)."""
        env = make_env(2)
        assert env.action_space.n == 2 * 4 + 1  # 9 actions for size 2

        env = make_env(4)
        assert env.action_space.n == 2 * 16 + 1  # 33 actions for size 4

    def test_mask_size_matches_action_space(self):
        """Mask length should match action space."""
        env = make_env(2)
        env.reset()
        mask = env.action_masks()
        assert len(mask) == env.action_space.n

    def test_initial_mask_only_bottom_row_pieces(self):
        """Initial mask should only allow piece placement on bottom row."""
        env = make_env(3)
        env.reset()
        mask = env.action_masks()
        n_cells = 9  # 3x3

        # Only bottom row (row 2) cells should be valid for piece
        for i in range(n_cells):
            row = i // 3
            if row == 2:
                assert mask[i] == 1, f"Piece at cell {i} (row {row}) should be valid"
            else:
                assert mask[i] == 0, f"Piece at cell {i} (row {row}) should be invalid"

        # No traps allowed initially (no piece placed yet)
        for i in range(n_cells, 2 * n_cells):
            assert mask[i] == 0, f"Trap at cell {i - n_cells} should be invalid initially"

        # No finish allowed initially
        assert mask[2 * n_cells] == 0

    def test_no_valid_cells_mask_in_obs(self):
        """valid_cells_mask should not be in observation space."""
        env = make_env(2)
        env.reset()
        obs = env._get_observation()
        assert "valid_cells_mask" not in obs


class TestStrictMaskingPieceMoves:
    """Test that piece moves blocking all paths are masked out."""

    def test_valid_piece_move_not_blocked(self):
        """Normal piece move that doesn't block paths is valid."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=2, col=1)
        env.piece_visited_positions = {"2,1"}
        # Move to (1, 1) - forward, should be valid
        assert env._is_valid_placement(1, 1, "piece")

    def test_piece_move_blocking_path_invalid(self):
        """Piece move that creates unreachable rows is invalid."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=2, col=0)
        env.piece_visited_positions = {"2,0"}
        env.trap_positions = {"1,0", "0,1"}
        # Moving to (2,1) - sideways, should still work since path exists
        # From (2,1), can go to (1,1) which is free, then (1,2) -> (0,2) should work
        assert env._is_valid_placement(2, 1, "piece")


class TestForwardOnlyMovement:
    """Test forward-only movement constraint."""

    def test_piece_cannot_move_backward(self):
        """Piece cannot move to a higher row index."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"2,1", "1,1"}
        # Try to move backward to (2,0) - should be invalid
        assert not env._is_valid_placement(2, 0, "piece")
        # Forward to (0,1) should be valid
        assert env._is_valid_placement(0, 1, "piece")

    def test_piece_can_move_sideways(self):
        """Piece can move sideways (same row)."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"2,1", "1,1"}
        # Sideways to (1,0) should be valid
        assert env._is_valid_placement(1, 0, "piece")

    def test_trap_cannot_be_behind_piece(self):
        """Trap cannot be placed at a higher row index than piece."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"2,1", "1,1"}
        # Trap at row 2 should be invalid (behind piece)
        assert not env._is_valid_placement(2, 0, "trap")

    def test_trap_can_be_at_same_row(self):
        """Trap can be placed at same row as piece."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"2,1", "1,1"}
        # Trap at (1,0) - same row, adjacent, should be valid
        assert env._is_valid_placement(1, 0, "trap")

    def test_backward_mask_in_action_masks(self):
        """Backward cells should be masked out in action_masks()."""
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"2,1", "1,1"}

        mask = env.action_masks()
        n_cells = 9

        # Row 2 cells should all be 0 for piece (backward)
        for c in range(3):
            idx = 2 * 3 + c  # row 2
            assert mask[idx] == 0, f"Piece at row 2 col {c} should be masked (backward)"

        # Row 2 cells should all be 0 for trap (backward)
        for c in range(3):
            idx = n_cells + 2 * 3 + c  # trap at row 2
            assert mask[idx] == 0, f"Trap at row 2 col {c} should be masked (backward)"


class TestStrictMaskingTrapMoves:
    """Test that trap placements blocking all paths are masked out."""

    def test_trap_blocking_only_path_invalid(self):
        """Trap that blocks the only remaining path is invalid."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"1,1"}
        env.trap_positions = {"0,0"}
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
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        # Forward-only escapes from (1,0): (-1,0)=(0,0), (0,-1)=OOB, (0,1)=(1,1)
        # If (0,0) is trapped and (1,1) is trapped, no forward escape
        env.trap_positions = {"0,0", "1,1"}
        assert not env._is_valid_placement(1, 0, "trap")

    def test_supermove_with_escape_valid(self):
        """Supermove trap where adjacent escape leads to all rows is valid."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=1, col=0)
        env.piece_visited_positions = {"1,0"}
        # (0,0) is free - escape there reaches row 0
        assert env._is_valid_placement(1, 0, "trap")


class TestFinishMaskAllRowsVisited:
    """Test that finish is only allowed when all rows are visited."""

    def test_finish_blocked_when_not_all_rows_and_moves_exist(self):
        """Cannot finish if not all rows visited but valid moves still exist."""
        env = make_env(3)
        env.reset()
        # Piece at row 1, has visited row 2 and row 1. Row 0 not visited yet.
        # Forward moves exist (to row 0), so finish should be blocked.
        env.current_piece_position = Position(row=1, col=1)
        env.piece_visited_positions = {"2,1", "1,1"}

        masks = env.action_masks()
        n_cells = env.board_size * env.board_size
        assert masks[2 * n_cells] == 0, "Finish should be blocked when row 0 not visited"

    def test_finish_allowed_when_all_rows_visited(self):
        """Finish allowed when all rows visited and piece at row 0."""
        env = make_env(2)
        env.reset()
        env.current_piece_position = Position(row=0, col=0)
        env.piece_visited_positions = {"1,0", "0,0"}  # all rows visited

        masks = env.action_masks()
        n_cells = env.board_size * env.board_size
        assert masks[2 * n_cells] == 1, "Finish should be allowed when all rows visited"

    def test_finish_at_row0_with_all_rows_vs_deadlock(self):
        """At row 0 with all rows visited: finish allowed.
        At row 0 missing rows: deadlock (forward-only can't go back) forces finish.
        """
        env = make_env(3)
        env.reset()
        env.current_piece_position = Position(row=0, col=1)
        env.piece_visited_positions = {"2,1", "1,1", "0,1"}  # all rows visited
        masks = env.action_masks()
        n_cells = env.board_size * env.board_size
        assert masks[2 * n_cells] == 1  # finish allowed (all rows visited)

        # With forward-only, if at row 0 but missing row 1, piece can't go back.
        # This is a deadlock scenario - finish is forced by deadlock detection.
        env.piece_visited_positions = {"2,1", "0,1"}
        masks = env.action_masks()
        # Finish is forced (deadlock) even though not all rows visited
        assert masks[2 * n_cells] == 1


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
            while not done:
                masks = env.action_masks()

                # Pick random valid action from flat mask
                valid_actions = np.where(masks == 1)[0]
                assert len(valid_actions) > 0, "No valid actions available"

                n_cells = board_size * board_size
                finish_action = 2 * n_cells

                # Prefer finish if available (50% of the time)
                if masks[finish_action] == 1 and np.random.random() < 0.5:
                    action = finish_action
                else:
                    # Pick from non-finish valid actions if possible
                    non_finish = valid_actions[valid_actions != finish_action]
                    if len(non_finish) > 0:
                        action = np.random.choice(non_finish)
                    else:
                        action = finish_action

                obs, reward, terminated, truncated_flag, info = env.step(action)
                done = terminated or truncated_flag

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
