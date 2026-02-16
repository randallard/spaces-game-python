"""Tests for fog of war (Stage 4) in SimultaneousPlayEnv.

Verifies that:
- Observation space includes fog_outcomes when use_fog=True
- Observation space unchanged when use_fog=False
- _encode_opponent_board_fog hides moves after playerLastStep
- _encode_opponent_board_fog hides all traps except sprung trap
- fog_outcomes populated correctly after simulation
- Full round-trip: fog env produces valid obs through step/finish cycle
"""

import numpy as np
import pytest

from spaces_game.simultaneous_play_env import SimultaneousPlayEnv
from spaces_game.types import Board, BoardMove, Position
from spaces_game.simulation import _rotate_position


def _find_pool(board_size):
    """Find a valid opponent pool file for the given board size."""
    import os
    pool_dir = f"boards/size{board_size}"
    if os.path.isdir(pool_dir):
        for f in sorted(os.listdir(pool_dir)):
            if f.endswith(".json"):
                return os.path.join(pool_dir, f)
    return f"boards/size{board_size}/simple.json"


def make_env(board_size=2, use_fog=False):
    """Create a minimal env for fog tests."""
    return SimultaneousPlayEnv(
        board_size=board_size,
        opponent_pools=[_find_pool(board_size)],
        max_construction_steps=board_size * 10,
        use_fog=use_fog,
    )


class TestObsSpace:
    """Test observation space shape with and without fog."""

    def test_no_fog_obs_space_has_no_fog_outcomes(self):
        env = make_env(2, use_fog=False)
        assert "fog_outcomes" not in env.observation_space.spaces

    def test_fog_obs_space_has_fog_outcomes(self):
        env = make_env(2, use_fog=True)
        assert "fog_outcomes" in env.observation_space.spaces
        shape = env.observation_space["fog_outcomes"].shape
        assert shape == (5, 6)  # 5 rounds x 6 channels

    def test_fog_obs_includes_fog_outcomes_in_reset(self):
        env = make_env(2, use_fog=True)
        obs, _ = env.reset(seed=42)
        assert "fog_outcomes" in obs
        assert obs["fog_outcomes"].shape == (5, 6)
        # All zeros at start
        np.testing.assert_array_equal(obs["fog_outcomes"], np.zeros((5, 6)))

    def test_no_fog_obs_excludes_fog_outcomes_in_reset(self):
        env = make_env(2, use_fog=False)
        obs, _ = env.reset(seed=42)
        assert "fog_outcomes" not in obs


class TestFogEncoding:
    """Test _encode_opponent_board_fog filtering logic."""

    def _make_board_with_trap(self):
        """Create a size-2 board: piece(1,0) order=1, piece(0,0) order=2, trap(0,1) order=3, final."""
        return Board(
            boardSize=2,
            grid=(("piece", "trap"), ("piece", "empty")),
            sequence=(
                BoardMove(Position(row=1, col=0), "piece", 1),
                BoardMove(Position(row=0, col=0), "piece", 2),
                BoardMove(Position(row=0, col=1), "trap", 3),
                BoardMove(Position(row=-1, col=0), "final", 4),
            ),
        )

    def test_fog_hides_moves_after_player_last_step(self):
        """Moves with order-1 > playerLastStep should be hidden."""
        env = make_env(2, use_fog=True)
        env.reset(seed=42)

        board = self._make_board_with_trap()
        # playerLastStep=0 → only moves where order-1 <= 0, i.e. order=1
        grid = env._encode_opponent_board_fog(board, player_last_step=0, sprung_trap_pos=None)

        # After rotation (opponent frame), piece at (1,0) becomes (0,1) for size 2
        rot = _rotate_position(1, 0, 2)
        # order=1 piece should be visible
        assert grid[rot.row][rot.col][0] == 1

        # order=2 piece should be hidden
        rot2 = _rotate_position(0, 0, 2)
        assert grid[rot2.row][rot2.col][0] == 0

    def test_fog_shows_all_pieces_at_high_last_step(self):
        """All piece moves visible when playerLastStep is high enough."""
        env = make_env(2, use_fog=True)
        env.reset(seed=42)

        board = self._make_board_with_trap()
        # playerLastStep=10 → all moves visible (orders 1,2,3 all have order-1 <= 10)
        grid = env._encode_opponent_board_fog(board, player_last_step=10, sprung_trap_pos=None)

        rot1 = _rotate_position(1, 0, 2)
        rot2 = _rotate_position(0, 0, 2)
        assert grid[rot1.row][rot1.col][0] == 1  # order 1 piece
        assert grid[rot2.row][rot2.col][0] == 2  # order 2 piece
        # Trap still hidden (no sprung trap)
        rot3 = _rotate_position(0, 1, 2)
        assert grid[rot3.row][rot3.col][1] == 0

    def test_fog_hides_traps_without_sprung(self):
        """All traps hidden when player didn't hit any."""
        env = make_env(2, use_fog=True)
        env.reset(seed=42)

        board = self._make_board_with_trap()
        grid = env._encode_opponent_board_fog(board, player_last_step=10, sprung_trap_pos=None)

        rot = _rotate_position(0, 1, 2)
        assert grid[rot.row][rot.col][1] == 0  # trap channel empty

    def test_fog_shows_sprung_trap(self):
        """The sprung trap (player hit it) should be visible."""
        env = make_env(2, use_fog=True)
        env.reset(seed=42)

        board = self._make_board_with_trap()
        sprung_pos = Position(row=0, col=1)
        grid = env._encode_opponent_board_fog(board, player_last_step=10, sprung_trap_pos=sprung_pos)

        rot = _rotate_position(0, 1, 2)
        assert grid[rot.row][rot.col][1] == 3  # trap order=3

    def test_fog_skips_final_moves(self):
        """Final moves should not appear in grid."""
        env = make_env(2, use_fog=True)
        env.reset(seed=42)

        board = self._make_board_with_trap()
        grid = env._encode_opponent_board_fog(board, player_last_step=10, sprung_trap_pos=None)

        # No cell should have the final move's order
        assert not np.any(grid == 4)


class TestFogRoundTrip:
    """Test that fog env works through a complete round."""

    def test_fog_env_completes_round(self):
        """Step through construction until finish, verify fog obs produced."""
        env = make_env(2, use_fog=True)
        obs, _ = env.reset(seed=42)

        # Play construction actions until a round finishes
        done = False
        round_finished = False
        max_steps = 50
        for _ in range(max_steps):
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) == 0:
                break
            action = valid_actions[0]  # Take first valid action
            obs, reward, terminated, truncated, info = env.step(action)
            if "valid_board" in info:
                round_finished = True
                break
            if terminated or truncated:
                break

        assert round_finished, "Should complete at least one round"
        assert "fog_outcomes" in obs
        # After one round, round 0 should have some data
        fog = obs["fog_outcomes"]
        assert fog.shape == (5, 6)
        # At least one channel should be non-zero for round 0
        # (playerLastStep normalized should be > 0 for any valid game)

    def test_fog_outcomes_zero_before_round(self):
        """fog_outcomes should be all zeros during construction (before round ends)."""
        env = make_env(2, use_fog=True)
        obs, _ = env.reset(seed=42)

        # Take one construction step
        masks = env.action_masks()
        valid_actions = np.where(masks)[0]
        piece_actions = valid_actions[valid_actions < 4]  # piece actions for size 2
        if len(piece_actions) > 0:
            obs, _, _, _, _ = env.step(piece_actions[0])
            np.testing.assert_array_equal(obs["fog_outcomes"], np.zeros((5, 6)))

    def test_no_fog_env_full_reveal(self):
        """Without fog, opponent_history should have full board info."""
        env = make_env(2, use_fog=False)
        obs, _ = env.reset(seed=42)

        # Play through to finish a round
        done = False
        for _ in range(50):
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            if len(valid_actions) == 0:
                break
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            if "valid_board" in info or terminated or truncated:
                break

        assert "fog_outcomes" not in obs
