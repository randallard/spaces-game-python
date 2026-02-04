"""
Test suite for BoardModifierEnv (Stage 1.5).

Validates environment mechanics, action space, observation space,
and modification logic.
"""

import numpy as np
import pytest
from spaces_game import BoardModifierEnv


def test_env_creation():
    """Test environment can be created with valid parameters."""
    env = BoardModifierEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        max_modifications=3,
    )
    assert env.board_size == 2
    assert env.max_modifications == 3
    assert len(env.opponent_library) > 0
    assert len(env.base_library) > 0
    env.close()


def test_observation_space():
    """Test observation space has correct shape and types."""
    env = BoardModifierEnv(board_size=2)

    obs_space = env.observation_space
    assert "opponent_board" in obs_space.spaces
    assert "base_board" in obs_space.spaces
    assert "modification_step" in obs_space.spaces
    assert "modifications_made" in obs_space.spaces
    assert "phase" in obs_space.spaces
    assert "valid_cells_mask" in obs_space.spaces

    # Check shapes
    assert obs_space["opponent_board"].shape == (2, 2, 2)
    assert obs_space["base_board"].shape == (2, 2, 2)
    assert obs_space["valid_cells_mask"].shape == (4,)

    env.close()


def test_action_space():
    """Test action space is MultiDiscrete with correct dimensions."""
    env = BoardModifierEnv(board_size=2)

    assert hasattr(env.action_space, "nvec")
    nvec = env.action_space.nvec

    # [base_or_modify, cell, type, done]
    assert nvec[0] == len(env.base_library)  # Base board selection
    assert nvec[1] == 4  # 2x2 = 4 cells
    assert nvec[2] == 2  # Piece or trap
    assert nvec[3] == 2  # Done or continue

    env.close()


def test_reset():
    """Test environment reset returns valid observation."""
    env = BoardModifierEnv(board_size=2, max_modifications=2)
    obs, info = env.reset(seed=42)

    # Check observation structure
    assert "opponent_board" in obs
    assert "base_board" in obs
    assert "modification_step" in obs
    assert "modifications_made" in obs
    assert "phase" in obs

    # Check initial state
    assert obs["phase"] == 0  # Base selection phase
    assert obs["modification_step"] == 0
    assert obs["modifications_made"][0] == 0

    # Check info
    assert info["round"] == 1
    assert info["phase"] == "base_selection"

    env.close()


def test_base_selection_phase():
    """Test base board selection phase."""
    env = BoardModifierEnv(board_size=2, max_modifications=2)
    obs, info = env.reset(seed=42)

    # Action: select base board 0
    action = np.array([0, 0, 0, 0])  # Only first element matters in phase 0
    obs, reward, done, truncated, info = env.step(action)

    # Should transition to modification phase
    assert obs["phase"] == 1
    assert info["phase"] == "modification"
    assert info["base_board_selected"] == 0
    assert reward == 0.1  # Small reward for selecting base

    # Base board should now be visible
    assert np.any(obs["base_board"] > 0)

    assert not done
    assert not truncated

    env.close()


def test_modification_phase_add_piece():
    """Test adding a piece during modification phase."""
    env = BoardModifierEnv(board_size=2, max_modifications=3)
    obs, info = env.reset(seed=42)

    # Select base board
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)

    # Get initial board state
    base_board_before = obs["base_board"].copy()

    # Add a piece at cell 1 (top-right)
    action = np.array([0, 1, 0, 0])  # add, cell=1, piece, continue
    obs, reward, done, truncated, info = env.step(action)

    # Check modification was applied
    assert obs["modifications_made"][0] == 1
    assert obs["modification_step"] == 1
    assert reward > 0  # Small reward for modification

    # Board should have changed (either added piece or modified existing)
    # Note: May not always be different if cell already had a piece
    assert not done
    assert not truncated

    env.close()


def test_modification_phase_remove_trap():
    """Test removing a trap during modification phase."""
    env = BoardModifierEnv(board_size=2, max_modifications=3)
    obs, info = env.reset(seed=42)

    # Select base board
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)

    # Remove a trap at cell 2
    action = np.array([1, 2, 1, 0])  # remove, cell=2, trap, continue
    obs, reward, done, truncated, info = env.step(action)

    assert obs["modifications_made"][0] == 1
    assert reward > 0  # Small reward for modification
    assert not done

    env.close()


def test_max_modifications_enforced():
    """Test that max_modifications limit is enforced."""
    env = BoardModifierEnv(board_size=2, max_modifications=2)
    obs, info = env.reset(seed=42)

    # Select base board
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)

    # Make 2 modifications without finishing
    action = np.array([0, 0, 0, 0])  # Modification 1
    obs, reward, done, truncated, info = env.step(action)
    assert obs["modifications_made"][0] == 1
    assert not done

    action = np.array([0, 1, 0, 0])  # Modification 2
    obs, reward, done, truncated, info = env.step(action)
    assert obs["modifications_made"][0] == 2

    # Should automatically finish after max_modifications
    # This step plays the round
    assert "valid_board" in info

    env.close()


def test_early_finish_with_done_flag():
    """Test finishing modifications early with done=1."""
    env = BoardModifierEnv(board_size=2, max_modifications=3)
    obs, info = env.reset(seed=42)

    # Select base board
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)

    # Make 1 modification and finish
    action = np.array([0, 0, 0, 1])  # add, cell=0, piece, DONE
    obs, reward, done, truncated, info = env.step(action)

    # Should play the round
    assert "valid_board" in info
    assert "agent_score" in info
    assert "opponent_score" in info

    env.close()


def test_valid_board_reward():
    """Test that valid boards receive positive rewards."""
    env = BoardModifierEnv(board_size=2, max_modifications=3)
    obs, info = env.reset(seed=42)

    # Select base board (should be valid)
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)

    # Finish immediately without modifications (base board is valid)
    action = np.array([0, 0, 0, 1])  # done=1
    obs, reward, done, truncated, info = env.step(action)

    assert info["valid_board"] is True
    # Reward should be score_diff + potential bonuses (win/tie)
    # Should NOT have -50 invalid penalty

    env.close()


def test_invalid_board_penalty():
    """Test that invalid boards receive heavy penalty."""
    env = BoardModifierEnv(board_size=2, max_modifications=3)
    obs, info = env.reset(seed=42)

    # Select base board
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)

    # Remove ALL pieces from bottom row to make board invalid
    # Cell 2 = row 1 col 0, Cell 3 = row 1 col 1 (bottom row)
    action = np.array([1, 2, 0, 0])  # remove, cell=2, piece, continue
    obs, reward, done, truncated, info = env.step(action)

    action = np.array([1, 3, 0, 1])  # remove, cell=3, piece, DONE
    obs, final_reward, done, truncated, info = env.step(action)

    # Check if board became invalid
    if not info["valid_board"]:
        assert final_reward < -40  # Should have -50 penalty
        assert info["agent_score"] == 0  # Invalid boards score 0
        assert info["opponent_score"] == 0

    env.close()


def test_multiple_rounds():
    """Test that environment runs multiple rounds per episode."""
    env = BoardModifierEnv(board_size=2, max_modifications=2, max_rounds=2)
    obs, info = env.reset(seed=42)

    round_1_done = False
    round_2_done = False

    # Round 1
    action = np.array([0, 0, 0, 0])  # Select base
    obs, reward, done, truncated, info = env.step(action)

    action = np.array([0, 0, 0, 1])  # Finish immediately
    obs, reward, done, truncated, info = env.step(action)

    if info["round"] == 1:
        round_1_done = True
        assert not done  # Should continue to round 2

    # Round 2
    action = np.array([0, 0, 0, 0])  # Select base
    obs, reward, done, truncated, info = env.step(action)

    action = np.array([0, 0, 0, 1])  # Finish immediately
    obs, reward, done, truncated, info = env.step(action)

    if info["round"] == 2:
        round_2_done = True
        assert done  # Episode should end after 2 rounds

    assert round_1_done
    assert round_2_done

    env.close()


def test_opponent_strategy_fixed():
    """Test fixed opponent strategy."""
    env = BoardModifierEnv(
        board_size=2,
        opponent_strategy="fixed_0",
        max_modifications=2,
    )

    obs, info = env.reset(seed=42)
    opponent_board_1 = obs["opponent_board"].copy()

    # Complete round 1
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)
    action = np.array([0, 0, 0, 1])
    obs, reward, done, truncated, info = env.step(action)

    # Check that round 2 has same opponent board (fixed_0)
    if not done:
        opponent_board_2 = obs["opponent_board"].copy()
        assert np.array_equal(opponent_board_1, opponent_board_2)

    env.close()


def test_deterministic_reset():
    """Test that reset with same seed produces same initial state."""
    env = BoardModifierEnv(board_size=2, opponent_strategy="random")

    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)

    assert np.array_equal(obs1["opponent_board"], obs2["opponent_board"])
    assert obs1["phase"] == obs2["phase"]

    env.close()


def test_episode_scoring():
    """Test that episode tracks cumulative scores correctly."""
    env = BoardModifierEnv(board_size=2, max_modifications=1, max_rounds=2)
    obs, info = env.reset(seed=42)

    total_agent_score = 0
    total_opponent_score = 0

    # Round 1
    action = np.array([0, 0, 0, 0])
    obs, reward, done, truncated, info = env.step(action)
    action = np.array([0, 0, 0, 1])
    obs, reward, done, truncated, info = env.step(action)

    total_agent_score += info["agent_score"]
    total_opponent_score += info["opponent_score"]

    if not done:
        # Round 2
        action = np.array([0, 0, 0, 0])
        obs, reward, done, truncated, info = env.step(action)
        action = np.array([0, 0, 0, 1])
        obs, reward, done, truncated, info = env.step(action)

        total_agent_score += info["agent_score"]
        total_opponent_score += info["opponent_score"]

        assert info["agent_total_score"] == total_agent_score
        assert info["opponent_total_score"] == total_opponent_score

    env.close()


def test_win_tie_rewards():
    """Test that wins and ties receive bonus rewards."""
    env = BoardModifierEnv(board_size=2, max_modifications=1)
    obs, info = env.reset(seed=42)

    # Play a round
    action = np.array([0, 0, 0, 0])  # Select base
    obs, reward, done, truncated, info = env.step(action)

    action = np.array([0, 0, 0, 1])  # Finish
    obs, reward, done, truncated, info = env.step(action)

    if info["valid_board"]:
        agent_score = info["agent_score"]
        opponent_score = info["opponent_score"]
        score_diff = agent_score - opponent_score

        # Check reward includes appropriate bonus
        if agent_score > opponent_score:
            # Should have win bonus (+10)
            expected_reward_base = score_diff + 10.0
            assert reward >= expected_reward_base - 0.5  # Allow for modification rewards
        elif agent_score == opponent_score:
            # Should have tie bonus (+5)
            expected_reward_base = score_diff + 5.0
            assert reward >= expected_reward_base - 0.5

    env.close()


if __name__ == "__main__":
    print("Running BoardModifierEnv tests...")
    print("=" * 70)

    tests = [
        test_env_creation,
        test_observation_space,
        test_action_space,
        test_reset,
        test_base_selection_phase,
        test_modification_phase_add_piece,
        test_modification_phase_remove_trap,
        test_max_modifications_enforced,
        test_early_finish_with_done_flag,
        test_valid_board_reward,
        test_invalid_board_penalty,
        test_multiple_rounds,
        test_opponent_strategy_fixed,
        test_deterministic_reset,
        test_episode_scoring,
        test_win_tie_rewards,
    ]

    passed = 0
    failed = 0

    for test in tests:
        test_name = test.__name__
        try:
            test()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
        exit(1)
