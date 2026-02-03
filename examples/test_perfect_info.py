"""
Quick test to verify perfect information mode works correctly.

Tests both partial and perfect information modes to ensure:
1. Environment initializes correctly
2. Observation space is correct
3. Actions work
4. Episodes complete successfully
"""

from spaces_game import SpacesGameEnv


def test_partial_info():
    """Test partial observability (default)."""
    print("Testing PARTIAL information mode...")
    env = SpacesGameEnv(
        board_pool_path="data/boards_size_2.json",
        deck_size=10,
        opponent_strategy="random",
        perfect_information=False,
    )

    obs, info = env.reset(seed=42)

    # Check observation keys
    expected_keys = {"round", "score_diff", "agent_score", "opponent_score", "first_picker", "agent_history", "opponent_history"}
    assert set(obs.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(obs.keys())}"
    assert "opponent_deck" not in obs, "Opponent deck should not be in partial info mode"

    # Run one episode
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

    assert done, "Episode should be done after 5 rounds"
    print("  ✓ Partial info mode works correctly")
    env.close()


def test_perfect_info():
    """Test perfect information mode."""
    print("\nTesting PERFECT information mode...")
    env = SpacesGameEnv(
        board_pool_path="data/boards_size_2.json",
        deck_size=10,
        opponent_strategy="random",
        perfect_information=True,
    )

    obs, info = env.reset(seed=42)

    # Check observation keys
    expected_keys = {"round", "score_diff", "agent_score", "opponent_score", "first_picker", "agent_history", "opponent_history", "opponent_deck"}
    assert set(obs.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(obs.keys())}"
    assert "opponent_deck" in obs, "Opponent deck should be in perfect info mode"

    # Check opponent deck shape (10 boards, 2x2 grid, 4 features)
    opponent_deck = obs["opponent_deck"]
    expected_shape = (10, 2, 2, 4)
    assert opponent_deck.shape == expected_shape, f"Expected shape {expected_shape}, got {opponent_deck.shape}"

    print(f"  Opponent deck shape: {opponent_deck.shape}")
    print(f"  Opponent deck dtype: {opponent_deck.dtype}")
    print(f"  Opponent deck range: [{opponent_deck.min():.1f}, {opponent_deck.max():.1f}]")

    # Run one episode
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Verify opponent deck is still present
        assert "opponent_deck" in obs, "Opponent deck should remain in observations"
        assert obs["opponent_deck"].shape == expected_shape

    assert done, "Episode should be done after 5 rounds"
    print("  ✓ Perfect info mode works correctly")
    env.close()


def test_observation_space_validation():
    """Test that SB3 can validate the observation space."""
    print("\nTesting observation space validation...")

    # Test partial info
    env_partial = SpacesGameEnv(
        board_pool_path="data/boards_size_2.json",
        perfect_information=False,
    )
    obs, _ = env_partial.reset(seed=42)
    assert env_partial.observation_space.contains(obs), "Partial info obs should be valid"
    print("  ✓ Partial info observation space valid")
    env_partial.close()

    # Test perfect info
    env_perfect = SpacesGameEnv(
        board_pool_path="data/boards_size_2.json",
        perfect_information=True,
    )
    obs, _ = env_perfect.reset(seed=42)
    assert env_perfect.observation_space.contains(obs), "Perfect info obs should be valid"
    print("  ✓ Perfect info observation space valid")
    env_perfect.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Perfect Information Implementation")
    print("=" * 70)

    test_partial_info()
    test_perfect_info()
    test_observation_space_validation()

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
    print("\nYou can now train with perfect information:")
    print("  python examples/train_basic.py --perfect-info --timesteps 100000")
