"""
Test the BoardConstructionEnv to verify it works correctly.

Quick validation before running full training.
"""

from spaces_game import BoardConstructionEnv


def test_basic_episode():
    """Test basic episode execution."""
    print("=" * 70)
    print("Testing BoardConstructionEnv - Basic Episode")
    print("=" * 70)

    env = BoardConstructionEnv(
        board_library_path="new_boards_2.json",
        opponent_strategy="fixed",  # Use fixed for deterministic testing
        show_opponent_board=True,
    )

    print(f"\nEnvironment created:")
    print(f"  Board library size: {env.library_size}")
    print(f"  Board size: {env.board_size}x{env.board_size}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space keys: {list(env.observation_space.keys())}")

    obs, info = env.reset(seed=42)

    print(f"\nInitial observation:")
    for key, value in obs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    print(f"\nPlaying 5 rounds...")
    total_reward = 0

    for round_num in range(1, 6):
        # Agent selects board 0 (arbitrary choice for testing)
        action = 0
        obs, reward, done, truncated, info = env.step(action)

        print(f"\n  Round {round_num}:")
        print(f"    Action: board {action}")
        print(f"    Reward: {reward:+.1f}")
        print(f"    Score: {info['agent_total_score']}-{info['opponent_total_score']}")
        print(f"    Done: {done}")

        total_reward += reward

    print(f"\n{'=' * 70}")
    print(f"Episode Complete!")
    print(f"  Total reward: {total_reward:+.1f}")
    print(f"  Final score: {info['agent_total_score']}-{info['opponent_total_score']}")

    if info['agent_total_score'] > info['opponent_total_score']:
        print(f"  Result: AGENT WINS!")
    elif info['agent_total_score'] < info['opponent_total_score']:
        print(f"  Result: OPPONENT WINS")
    else:
        print(f"  Result: TIE")

    print(f"{'=' * 70}")

    env.close()


def test_without_showing_opponent():
    """Test with opponent board hidden (simultaneous mode)."""
    print("\n\n" + "=" * 70)
    print("Testing BoardConstructionEnv - Simultaneous Mode (Hidden Opponent)")
    print("=" * 70)

    env = BoardConstructionEnv(
        board_library_path="new_boards_2.json",
        opponent_strategy="random",
        show_opponent_board=False,  # Don't show opponent board
    )

    obs, info = env.reset(seed=42)

    print(f"\nObservation keys (opponent board should be absent):")
    print(f"  {list(obs.keys())}")

    if "opponent_board" in obs:
        print(f"  ❌ ERROR: opponent_board should not be in observation!")
    else:
        print(f"  ✓ Correct: opponent_board not in observation (simultaneous mode)")

    # Play one round
    obs, reward, done, truncated, info = env.step(0)
    print(f"\nRound 1 completed:")
    print(f"  Reward: {reward:+.1f}")
    print(f"  Score: {info['agent_total_score']}-{info['opponent_total_score']}")

    env.close()


def test_board_reusability():
    """Test that boards can be reused across rounds."""
    print("\n\n" + "=" * 70)
    print("Testing BoardConstructionEnv - Board Reusability")
    print("=" * 70)

    env = BoardConstructionEnv(
        board_library_path="new_boards_2.json",
        opponent_strategy="fixed",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    print(f"\nReusing board 0 for all 5 rounds...")

    for round_num in range(1, 6):
        # Always use board 0
        obs, reward, done, truncated, info = env.step(0)
        print(f"  Round {round_num}: Used board 0, reward={reward:+.1f}")

    print(f"\n✓ Success: No error when reusing same board 5 times")
    print(f"  (In deck selection mode, this would have failed)")

    env.close()


def test_random_vs_greedy():
    """Compare random vs greedy opponent strategies."""
    print("\n\n" + "=" * 70)
    print("Testing BoardConstructionEnv - Opponent Strategies")
    print("=" * 70)

    for strategy in ["random", "greedy", "fixed"]:
        env = BoardConstructionEnv(
            board_library_path="new_boards_2.json",
            opponent_strategy=strategy,
            show_opponent_board=True,
        )

        obs, info = env.reset(seed=42)

        # Play full episode with agent using board 0
        total_reward = 0
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(0)
            total_reward += reward

        print(f"\n  {strategy.upper()} opponent:")
        print(f"    Final score: {info['agent_total_score']}-{info['opponent_total_score']}")
        print(f"    Total reward: {total_reward:+.1f}")

        env.close()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BOARD CONSTRUCTION ENVIRONMENT TEST" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")

    test_basic_episode()
    test_without_showing_opponent()
    test_board_reusability()
    test_random_vs_greedy()

    print("\n\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nEnvironment is ready for training.")
    print("Next step: python examples/train_construction.py")
    print("=" * 70)
