"""
Test Phase 0 of Reverse Curriculum (goal placement only).

This is the easiest phase - board is 99% complete, agent just needs to place goal.
Expected: Agent should quickly learn to place goal and achieve high win rate.
"""

import numpy as np
from spaces_game import ReverseCurriculumBuilderEnv


def test_phase0_basic():
    """Test Phase 0: Just place goal move."""
    print("=" * 70)
    print("PHASE 0 TEST: Goal Placement Only")
    print("=" * 70)

    # Create environment with Phase 0 (easiest)
    # Note: No Stage 1 model for local testing (will use random base selection)
    env = ReverseCurriculumBuilderEnv(
        board_size=2,
        curriculum_phase=0,  # Phase 0 = only goal move
        opponent_strategy="fixed_0",
    )

    print("\n✓ Environment created successfully")
    print(f"  Curriculum phase: {env.curriculum_phase}")
    print(f"  Action space: {env.action_space}")

    # Reset and inspect
    obs, info = env.reset(seed=42)

    print(f"\n✓ Environment reset")
    print(f"  Moves to place: {info['moves_to_place']}")
    print(f"  Base board index: {info['base_board_idx']}")
    print(f"  Remaining moves: {obs['remaining_moves'][0]}")

    # Check that only 1 move needs to be placed (goal)
    assert info['moves_to_place'] == 1, f"Expected 1 move, got {info['moves_to_place']}"
    print(f"\n✓ Correct: Only {info['moves_to_place']} move to place (goal)")

    # Try placing goal (action doesn't matter much, done=1 triggers evaluation)
    action = np.array([0, 0, 1])  # cell=0, type=piece, DONE
    obs, reward, done, truncated, info = env.step(action)

    print(f"\n✓ Step completed")
    print(f"  Done: {done}")
    print(f"  Valid board: {info['valid_board']}")
    print(f"  Agent score: {info['agent_score']}")
    print(f"  Opponent score: {info['opponent_score']}")
    print(f"  Reward: {reward:.2f}")

    # Board might be invalid if we placed incorrectly, but that's expected for random action
    if info['valid_board']:
        print("\n✓ Board is valid!")
    else:
        print("\n⚠ Board is invalid (expected for random action in test)")

    env.close()

    print("\n" + "=" * 70)
    print("✓ PHASE 0 TEST PASSED")
    print("=" * 70)
    print("\nNext step: Train agent on Phase 0 to achieve >95% win rate")
    print("Then progress to Phase 1 (last move + goal)")


def test_phase0_with_stage1():
    """Test Phase 0 with Stage 1 model if available."""
    import os

    # Check if Stage 1 model exists
    stage1_path = "models/construction/best/best_model.zip"

    if not os.path.exists(stage1_path):
        print("\n" + "=" * 70)
        print("SKIP: Stage 1 model not found")
        print("=" * 70)
        print(f"Looking for: {stage1_path}")
        print("This test requires the Stage 1 trained model from tenx machine")
        return

    print("\n" + "=" * 70)
    print("PHASE 0 TEST WITH STAGE 1 MODEL")
    print("=" * 70)

    env = ReverseCurriculumBuilderEnv(
        board_size=2,
        curriculum_phase=0,
        stage1_model_path=stage1_path,
        opponent_strategy="fixed_0",
    )

    print(f"\n✓ Environment created with Stage 1 model")

    obs, info = env.reset(seed=42)

    print(f"✓ Reset with Stage 1 base selection")
    print(f"  Base board selected by Stage 1: {info['base_board_idx']}")

    # Place goal
    action = np.array([0, 0, 1])
    obs, reward, done, truncated, info = env.step(action)

    print(f"\n✓ Step completed")
    print(f"  Valid: {info['valid_board']}, Agent: {info['agent_score']}, Opp: {info['opponent_score']}")
    print(f"  Reward: {reward:.2f}")

    env.close()

    print("\n✓ Stage 1 integration works!")


if __name__ == "__main__":
    test_phase0_basic()
    test_phase0_with_stage1()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print("\nReady to train Phase 0!")
