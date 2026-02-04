"""
Simple smoke test for BoardModifierEnv.
Just validates it can be created and run for a few steps.
"""

import numpy as np
from spaces_game import BoardModifierEnv


def test_basic_workflow():
    """Test basic workflow: create, reset, a few steps."""
    print("Creating environment...")
    env = BoardModifierEnv(board_size=2, max_modifications=2)

    print("Resetting...")
    obs, info = env.reset(seed=42)
    print(f"Reset successful. Phase: {obs['phase']}")

    print("Selecting base board...")
    action = np.array([0, 0, 0, 0])  # Select base board 0
    obs, reward, done, truncated, info = env.step(action)
    print(f"Base selected. Phase: {obs['phase']}, Reward: {reward}")

    print("Finishing (no modifications)...")
    action = np.array([0, 0, 0, 1])  # Finish immediately
    obs, reward, done, truncated, info = env.step(action)
    print(f"Round complete. Valid: {info.get('valid_board')}, Agent: {info.get('agent_score')}, Opp: {info.get('opponent_score')}")

    env.close()
    print("\n✓ Basic workflow test passed!")


if __name__ == "__main__":
    test_basic_workflow()
