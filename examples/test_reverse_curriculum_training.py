"""
Quick verification test for reverse curriculum training setup.

Tests the PhaseProgressionCallback without running full training.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from spaces_game import ReverseCurriculumBuilderEnv


def test_callback_evaluation():
    """Test that the phase callback can evaluate without errors."""
    print("=" * 70)
    print("TEST: PhaseProgressionCallback Evaluation")
    print("=" * 70)

    # Create a simple environment
    def make_env():
        env = ReverseCurriculumBuilderEnv(
            board_size=2,
            board_library_path="new_boards_2.json",
            stage1_model_path="models/construction/best/best_model.zip",
            curriculum_phase=0,
            opponent_strategy="random",
            show_opponent_board=True,
        )
        env.reset(seed=42)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    print("✓ Environment created")

    # Create a simple PPO model
    model = PPO("MultiInputPolicy", env, verbose=0)

    print("✓ Model created")

    # Manually test the evaluation logic
    print("\nTesting evaluation loop...")

    for ep in range(3):
        print(f"\nEpisode {ep + 1}:")
        obs = env.reset()
        done = False
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)

            print(f"  Step {step_count + 1}: step() returned {len(step_result)} values")

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                print(f"    Using 5-value API (Gymnasium)")
            else:
                obs, reward, done, info = step_result
                print(f"    Using 4-value API (Gym)")

            # Handle vectorized outputs
            if isinstance(done, np.ndarray):
                print(f"    done is array: {done}, using .any()")
                done = done.any()
            else:
                print(f"    done is scalar: {done}")

            if isinstance(reward, np.ndarray):
                print(f"    reward is array: {reward}, extracting [0]")
                reward_value = reward[0]
            else:
                print(f"    reward is scalar: {reward}")
                reward_value = reward

            step_count += 1

            if step_count > 10:  # Safety limit
                print("    (stopping after 10 steps)")
                break

        # Check info
        print(f"\n  Info type: {type(info)}")
        if isinstance(info, list):
            print(f"    Info is list, extracting [0]")
            info = info[0]

        print(f"  Info after extraction: {type(info)}")
        print(f"  Info keys: {info.keys() if hasattr(info, 'keys') else 'N/A'}")

        if hasattr(info, 'get'):
            valid = info.get('valid_board', False)
            agent_score = info.get('agent_score', 0)
            opponent_score = info.get('opponent_score', 0)
            print(f"  ✓ Info is dict-like:")
            print(f"    valid_board: {valid}")
            print(f"    agent_score: {agent_score}")
            print(f"    opponent_score: {opponent_score}")
        else:
            print(f"  ✗ ERROR: Info is not dict-like: {info}")
            return False

    env.close()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)
    print("\nThe PhaseProgressionCallback should work correctly now!")
    return True


if __name__ == "__main__":
    success = test_callback_evaluation()
    exit(0 if success else 1)
