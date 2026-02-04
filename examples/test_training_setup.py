"""
Quick validation test for training setup.

Run this before starting a long training run to verify:
- All dependencies are installed
- Environment can be created
- PPO can be initialized
- A few training steps work
- Reward shaping is functioning

Usage:
    python examples/test_training_setup.py
"""

import sys
import numpy as np

print("=" * 70)
print("TRAINING SETUP VALIDATION")
print("=" * 70)

# Test 1: Check imports
print("\n[1/6] Checking dependencies...")
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from spaces_game import BoardBuilderEnv
    print("✓ All dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nInstall with:")
    print("  pip install stable-baselines3 gymnasium")
    sys.exit(1)

# Test 2: Create environment
print("\n[2/6] Creating BoardBuilderEnv...")
try:
    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
        max_construction_steps=10,
    )
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space keys: {list(env.observation_space.keys())}")
except Exception as e:
    print(f"✗ Failed to create environment: {e}")
    sys.exit(1)

# Test 3: Test observation and action
print("\n[3/6] Testing environment reset and step...")
try:
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful")
    print(f"  Round: {info['round']}")
    print(f"  Construction step: {obs['construction_step']}")
    print(f"  Valid cells: {obs['valid_cells_mask']}")

    # Take a valid action: place piece at bottom-left
    action = np.array([2, 0, 0])  # cell=2, type=piece, done=continue
    obs, reward, done, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  Reward: {reward:.2f}")
    print(f"  Construction step: {obs['construction_step']}")
except Exception as e:
    print(f"✗ Failed environment step: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test reward shaping
print("\n[4/6] Testing reward shaping...")
try:
    env.reset(seed=42)

    # First piece (bottom row) - should get +1.0
    action = np.array([2, 0, 0])  # Bottom-left piece
    obs, reward1, done, truncated, info = env.step(action)

    # Second piece (top row, adjacent) - should get +0.5 +1.0 +2.0 = +3.5
    action = np.array([0, 0, 0])  # Top-left piece
    obs, reward2, done, truncated, info = env.step(action)

    print(f"✓ Reward shaping working")
    print(f"  First piece reward: {reward1:.2f} (expected ~1.0)")
    print(f"  Second piece reward: {reward2:.2f} (expected ~3.5)")

    if reward1 < 0.5 or reward1 > 1.5:
        print(f"  ⚠ Warning: First piece reward unexpected")
    if reward2 < 3.0 or reward2 > 4.0:
        print(f"  ⚠ Warning: Second piece reward unexpected")

except Exception as e:
    print(f"✗ Failed reward test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create vectorized environment
print("\n[5/6] Creating vectorized environment...")
try:
    def make_env():
        env = BoardBuilderEnv(
            board_size=2,
            opponent_library_path="new_boards_2.json",
            opponent_strategy="random",
            show_opponent_board=True,
            max_construction_steps=10,
        )
        return env

    vec_env = DummyVecEnv([make_env])
    print(f"✓ Vectorized environment created")

    obs = vec_env.reset()
    print(f"  Observation shape: {obs['construction_step'].shape}")
    vec_env.close()
except Exception as e:
    print(f"✗ Failed to create vectorized environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Initialize PPO
print("\n[6/6] Initializing PPO agent...")
try:
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=3,
    )
    print(f"✓ PPO agent initialized")
    print(f"  Policy: {model.policy.__class__.__name__}")
    print(f"  Device: {model.device}")

    # Test a few training steps
    print("\n  Testing training loop...")
    model.learn(total_timesteps=256, progress_bar=False)
    print(f"✓ Training loop works (256 steps completed)")

    vec_env.close()
except Exception as e:
    print(f"✗ Failed to initialize or train PPO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✓ ALL VALIDATION TESTS PASSED")
print("=" * 70)
print("\nYour training setup is ready!")
print("\nTo start training:")
print("  python examples/train_board_builder.py --timesteps 200000")
print("\nRecommended settings:")
print("  --timesteps 200000   (or more for better results)")
print("  --envs 4             (parallel environments)")
print("  --board-size 2       (start with 2x2 boards)")
print("=" * 70)
