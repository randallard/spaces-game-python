"""
Train Phase 0 only (goal placement) - simplified training script.

This is a minimal version focusing only on Phase 0 to verify the basic
setup works before adding curriculum complexity.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from spaces_game import ReverseCurriculumBuilderEnv


def make_env(rank: int, seed: int = 0, stage1_model_path: str = None):
    """Create environment for Phase 0."""
    def _init():
        env = ReverseCurriculumBuilderEnv(
            board_size=2,
            board_library_path="new_boards_2.json",
            stage1_model_path=stage1_model_path,
            curriculum_phase=0,  # Phase 0 only
            opponent_strategy="random",
            show_opponent_board=True,
            max_construction_steps=20,
        )
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def train_phase0(
    total_timesteps: int = 50_000,
    n_envs: int = 4,
    stage1_model_path: str = "models/construction/best/best_model.zip",
):
    """Train Phase 0 only."""
    print("=" * 70)
    print("PHASE 0 TRAINING (Goal Placement Only)")
    print("=" * 70)
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Stage 1 model:     {stage1_model_path}")
    print("=" * 70)

    # Create directories
    os.makedirs("models/phase0", exist_ok=True)
    os.makedirs("logs/phase0", exist_ok=True)
    os.makedirs("eval/phase0", exist_ok=True)

    # Create training environment
    print("\nCreating environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, stage1_model_path=stage1_model_path) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, stage1_model_path=stage1_model_path)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(rank=1000, seed=42, stage1_model_path=stage1_model_path)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000 // n_envs,
        save_path="models/phase0",
        name_prefix="ppo_phase0",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/phase0/best",
        log_path="eval/phase0",
        eval_freq=2_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/phase0",
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
    )

    # Train
    print("\nStarting training...")
    print("Monitor: tensorboard --logdir logs/phase0/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = "models/phase0/ppo_phase0_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("PHASE 0 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model: {final_path}")
    print(f"Best model:  models/phase0/best/best_model.zip")
    print("=" * 70)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Phase 0 (goal placement)")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total training timesteps (default: 50,000)",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default="models/construction/best/best_model.zip",
        help="Path to Stage 1 model",
    )

    args = parser.parse_args()

    train_phase0(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        stage1_model_path=args.stage1_model,
    )
