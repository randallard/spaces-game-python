"""
Train an agent for board building (Stage 2 of curriculum).

The agent learns to BUILD boards from scratch using sequential actions
(place pieces/traps step-by-step) rather than selecting from a library.

Usage:
    python examples/train_board_builder.py

Training will save checkpoints to models/builder/ and log to tensorboard.
Monitor progress with:
    tensorboard --logdir logs/builder/
"""

import os
import random as py_random
import gymnasium as gym
from typing import Any, Dict, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from spaces_game import BoardBuilderEnv


class FixedBoardCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that cycles through fixed opponent boards for balanced training.

    Ensures agent learns to build counters for EACH opponent board
    by giving equal training time to each opponent board.
    """

    def __init__(self, env: BoardBuilderEnv, curriculum_mode: str = "cycle"):
        super().__init__(env)
        self.library_size = len(env.opponent_library)
        self.curriculum_mode = curriculum_mode
        self.episode_count = 0
        self.current_board_idx = 0

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.curriculum_mode == "cycle":
            # Cycle through boards sequentially: 0, 1, 2, ..., 7, 0, 1, ...
            self.current_board_idx = self.episode_count % self.library_size
        elif self.curriculum_mode == "random":
            # Random selection (fallback for comparison)
            self.current_board_idx = py_random.randint(0, self.library_size - 1)

        # Override opponent selection to use specific board
        self.env.opponent_strategy = f"fixed_{self.current_board_idx}"

        self.episode_count += 1
        return self.env.reset(**kwargs)


def make_env(rank: int, seed: int = 0, curriculum: bool = True, board_size: int = 2):
    """
    Create a single environment instance.

    Args:
        rank: Environment index (for parallel training)
        seed: Random seed offset
        curriculum: If True, use fixed-board curriculum (cycle through all boards)
        board_size: Size of boards to build (2 for 2x2, 3 for 3x3, etc.)

    Returns:
        Function that creates the environment
    """
    def _init():
        env = BoardBuilderEnv(
            board_size=board_size,
            opponent_library_path="new_boards_2.json",
            opponent_strategy="random",  # Will be overridden by wrapper
            show_opponent_board=True,    # Perfect information (Stage 2)
            max_construction_steps=20,   # Allow plenty of steps for learning
        )

        # Wrap to cycle through fixed boards for balanced training
        if curriculum:
            env = FixedBoardCurriculumWrapper(env, curriculum_mode="cycle")

        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    eval_freq: int = 10_000,
    save_freq: int = 20_000,
    board_size: int = 2,
):
    """
    Train PPO agent for board building.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency (timesteps)
        save_freq: Checkpoint save frequency (timesteps)
        board_size: Size of boards to build (2 for 2x2, 3 for 3x3, etc.)
    """
    print("=" * 70)
    print("BOARD BUILDER TRAINING (Stage 2)")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs:   {n_envs}")
    print(f"Eval frequency:  {eval_freq:,} steps")
    print(f"Save frequency:  {save_freq:,} steps")
    print(f"Board size:      {board_size}x{board_size}")
    print(f"Curriculum:      Cycling through all 8 opponent boards")
    print("=" * 70)

    # Create directories
    os.makedirs("models/builder", exist_ok=True)
    os.makedirs("logs/builder", exist_ok=True)
    os.makedirs("eval/builder", exist_ok=True)

    # Create vectorized training environment
    print("\nCreating training environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, board_size=board_size) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, board_size=board_size)])

    # Create evaluation environment (single, deterministic)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(rank=1000, seed=42, board_size=board_size)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for parallel envs
        save_path="models/builder",
        name_prefix="ppo_builder",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/builder/best",
        log_path="eval/builder",
        eval_freq=eval_freq // n_envs,  # Adjust for parallel envs
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Create PPO agent
    print("\nInitializing PPO agent...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/builder",
        learning_rate=3e-4,
        n_steps=2048 // n_envs,  # Adjust for parallel envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher exploration for construction
        vf_coef=0.5,
    )

    # Train
    print("\nStarting training...")
    print("Monitor progress with: tensorboard --logdir logs/builder/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = "models/builder/ppo_builder_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: models/builder/best/best_model.zip")
    print(f"\nEvaluate with:")
    print(f"  python examples/evaluate_board_builder.py {final_path}")
    print("=" * 70)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train board builder agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200,000)",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in timesteps (default: 10,000)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=20_000,
        help="Checkpoint save frequency in timesteps (default: 20,000)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=2,
        help="Size of boards to build (default: 2 for 2x2)",
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        board_size=args.board_size,
    )
