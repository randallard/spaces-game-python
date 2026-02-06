"""
Train an agent for board construction (Stage 1 of curriculum).

The agent learns to select counter-boards from a library when it can
see the opponent's board. This is the foundation for strategic play.

Usage:
    python examples/train_construction.py --size 2
    python examples/train_construction.py --size 3 --timesteps 200000
    python examples/train_construction.py --size 3 --board-library new_boards_3.json

Training will save checkpoints to models/size{N}/stage1/ and log to
tensorboard. Monitor progress with:
    tensorboard --logdir logs/size{N}_stage1/
"""

import os
import random as py_random
import gymnasium as gym
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from spaces_game import BoardConstructionEnv


class FixedBoardCurriculumWrapper(gym.Wrapper):
    """
    Wrapper that cycles through fixed opponent boards for balanced training.

    Ensures agent learns optimal counter for EACH board in the library
    by giving equal training time to each opponent board.
    """

    def __init__(self, env: BoardConstructionEnv, curriculum_mode: str = "cycle"):
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


def make_env(rank: int, seed: int = 0, curriculum: bool = True, board_library_path: str = "new_boards_2.json"):
    """
    Create a single environment instance.

    Args:
        rank: Environment index (for parallel training)
        seed: Random seed offset
        curriculum: If True, use fixed-board curriculum (cycle through all boards)
        board_library_path: Path to board library JSON

    Returns:
        Function that creates the environment
    """
    def _init():
        env = BoardConstructionEnv(
            board_library_path=board_library_path,
            opponent_strategy="random",  # Will be overridden by wrapper
            show_opponent_board=True,    # Perfect information (Stage 1)
        )

        # Wrap to cycle through fixed boards for balanced training
        if curriculum:
            env = FixedBoardCurriculumWrapper(env, curriculum_mode="cycle")

        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def train(
    board_size: int = 2,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    eval_freq: int = 5_000,
    save_freq: int = 10_000,
    board_library_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """
    Train PPO agent for board construction.

    Args:
        board_size: Board size (derived from library, used for path defaults)
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency (timesteps)
        save_freq: Checkpoint save frequency (timesteps)
        board_library_path: Path to board library JSON
        output_dir: Output directory for models
    """
    if board_library_path is None:
        board_library_path = f"new_boards_{board_size}.json"
    if output_dir is None:
        output_dir = f"models/size{board_size}/stage1"

    log_dir = f"logs/size{board_size}_stage1"
    eval_dir = f"eval/size{board_size}_stage1"

    # Check if board library exists
    if not Path(board_library_path).exists():
        print(f"\nERROR: Board library not found: {board_library_path}")
        print(f"  Create it with curated size-{board_size} boards before training.")
        return

    # Count boards
    import json
    with open(board_library_path) as f:
        n_boards = len(json.load(f))

    print("=" * 70)
    print("BOARD CONSTRUCTION TRAINING (Stage 1)")
    print("=" * 70)
    print(f"Board size:      {board_size}x{board_size}")
    print(f"Board library:   {board_library_path} ({n_boards} boards)")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs:   {n_envs}")
    print(f"Eval frequency:  {eval_freq:,} steps")
    print(f"Save frequency:  {save_freq:,} steps")
    print(f"Curriculum:      Cycling through all {n_boards} boards (equal training per board)")
    print(f"Output dir:      {output_dir}")
    print("=" * 70)

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Create vectorized training environment
    print("\nCreating training environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, board_library_path=board_library_path) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, board_library_path=board_library_path)])

    # Create evaluation environment (single, deterministic)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(rank=1000, seed=42, board_library_path=board_library_path)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for parallel envs
        save_path=output_dir,
        name_prefix="ppo_stage1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best",
        log_path=eval_dir,
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
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,  # Adjust for parallel envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
    )

    # Train
    print("\nStarting training...")
    print(f"Monitor progress with: tensorboard --logdir {log_dir}/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{output_dir}/ppo_stage1_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir}/best/best_model.zip")
    print(f"\nThis model is Stage 1 for size {board_size}.")
    print(f"Use it for Stage 2 training:")
    print(f"  python examples/train_reverse_curriculum.py --size {board_size}")
    print("=" * 70)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train board construction agent (Stage 1)")
    parser.add_argument(
        "--size",
        type=int,
        default=2,
        help="Board size (default: 2)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100,000)",
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
        default=5_000,
        help="Evaluation frequency in timesteps (default: 5,000)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency in timesteps (default: 10,000)",
    )
    parser.add_argument(
        "--board-library",
        type=str,
        default=None,
        help="Path to board library JSON (default: new_boards_{N}.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/size{N}/stage1)",
    )

    args = parser.parse_args()

    train(
        board_size=args.size,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        board_library_path=args.board_library,
        output_dir=args.output_dir,
    )
