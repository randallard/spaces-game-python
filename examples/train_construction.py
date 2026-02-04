"""
Train an agent for board construction (Stage 1 of curriculum).

The agent learns to select counter-boards from a library when it can
see the opponent's board. This is the foundation for strategic play.

Usage:
    python examples/train_construction.py

Training will save checkpoints to models/construction/ and log to
tensorboard. Monitor progress with:
    tensorboard --logdir logs/construction/
"""

import os
import random as py_random
import gymnasium as gym
from typing import Any, Dict, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from spaces_game import BoardConstructionEnv


class MixedOpponentWrapper(gym.Wrapper):
    """
    Wrapper that randomizes opponent strategy each episode.

    Forces agent to learn adaptive counter-play rather than
    memorizing one fixed strategy.
    """

    def __init__(self, env: BoardConstructionEnv, strategy_weights: Dict[str, float] = None):
        super().__init__(env)
        self.strategies = ["random", "greedy", "fixed"]

        if strategy_weights is None:
            # Default: favor random, but include fixed opponents for counter-learning
            self.weights = [0.4, 0.3, 0.3]
        else:
            self.weights = [strategy_weights.get(s, 0.0) for s in self.strategies]

    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Pick new opponent strategy for this episode
        strategy = py_random.choices(self.strategies, weights=self.weights, k=1)[0]
        self.env.opponent_strategy = strategy
        return self.env.reset(**kwargs)


def make_env(rank: int, seed: int = 0, mixed_opponents: bool = True):
    """
    Create a single environment instance.

    Args:
        rank: Environment index (for parallel training)
        seed: Random seed offset
        mixed_opponents: If True, randomize opponent strategy each episode

    Returns:
        Function that creates the environment
    """
    def _init():
        env = BoardConstructionEnv(
            board_library_path="new_boards_2.json",
            opponent_strategy="random",  # Will be overridden by wrapper
            show_opponent_board=True,    # Perfect information (Stage 1)
        )

        # Wrap to randomize opponent each episode
        if mixed_opponents:
            env = MixedOpponentWrapper(env)

        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    eval_freq: int = 5_000,
    save_freq: int = 10_000,
):
    """
    Train PPO agent for board construction.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency (timesteps)
        save_freq: Checkpoint save frequency (timesteps)
    """
    print("=" * 70)
    print("BOARD CONSTRUCTION TRAINING (Stage 1)")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs:   {n_envs}")
    print(f"Eval frequency:  {eval_freq:,} steps")
    print(f"Save frequency:  {save_freq:,} steps")
    print(f"Opponent mix:    40% random, 30% greedy, 30% fixed")
    print("=" * 70)

    # Create directories
    os.makedirs("models/construction", exist_ok=True)
    os.makedirs("logs/construction", exist_ok=True)
    os.makedirs("eval/construction", exist_ok=True)

    # Create vectorized training environment
    print("\nCreating training environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Create evaluation environment (single, deterministic)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(rank=1000, seed=42)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Adjust for parallel envs
        save_path="models/construction",
        name_prefix="ppo_construction",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/construction/best",
        log_path="eval/construction",
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
        tensorboard_log="logs/construction",
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
    print("Monitor progress with: tensorboard --logdir logs/construction/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = "models/construction/ppo_construction_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: models/construction/best/best_model.zip")
    print(f"\nEvaluate with:")
    print(f"  python examples/evaluate_construction.py {final_path}")
    print("=" * 70)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train board construction agent")
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

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )
