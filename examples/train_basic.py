"""
Basic RL training script for Spaces Game using Stable Baselines3.

This script starts simple with PPO on size-3 boards, designed for the
8GB RAM constraint on the training machine.

Usage:
    python examples/train_basic.py --timesteps 100000 --save-freq 10000
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from spaces_game import SpacesGameEnv


def make_env(board_pool_path: str, opponent: str = "random", seed: int = 0, perfect_info: bool = False):
    """
    Create a single environment instance.

    Args:
        board_pool_path: Path to board pool JSON
        opponent: Opponent strategy ("random" or "greedy")
        seed: Random seed
        perfect_info: Enable perfect information (agent sees opponent deck)

    Returns:
        Callable that creates the environment
    """
    def _init():
        env = SpacesGameEnv(
            board_pool_path=board_pool_path,
            deck_size=10,
            opponent_strategy=opponent,
            perfect_information=perfect_info,
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(
        description="Train RL agent to play Spaces Game"
    )
    parser.add_argument(
        "--board-pool",
        type=str,
        default="data/boards_size_2.json",
        help="Path to board pool (default: data/boards_size_2.json - start simple!)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "greedy"],
        help="Opponent strategy (default: random)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps to train (default: 100000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4 for 8GB RAM)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models/)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save model every N timesteps (default: 10000)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluate every N timesteps (default: 5000)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to model to continue training from",
    )
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="Use SubprocVecEnv instead of DummyVecEnv (more parallel but more RAM)",
    )
    parser.add_argument(
        "--perfect-info",
        action="store_true",
        help="Enable perfect information (agent can see opponent's full deck)",
    )

    args = parser.parse_args()

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Spaces Game RL Training - Basic PPO")
    print("=" * 70)
    print(f"Board Pool:       {args.board_pool}")
    print(f"Opponent:         {args.opponent}")
    print(f"Perfect Info:     {'YES - Agent sees opponent deck' if args.perfect_info else 'NO - Partial observability'}")
    print(f"Total Timesteps:  {args.timesteps:,}")
    print(f"Parallel Envs:    {args.n_envs}")
    print(f"Save Directory:   {args.save_dir}")
    print(f"Save Frequency:   {args.save_freq:,} steps")
    print(f"Eval Frequency:   {args.eval_freq:,} steps")
    print(f"Eval Episodes:    {args.eval_episodes}")
    print(f"Seed:             {args.seed}")
    if args.load_model:
        print(f"Loading Model:    {args.load_model}")
    print("=" * 70)
    print()

    # Create vectorized environment
    print(f"Creating {args.n_envs} parallel environments...")

    if args.use_subprocess:
        print("  Using SubprocVecEnv (more parallel, more RAM)")
        env = SubprocVecEnv([
            make_env(args.board_pool, args.opponent, args.seed + i, args.perfect_info)
            for i in range(args.n_envs)
        ])
    else:
        print("  Using DummyVecEnv (less parallel, less RAM)")
        env = DummyVecEnv([
            make_env(args.board_pool, args.opponent, args.seed + i, args.perfect_info)
            for i in range(args.n_envs)
        ])

    # Create evaluation environment (single env)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(args.board_pool, args.opponent, args.seed + 1000, args.perfect_info)
    ])

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,  # Divide by n_envs for per-env steps
        save_path=str(save_dir),
        name_prefix="ppo_spacegame",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(log_dir),
        eval_freq=args.eval_freq // args.n_envs,  # Divide by n_envs
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # Create or load model
    if args.load_model:
        print(f"Loading model from {args.load_model}...")
        model = PPO.load(args.load_model, env=env)
        print("Model loaded. Continuing training...")
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            seed=args.seed,
            # Conservative hyperparameters for 8GB RAM
            n_steps=2048,          # Default, works well
            batch_size=64,         # Smaller batch for less RAM
            n_epochs=10,           # Default
            learning_rate=3e-4,    # Default
            ent_coef=0.01,         # Small entropy for exploration
            tensorboard_log=str(log_dir),
        )
        print("Model created.")

    print()
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print(f"Monitor progress: tensorboard --logdir {log_dir}")
    print()

    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        print()
        print("=" * 70)
        print("Training complete!")
        print("=" * 70)

        # Save final model
        final_path = save_dir / "ppo_spacegame_final.zip"
        model.save(str(final_path))
        print(f"Final model saved to: {final_path}")

    except KeyboardInterrupt:
        print()
        print("=" * 70)
        print("Training interrupted by user")
        print("=" * 70)

        # Save interrupted model
        interrupted_path = save_dir / "ppo_spacegame_interrupted.zip"
        model.save(str(interrupted_path))
        print(f"Model saved to: {interrupted_path}")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
