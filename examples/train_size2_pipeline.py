"""
Size 2 training pipeline: Stage 2 (construction) then Stage 3 (simultaneous play).

Runs both stages sequentially in one script. Stage 3 starts from scratch
(different observation space, no weight transfer).

Usage:
    python examples/train_size2_pipeline.py
    python examples/train_size2_pipeline.py --stage2-steps 100000 --stage3-steps 300000
    python examples/train_size2_pipeline.py --skip-stage2  # only run Stage 3

Monitor:
    tensorboard --logdir logs/
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add examples/ dir to path so sibling scripts can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_stage2(timesteps: int = 100_000, n_envs: int = 4):
    """Train Stage 2: reverse curriculum board construction."""
    print("\n" + "=" * 70)
    print("PIPELINE STAGE 2: Reverse Curriculum Construction")
    print("=" * 70)

    from train_reverse_curriculum import train
    train(
        board_size=2,
        total_timesteps=timesteps,
        n_envs=n_envs,
        eval_freq=2000,
        save_freq=10_000,
        board_library_path="new_boards_2.json",
        output_dir="models/size2/stage2",
    )

    model_path = "models/size2/stage2/ppo_stage2_final.zip"
    if Path(model_path).exists():
        print(f"\nStage 2 complete: {model_path}")
        return model_path

    best_path = "models/size2/stage2/best/best_model.zip"
    if Path(best_path).exists():
        print(f"\nStage 2 complete (best model): {best_path}")
        return best_path

    print("\nWARNING: No Stage 2 model found after training!")
    return None


def run_stage3(timesteps: int = 300_000, n_envs: int = 4):
    """Train Stage 3: simultaneous 5-round play."""
    print("\n" + "=" * 70)
    print("PIPELINE STAGE 3: Simultaneous 5-Round Play")
    print("=" * 70)

    from train_simultaneous import train
    train(
        board_size=2,
        total_timesteps=timesteps,
        n_envs=n_envs,
        eval_freq=2000,
        save_freq=10_000,
        output_dir="models/size2/stage3",
    )

    model_path = "models/size2/stage3/ppo_stage3_final.zip"
    if Path(model_path).exists():
        print(f"\nStage 3 complete: {model_path}")
        return model_path

    best_path = "models/size2/stage3/best/best_model.zip"
    if Path(best_path).exists():
        print(f"\nStage 3 complete (best model): {best_path}")
        return best_path

    print("\nWARNING: No Stage 3 model found after training!")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Size 2 training pipeline (Stage 2 + Stage 3)",
    )
    parser.add_argument(
        "--stage2-steps", type=int, default=100_000,
        help="Stage 2 timesteps (default: 100,000)",
    )
    parser.add_argument(
        "--stage3-steps", type=int, default=300_000,
        help="Stage 3 timesteps (default: 300,000)",
    )
    parser.add_argument(
        "--envs", type=int, default=4,
        help="Parallel environments (default: 4)",
    )
    parser.add_argument(
        "--skip-stage2", action="store_true",
        help="Skip Stage 2, only run Stage 3",
    )
    parser.add_argument(
        "--skip-stage3", action="store_true",
        help="Skip Stage 3, only run Stage 2",
    )
    args = parser.parse_args()

    start = time.time()

    print("=" * 70)
    print("SIZE 2 TRAINING PIPELINE")
    print("=" * 70)
    print(f"Stage 2: {'SKIP' if args.skip_stage2 else f'{args.stage2_steps:,} steps'}")
    print(f"Stage 3: {'SKIP' if args.skip_stage3 else f'{args.stage3_steps:,} steps'}")
    print(f"Parallel envs: {args.envs}")
    print(f"Trap limit: board_size - 1 = 1")
    print("=" * 70)

    # Stage 2
    if not args.skip_stage2:
        s2_start = time.time()
        s2_model = run_stage2(timesteps=args.stage2_steps, n_envs=args.envs)
        s2_time = time.time() - s2_start
        print(f"Stage 2 time: {s2_time / 60:.1f} min")
        if s2_model is None:
            print("Stage 2 failed — aborting pipeline.")
            sys.exit(1)

    # Stage 3
    if not args.skip_stage3:
        s3_start = time.time()
        s3_model = run_stage3(timesteps=args.stage3_steps, n_envs=args.envs)
        s3_time = time.time() - s3_start
        print(f"Stage 3 time: {s3_time / 60:.1f} min")

    total = time.time() - start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {total / 60:.1f} min")
    if not args.skip_stage2:
        print(f"Stage 2 model: models/size2/stage2/ppo_stage2_final.zip")
    if not args.skip_stage3:
        print(f"Stage 3 model: models/size2/stage3/ppo_stage3_final.zip")
    print(f"\nVerify with:")
    print(f"  python examples/play_against_agent.py --size 2 --rounds 5")
    print("=" * 70)


if __name__ == "__main__":
    main()
