"""
Check Gymnasium environment compatibility.

This script uses Gymnasium's built-in environment checker to verify
that SpacesGameEnv follows all Gymnasium API conventions.
"""

import argparse

from gymnasium.utils.env_checker import check_env

from spaces_game import SpacesGameEnv


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check SpacesGameEnv Gymnasium compatibility"
    )
    parser.add_argument(
        "--board-pool",
        type=str,
        default="data/boards_size_3.json",
        help="Path to board pool JSON file (default: data/boards_size_3.json)",
    )
    parser.add_argument(
        "--deck-size",
        type=int,
        default=10,
        help="Number of boards in each deck (default: 10)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "greedy"],
        help="Opponent strategy (default: random)",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("Gymnasium Environment Compatibility Check")
    print(f"{'=' * 60}")
    print(f"Board Pool: {args.board_pool}")
    print(f"Deck Size: {args.deck_size}")
    print(f"Opponent: {args.opponent}")
    print(f"{'=' * 60}\n")

    # Create environment
    env = SpacesGameEnv(
        board_pool_path=args.board_pool,
        deck_size=args.deck_size,
        opponent_strategy=args.opponent,
    )

    # Run Gymnasium's environment checker
    print("Running Gymnasium environment checker...\n")
    try:
        check_env(env.unwrapped, skip_render_check=False)
        print(f"\n{'=' * 60}")
        print("✓ All checks passed!")
        print(f"{'=' * 60}\n")
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"✗ Check failed: {e}")
        print(f"{'=' * 60}\n")
        raise

    env.close()


if __name__ == "__main__":
    main()
