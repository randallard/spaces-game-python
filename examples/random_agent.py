"""
Example: Random Agent playing Spaces Game.

This script demonstrates how to use the SpacesGameEnv gymnasium environment
with a simple random agent that selects boards uniformly at random.
"""

import argparse
from typing import Dict, Any

import numpy as np

from spaces_game import SpacesGameEnv


def run_random_agent(
    num_episodes: int = 10,
    board_pool_path: str = "data/boards_size_3.json",
    deck_size: int = 10,
    opponent_strategy: str = "random",
    render: bool = False,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Run a random agent for multiple episodes.

    Args:
        num_episodes: Number of episodes to run
        board_pool_path: Path to board pool JSON file
        deck_size: Number of boards in each deck
        opponent_strategy: Opponent strategy ("random", "greedy")
        render: Whether to render each episode
        seed: Random seed for reproducibility

    Returns:
        Statistics dict with win/loss/tie counts and average scores
    """
    # Create environment
    render_mode = "human" if render else None
    env = SpacesGameEnv(
        board_pool_path=board_pool_path,
        deck_size=deck_size,
        opponent_strategy=opponent_strategy,
        render_mode=render_mode,
    )

    # Statistics tracking
    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0
    episode_rewards = []

    # Run episodes
    for episode in range(num_episodes):
        # Reset environment
        episode_seed = seed + episode if seed is not None else None
        obs, info = env.reset(seed=episode_seed)

        if render:
            print(f"\n{'=' * 60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'=' * 60}")
            env.render()

        # Track episode reward
        episode_reward = 0.0
        terminated = False
        available_boards = list(range(deck_size))

        # Play episode (5 rounds)
        while not terminated:
            # Random agent: select random board from unused boards
            if len(available_boards) == 0:
                # Shouldn't happen in 5-round game with 10 boards, but handle it
                available_boards = list(range(deck_size))

            action = np.random.choice(available_boards)
            available_boards.remove(action)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

        # Update statistics
        episode_rewards.append(episode_reward)
        total_agent_score += info["agent_total_score"]
        total_opponent_score += info["opponent_total_score"]

        if info["agent_total_score"] > info["opponent_total_score"]:
            wins += 1
        elif info["agent_total_score"] < info["opponent_total_score"]:
            losses += 1
        else:
            ties += 1

        if not render:
            # Print episode summary
            result = "WIN" if wins == episode + 1 - (losses + ties) else ("LOSS" if losses == episode + 1 - (wins + ties) else "TIE")
            print(
                f"Episode {episode + 1:3d}/{num_episodes}: {result:4s} | "
                f"Score: {info['agent_total_score']:3d} - {info['opponent_total_score']:3d} | "
                f"Reward: {episode_reward:+7.1f}"
            )

    env.close()

    # Calculate statistics
    stats = {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / num_episodes,
        "avg_agent_score": total_agent_score / num_episodes,
        "avg_opponent_score": total_opponent_score / num_episodes,
        "avg_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
    }

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run random agent on Spaces Game environment"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render each episode to console",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set numpy seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Run random agent
    print(f"\n{'=' * 60}")
    print(f"Random Agent vs {args.opponent.capitalize()} Opponent")
    print(f"Board Pool: {args.board_pool}")
    print(f"Deck Size: {args.deck_size}")
    print(f"Episodes: {args.episodes}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    stats = run_random_agent(
        num_episodes=args.episodes,
        board_pool_path=args.board_pool,
        deck_size=args.deck_size,
        opponent_strategy=args.opponent,
        render=args.render,
        seed=args.seed,
    )

    # Print final statistics
    print(f"\n{'=' * 60}")
    print("FINAL STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total Episodes: {stats['num_episodes']}")
    print(f"Wins:           {stats['wins']} ({stats['win_rate']:.1%})")
    print(f"Losses:         {stats['losses']} ({stats['losses']/stats['num_episodes']:.1%})")
    print(f"Ties:           {stats['ties']} ({stats['ties']/stats['num_episodes']:.1%})")
    print(f"")
    print(f"Avg Agent Score:    {stats['avg_agent_score']:.1f}")
    print(f"Avg Opponent Score: {stats['avg_opponent_score']:.1f}")
    print(f"Avg Score Diff:     {stats['avg_agent_score'] - stats['avg_opponent_score']:+.1f}")
    print(f"")
    print(f"Avg Episode Reward: {stats['avg_episode_reward']:+.1f} ± {stats['std_episode_reward']:.1f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
