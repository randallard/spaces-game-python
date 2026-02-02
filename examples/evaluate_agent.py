"""
Evaluate a trained RL agent against different baselines.

Usage:
    python examples/evaluate_agent.py models/ppo_spacegame_final.zip --episodes 1000
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from spaces_game import SpacesGameEnv


def evaluate_agent(
    model_path: str,
    board_pool: str,
    opponent: str,
    n_episodes: int,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Evaluate agent performance.

    Args:
        model_path: Path to trained model
        board_pool: Path to board pool
        opponent: Opponent strategy
        n_episodes: Number of episodes to evaluate
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = SpacesGameEnv(
        board_pool_path=board_pool,
        deck_size=10,
        opponent_strategy=opponent,
    )

    # Track statistics
    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0
    score_diffs = []

    # Run episodes
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        # Record results
        agent_score = info["agent_total_score"]
        opponent_score = info["opponent_total_score"]
        score_diff = agent_score - opponent_score

        total_agent_score += agent_score
        total_opponent_score += opponent_score
        score_diffs.append(score_diff)

        if agent_score > opponent_score:
            wins += 1
        elif agent_score < opponent_score:
            losses += 1
        else:
            ties += 1

        if verbose and (ep + 1) % 100 == 0:
            win_rate = wins / (ep + 1) * 100
            print(f"  Episode {ep + 1}/{n_episodes}: Win rate = {win_rate:.1f}%")

    env.close()

    # Calculate statistics
    win_rate = wins / n_episodes * 100
    loss_rate = losses / n_episodes * 100
    tie_rate = ties / n_episodes * 100
    avg_score_diff = np.mean(score_diffs)
    std_score_diff = np.std(score_diffs)

    return {
        "n_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "tie_rate": tie_rate,
        "avg_agent_score": total_agent_score / n_episodes,
        "avg_opponent_score": total_opponent_score / n_episodes,
        "avg_score_diff": avg_score_diff,
        "std_score_diff": std_score_diff,
    }


def evaluate_random_baseline(
    board_pool: str,
    opponent: str,
    n_episodes: int,
    seed: int = 0,
) -> dict:
    """
    Evaluate random baseline (agent picks randomly).

    Args:
        board_pool: Path to board pool
        opponent: Opponent strategy
        n_episodes: Number of episodes
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    env = SpacesGameEnv(
        board_pool_path=board_pool,
        deck_size=10,
        opponent_strategy=opponent,
    )

    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False

        while not done:
            action = env.action_space.sample()  # Random selection
            obs, reward, done, truncated, info = env.step(action)

        agent_score = info["agent_total_score"]
        opponent_score = info["opponent_total_score"]

        total_agent_score += agent_score
        total_opponent_score += opponent_score

        if agent_score > opponent_score:
            wins += 1
        elif agent_score < opponent_score:
            losses += 1
        else:
            ties += 1

    env.close()

    win_rate = wins / n_episodes * 100

    return {
        "n_episodes": n_episodes,
        "wins": wins,
        "win_rate": win_rate,
        "avg_agent_score": total_agent_score / n_episodes,
        "avg_opponent_score": total_opponent_score / n_episodes,
    }


def main():
    """Main evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--board-pool",
        type=str,
        default="data/boards_size_2.json",
        help="Path to board pool (default: data/boards_size_2.json)",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "greedy"],
        help="Opponent strategy (default: random)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also evaluate random baseline for comparison",
    )

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        return

    print("=" * 70)
    print("Spaces Game RL Agent Evaluation")
    print("=" * 70)
    print(f"Model:       {args.model_path}")
    print(f"Board Pool:  {args.board_pool}")
    print(f"Opponent:    {args.opponent}")
    print(f"Episodes:    {args.episodes:,}")
    print(f"Seed:        {args.seed}")
    print("=" * 70)
    print()

    # Evaluate trained agent
    print("Evaluating trained agent...")
    results = evaluate_agent(
        args.model_path,
        args.board_pool,
        args.opponent,
        args.episodes,
        args.seed,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("TRAINED AGENT RESULTS")
    print("=" * 70)
    print(f"Episodes:           {results['n_episodes']:,}")
    print(f"Wins:               {results['wins']:,} ({results['win_rate']:.1f}%)")
    print(f"Losses:             {results['losses']:,} ({results['loss_rate']:.1f}%)")
    print(f"Ties:               {results['ties']:,} ({results['tie_rate']:.1f}%)")
    print()
    print(f"Avg Agent Score:    {results['avg_agent_score']:.2f}")
    print(f"Avg Opponent Score: {results['avg_opponent_score']:.2f}")
    print(f"Avg Score Diff:     {results['avg_score_diff']:.2f} ± {results['std_score_diff']:.2f}")
    print("=" * 70)

    # Evaluate baseline if requested
    if args.baseline:
        print()
        print("Evaluating random baseline...")
        baseline = evaluate_random_baseline(
            args.board_pool,
            args.opponent,
            args.episodes,
            args.seed + 10000,
        )

        print()
        print("=" * 70)
        print("RANDOM BASELINE RESULTS")
        print("=" * 70)
        print(f"Episodes:           {baseline['n_episodes']:,}")
        print(f"Wins:               {baseline['wins']:,} ({baseline['win_rate']:.1f}%)")
        print(f"Avg Agent Score:    {baseline['avg_agent_score']:.2f}")
        print(f"Avg Opponent Score: {baseline['avg_opponent_score']:.2f}")
        print("=" * 70)

        print()
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        improvement = results['win_rate'] - baseline['win_rate']
        print(f"Win Rate Improvement: {improvement:+.1f} percentage points")
        print(f"Trained: {results['win_rate']:.1f}% vs Random: {baseline['win_rate']:.1f}%")
        print("=" * 70)


if __name__ == "__main__":
    main()
