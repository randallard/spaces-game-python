"""
Evaluate a trained board construction agent.

Tests the agent against different opponent strategies and reports
win rate, average score differential, and decision quality.

Usage:
    python examples/evaluate_construction.py models/construction/ppo_construction_final.zip
    python examples/evaluate_construction.py models/construction/best/best_model.zip
"""

import sys
import numpy as np
from stable_baselines3 import PPO

from spaces_game import BoardConstructionEnv


def evaluate_agent(
    model_path: str,
    opponent_strategy: str = "random",
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Evaluate trained agent against specified opponent.

    Args:
        model_path: Path to trained model .zip file
        opponent_strategy: Opponent strategy ("random", "greedy", or "fixed")
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        verbose: Print detailed results

    Returns:
        dict with evaluation metrics
    """
    # Load trained model
    model = PPO.load(model_path)

    # Create evaluation environment
    env = BoardConstructionEnv(
        board_library_path="new_boards_2.json",
        opponent_strategy=opponent_strategy,
        show_opponent_board=True,
    )

    # Evaluation metrics
    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0
    score_diffs = []

    if verbose:
        print(f"\nEvaluating against {opponent_strategy.upper()} opponent...")
        print(f"Episodes: {n_episodes}")
        print("-" * 60)

    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False

        while not done:
            # Agent selects action using trained policy
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        # Record results
        agent_score = info['agent_total_score']
        opponent_score = info['opponent_total_score']
        score_diff = agent_score - opponent_score

        total_agent_score += agent_score
        total_opponent_score += opponent_score
        score_diffs.append(score_diff)

        if agent_score > opponent_score:
            wins += 1
            result = "WIN"
        elif agent_score < opponent_score:
            losses += 1
            result = "LOSS"
        else:
            ties += 1
            result = "TIE"

        if verbose and episode < 10:  # Show first 10 episodes
            print(f"Episode {episode+1:3d}: {agent_score:3d}-{opponent_score:3d} "
                  f"({result:4s}) Reward: {episode_reward:+7.1f}")

    env.close()

    # Calculate metrics
    win_rate = wins / n_episodes
    loss_rate = losses / n_episodes
    tie_rate = ties / n_episodes
    avg_agent_score = total_agent_score / n_episodes
    avg_opponent_score = total_opponent_score / n_episodes
    avg_score_diff = np.mean(score_diffs)
    std_score_diff = np.std(score_diffs)

    if verbose:
        if n_episodes > 10:
            print(f"... ({n_episodes - 10} more episodes)")
        print("-" * 60)
        print(f"\nRESULTS vs {opponent_strategy.upper()}:")
        print(f"  Wins:   {wins:3d} ({win_rate*100:5.1f}%)")
        print(f"  Losses: {losses:3d} ({loss_rate*100:5.1f}%)")
        print(f"  Ties:   {ties:3d} ({tie_rate*100:5.1f}%)")
        print(f"\nSCORES:")
        print(f"  Avg Agent Score:    {avg_agent_score:6.1f}")
        print(f"  Avg Opponent Score: {avg_opponent_score:6.1f}")
        print(f"  Avg Differential:   {avg_score_diff:+6.1f} ± {std_score_diff:.1f}")

    return {
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'tie_rate': tie_rate,
        'avg_agent_score': avg_agent_score,
        'avg_opponent_score': avg_opponent_score,
        'avg_score_diff': avg_score_diff,
        'std_score_diff': std_score_diff,
        'opponent_strategy': opponent_strategy,
        'n_episodes': n_episodes,
    }


def evaluate_all_strategies(model_path: str, n_episodes: int = 100):
    """
    Evaluate agent against all opponent strategies.

    Args:
        model_path: Path to trained model
        n_episodes: Episodes per strategy
    """
    print("=" * 70)
    print("BOARD CONSTRUCTION AGENT EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes per strategy: {n_episodes}")

    strategies = ["random", "greedy", "fixed"]
    results = {}

    for strategy in strategies:
        results[strategy] = evaluate_agent(
            model_path,
            opponent_strategy=strategy,
            n_episodes=n_episodes,
            verbose=True,
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<12} {'Win Rate':<12} {'Avg Diff':<15} {'Agent/Opp Score'}")
    print("-" * 70)

    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy.upper():<12} "
              f"{r['win_rate']*100:5.1f}%       "
              f"{r['avg_score_diff']:+6.1f} ± {r['std_score_diff']:4.1f}    "
              f"{r['avg_agent_score']:5.1f} / {r['avg_opponent_score']:5.1f}")

    print("=" * 70)

    # Overall assessment
    avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
    print(f"\nOVERALL WIN RATE: {avg_win_rate*100:.1f}%")

    if avg_win_rate >= 0.95:
        print("✓ EXCELLENT! Agent has mastered board construction.")
    elif avg_win_rate >= 0.80:
        print("✓ GOOD! Agent performs well but has room for improvement.")
    elif avg_win_rate >= 0.60:
        print("⚠ FAIR. Agent shows learning but needs more training.")
    else:
        print("✗ POOR. Agent needs significantly more training.")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate board construction agent")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["random", "greedy", "fixed", "all"],
        default="all",
        help="Opponent strategy to evaluate against (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.strategy == "all":
        evaluate_all_strategies(args.model_path, n_episodes=args.episodes)
    else:
        evaluate_agent(
            args.model_path,
            opponent_strategy=args.strategy,
            n_episodes=args.episodes,
            seed=args.seed,
            verbose=True,
        )
