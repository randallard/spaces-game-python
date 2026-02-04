"""
Evaluate a trained board builder agent.

Tests the agent's ability to build valid boards that counter opponent strategies.
Reports win rate, construction quality, and validity metrics.

Usage:
    python examples/evaluate_board_builder.py models/builder/ppo_builder_final.zip
    python examples/evaluate_board_builder.py models/builder/best/best_model.zip
"""

import sys
import numpy as np
from stable_baselines3 import PPO

from spaces_game import BoardBuilderEnv


def evaluate_agent(
    model_path: str,
    opponent_strategy: str = "random",
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True,
    board_size: int = 2,
):
    """
    Evaluate trained builder agent against specified opponent.

    Args:
        model_path: Path to trained model .zip file
        opponent_strategy: Opponent strategy (\"random\", \"greedy\", or \"fixed_N\")
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        verbose: Print detailed results
        board_size: Size of boards to build

    Returns:
        dict with evaluation metrics
    """
    # Load trained model
    model = PPO.load(model_path)

    # Create evaluation environment
    env = BoardBuilderEnv(
        board_size=board_size,
        opponent_library_path="new_boards_2.json",
        opponent_strategy=opponent_strategy,
        show_opponent_board=True,
        max_construction_steps=20,
    )

    # Evaluation metrics
    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0
    score_diffs = []

    # Construction quality metrics
    invalid_boards = 0
    construction_steps_list = []
    piece_counts = []
    trap_counts = []

    if verbose:
        print(f"\\nEvaluating against {opponent_strategy.upper()} opponent...")
        print(f"Episodes: {n_episodes}")
        print("-" * 60)

    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False

        # Track construction for this episode
        episode_construction_steps = []
        episode_invalid = False

        while not done:
            # Agent selects action using trained policy
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            # Track construction metrics
            if info.get('invalid_board', False):
                episode_invalid = True

            construction_steps_list.append(info.get('construction_step', 0))
            piece_counts.append(info.get('piece_count', 0))
            trap_counts.append(info.get('trap_count', 0))

        # Record results
        agent_score = info['agent_total_score']
        opponent_score = info['opponent_total_score']
        score_diff = agent_score - opponent_score

        total_agent_score += agent_score
        total_opponent_score += opponent_score
        score_diffs.append(score_diff)

        if episode_invalid:
            invalid_boards += 1

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
    invalid_rate = invalid_boards / n_episodes
    avg_pieces = np.mean(piece_counts) if piece_counts else 0
    avg_traps = np.mean(trap_counts) if trap_counts else 0

    if verbose:
        if n_episodes > 10:
            print(f"... ({n_episodes - 10} more episodes)")
        print("-" * 60)
        print(f"\\nRESULTS vs {opponent_strategy.upper()}:")
        print(f"  Wins:   {wins:3d} ({win_rate*100:5.1f}%)")
        print(f"  Losses: {losses:3d} ({loss_rate*100:5.1f}%)")
        print(f"  Ties:   {ties:3d} ({tie_rate*100:5.1f}%)")
        print(f"\\nSCORES:")
        print(f"  Avg Agent Score:    {avg_agent_score:6.1f}")
        print(f"  Avg Opponent Score: {avg_opponent_score:6.1f}")
        print(f"  Avg Differential:   {avg_score_diff:+6.1f} ± {std_score_diff:.1f}")
        print(f"\\nCONSTRUCTION QUALITY:")
        print(f"  Invalid Boards:     {invalid_boards} ({invalid_rate*100:.1f}%)")
        print(f"  Avg Pieces:         {avg_pieces:.1f}")
        print(f"  Avg Traps:          {avg_traps:.1f}")

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
        'invalid_boards': invalid_boards,
        'invalid_rate': invalid_rate,
        'avg_pieces': avg_pieces,
        'avg_traps': avg_traps,
    }


def evaluate_curriculum(model_path: str, n_episodes_per_board: int = 10, board_size: int = 2):
    """
    Evaluate agent against all boards in curriculum (fixed_0 through fixed_7).

    Args:
        model_path: Path to trained model
        n_episodes_per_board: Episodes per opponent board
        board_size: Size of boards to build

    Returns:
        dict with detailed per-board results
    """
    print("=" * 70)
    print("CURRICULUM EVALUATION (All 8 Opponent Boards)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes per board: {n_episodes_per_board}")
    print(f"Board size: {board_size}x{board_size}")

    board_results = {}

    for board_idx in range(8):
        strategy = f"fixed_{board_idx}"
        print(f"\\n{'='*70}")
        print(f"Evaluating vs Opponent Board {board_idx}")
        print(f"{'='*70}")

        result = evaluate_agent(
            model_path,
            opponent_strategy=strategy,
            n_episodes=n_episodes_per_board,
            seed=42,
            verbose=True,
            board_size=board_size,
        )
        board_results[board_idx] = result

    # Summary table
    print("\\n" + "=" * 70)
    print("PER-BOARD SUMMARY")
    print("=" * 70)
    print(f"{'Opp Board':<10} {'Win Rate':<12} {'Invalid%':<10} {'Avg Diff':<15} {'Agent/Opp Score'}")
    print("-" * 70)

    for board_idx in range(8):
        r = board_results[board_idx]
        print(f"Board {board_idx:<4} "
              f"{r['win_rate']*100:5.1f}%       "
              f"{r['invalid_rate']*100:4.1f}%     "
              f"{r['avg_score_diff']:+6.1f} ± {r['std_score_diff']:4.1f}    "
              f"{r['avg_agent_score']:5.1f} / {r['avg_opponent_score']:5.1f}")

    # Overall stats
    overall_win_rate = np.mean([r['win_rate'] for r in board_results.values()])
    overall_invalid_rate = np.mean([r['invalid_rate'] for r in board_results.values()])
    overall_avg_diff = np.mean([r['avg_score_diff'] for r in board_results.values()])

    print("=" * 70)
    print(f"OVERALL (all 8 boards):")
    print(f"  Win rate:      {overall_win_rate*100:.1f}%")
    print(f"  Invalid rate:  {overall_invalid_rate*100:.1f}%")
    print(f"  Avg diff:      {overall_avg_diff:+.1f}")

    # Assessment
    if overall_invalid_rate > 0.5:
        print("✗ POOR. Agent building mostly invalid boards.")
    elif overall_invalid_rate > 0.2:
        print("⚠ FAIR. Agent needs to improve board validity.")
    elif overall_win_rate >= 0.95:
        print("✓ EXCELLENT! Agent mastered board building.")
    elif overall_win_rate >= 0.80:
        print("✓ GOOD! Agent builds effective counter-boards.")
    elif overall_win_rate >= 0.60:
        print("⚠ FAIR. Agent shows learning but needs refinement.")
    else:
        print("✗ POOR. Agent hasn't learned effective construction.")

    print("=" * 70)
    return board_results


def evaluate_all_strategies(model_path: str, n_episodes: int = 100, board_size: int = 2):
    """
    Evaluate agent against all opponent strategies.

    Args:
        model_path: Path to trained model
        n_episodes: Episodes per strategy
        board_size: Size of boards to build
    """
    print("=" * 70)
    print("BOARD BUILDER AGENT EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes per strategy: {n_episodes}")
    print(f"Board size: {board_size}x{board_size}")

    strategies = ["random", "greedy", "fixed"]
    results = {}

    for strategy in strategies:
        results[strategy] = evaluate_agent(
            model_path,
            opponent_strategy=strategy,
            n_episodes=n_episodes,
            verbose=True,
            board_size=board_size,
        )

    # Summary
    print("\\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':<12} {'Win Rate':<12} {'Invalid%':<10} {'Avg Diff':<15} {'Agent/Opp Score'}")
    print("-" * 70)

    for strategy in strategies:
        r = results[strategy]
        print(f"{strategy.upper():<12} "
              f"{r['win_rate']*100:5.1f}%       "
              f"{r['invalid_rate']*100:4.1f}%     "
              f"{r['avg_score_diff']:+6.1f} ± {r['std_score_diff']:4.1f}    "
              f"{r['avg_agent_score']:5.1f} / {r['avg_opponent_score']:5.1f}")

    print("=" * 70)

    # Overall assessment
    avg_win_rate = np.mean([r['win_rate'] for r in results.values()])
    avg_invalid_rate = np.mean([r['invalid_rate'] for r in results.values()])
    print(f"\\nOVERALL:")
    print(f"  Win Rate:     {avg_win_rate*100:.1f}%")
    print(f"  Invalid Rate: {avg_invalid_rate*100:.1f}%")

    if avg_invalid_rate > 0.5:
        print("✗ POOR. Agent building mostly invalid boards.")
    elif avg_invalid_rate > 0.2:
        print("⚠ FAIR. Agent needs to improve board validity.")
    elif avg_win_rate >= 0.95:
        print("✓ EXCELLENT! Agent mastered board building.")
    elif avg_win_rate >= 0.80:
        print("✓ GOOD! Agent builds effective boards.")
    elif avg_win_rate >= 0.60:
        print("⚠ FAIR. Agent shows learning but needs more training.")
    else:
        print("✗ POOR. Agent needs significantly more training.")

    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate board builder agent")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["random", "greedy", "fixed", "all", "curriculum"],
        default="curriculum",
        help="Opponent strategy to evaluate against (default: curriculum)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100, or per-board if curriculum)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=2,
        help="Size of boards to build (default: 2 for 2x2)",
    )

    args = parser.parse_args()

    if args.strategy == "curriculum":
        # Evaluate against all 8 boards (episodes = per board)
        evaluate_curriculum(args.model_path, n_episodes_per_board=args.episodes, board_size=args.board_size)
    elif args.strategy == "all":
        evaluate_all_strategies(args.model_path, n_episodes=args.episodes, board_size=args.board_size)
    else:
        evaluate_agent(
            args.model_path,
            opponent_strategy=args.strategy,
            n_episodes=args.episodes,
            seed=args.seed,
            verbose=True,
            board_size=args.board_size,
        )
