"""
Test if agent can select the optimal board when given perfect information.

This is a controlled test to validate that the agent understands board matchups
before moving to board construction.

Usage:
    python examples/test_board_selection.py models/ppo_spacegame_final.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from spaces_game import SpacesGameEnv
from spaces_game.simulation import simulate_round


def analyze_matchup(agent_board, opponent_board, round_num=1):
    """Simulate a single matchup and return detailed results."""
    result = simulate_round(round_num, agent_board, opponent_board, silent=True)

    return {
        "agent_points": result.playerPoints,
        "opponent_points": result.opponentPoints,
        "differential": result.playerPoints - result.opponentPoints,
        "winner": result.winner,
        "agent_trapped": result.simulationDetails.playerHitTrap,
        "opponent_trapped": result.simulationDetails.opponentHitTrap,
    }


def find_optimal_board(agent_deck, opponent_board, round_num=1):
    """
    Find the objectively best board from agent's deck against opponent's board.

    Returns:
        best_idx: Index of best board
        best_diff: Score differential of best board
        all_results: List of all matchup results
    """
    results = []

    for idx, agent_board in enumerate(agent_deck):
        matchup = analyze_matchup(agent_board, opponent_board, round_num)
        matchup["board_idx"] = idx
        results.append(matchup)

    # Sort by score differential (highest first)
    results.sort(key=lambda x: x["differential"], reverse=True)

    best = results[0]
    return best["board_idx"], best["differential"], results


def test_agent_selection(model_path, board_pool_path, perfect_info=True, verbose=True):
    """
    Test agent's board selection against all possible opponent boards.

    For each opponent board:
    - Show agent the opponent's board (perfect info)
    - Agent selects best counter from its deck
    - Compare to optimal selection
    """

    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create environment
    env = SpacesGameEnv(
        board_pool_path=board_pool_path,
        deck_size=8,  # Using all 8 boards from new_boards_2.json
        opponent_strategy="random",  # Doesn't matter, we'll override
        perfect_information=perfect_info,
    )

    # Reset to get initial decks
    obs, info = env.reset(seed=42)

    agent_deck = env.agent_deck
    opponent_deck = env.opponent_deck

    if verbose:
        print(f"\nAgent has {len(agent_deck)} boards")
        print(f"Opponent has {len(opponent_deck)} boards")
        print("\n" + "=" * 80)
        print("TESTING AGENT'S BOARD SELECTION")
        print("=" * 80)

    total_tests = 0
    correct_selections = 0
    total_score_diff = 0
    total_optimal_diff = 0

    results_log = []

    # Test against each opponent board
    for opp_idx, opponent_board in enumerate(opponent_deck):
        if verbose:
            print(f"\n--- Test {opp_idx + 1}/{len(opponent_deck)} ---")
            print(f"Opponent plays Board {opp_idx}")

        # Find optimal selection
        optimal_idx, optimal_diff, all_matchups = find_optimal_board(
            agent_deck, opponent_board, round_num=1
        )

        # Create observation for agent with this specific opponent board
        # Simulate round 1 where opponent has already "selected" their board
        env.opponent_deck = [opponent_board] * len(opponent_deck)  # All same for consistency
        env.current_round = 1
        env.agent_history = []
        env.opponent_history = []
        env.agent_total_score = 0
        env.opponent_total_score = 0

        # Get observation (agent sees opponent's full deck in perfect info mode)
        obs = env._get_observation()

        # Agent selects board
        agent_action, _ = model.predict(obs, deterministic=True)
        agent_idx = int(agent_action)

        # Validate selection is in range
        if agent_idx < 0 or agent_idx >= len(agent_deck):
            print(f"ERROR: Agent selected invalid board index {agent_idx}")
            continue

        # Get agent's actual result
        agent_matchup = all_matchups[agent_idx]
        agent_diff = agent_matchup["differential"]

        # Check if selection was optimal
        is_optimal = (agent_idx == optimal_idx)
        correct_selections += is_optimal
        total_tests += 1
        total_score_diff += agent_diff
        total_optimal_diff += optimal_diff

        if verbose:
            print(f"  Optimal selection: Board {optimal_idx} (diff: {optimal_diff:+d})")
            print(f"  Agent selected:    Board {agent_idx} (diff: {agent_diff:+d})")
            if is_optimal:
                print(f"  ✓ CORRECT!")
            else:
                print(f"  ✗ SUBOPTIMAL (missed {optimal_diff - agent_diff} points)")

        results_log.append({
            "opponent_board": opp_idx,
            "optimal_board": optimal_idx,
            "optimal_diff": optimal_diff,
            "agent_board": agent_idx,
            "agent_diff": agent_diff,
            "is_optimal": is_optimal,
        })

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total tests:           {total_tests}")
        print(f"Correct selections:    {correct_selections} / {total_tests}")
        print(f"Accuracy:              {correct_selections / total_tests * 100:.1f}%")
        print(f"Avg agent score diff:  {total_score_diff / total_tests:.2f}")
        print(f"Avg optimal diff:      {total_optimal_diff / total_tests:.2f}")
        print(f"Efficiency:            {total_score_diff / total_optimal_diff * 100:.1f}%")
        print("=" * 80)

    return {
        "accuracy": correct_selections / total_tests if total_tests > 0 else 0,
        "total_tests": total_tests,
        "correct": correct_selections,
        "avg_agent_diff": total_score_diff / total_tests if total_tests > 0 else 0,
        "avg_optimal_diff": total_optimal_diff / total_tests if total_tests > 0 else 0,
        "results": results_log,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test agent's board selection accuracy"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--board-pool",
        type=str,
        default="new_boards_2.json",
        help="Path to board pool (default: new_boards_2.json)",
    )
    parser.add_argument(
        "--perfect-info",
        action="store_true",
        default=True,
        help="Use perfect information mode (default: True)",
    )

    args = parser.parse_args()

    # Check files exist
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        return 1

    if not Path(args.board_pool).exists():
        print(f"Error: Board pool not found at {args.board_pool}")
        return 1

    # Run test
    try:
        results = test_agent_selection(
            args.model_path,
            args.board_pool,
            perfect_info=args.perfect_info,
            verbose=True,
        )

        # Exit code based on accuracy
        if results["accuracy"] >= 0.8:
            print("\n✓ PASS: Agent achieves >= 80% optimal selection")
            return 0
        else:
            print(f"\n✗ FAIL: Agent only achieves {results['accuracy']*100:.1f}% optimal selection")
            return 1

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
