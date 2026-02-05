"""
Evaluate a trained reverse curriculum board builder agent.

Tests the agent's ability to construct boards at different curriculum phases
and reports validity rate, win rate, and score differentials.

Usage:
    python examples/evaluate_reverse_curriculum.py models/reverse_curriculum/best/best_model.zip
    python examples/evaluate_reverse_curriculum.py models/reverse_curriculum/phase_3_checkpoint.zip --phase 3
    python examples/evaluate_reverse_curriculum.py models/reverse_curriculum/best/best_model.zip --all-phases
"""

import sys
import json
import numpy as np
from pathlib import Path

from spaces_game import ReverseCurriculumBuilderEnv


def _load_model(model_path: str):
    """Load model, trying MaskablePPO first then falling back to PPO."""
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        return model, True  # (model, uses_action_masks)
    except Exception:
        pass
    from stable_baselines3 import PPO
    return PPO.load(model_path), False


def evaluate_phase(
    model_path: str,
    phase: int,
    n_episodes: int = 50,
    seed: int = 42,
    stage1_model_path: str = "models/construction/best/best_model.zip",
    verbose: bool = True,
):
    """
    Evaluate trained agent on a specific curriculum phase.

    Args:
        model_path: Path to trained model .zip file
        phase: Curriculum phase to evaluate (0=easiest, higher=harder)
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        stage1_model_path: Path to Stage 1 model for base board selection
        verbose: Print detailed results

    Returns:
        dict with evaluation metrics
    """
    # Load trained model
    model, uses_masks = _load_model(model_path)

    # Create evaluation environment
    env = ReverseCurriculumBuilderEnv(
        board_size=2,
        board_library_path="new_boards_2.json",
        stage1_model_path=stage1_model_path if Path(stage1_model_path).exists() else None,
        curriculum_phase=phase,
        opponent_strategy="random",
        show_opponent_board=True,
    )

    # Evaluation metrics
    valid_boards = 0
    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0
    score_diffs = []
    moves_placed = []

    if verbose:
        print(f"\nEvaluating Phase {phase} (place last {phase+1} move{'s' if phase > 0 else ''} + goal)...")
        print(f"Episodes: {n_episodes}")
        print("-" * 60)

    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False

        while not done:
            # Agent selects action using trained policy
            if uses_masks:
                action_masks = env.action_masks()
                action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        # Record results
        is_valid = info.get('valid_board', False)
        agent_score = info.get('agent_score', 0)
        opponent_score = info.get('opponent_score', 0)
        score_diff = agent_score - opponent_score
        num_moves = info.get('moves_placed', 0)

        if is_valid:
            valid_boards += 1
            total_agent_score += agent_score
            total_opponent_score += opponent_score
            score_diffs.append(score_diff)
            moves_placed.append(num_moves)

            if agent_score > opponent_score:
                wins += 1
                result = "WIN"
            elif agent_score < opponent_score:
                losses += 1
                result = "LOSS"
            else:
                ties += 1
                result = "TIE"
        else:
            result = "INVALID"

        if verbose and episode < 10:  # Show first 10 episodes
            print(f"Episode {episode+1:3d}: {agent_score:3d}-{opponent_score:3d} "
                  f"({result:7s}) Moves: {num_moves:2d}/{info.get('moves_required', 0):2d} "
                  f"Reward: {episode_reward:+7.1f}")

    env.close()

    # Calculate metrics
    valid_rate = valid_boards / n_episodes
    win_rate = wins / valid_boards if valid_boards > 0 else 0.0
    loss_rate = losses / valid_boards if valid_boards > 0 else 0.0
    tie_rate = ties / valid_boards if valid_boards > 0 else 0.0
    avg_agent_score = total_agent_score / valid_boards if valid_boards > 0 else 0.0
    avg_opponent_score = total_opponent_score / valid_boards if valid_boards > 0 else 0.0
    avg_score_diff = np.mean(score_diffs) if score_diffs else 0.0
    std_score_diff = np.std(score_diffs) if score_diffs else 0.0
    avg_moves = np.mean(moves_placed) if moves_placed else 0.0

    if verbose:
        if n_episodes > 10:
            print(f"... ({n_episodes - 10} more episodes)")
        print("-" * 60)
        print(f"\nRESULTS FOR PHASE {phase}:")
        print(f"  Valid boards: {valid_boards:3d} ({valid_rate*100:5.1f}%)")
        if valid_boards > 0:
            print(f"  Wins:         {wins:3d} ({win_rate*100:5.1f}% of valid)")
            print(f"  Losses:       {losses:3d} ({loss_rate*100:5.1f}% of valid)")
            print(f"  Ties:         {ties:3d} ({tie_rate*100:5.1f}% of valid)")
            print(f"\nSCORES (valid boards only):")
            print(f"  Avg Agent Score:    {avg_agent_score:6.1f}")
            print(f"  Avg Opponent Score: {avg_opponent_score:6.1f}")
            print(f"  Avg Differential:   {avg_score_diff:+6.1f} ± {std_score_diff:.1f}")
            print(f"  Avg Moves Placed:   {avg_moves:.1f}")

    return {
        'phase': phase,
        'valid_boards': valid_boards,
        'valid_rate': valid_rate,
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
        'avg_moves': avg_moves,
        'n_episodes': n_episodes,
    }


def evaluate_all_phases(
    model_path: str,
    max_phase: int = 5,
    n_episodes_per_phase: int = 50,
    stage1_model_path: str = "models/construction/best/best_model.zip",
):
    """
    Evaluate agent across all curriculum phases.

    Args:
        model_path: Path to trained model
        max_phase: Maximum phase to evaluate (0 to max_phase inclusive)
        n_episodes_per_phase: Episodes per phase
        stage1_model_path: Path to Stage 1 model

    Returns:
        dict with detailed per-phase results
    """
    print("=" * 70)
    print("REVERSE CURRICULUM EVALUATION (All Phases)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes per phase: {n_episodes_per_phase}")
    print(f"Phases: 0 to {max_phase}")

    phase_results = {}

    for phase in range(max_phase + 1):
        print(f"\n{'='*70}")
        print(f"Evaluating Phase {phase} (place last {phase+1} move{'s' if phase > 0 else ''} + goal)")
        print(f"{'='*70}")

        result = evaluate_phase(
            model_path,
            phase=phase,
            n_episodes=n_episodes_per_phase,
            seed=42,
            stage1_model_path=stage1_model_path,
            verbose=True,
        )
        phase_results[phase] = result

    # Summary table
    print("\n" + "=" * 70)
    print("PER-PHASE SUMMARY")
    print("=" * 70)
    print(f"{'Phase':<8} {'Valid%':<10} {'Win%':<10} {'Avg Diff':<15} {'Moves'}")
    print("-" * 70)

    for phase in range(max_phase + 1):
        r = phase_results[phase]
        valid_pct = r['valid_rate'] * 100
        win_pct = r['win_rate'] * 100 if r['valid_boards'] > 0 else 0
        print(f"{phase:<8} "
              f"{valid_pct:5.1f}%     "
              f"{win_pct:5.1f}%     "
              f"{r['avg_score_diff']:+6.1f} ± {r['std_score_diff']:4.1f}    "
              f"{r['avg_moves']:4.1f}")

    # Overall stats
    overall_valid_rate = np.mean([r['valid_rate'] for r in phase_results.values()])
    valid_phases = [r for r in phase_results.values() if r['valid_boards'] > 0]
    overall_win_rate = np.mean([r['win_rate'] for r in valid_phases]) if valid_phases else 0.0
    overall_avg_diff = np.mean([r['avg_score_diff'] for r in valid_phases]) if valid_phases else 0.0

    print("=" * 70)
    print(f"OVERALL (phases 0-{max_phase}):")
    print(f"  Valid rate: {overall_valid_rate*100:.1f}%")
    print(f"  Win rate:   {overall_win_rate*100:.1f}% (of valid boards)")
    print(f"  Avg diff:   {overall_avg_diff:+.1f}")

    # Assessment
    if overall_valid_rate >= 0.95 and overall_win_rate >= 0.80:
        print("\n✓ EXCELLENT! Agent masters reverse curriculum across all phases.")
    elif overall_valid_rate >= 0.80 and overall_win_rate >= 0.70:
        print("\n✓ GOOD! Agent performs well but has room for improvement.")
    elif overall_valid_rate >= 0.60:
        print("\n⚠ FAIR. Agent shows learning but needs more training.")
    else:
        print("\n✗ POOR. Agent produces many invalid boards.")

    print("=" * 70)
    return phase_results


def compare_with_stage1(
    reverse_model_path: str,
    stage1_model_path: str = "models/construction/best/best_model.zip",
    n_episodes: int = 100,
):
    """
    Compare reverse curriculum agent's constructed boards vs Stage 1 direct selection.

    Args:
        reverse_model_path: Path to reverse curriculum model
        stage1_model_path: Path to Stage 1 model
        n_episodes: Number of comparison episodes
    """
    print("=" * 70)
    print("COMPARISON: Reverse Curriculum vs Stage 1 Direct Selection")
    print("=" * 70)

    from spaces_game import BoardConstructionEnv
    from stable_baselines3 import PPO

    # Evaluate Stage 1 (direct selection)
    print("\n[1] Evaluating Stage 1 (Direct Board Selection)...")
    stage1_model = PPO.load(stage1_model_path)
    stage1_env = BoardConstructionEnv(
        board_library_path="new_boards_2.json",
        opponent_strategy="random",
        show_opponent_board=True,
    )

    stage1_wins = 0
    stage1_total_score = 0
    stage1_opponent_score = 0

    for ep in range(n_episodes):
        obs, info = stage1_env.reset(seed=42 + ep)
        done = False
        while not done:
            action, _ = stage1_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = stage1_env.step(action)

        agent_score = info['agent_total_score']
        opponent_score = info['opponent_total_score']
        stage1_total_score += agent_score
        stage1_opponent_score += opponent_score
        if agent_score > opponent_score:
            stage1_wins += 1

    stage1_env.close()

    # Evaluate Reverse Curriculum (Phase 5 - harder construction)
    print("\n[2] Evaluating Reverse Curriculum (Phase 5)...")
    reverse_result = evaluate_phase(
        reverse_model_path,
        phase=5,
        n_episodes=n_episodes,
        seed=42,
        stage1_model_path=stage1_model_path,
        verbose=False,
    )

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Approach':<25} {'Win Rate':<12} {'Avg Score':<15} {'Valid%'}")
    print("-" * 70)

    stage1_win_rate = stage1_wins / n_episodes
    stage1_avg_score = stage1_total_score / n_episodes
    reverse_win_rate = reverse_result['win_rate']
    reverse_avg_score = reverse_result['avg_agent_score']
    reverse_valid_rate = reverse_result['valid_rate']

    print(f"{'Stage 1 (Selection)':<25} "
          f"{stage1_win_rate*100:5.1f}%       "
          f"{stage1_avg_score:6.1f}          100.0%")

    print(f"{'Reverse Curriculum (P5)':<25} "
          f"{reverse_win_rate*100:5.1f}%       "
          f"{reverse_avg_score:6.1f}          {reverse_valid_rate*100:5.1f}%")

    print("=" * 70)

    if reverse_win_rate >= stage1_win_rate * 0.95:
        print("✓ Reverse curriculum matches Stage 1 performance!")
        print("  Agent has learned to construct competitive boards.")
    elif reverse_win_rate >= stage1_win_rate * 0.80:
        print("⚠ Reverse curriculum is close but not quite matching Stage 1.")
    else:
        print("✗ Reverse curriculum needs more training to match Stage 1.")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate reverse curriculum board builder")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Specific phase to evaluate (default: None, uses --all-phases)",
    )
    parser.add_argument(
        "--all-phases",
        action="store_true",
        help="Evaluate all phases from 0 to --max-phase",
    )
    parser.add_argument(
        "--max-phase",
        type=int,
        default=5,
        help="Maximum phase to evaluate when using --all-phases (default: 5)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per phase (default: 50)",
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default="models/construction/best/best_model.zip",
        help="Path to Stage 1 model (default: models/construction/best/best_model.zip)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare reverse curriculum vs Stage 1 direct selection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    if args.compare:
        compare_with_stage1(
            args.model_path,
            stage1_model_path=args.stage1_model,
            n_episodes=args.episodes,
        )
    elif args.all_phases:
        evaluate_all_phases(
            args.model_path,
            max_phase=args.max_phase,
            n_episodes_per_phase=args.episodes,
            stage1_model_path=args.stage1_model,
        )
    elif args.phase is not None:
        evaluate_phase(
            args.model_path,
            phase=args.phase,
            n_episodes=args.episodes,
            seed=args.seed,
            stage1_model_path=args.stage1_model,
            verbose=True,
        )
    else:
        # Default: evaluate all phases
        evaluate_all_phases(
            args.model_path,
            max_phase=args.max_phase,
            n_episodes_per_phase=args.episodes,
            stage1_model_path=args.stage1_model,
        )
