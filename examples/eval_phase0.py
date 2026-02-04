"""
Simple evaluation for Phase 0 models.
"""

import sys
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

from spaces_game import ReverseCurriculumBuilderEnv


def eval_phase0(
    model_path: str,
    n_episodes: int = 100,
    stage1_model_path: str = "models/construction/best/best_model.zip",
):
    """Evaluate Phase 0 model."""
    print("=" * 70)
    print("PHASE 0 EVALUATION (Goal Placement)")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print("=" * 70)

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = ReverseCurriculumBuilderEnv(
        board_size=2,
        board_library_path="new_boards_2.json",
        stage1_model_path=stage1_model_path if Path(stage1_model_path).exists() else None,
        curriculum_phase=0,
        opponent_strategy="random",
        show_opponent_board=True,
    )

    # Evaluation
    valid_boards = 0
    wins = 0
    losses = 0
    ties = 0
    total_agent_score = 0
    total_opponent_score = 0
    score_diffs = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        # Check results
        is_valid = info.get('valid_board', False)
        agent_score = info.get('agent_score', 0)
        opponent_score = info.get('opponent_score', 0)

        if is_valid:
            valid_boards += 1
            total_agent_score += agent_score
            total_opponent_score += opponent_score
            score_diff = agent_score - opponent_score
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
        else:
            result = "INVALID"

        if ep < 10:
            print(f"Episode {ep+1:3d}: {agent_score:3d}-{opponent_score:3d} ({result:7s}) Reward: {episode_reward:+7.1f}")

    env.close()

    # Results
    valid_rate = valid_boards / n_episodes
    win_rate = wins / valid_boards if valid_boards > 0 else 0.0
    avg_agent_score = total_agent_score / valid_boards if valid_boards > 0 else 0.0
    avg_opponent_score = total_opponent_score / valid_boards if valid_boards > 0 else 0.0
    avg_score_diff = np.mean(score_diffs) if score_diffs else 0.0
    std_score_diff = np.std(score_diffs) if score_diffs else 0.0

    if n_episodes > 10:
        print(f"... ({n_episodes - 10} more episodes)")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Valid boards:   {valid_boards}/{n_episodes} ({valid_rate*100:.1f}%)")
    if valid_boards > 0:
        print(f"Wins:           {wins} ({win_rate*100:.1f}% of valid)")
        print(f"Losses:         {losses}")
        print(f"Ties:           {ties}")
        print(f"\nAgent score:    {avg_agent_score:.1f}")
        print(f"Opponent score: {avg_opponent_score:.1f}")
        print(f"Score diff:     {avg_score_diff:+.1f} ± {std_score_diff:.1f}")

    print("=" * 70)

    # Assessment
    if valid_rate >= 0.95 and win_rate >= 0.80:
        print("✓ EXCELLENT! Phase 0 mastered - ready for Phase 1")
    elif valid_rate >= 0.90 and win_rate >= 0.70:
        print("✓ GOOD! Phase 0 performing well")
    elif valid_rate >= 0.70:
        print("⚠ FAIR. Needs more training")
    else:
        print("✗ POOR. Most boards invalid")

    print("=" * 70)

    return {
        'valid_rate': valid_rate,
        'win_rate': win_rate,
        'avg_score_diff': avg_score_diff,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Phase 0 model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default="models/construction/best/best_model.zip",
        help="Path to Stage 1 model",
    )

    args = parser.parse_args()

    eval_phase0(
        args.model_path,
        n_episodes=args.episodes,
        stage1_model_path=args.stage1_model,
    )
