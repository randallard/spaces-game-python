"""
Debug Phase 0 to see what the agent is actually doing.
"""

import numpy as np
from stable_baselines3 import PPO
from spaces_game import ReverseCurriculumBuilderEnv


def debug_episode():
    """Run one episode with detailed logging."""
    print("=" * 70)
    print("DEBUG: Phase 0 Episode")
    print("=" * 70)

    # Load model
    model = PPO.load("models/phase0/best/best_model.zip")

    # Create environment
    env = ReverseCurriculumBuilderEnv(
        board_size=2,
        board_library_path="new_boards_2.json",
        stage1_model_path="models/construction/best/best_model.zip",
        curriculum_phase=0,
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    print(f"\nReset Info:")
    print(f"  Moves to place: {info['moves_to_place']}")
    print(f"  Target sequence length: {len(env.target_sequence)}")
    print(f"  Target sequence: {[(m.type, m.order) for m in env.target_sequence]}")
    print(f"  Construction step: {env.construction_step}")
    print(f"  Placed moves: {len(env.placed_moves)}")

    step_num = 0
    done = False
    total_reward = 0

    while not done and step_num < 10:
        step_num += 1
        action, _states = model.predict(obs, deterministic=True)

        cell = action[0] % 4
        piece_or_trap = action[1] % 2
        done_flag = action[2] > 0

        print(f"\nStep {step_num}:")
        print(f"  Action: cell={cell}, type={'piece' if piece_or_trap==0 else 'trap'}, done={done_flag}")
        print(f"  Before: placed_moves={len(env.placed_moves)}, target={len(env.target_sequence)}")

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(f"  After:  placed_moves={len(env.placed_moves)}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")

        if done:
            print(f"\n  Final Info:")
            print(f"    Valid board: {info.get('valid_board', False)}")
            print(f"    Agent score: {info.get('agent_score', 0)}")
            print(f"    Opponent score: {info.get('opponent_score', 0)}")
            print(f"    Moves placed: {info.get('moves_placed', 0)}")
            print(f"    Moves required: {info.get('moves_required', 0)}")

            # Show constructed board
            print(f"\n  Constructed board details:")
            print(f"    Placed moves: {env.placed_moves}")

            # Try to get the actual board
            agent_board = env._construct_board_from_state()
            print(f"    Board sequence ({len(agent_board.sequence)} moves):")
            for move in agent_board.sequence:
                print(f"      {move.order}. {move.type} at ({move.position.row}, {move.position.col})")

    env.close()

    print(f"\nTotal reward: {total_reward:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    debug_episode()
