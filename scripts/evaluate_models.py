"""Head-to-head model evaluation: two models build boards and play against each other.

Both models construct boards blind each round, then simulate_round() determines
the outcome. This is true self-play evaluation, not pool-based.

Usage:
    python scripts/evaluate_models.py \
        --model-a models/size9/stage4/ppo_stage3_17040000_steps.zip \
        --model-b models/size9/stage4/best/best_model.zip \
        --size 9 --fog --games 10000 --report-every 1000
"""
import argparse
import sys
import time

import numpy as np
from sb3_contrib import MaskablePPO

from spaces_game.callbacks.pool_utils import discover_pools, build_phase_map
from spaces_game.simulation import simulate_round
from spaces_game.simultaneous_play_env import SimultaneousPlayEnv
from spaces_game.validation import is_board_playable


def build_board(model, board_size, opponent_pools, phase_map, use_fog,
                round_num, my_score, opp_score, opp_history_grids, fog_outcomes,
                seed):
    """Have a model construct a board blind, returning a Board or None."""
    max_steps = board_size * 10  # Must match training config
    env = SimultaneousPlayEnv(
        board_size=board_size,
        opponent_pools=opponent_pools,
        phase_map=phase_map,
        use_fog=use_fog,
        max_construction_steps=max_steps,
    )
    env.reset(seed=seed)

    # Set game context so the model sees correct state
    env.current_round = round_num
    env.agent_total_score = my_score
    env.opponent_total_score = opp_score
    env.opponent_history_grids = opp_history_grids.copy()
    if use_fog:
        env.fog_outcomes_data = fog_outcomes.copy()

    obs = env._get_observation()
    max_steps = env.max_construction_steps

    for _ in range(max_steps):
        masks = env.action_masks()
        action, _ = model.predict(obs, deterministic=True, action_masks=masks)

        act = int(action)
        n_cells = board_size * board_size
        if act >= 2 * n_cells:
            break  # finish

        cell = act % n_cells
        move_type = "piece" if act < n_cells else "trap"
        row = cell // board_size
        col = cell % board_size
        order = env.construction_step + 1

        if env._is_valid_placement(row, col, move_type):
            env.construction_sequence.append({
                "row": row, "col": col, "type": move_type, "order": order,
            })
            if move_type == "piece":
                env.building_grid[row, col, 0] = order
                if env.supermove_active:
                    env.supermove_active = False
                    env.supermove_position = None
                from spaces_game.types import Position
                env.piece_visited_positions.add(f"{row},{col}")
                env.current_piece_position = Position(row=row, col=col)
            elif move_type == "trap":
                env.building_grid[row, col, 1] = order
                env.trap_positions.add(f"{row},{col}")
                from spaces_game.types import Position
                if (env.current_piece_position is not None and
                        row == env.current_piece_position.row and
                        col == env.current_piece_position.col):
                    env.supermove_active = True
                    env.supermove_position = Position(row=row, col=col)
            env.construction_step += 1

        obs = env._get_observation()

    board = env._construct_board_from_state()
    env.close()

    if is_board_playable(board):
        return board
    return None


def play_game(model_a, model_b, board_size, opponent_pools, phase_map, use_fog, seed):
    """Play one 5-round game. Returns 'a', 'b', or 'tie'."""
    ROUNDS = 5
    a_score, b_score = 0, 0
    a_boards, b_boards = [], []

    # Each model's view of opponent history (rotated boards from opponent)
    a_opp_history = np.zeros((ROUNDS, board_size, board_size, 2), dtype=np.int32)
    b_opp_history = np.zeros((ROUNDS, board_size, board_size, 2), dtype=np.int32)
    a_fog = np.zeros((ROUNDS, 6), dtype=np.float32)
    b_fog = np.zeros((ROUNDS, 6), dtype=np.float32)

    # We need a temp env for encoding helpers
    enc_env = SimultaneousPlayEnv(
        board_size=board_size, opponent_pools=opponent_pools,
        phase_map=phase_map, use_fog=use_fog,
        max_construction_steps=board_size * 10,
    )
    enc_env.reset(seed=seed)

    for rnd in range(ROUNDS):
        board_a = build_board(
            model_a, board_size, opponent_pools, phase_map, use_fog,
            rnd, a_score, b_score, a_opp_history, a_fog,
            seed=seed * 10 + rnd * 2,
        )
        board_b = build_board(
            model_b, board_size, opponent_pools, phase_map, use_fog,
            rnd, b_score, a_score, b_opp_history, b_fog,
            seed=seed * 10 + rnd * 2 + 1,
        )

        if board_a is None or board_b is None:
            # Invalid board = loss for that model
            if board_a is None and board_b is None:
                pass  # no score change
            elif board_a is None:
                b_score += board_size  # award opponent
            else:
                a_score += board_size
            a_boards.append(board_a)
            b_boards.append(board_b)
            continue

        result = simulate_round(rnd + 1, board_a, board_b, silent=True)
        a_score += result.playerPoints
        b_score += result.opponentPoints
        details = result.simulationDetails

        # Encode opponent boards into history for next round
        if use_fog:
            # A sees B's board (fog-filtered from A's perspective)
            a_opp_history[rnd] = enc_env._encode_opponent_board_fog(
                board_b, details.playerLastStep,
                details.playerTrapPosition,
            )
            # B sees A's board (fog-filtered from B's perspective)
            b_opp_history[rnd] = enc_env._encode_opponent_board_fog(
                board_a, details.opponentLastStep,
                details.opponentTrapPosition,
            )

            # Fog outcomes for A
            max_steps_b = max(len([m for m in board_b.sequence if m.type != "final"]), 1)
            max_traps = board_size - 1
            a_fog[rnd] = np.array([
                details.playerLastStep / max(max_steps_b, 1),
                float(details.opponentHitTrap),
                float(details.playerHitTrap),
                float(result.collision),
                float(result.opponentPoints > 0 and not result.collision and not details.opponentHitTrap),
                (1 if details.playerHitTrap else 0) / max(max_traps, 1),
            ], dtype=np.float32)

            # Fog outcomes for B (swapped perspective)
            max_steps_a = max(len([m for m in board_a.sequence if m.type != "final"]), 1)
            b_fog[rnd] = np.array([
                details.opponentLastStep / max(max_steps_a, 1),
                float(details.playerHitTrap),
                float(details.opponentHitTrap),
                float(result.collision),
                float(result.playerPoints > 0 and not result.collision and not details.playerHitTrap),
                (1 if details.opponentHitTrap else 0) / max(max_traps, 1),
            ], dtype=np.float32)
        else:
            a_opp_history[rnd] = enc_env._encode_opponent_board(board_b)
            b_opp_history[rnd] = enc_env._encode_opponent_board(board_a)

        a_boards.append(board_a)
        b_boards.append(board_b)

    enc_env.close()

    if a_score > b_score:
        return "a"
    elif b_score > a_score:
        return "b"
    return "tie"


def main():
    parser = argparse.ArgumentParser(description="Head-to-head model evaluation")
    parser.add_argument("--model-a", required=True, help="Path to model A")
    parser.add_argument("--model-b", required=True, help="Path to model B")
    parser.add_argument("--size", type=int, required=True, help="Board size")
    parser.add_argument("--fog", action="store_true", help="Enable fog of war")
    parser.add_argument("--games", type=int, default=10000, help="Number of games")
    parser.add_argument("--report-every", type=int, default=1000, help="Report interval")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    pool_paths = discover_pools(args.size)
    phase_map = build_phase_map(len(pool_paths))

    print(f"Loading Model A: {args.model_a}")
    model_a = MaskablePPO.load(args.model_a)
    print(f"Loading Model B: {args.model_b}")
    model_b = MaskablePPO.load(args.model_b)

    print(f"\nSize {args.size} | Fog: {args.fog} | Games: {args.games}")
    print(f"{'='*70}")

    a_wins, b_wins, ties = 0, 0, 0
    start = time.time()

    for game in range(1, args.games + 1):
        result = play_game(
            model_a, model_b, args.size, pool_paths, phase_map, args.fog,
            seed=args.seed + game,
        )
        if result == "a":
            a_wins += 1
        elif result == "b":
            b_wins += 1
        else:
            ties += 1

        if game % args.report_every == 0:
            elapsed = time.time() - start
            gps = game / elapsed
            a_wr = a_wins / game * 100
            print(
                f"Games {game:>6} | "
                f"A wins: {a_wins:>5}  A losses: {b_wins:>5}  Ties: {ties:>4} | "
                f"A WR: {a_wr:5.1f}% | {gps:.1f} games/s"
            )

    print(f"{'='*70}")
    print(
        f"Final: Model A wins {a_wins}/{args.games} ({a_wins/args.games*100:.1f}%), "
        f"Model B wins {b_wins}/{args.games} ({b_wins/args.games*100:.1f}%), "
        f"Ties {ties} ({ties/args.games*100:.1f}%)"
    )
    print(f"\nModel A: {args.model_a}")
    print(f"Model B: {args.model_b}")


if __name__ == "__main__":
    main()
