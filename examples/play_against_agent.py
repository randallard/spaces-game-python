"""
Play against the trained reverse curriculum agent.

Build your board interactively, then watch the AI construct a counter-board
and see who wins in simulation.

Usage:
    python examples/play_against_agent.py
    python examples/play_against_agent.py --size 2
    python examples/play_against_agent.py --model models/reverse_curriculum/ppo_reverse_curriculum_90000_steps.zip
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

import click

from spaces_game import ReverseCurriculumBuilderEnv, BoardConstructionEnv
from spaces_game.interactive_builder import build_board_interactive, _render_board_state
from spaces_game.board_loader import load_boards_from_json
from spaces_game.simulation import simulate_round
from spaces_game.types import Board


def _load_agent(model_path: str):
    """Load the trained agent model."""
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path), True
    except Exception:
        pass
    from stable_baselines3 import PPO
    return PPO.load(model_path), False


def _agent_build_board(
    player_board: Board,
    model,
    uses_masks: bool,
    board_size: int = 2,
    board_library_path: str = "new_boards_2.json",
    stage1_model_path: str = "models/construction/best/best_model.zip",
) -> Board:
    """Have the agent build a counter-board against the player's board."""
    env = ReverseCurriculumBuilderEnv(
        board_size=board_size,
        board_library_path=board_library_path,
        stage1_model_path=stage1_model_path if Path(stage1_model_path).exists() else None,
        curriculum_phase=10,  # Full construction
        opponent_strategy="fixed_0",  # Will override with player board
        show_opponent_board=True,
    )

    # Override the opponent board with the player's board
    env.opponent_board = player_board
    obs, info = env.reset(seed=42)
    env.opponent_board = player_board  # Re-set after reset

    done = False
    while not done:
        if uses_masks:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    agent_board = env._construct_board_from_state()
    env.close()
    return agent_board


def _display_board_simple(board: Board, title: str):
    """Display a board with a simple grid."""
    click.echo(_render_board_state(board, current_position=None, title=title))


def _display_result(player_board: Board, agent_board: Board, result):
    """Display the simulation result."""
    player_score = result.playerPoints
    agent_score = result.opponentPoints

    click.echo(click.style("\n" + "=" * 50, bold=True))
    click.echo(click.style("  SIMULATION RESULT", bold=True))
    click.echo(click.style("=" * 50, bold=True))

    click.echo(f"\n  You:   {player_score} points")
    click.echo(f"  Agent: {agent_score} points")

    if player_score > agent_score:
        click.echo(click.style(f"\n  YOU WIN! (+{player_score - agent_score})", fg="green", bold=True))
    elif agent_score > player_score:
        click.echo(click.style(f"\n  AGENT WINS! (+{agent_score - player_score})", fg="red", bold=True))
    else:
        click.echo(click.style("\n  TIE!", fg="yellow", bold=True))

    click.echo(click.style("=" * 50 + "\n", bold=True))


def play(
    board_size: int = 2,
    model_path: str = "models/reverse_curriculum/ppo_reverse_curriculum_90000_steps.zip",
    stage1_model_path: str = "models/construction/best/best_model.zip",
    board_library_path: str = "new_boards_2.json",
):
    """Main play loop."""
    click.echo(click.style("\n" + "=" * 50, bold=True))
    click.echo(click.style("  SPACES GAME - Play vs Agent", bold=True))
    click.echo(click.style("=" * 50, bold=True))
    click.echo(f"\n  Board size: {board_size}x{board_size}")
    click.echo(f"  Agent model: {model_path}")

    # Load agent
    if not Path(model_path).exists():
        click.echo(click.style(f"\n  Model not found: {model_path}", fg="red"))
        click.echo("  Train first with: python examples/train_reverse_curriculum.py")
        sys.exit(1)

    click.echo("\n  Loading agent model...")
    model, uses_masks = _load_agent(model_path)
    click.echo(click.style("  Agent loaded!", fg="green"))

    while True:
        click.echo(click.style("\n" + "-" * 50, bold=True))
        click.echo(click.style("  NEW GAME", bold=True))
        click.echo(click.style("-" * 50, bold=True))

        # Choose mode
        click.echo("\n  How do you want to play?")
        click.echo("    1) Build a board interactively")
        click.echo("    2) Pick a board from the library")
        click.echo("    q) Quit")

        choice = click.prompt("\n  Choice", type=str, default="1")

        if choice.lower() in ("q", "quit"):
            click.echo("\nGoodbye!")
            break

        player_board = None

        if choice == "1":
            # Interactive building
            click.echo(click.style("\n  Build your board!\n", fg="cyan", bold=True))
            player_board = build_board_interactive(size=board_size)
            if player_board is None:
                click.echo("  Board building cancelled.")
                continue

        elif choice == "2":
            # Pick from library
            try:
                boards = load_boards_from_json(board_library_path)
            except Exception as e:
                click.echo(click.style(f"  Could not load library: {e}", fg="red"))
                continue

            click.echo(f"\n  Available boards (from {board_library_path}):\n")
            for i, board in enumerate(boards):
                n_pieces = sum(1 for m in board.sequence if m.type == "piece")
                n_traps = sum(1 for m in board.sequence if m.type == "trap")
                click.echo(f"    {i}: {n_pieces} pieces, {n_traps} traps, {len(board.sequence)} total moves")

            try:
                idx = click.prompt("\n  Select board #", type=int, default=0)
                if idx < 0 or idx >= len(boards):
                    click.echo(click.style("  Invalid selection.", fg="red"))
                    continue
                player_board = boards[idx]
            except (ValueError, KeyboardInterrupt):
                continue
        else:
            click.echo("  Invalid choice.")
            continue

        # Show player's board
        _display_board_simple(player_board, "Your Board")

        # Agent builds counter-board
        click.echo(click.style("\n  Agent is building a counter-board...", fg="yellow"))
        agent_board = _agent_build_board(
            player_board, model, uses_masks,
            board_size=board_size,
            board_library_path=board_library_path,
            stage1_model_path=stage1_model_path,
        )

        # Show agent's board
        _display_board_simple(agent_board, "Agent's Board")

        # Simulate: player is "player", agent is "opponent"
        result = simulate_round(0, player_board, agent_board, silent=True)
        _display_result(player_board, agent_board, result)

        # Play again?
        again = click.prompt("  Play again?", type=str, default="y")
        if again.lower() not in ("y", "yes"):
            click.echo("\nGoodbye!")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play against the trained agent")
    parser.add_argument(
        "--size", type=int, default=2,
        help="Board size (default: 2)",
    )
    parser.add_argument(
        "--model", type=str,
        default="models/reverse_curriculum/ppo_reverse_curriculum_90000_steps.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--stage1-model", type=str,
        default="models/construction/best/best_model.zip",
        help="Path to Stage 1 model",
    )
    parser.add_argument(
        "--board-library", type=str,
        default="new_boards_2.json",
        help="Path to board library JSON",
    )

    args = parser.parse_args()

    play(
        board_size=args.size,
        model_path=args.model,
        stage1_model_path=args.stage1_model,
        board_library_path=args.board_library,
    )
