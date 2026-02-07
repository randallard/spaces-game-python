"""
Play against the trained reverse curriculum agent.

Build your board interactively, then watch the AI construct a counter-board
and see who wins in simulation.

Usage:
    python examples/play_against_agent.py
    python examples/play_against_agent.py --size 2
    python examples/play_against_agent.py --model models/reverse_curriculum/phase_3_checkpoint.zip
"""

import os
import random
import re
import sys
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Optional

import click

from spaces_game import ReverseCurriculumBuilderEnv, BoardConstructionEnv
from spaces_game.interactive_builder import build_board_interactive, _render_board_state
from spaces_game.board_loader import load_boards_from_json
from spaces_game.simulation import simulate_round
from spaces_game.validation import is_board_playable
from spaces_game.types import Board
from spaces_game.cli import _render_result_details, _render_board


def _get_model_board_size(model) -> Optional[int]:
    """Extract board size from a loaded model's observation space."""
    try:
        obs_space = model.observation_space
        if hasattr(obs_space, 'spaces') and 'building_board' in obs_space.spaces:
            return obs_space.spaces['building_board'].shape[0]
        if hasattr(obs_space, 'shape') and obs_space.shape is not None:
            return obs_space.shape[0]
    except Exception:
        pass
    return None


def _load_agent(model_path: str):
    """Load the trained agent model."""
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path), True
    except Exception:
        pass
    from stable_baselines3 import PPO
    return PPO.load(model_path), False


def _discover_training_runs(base_dir: str = "models") -> list:
    """Scan for all directories containing model .zip files.

    Returns list of dicts: {"path": str, "label": str, "model_count": int, "newest": float}
    """
    base = Path(base_dir)
    if not base.exists():
        return []

    # Find all directories that directly contain .zip files
    runs = {}
    for zip_file in base.rglob("*.zip"):
        parent = zip_file.parent
        # Skip "best/" subdirectories - attribute those to parent
        if parent.name == "best":
            parent = parent.parent
        key = str(parent)
        if key not in runs:
            runs[key] = {"path": key, "zips": [], "newest": 0.0}
        runs[key]["zips"].append(zip_file)
        mtime = zip_file.stat().st_mtime
        if mtime > runs[key]["newest"]:
            runs[key]["newest"] = mtime

    # Build labeled results
    results = []
    for key, info in runs.items():
        rel = os.path.relpath(info["path"], ".")
        n_models = len(info["zips"])
        date_str = datetime.fromtimestamp(info["newest"]).strftime("%Y-%m-%d %H:%M")
        results.append({
            "path": info["path"],
            "label": rel,
            "model_count": n_models,
            "date": date_str,
            "newest": info["newest"],
        })

    # Sort by most recent first
    results.sort(key=lambda r: r["newest"], reverse=True)
    return results


def _discover_models(model_dir: str = "models/reverse_curriculum") -> dict:
    """Scan model directory and return available models organized by type."""
    result = {
        "best": None,
        "final": None,
        "phase_checkpoints": {},  # phase -> path
        "step_checkpoints": {},  # steps -> path
    }
    model_path = Path(model_dir)
    if not model_path.exists():
        return result

    # Best model
    best = model_path / "best" / "best_model.zip"
    if best.exists():
        result["best"] = str(best)

    # Final model - match any *_final.zip
    for f in model_path.glob("*_final.zip"):
        result["final"] = str(f)
        break

    # Phase checkpoints
    for f in model_path.glob("phase_*_checkpoint.zip"):
        m = re.match(r"phase_(\d+)_checkpoint\.zip", f.name)
        if m:
            phase = int(m.group(1))
            result["phase_checkpoints"][phase] = str(f)

    # Step checkpoints - match any *_DIGITS_steps.zip pattern
    for f in model_path.glob("*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", f.name)
        if m:
            steps = int(m.group(1))
            result["step_checkpoints"][steps] = str(f)

    return result


def _select_training_run(base_dir: str = "models") -> Optional[str]:
    """Interactive training run selection. Returns chosen directory path or None."""
    runs = _discover_training_runs(base_dir)

    if not runs:
        click.echo(click.style(f"\n  No models found in {base_dir}/", fg="red"))
        click.echo("  Train first with: python examples/train_reverse_curriculum.py")
        return None

    # If there's only one run, use it directly
    if len(runs) == 1:
        click.echo(f"\n  Using training: {click.style(runs[0]['label'], fg='cyan')}")
        return runs[0]["path"]

    click.echo(click.style("\n  Available training runs:", bold=True))
    click.echo()
    for i, run in enumerate(runs):
        label = click.style(run["label"], fg="cyan")
        click.echo(f"    {i:>2}) {label}  ({run['model_count']} models, latest {run['date']})")

    click.echo()
    try:
        idx = click.prompt("  Select training run #", type=int, default=0)
        if idx < 0 or idx >= len(runs):
            click.echo(click.style("  Invalid selection.", fg="red"))
            return None
        return runs[idx]["path"]
    except (ValueError, KeyboardInterrupt):
        return None


def _select_model(model_dir: str = "models/reverse_curriculum") -> Optional[str]:
    """Interactive model selection menu. Returns chosen model path or None.

    If model_dir is the default, first prompts user to pick a training run.
    """
    # If using default, let user pick a training run first
    if model_dir == "models/reverse_curriculum":
        chosen_dir = _select_training_run()
        if chosen_dir is None:
            return None
        model_dir = chosen_dir

    discovered = _discover_models(model_dir)

    has_models = (
        discovered["best"]
        or discovered["final"]
        or discovered["step_checkpoints"]
        or discovered["phase_checkpoints"]
    )
    if not has_models:
        click.echo(click.style(f"\n  No models found in {model_dir}/", fg="red"))
        click.echo("  Train first with: python examples/train_reverse_curriculum.py")
        return None

    dir_label = os.path.relpath(model_dir, ".")

    # Build menu options
    options = []

    if discovered["best"]:
        options.append({"label": "Best model (EvalCallback)", "path": discovered["best"]})

    if discovered["final"]:
        options.append({"label": "Final model (end of training)", "path": discovered["final"]})

    # Phase checkpoints
    phase_checkpoints = discovered["phase_checkpoints"]
    if phase_checkpoints:
        for phase in sorted(phase_checkpoints.keys()):
            options.append({"label": f"Phase {phase} checkpoint", "path": phase_checkpoints[phase]})

    # Sample ~5 step checkpoints evenly distributed
    step_checkpoints = discovered["step_checkpoints"]
    if step_checkpoints:
        all_steps = sorted(step_checkpoints.keys())
        max_steps = max(all_steps)

        if len(all_steps) <= 5:
            # Show all if 5 or fewer
            sampled_steps = all_steps
        else:
            # Sample ~5 evenly spaced checkpoints
            interval = max_steps // 5
            target_steps = [interval * i for i in range(1, 6)]
            sampled_steps = []
            for target in target_steps:
                # Find closest available step
                closest = min(all_steps, key=lambda s: abs(s - target))
                if closest not in sampled_steps:
                    sampled_steps.append(closest)
            sampled_steps.sort()

        for steps in sampled_steps:
            options.append({"label": f"{steps:,} steps", "path": step_checkpoints[steps]})

        # Add "other" option if there are more checkpoints
        if len(all_steps) > len(sampled_steps):
            options.append({"label": "Enter specific step count...", "path": "__custom__"})

    click.echo(click.style(f"\n  Models in {dir_label}:", bold=True))
    click.echo()
    for i, opt in enumerate(options):
        click.echo(f"    {i:>2}) {opt['label']}")

    click.echo()
    try:
        idx = click.prompt("  Select agent #", type=int, default=0)
        if idx < 0 or idx >= len(options):
            click.echo(click.style("  Invalid selection.", fg="red"))
            return None

        selected = options[idx]

        if selected["path"] == "__custom__":
            # Let user enter specific step count
            all_steps = sorted(step_checkpoints.keys())
            click.echo(f"\n  Available steps: {', '.join(f'{s:,}' for s in all_steps)}")
            try:
                desired = click.prompt("  Enter step count", type=int)
                if desired in step_checkpoints:
                    return step_checkpoints[desired]
                else:
                    # Find closest
                    closest = min(all_steps, key=lambda s: abs(s - desired))
                    click.echo(f"  Step {desired:,} not found, using closest: {closest:,}")
                    return step_checkpoints[closest]
            except (ValueError, KeyboardInterrupt):
                return None

        return selected["path"]
    except (ValueError, KeyboardInterrupt):
        return None


def _agent_build_board(
    player_board: Board,
    model,
    uses_masks: bool,
    deterministic: bool = True,
    board_size: int = 2,
    board_library_path: str = "new_boards_2.json",
    stage1_model_path: str = "models/construction/best/best_model.zip",
    max_retries: int = 5,
) -> Board:
    """Have the agent build a counter-board against the player's board.

    Retries up to max_retries times if the agent produces an invalid board.
    """
    env = ReverseCurriculumBuilderEnv(
        board_size=board_size,
        board_library_path=board_library_path,
        stage1_model_path=stage1_model_path if Path(stage1_model_path).exists() else None,
        curriculum_phase=10,  # Full construction
        opponent_strategy="random",
        show_opponent_board=True,
    )

    # Override opponent selection so Stage 1 picks a counter to the PLAYER's board
    env._select_opponent_board = lambda: player_board

    best_board = None
    for attempt in range(max_retries):
        seed = 42 + attempt if deterministic else random.randint(0, 100000)
        obs, info = env.reset(seed=seed)

        done = False
        while not done:
            if uses_masks:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        board = env._construct_board_from_state()
        if is_board_playable(board):
            env.close()
            if attempt > 0:
                click.echo(click.style(f"  (valid board on attempt {attempt + 1})", fg="cyan"))
            return board
        best_board = board

    env.close()
    # All retries failed - return the last attempt so caller can show it as invalid
    click.echo(click.style(f"  (no valid board after {max_retries} attempts)", fg="red"))
    return best_board


def _display_board_simple(board: Board, title: str):
    """Display a board with a simple grid using CLI's render function."""
    click.echo(_render_board(board, title))


def _validate_and_display(board: Board, label: str) -> bool:
    """Validate a board and display result. Returns True if valid."""
    is_valid = is_board_playable(board)
    if is_valid:
        click.echo(click.style(f"  {label} board: Valid", fg="green"))
    else:
        click.echo(click.style(f"  {label} board: INVALID", fg="red", bold=True))
    return is_valid


def play(
    board_size: int = 2,
    model_path: Optional[str] = None,
    stage1_model_path: str = "models/construction/best/best_model.zip",
    board_library_path: str = "new_boards_2.json",
    deterministic: bool = True,
):
    """Main play loop."""
    click.echo(click.style("\n" + "=" * 50, bold=True))
    click.echo(click.style("  SPACES GAME - Play vs Agent", bold=True))
    click.echo(click.style("=" * 50, bold=True))
    click.echo(f"\n  Board size: {board_size}x{board_size}")
    mode_str = "Deterministic" if deterministic else "Stochastic"
    mode_color = "cyan" if deterministic else "magenta"
    toggle_flag = "--stochastic" if deterministic else "--deterministic"
    click.echo(f"  Mode: {click.style(mode_str, fg=mode_color)} (use {toggle_flag} to change)")

    # Select model if not provided via --model
    if model_path is None:
        model_path = _select_model()
        if model_path is None:
            sys.exit(1)
    elif not Path(model_path).exists():
        click.echo(click.style(f"\n  Model not found: {model_path}", fg="red"))
        click.echo("  Train first with: python examples/train_reverse_curriculum.py")
        sys.exit(1)

    click.echo(f"\n  Agent: {model_path}")
    click.echo("  Loading agent model...")
    model, uses_masks = _load_agent(model_path)
    model_board_size = _get_model_board_size(model)
    if model_board_size:
        size_color = "green" if model_board_size == board_size else "red"
        click.echo(f"  Agent loaded! (trained on size {click.style(str(model_board_size), fg=size_color)} boards)")
        if model_board_size != board_size:
            click.echo(click.style(
                f"  WARNING: Model expects size {model_board_size} but game is size {board_size}. "
                f"Use --size {model_board_size} or pick a different model.",
                fg="red", bold=True,
            ))
    else:
        click.echo(click.style("  Agent loaded!", fg="green"))

    while True:
        click.echo(click.style("\n" + "-" * 50, bold=True))
        click.echo(click.style("  NEW GAME", bold=True))
        click.echo(click.style("-" * 50, bold=True))

        # Choose mode
        click.echo("\n  How do you want to play?")
        click.echo("    1) Build a board interactively")
        click.echo("    2) Pick a board from the library")
        click.echo("    3) Switch agent")
        click.echo("    q) Quit")

        choice = click.prompt("\n  Choice", type=str, default="1")

        if choice.lower() in ("q", "quit"):
            click.echo("\nGoodbye!")
            break

        if choice == "3":
            new_path = _select_model()
            if new_path is not None:
                click.echo(f"\n  Loading {new_path}...")
                model, uses_masks = _load_agent(new_path)
                model_path = new_path
                click.echo(click.style("  Agent switched!", fg="green"))
            continue

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

        # Validate player's board
        click.echo()
        player_valid = _validate_and_display(player_board, "Your")
        if not player_valid:
            click.echo(click.style("  Cannot play with invalid board.", fg="red"))
            continue

        # Agent builds counter-board
        click.echo(click.style("\n  Agent is building a counter-board...", fg="yellow"))
        try:
            agent_board = _agent_build_board(
                player_board, model, uses_masks,
                deterministic=deterministic,
                board_size=board_size,
                board_library_path=board_library_path,
                stage1_model_path=stage1_model_path,
            )
        except ValueError as e:
            if "observation shape" in str(e).lower() or "unexpected observation" in str(e).lower():
                click.echo(click.style(
                    f"\n  Board size mismatch: model was trained on a different size.",
                    fg="red", bold=True,
                ))
                if model_board_size:
                    click.echo(f"  Use --size {model_board_size} or pick a model trained on size {board_size}.")
                continue
            raise

        # Validate agent's board
        agent_valid = _validate_and_display(agent_board, "Agent")

        if not agent_valid:
            click.echo(click.style("\n  Agent produced an invalid board!", fg="red", bold=True))
            click.echo("\n" + _render_board(agent_board, "Agent's Invalid Board"))
            click.echo(click.style("\n  You win by default (agent failed to build valid board)", fg="green", bold=True))
            continue

        # Simulate using CLI's detailed output: player is "player", agent is "opponent"
        result = simulate_round(1, player_board, agent_board, silent=True)
        _render_result_details(result, fog_of_war=False)

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
        "--model", type=str, default=None,
        help="Path to trained model (interactive selection if omitted)",
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
    parser.add_argument(
        "-d", "--deterministic", action="store_true",
        help="Agent always plays the same response (default)",
    )
    parser.add_argument(
        "-s", "--stochastic", action="store_true",
        help="Agent samples from policy (varied responses)",
    )

    args = parser.parse_args()

    # Determine mode: stochastic flag overrides deterministic default
    deterministic = not args.stochastic

    play(
        board_size=args.size,
        model_path=args.model,
        stage1_model_path=args.stage1_model,
        board_library_path=args.board_library,
        deterministic=deterministic,
    )
