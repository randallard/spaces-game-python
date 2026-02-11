"""
Play against the trained agent - single round or 5-round match.

Single round: You build a board, agent builds a counter-board, simulate.
5-round game: Both build blind each round, scores accumulate, best total wins.

Usage:
    python examples/play_against_agent.py
    python examples/play_against_agent.py --size 2
    python examples/play_against_agent.py --size 2 --rounds 5
    python examples/play_against_agent.py --model models/size2/stage3/ppo_stage3_final.zip --rounds 5
    python examples/play_against_agent.py --rounds 5 --fog
"""

import os
import random
import re
import sys
from datetime import datetime
from typing import List, Optional

import click
import numpy as np
from pathlib import Path

from spaces_game import ReverseCurriculumBuilderEnv, SimultaneousPlayEnv
from spaces_game.interactive_builder import build_board_interactive, _render_board_state
from spaces_game.board_loader import load_boards_from_json
from spaces_game.simulation import simulate_round, _rotate_position
from spaces_game.validation import is_board_playable
from spaces_game.types import Board, Position
from spaces_game.cli import _render_result_details, _render_board


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

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


def _is_stage3_model(model) -> bool:
    """Check if model was trained with SimultaneousPlayEnv (has opponent_history)."""
    try:
        return 'opponent_history' in model.observation_space.spaces
    except Exception:
        return False


def _load_agent(model_path: str):
    """Load the trained agent model."""
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(model_path), True
    except Exception:
        pass
    from stable_baselines3 import PPO
    return PPO.load(model_path), False


# ---------------------------------------------------------------------------
# Model / training-run discovery & selection
# ---------------------------------------------------------------------------

def _discover_training_runs(base_dir: str = "models") -> list:
    """Scan for all directories containing model .zip files."""
    base = Path(base_dir)
    if not base.exists():
        return []

    runs = {}
    for zip_file in base.rglob("*.zip"):
        parent = zip_file.parent
        if parent.name == "best":
            parent = parent.parent
        key = str(parent)
        if key not in runs:
            runs[key] = {"path": key, "zips": [], "newest": 0.0}
        runs[key]["zips"].append(zip_file)
        mtime = zip_file.stat().st_mtime
        if mtime > runs[key]["newest"]:
            runs[key]["newest"] = mtime

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

    results.sort(key=lambda r: r["newest"], reverse=True)
    return results


def _discover_models(model_dir: str = "models/reverse_curriculum") -> dict:
    """Scan model directory and return available models organized by type."""
    result = {
        "best": None,
        "final": None,
        "phase_checkpoints": {},
        "step_checkpoints": {},
    }
    model_path = Path(model_dir)
    if not model_path.exists():
        return result

    best = model_path / "best" / "best_model.zip"
    if best.exists():
        result["best"] = str(best)

    for f in model_path.glob("*_final.zip"):
        result["final"] = str(f)
        break

    for f in model_path.glob("phase_*_checkpoint.zip"):
        m = re.match(r"phase_(\d+)_checkpoint\.zip", f.name)
        if m:
            result["phase_checkpoints"][int(m.group(1))] = str(f)

    for f in model_path.glob("*_steps.zip"):
        m = re.search(r"_(\d+)_steps\.zip$", f.name)
        if m:
            result["step_checkpoints"][int(m.group(1))] = str(f)

    return result


def _select_training_run(base_dir: str = "models") -> Optional[str]:
    """Interactive training run selection."""
    runs = _discover_training_runs(base_dir)
    if not runs:
        click.echo(click.style(f"\n  No models found in {base_dir}/", fg="red"))
        click.echo("  Train first with: python examples/train_reverse_curriculum.py")
        return None

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
    """Interactive model selection menu."""
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
    options = []

    def _model_date(path: str) -> str:
        """Get modification date string for a model file."""
        try:
            mtime = os.path.getmtime(path)
            from datetime import datetime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return ""

    if discovered["best"]:
        date = _model_date(discovered["best"])
        options.append({"label": f"Best model (EvalCallback)  {date}", "path": discovered["best"]})
    if discovered["final"]:
        date = _model_date(discovered["final"])
        options.append({"label": f"Final model (end of training)  {date}", "path": discovered["final"]})

    for phase in sorted(discovered["phase_checkpoints"].keys()):
        path = discovered["phase_checkpoints"][phase]
        date = _model_date(path)
        options.append({"label": f"Phase {phase} checkpoint  {date}", "path": path})

    step_checkpoints = discovered["step_checkpoints"]
    if step_checkpoints:
        all_steps = sorted(step_checkpoints.keys())
        max_steps = max(all_steps)
        if len(all_steps) <= 5:
            sampled_steps = all_steps
        else:
            interval = max_steps // 5
            target_steps = [interval * i for i in range(1, 6)]
            sampled_steps = []
            for target in target_steps:
                closest = min(all_steps, key=lambda s: abs(s - target))
                if closest not in sampled_steps:
                    sampled_steps.append(closest)
            sampled_steps.sort()

        for steps in sampled_steps:
            date = _model_date(step_checkpoints[steps])
            options.append({"label": f"{steps:,} steps  {date}", "path": step_checkpoints[steps]})

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
            all_steps = sorted(step_checkpoints.keys())
            click.echo(f"\n  Available steps: {', '.join(f'{s:,}' for s in all_steps)}")
            try:
                desired = click.prompt("  Enter step count", type=int)
                if desired in step_checkpoints:
                    return step_checkpoints[desired]
                closest = min(all_steps, key=lambda s: abs(s - desired))
                click.echo(f"  Step {desired:,} not found, using closest: {closest:,}")
                return step_checkpoints[closest]
            except (ValueError, KeyboardInterrupt):
                return None

        return selected["path"]
    except (ValueError, KeyboardInterrupt):
        return None


# ---------------------------------------------------------------------------
# Opponent pool discovery
# ---------------------------------------------------------------------------

def _discover_opponent_pools(board_size: int) -> List[str]:
    """Auto-discover opponent board pools in boards/size{N}/."""
    pools = []
    base = Path(f"boards/size{board_size}")
    if base.exists():
        for f in sorted(base.glob("*.json")):
            pools.append(str(f))
    return pools


# ---------------------------------------------------------------------------
# Board building helpers
# ---------------------------------------------------------------------------

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
    """Have a Stage 2 agent build a counter-board (sees player's board)."""
    env = ReverseCurriculumBuilderEnv(
        board_size=board_size,
        board_library_path=board_library_path,
        stage1_model_path=stage1_model_path if Path(stage1_model_path).exists() else None,
        curriculum_phase=10,
        opponent_strategy="random",
        show_opponent_board=True,
    )
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
    click.echo(click.style(f"  (no valid board after {max_retries} attempts)", fg="red"))
    return best_board


def _agent_build_board_blind(
    model,
    uses_masks: bool,
    board_size: int,
    round_num: int,
    agent_score: float,
    opponent_score: float,
    opponent_history_grids: np.ndarray,
    opponent_pools: List[str],
    deterministic: bool = True,
    max_retries: int = 5,
) -> Board:
    """Have a Stage 3 agent build a board blind (simultaneous play).

    Manually drives construction using the SimultaneousPlayEnv's observation
    space and action masking, but does NOT simulate or advance rounds.
    """
    max_steps = board_size * 10
    env = SimultaneousPlayEnv(
        board_size=board_size,
        opponent_pools=opponent_pools,
        max_construction_steps=max_steps,
    )

    for attempt in range(max_retries):
        seed = 42 + attempt if deterministic else random.randint(0, 100000)
        env.reset(seed=seed)

        # Override game state to match current round context
        env.current_round = round_num
        env.agent_total_score = agent_score
        env.opponent_total_score = opponent_score
        env.opponent_history_grids = opponent_history_grids.copy()

        obs = env._get_observation()

        for _ in range(max_steps):
            if uses_masks:
                masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            # Check done flag before calling step (avoid _finish_round)
            if int(action[2]) > 0:
                break

            # Apply construction step manually
            cell = int(action[0]) % (board_size * board_size)
            piece_or_trap = int(action[1]) % 2
            row = cell // board_size
            col = cell % board_size
            move_type = "piece" if piece_or_trap == 0 else "trap"
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
                    env.piece_visited_positions.add(f"{row},{col}")
                    env.current_piece_position = Position(row=row, col=col)
                elif move_type == "trap":
                    env.building_grid[row, col, 1] = order
                    env.trap_positions.add(f"{row},{col}")
                    if (env.current_piece_position is not None and
                            row == env.current_piece_position.row and
                            col == env.current_piece_position.col):
                        env.supermove_active = True
                        env.supermove_position = Position(row=row, col=col)
                env.construction_step += 1

            obs = env._get_observation()

        board = env._construct_board_from_state()
        if is_board_playable(board):
            env.close()
            if attempt > 0:
                click.echo(click.style(f"  (valid board on attempt {attempt + 1})", fg="cyan"))
            return board

    env.close()
    click.echo(click.style(f"  (no valid board after {max_retries} attempts)", fg="red"))
    return board


def _encode_board_for_agent(board: Board, board_size: int) -> np.ndarray:
    """Encode a board rotated 180 degrees (opponent's perspective for agent)."""
    grid = np.zeros((board_size, board_size, 2), dtype=np.int32)
    for move in board.sequence:
        if move.type == "final":
            continue
        rot = _rotate_position(move.position.row, move.position.col, board_size)
        if rot.row < 0 or rot.row >= board_size or rot.col < 0 or rot.col >= board_size:
            continue
        channel = 0 if move.type == "piece" else 1
        grid[rot.row][rot.col][channel] = move.order
    return grid


# ---------------------------------------------------------------------------
# Board selection for a single round (shared by single-round and multi-round)
# ---------------------------------------------------------------------------

def _get_player_board(
    board_size: int,
    board_library_path: str,
    prompt_text: str = "  Choice",
) -> Optional[Board]:
    """Prompt the player to build or pick a board. Returns Board or None."""
    click.echo("\n  How to build your board?")
    click.echo("    1) Build interactively")
    click.echo("    2) Pick from library")

    choice = click.prompt(prompt_text, type=str, default="1")

    if choice == "1":
        click.echo(click.style("\n  Build your board!\n", fg="cyan", bold=True))
        board = build_board_interactive(size=board_size)
        if board is None:
            click.echo("  Board building cancelled.")
            return None
        return board

    elif choice == "2":
        try:
            boards = load_boards_from_json(board_library_path)
        except Exception as e:
            click.echo(click.style(f"  Could not load library: {e}", fg="red"))
            return None

        click.echo(f"\n  Available boards (from {board_library_path}):\n")
        for i, b in enumerate(boards):
            n_pieces = sum(1 for m in b.sequence if m.type == "piece")
            n_traps = sum(1 for m in b.sequence if m.type == "trap")
            click.echo(f"    {i}: {n_pieces} pieces, {n_traps} traps, {len(b.sequence)} total moves")

        try:
            idx = click.prompt("\n  Select board #", type=int, default=0)
            if idx < 0 or idx >= len(boards):
                click.echo(click.style("  Invalid selection.", fg="red"))
                return None
            return boards[idx]
        except (ValueError, KeyboardInterrupt):
            return None

    click.echo("  Invalid choice.")
    return None


def _validate_and_display(board: Board, label: str) -> bool:
    """Validate a board and display result."""
    is_valid = is_board_playable(board)
    if is_valid:
        click.echo(click.style(f"  {label} board: Valid", fg="green"))
    else:
        click.echo(click.style(f"  {label} board: INVALID", fg="red", bold=True))
    return is_valid


# ---------------------------------------------------------------------------
# Multi-round game
# ---------------------------------------------------------------------------

def _play_multi_round(
    model,
    uses_masks: bool,
    is_stage3: bool,
    board_size: int,
    board_library_path: str,
    stage1_model_path: str,
    opponent_pools: List[str],
    deterministic: bool = True,
    rounds: int = 5,
    fog_of_war: bool = False,
):
    """Play a multi-round game against the agent."""
    play_mode = "simultaneous (blind)" if is_stage3 else "counter-play (agent sees your board)"
    click.echo(click.style(f"\n  {rounds}-ROUND GAME", bold=True))
    click.echo(f"  Mode: {play_mode}")
    if fog_of_war:
        click.echo(f"  Fog of war: {click.style('ON', fg='yellow')} (opponent traps hidden unless you hit them)")
    click.echo()

    player_total = 0
    agent_total = 0

    # For Stage 3: track opponent history (human's boards encoded for agent)
    opponent_history_grids = np.zeros(
        (5, board_size, board_size, 2), dtype=np.int32,
    )

    round_results = []

    for round_num in range(rounds):
        click.echo(click.style(f"\n{'=' * 50}", bold=True))
        click.echo(click.style(f"  ROUND {round_num + 1} of {rounds}", bold=True))
        click.echo(click.style(f"{'=' * 50}", bold=True))
        if round_num > 0:
            click.echo(f"  Score: You {player_total} - Agent {agent_total}")

        # Human builds/picks board
        player_board = _get_player_board(board_size, board_library_path)
        if player_board is None:
            click.echo(click.style("  Skipping round (no board).", fg="yellow"))
            agent_total += 5  # forfeit
            round_results.append(("forfeit", 0, 5))
            continue

        # Validate player's board
        click.echo()
        if not _validate_and_display(player_board, "Your"):
            click.echo(click.style("  Cannot play with invalid board - round forfeited.", fg="red"))
            agent_total += 5
            round_results.append(("forfeit", 0, 5))
            continue

        # Agent builds board
        if is_stage3:
            click.echo(click.style("\n  Agent is building a board (blind)...", fg="yellow"))
            try:
                agent_board = _agent_build_board_blind(
                    model, uses_masks, board_size,
                    round_num=round_num,
                    agent_score=agent_total,
                    opponent_score=player_total,
                    opponent_history_grids=opponent_history_grids,
                    opponent_pools=opponent_pools,
                    deterministic=deterministic,
                )
            except ValueError as e:
                if "observation shape" in str(e).lower():
                    click.echo(click.style(
                        "\n  Board size mismatch: model was trained on a different size.",
                        fg="red", bold=True,
                    ))
                    return
                raise
        else:
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
                if "observation shape" in str(e).lower():
                    click.echo(click.style(
                        "\n  Board size mismatch: model was trained on a different size.",
                        fg="red", bold=True,
                    ))
                    return
                raise

        # Validate agent's board
        agent_valid = _validate_and_display(agent_board, "Agent")

        if not agent_valid:
            click.echo(click.style("  Agent produced an invalid board!", fg="red", bold=True))
            click.echo("\n" + _render_board(agent_board, "Agent's Invalid Board"))
            click.echo(click.style("  You win this round by default! (+5 points)", fg="green", bold=True))
            player_total += 5
            round_results.append(("player_default", 5, 0))
        else:
            # Simulate: human is "player", agent is "opponent"
            result = simulate_round(round_num + 1, player_board, agent_board, silent=True)
            _render_result_details(result, fog_of_war=fog_of_war)

            player_total += result.playerPoints
            agent_total += result.opponentPoints
            round_results.append((result.winner, result.playerPoints, result.opponentPoints))

        # Encode human's board into opponent_history for agent's next round
        if is_stage3 and round_num < 5:
            opponent_history_grids[round_num] = _encode_board_for_agent(
                player_board, board_size,
            )

        # Round summary
        click.echo(click.style(
            f"\n  Game Score: You {player_total} - Agent {agent_total}",
            bold=True,
        ))

        # Pause between rounds (except last)
        if round_num < rounds - 1:
            try:
                click.prompt("  Press Enter for next round", default="", show_default=False)
            except (KeyboardInterrupt, EOFError):
                click.echo("\n  Game aborted.")
                return

    # Final result
    click.echo(click.style(f"\n{'=' * 50}", bold=True))
    click.echo(click.style("  GAME RESULT", bold=True))
    click.echo(click.style(f"{'=' * 50}", bold=True))

    # Round-by-round recap
    click.echo()
    for i, (winner, p_pts, a_pts) in enumerate(round_results):
        marker = {
            "player": click.style(">>", fg="green"),
            "player_default": click.style(">>", fg="green"),
            "opponent": click.style("<<", fg="red"),
            "tie": click.style("==", fg="yellow"),
            "forfeit": click.style("FF", fg="red"),
        }.get(winner, "  ")
        click.echo(f"  Round {i+1}: You {p_pts:2d} {marker} {a_pts:2d} Agent")

    click.echo(click.style(f"\n  Final: You {player_total} - Agent {agent_total}", bold=True))

    if player_total > agent_total:
        click.echo(click.style("\n  YOU WIN THE GAME!", fg="green", bold=True))
    elif player_total < agent_total:
        click.echo(click.style("\n  AGENT WINS THE GAME!", fg="red", bold=True))
    else:
        click.echo(click.style("\n  TIE GAME!", fg="yellow", bold=True))

    click.echo(click.style(f"{'=' * 50}\n", bold=True))


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

def play(
    board_size: int = 2,
    model_path: Optional[str] = None,
    stage1_model_path: str = "models/construction/best/best_model.zip",
    board_library_path: str = "new_boards_2.json",
    deterministic: bool = True,
    rounds: int = 1,
    fog_of_war: bool = False,
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

    # Select model if not provided via --model or --difficulty
    if model_path is None:
        # Check for difficulty models first
        diff_dir = Path(f"models/size{board_size}/stage3/difficulty")
        available_difficulties = []
        if diff_dir.exists():
            for level in ["beginner", "intermediate", "expert"]:
                if (diff_dir / f"{level}.zip").exists():
                    available_difficulties.append(level)

        if available_difficulties:
            click.echo(click.style("\n  Difficulty levels available:", bold=True))
            click.echo()
            for i, level in enumerate(available_difficulties):
                click.echo(f"    {i}) {level.capitalize()}")
            click.echo(f"    {len(available_difficulties)}) Choose specific model instead...")
            click.echo()
            try:
                idx = click.prompt("  Select difficulty #", type=int, default=0)
                if 0 <= idx < len(available_difficulties):
                    model_path = str(diff_dir / f"{available_difficulties[idx]}.zip")
                else:
                    model_path = _select_model()
            except (ValueError, KeyboardInterrupt):
                model_path = _select_model()
        else:
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
    is_stage3 = _is_stage3_model(model)

    if model_board_size:
        size_color = "green" if model_board_size == board_size else "red"
        stage_label = "Stage 3 simultaneous" if is_stage3 else "Stage 2 construction"
        click.echo(
            f"  Agent loaded! ({stage_label}, "
            f"size {click.style(str(model_board_size), fg=size_color)} boards)"
        )
        if model_board_size != board_size:
            click.echo(click.style(
                f"  WARNING: Model expects size {model_board_size} but game is size {board_size}. "
                f"Use --size {model_board_size} or pick a different model.",
                fg="red", bold=True,
            ))
    else:
        click.echo(click.style("  Agent loaded!", fg="green"))

    # Discover opponent pools for Stage 3
    opponent_pools = _discover_opponent_pools(board_size)
    if is_stage3 and not opponent_pools:
        click.echo(click.style(
            f"  Warning: No opponent pools in boards/size{board_size}/. "
            f"Using board library as fallback.",
            fg="yellow",
        ))
        opponent_pools = [board_library_path]

    while True:
        click.echo(click.style("\n" + "-" * 50, bold=True))
        click.echo(click.style("  NEW GAME", bold=True))
        click.echo(click.style("-" * 50, bold=True))

        # Choose mode
        click.echo("\n  How do you want to play?")
        click.echo("    1) Build a board (single round)")
        click.echo("    2) Pick a board from library (single round)")
        click.echo(f"    3) {rounds}-round game")
        click.echo("    4) Switch agent")
        click.echo("    q) Quit")

        choice = click.prompt("\n  Choice", type=str, default="3" if rounds > 1 else "1")

        if choice.lower() in ("q", "quit"):
            click.echo("\nGoodbye!")
            break

        if choice == "4":
            new_path = _select_model()
            if new_path is not None:
                click.echo(f"\n  Loading {new_path}...")
                model, uses_masks = _load_agent(new_path)
                model_path = new_path
                is_stage3 = _is_stage3_model(model)
                stage_label = "Stage 3 simultaneous" if is_stage3 else "Stage 2 construction"
                click.echo(click.style(f"  Agent switched! ({stage_label})", fg="green"))
                opponent_pools = _discover_opponent_pools(board_size)
                if is_stage3 and not opponent_pools:
                    opponent_pools = [board_library_path]
            continue

        # --- Multi-round game ---
        if choice == "3":
            _play_multi_round(
                model, uses_masks, is_stage3, board_size,
                board_library_path, stage1_model_path, opponent_pools,
                deterministic=deterministic,
                rounds=rounds,
                fog_of_war=fog_of_war,
            )
            continue

        # --- Single round ---
        player_board = None

        if choice == "1":
            click.echo(click.style("\n  Build your board!\n", fg="cyan", bold=True))
            player_board = build_board_interactive(size=board_size)
            if player_board is None:
                click.echo("  Board building cancelled.")
                continue
        elif choice == "2":
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
        if not _validate_and_display(player_board, "Your"):
            click.echo(click.style("  Cannot play with invalid board.", fg="red"))
            continue

        # Agent builds board (blind for Stage 3, counter for Stage 2)
        if is_stage3:
            click.echo(click.style("\n  Agent is building a board (blind)...", fg="yellow"))
            try:
                agent_board = _agent_build_board_blind(
                    model, uses_masks, board_size,
                    round_num=0, agent_score=0, opponent_score=0,
                    opponent_history_grids=np.zeros(
                        (5, board_size, board_size, 2), dtype=np.int32,
                    ),
                    opponent_pools=opponent_pools,
                    deterministic=deterministic,
                )
            except ValueError as e:
                if "observation shape" in str(e).lower():
                    click.echo(click.style(
                        "\n  Board size mismatch: model was trained on a different size.",
                        fg="red", bold=True,
                    ))
                    continue
                raise
        else:
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
                if "observation shape" in str(e).lower():
                    click.echo(click.style(
                        "\n  Board size mismatch: model was trained on a different size.",
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

        # Simulate
        result = simulate_round(1, player_board, agent_board, silent=True)
        _render_result_details(result, fog_of_war=fog_of_war)

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
    parser.add_argument(
        "--rounds", type=int, default=5,
        help="Number of rounds for multi-round game (default: 5)",
    )
    parser.add_argument(
        "--fog", action="store_true",
        help="Enable fog of war (hide opponent traps you didn't hit)",
    )
    parser.add_argument(
        "--difficulty", type=str, default=None,
        choices=["beginner", "intermediate", "expert"],
        help="Difficulty level (auto-selects model from models/size{N}/stage3/difficulty/)",
    )

    args = parser.parse_args()
    deterministic = not args.stochastic

    # Resolve difficulty to model path
    model_path = args.model
    if args.difficulty and model_path is None:
        model_path = f"models/size{args.size}/stage3/difficulty/{args.difficulty}.zip"

    play(
        board_size=args.size,
        model_path=model_path,
        stage1_model_path=args.stage1_model,
        board_library_path=args.board_library,
        deterministic=deterministic,
        rounds=args.rounds,
        fog_of_war=args.fog,
    )
