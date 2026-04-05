"""
Commentary Lab — Run games between agents and collect observations for building
the coach/commentator knowledge base.

Two agents play a 5-round game. After each round, the boards are displayed in
ASCII with full analysis, and you can type notes. Everything is saved to a
session file for later reference.

Usage:
    # Scripted vs scripted (no model loading)
    python examples/commentary_lab.py --size 3 --agent-a scripted_1 --agent-b scripted_3

    # Play as Agent A against a scripted opponent
    python examples/commentary_lab.py --size 3 --agent-a human --agent-b scripted_3

    # Play against an RL model
    python examples/commentary_lab.py --size 3 --agent-a human --agent-b expert

    # RL model vs scripted
    python examples/commentary_lab.py --size 3 --agent-a intermediate --agent-b scripted_4

    # RL vs RL
    python examples/commentary_lab.py --size 3 --agent-a beginner --agent-b expert

Available agents: human, beginner, intermediate, expert, scripted_1 through scripted_5
"""

import argparse
import click
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from spaces_game.types import Board, BoardMove, Position, RoundResult
from spaces_game.simulation import simulate_round, _rotate_position
from spaces_game.validation import is_board_playable
from spaces_game.interactive_builder import build_board_interactive as _build_interactive, _render_board_state
from spaces_game.board_loader import load_boards_from_json
from inference_server.scripted_agents import scripted_board, build_simple_board
from inference_server.inference import (
    load_agent,
    build_board_for_round,
    encode_board_for_agent,
    discover_opponent_pools,
)


# ── Board rendering ──────────────────────────────────────────────────────────

def render_board_styled(
    board: Board,
    label: str,
    rotated: bool = False,
    fog: bool = False,
    sprung_trap_pos: Optional[Position] = None,
) -> str:
    """Render a board using the builder-style display (colored step numbers).

    Uses the same visual style as the interactive builder: 1●, 2X, 3● etc.

    Args:
        board: The board to render.
        label: Display label (e.g. "Agent A (You)").
        rotated: If True, show the board from the opponent's rotated perspective.
        fog: If True, apply fog of war — all pieces visible, traps hidden
             unless the opponent's piece hit them.
        sprung_trap_pos: Under fog, the only trap position to reveal (in the
             board's native coordinates). None hides all traps.

    Returns:
        Multi-line string with the styled board.
    """
    size = board.boardSize
    fog_label = click.style(" [FOG]", fg="yellow") if fog else ""
    lines = []
    lines.append(click.style(f"\n  {label}", bold=True) + fog_label)

    # Collect visible pieces and traps per cell
    # Key: (display_row, display_col) -> (piece_orders, trap_orders)
    position_contents: dict[tuple[int, int], tuple[list[int], list[int]]] = {}

    for move in board.sequence:
        if move.type == "final":
            continue

        # Fog: pieces always visible, traps hidden unless sprung
        if fog and move.type == "trap":
            if sprung_trap_pos is None:
                continue
            if (move.position.row != sprung_trap_pos.row or
                    move.position.col != sprung_trap_pos.col):
                continue

        r, c = move.position.row, move.position.col
        if rotated:
            rot = _rotate_position(r, c, size)
            r, c = rot.row, rot.col
        if not (0 <= r < size and 0 <= c < size):
            continue

        pos = (r, c)
        if pos not in position_contents:
            position_contents[pos] = ([], [])
        pieces, traps = position_contents[pos]
        if move.type == "piece":
            pieces.append(move.order)
        elif move.type == "trap":
            traps.append(move.order)

    # Build grid
    lines.append("  ┌" + "─────────┬" * (size - 1) + "─────────┐")

    for row_idx in range(size):
        row_items = []
        for col_idx in range(size):
            pos = (row_idx, col_idx)
            pieces, traps = position_contents.get(pos, ([], []))

            if pieces and traps:
                # Supermove cell
                piece_str = click.style(f"{pieces[0]}●", fg="blue")
                trap_str = click.style(f"{traps[0]}X", fg="red")
                content = f"{piece_str},{trap_str}"
                visible_len = len(str(pieces[0])) + 1 + 1 + len(str(traps[0])) + 1
                padding = 9 - 1 - visible_len
                cell = f" {content}{' ' * padding}"
            elif pieces:
                num_str = click.style(f"{pieces[0]}●", fg="blue")
                visible_len = len(str(pieces[0])) + 1
                padding = 9 - 2 - visible_len
                cell = f"  {num_str}{' ' * padding}"
            elif traps:
                num_str = click.style(f"{traps[0]}X", fg="red")
                visible_len = len(str(traps[0])) + 1
                padding = 9 - 2 - visible_len
                cell = f"  {num_str}{' ' * padding}"
            elif fog:
                cell = click.style("    ·    ", fg="bright_black")
            else:
                cell = "         "

            row_items.append(cell)

        lines.append("  │" + "│".join(row_items) + "│")
        if row_idx < size - 1:
            lines.append("  ├" + "─────────┼" * (size - 1) + "─────────┤")

    lines.append("  └" + "─────────┴" * (size - 1) + "─────────┘")

    # Footer
    if fog:
        visible_pieces = sum(len(p) for p, _ in position_contents.values())
        max_traps = size - 1
        trap_note = "1 trap revealed" if sprung_trap_pos else f"up to {max_traps} traps hidden"
        lines.append(f"  Visible: {visible_pieces} moves — {trap_note}")
    else:
        trap_count = sum(1 for m in board.sequence if m.type == "trap")
        piece_count = sum(1 for m in board.sequence if m.type == "piece")
        lines.append(f"  {piece_count} pieces, {trap_count} traps")

    return "\n".join(lines)


def render_simulation_trace(
    result: RoundResult,
    fog_a: bool = False,
    fog_b: bool = False,
) -> str:
    """Render a step-by-step simulation trace.

    Each side's sequence is shown up to that player's last step (when they
    got trapped, collided, or reached the goal). No steps shown past that.

    Under fog for a given side:
      - Piece moves shown with coordinates (they're visible on the board)
      - Trap placements shown as "trap placed" (no coordinates) unless the
        opposing piece hit that trap, in which case full position is shown
      - Goal shown normally

    Args:
        fog_a: Apply fog to Agent A's board (hide A's trap positions).
        fog_b: Apply fog to Agent B's board (hide B's trap positions).
    """
    size = result.playerBoard.boardSize
    sd = result.simulationDetails
    lines = ["  Step-by-step:"]

    p_seq = list(result.playerBoard.sequence)
    o_seq = list(result.opponentBoard.sequence)

    # Each side only shows steps up to their last executed step
    p_last = sd.playerLastStep
    o_last = sd.opponentLastStep
    max_steps = max(p_last + 1, o_last + 1)

    # Trap positions from simulation are in rotated coords. Un-rotate to
    # native board coords so we can compare against move.position.
    a_sprung = None  # trap on A's board that B hit (native A coords)
    if sd.opponentHitTrap and sd.opponentTrapPosition:
        a_sprung = _rotate_position(sd.opponentTrapPosition.row, sd.opponentTrapPosition.col, size)
    b_sprung = None  # trap on B's board that A hit (native B coords)
    if sd.playerHitTrap and sd.playerTrapPosition:
        b_sprung = _rotate_position(sd.playerTrapPosition.row, sd.playerTrapPosition.col, size)

    p_pos = None
    o_pos = None

    for step in range(max_steps):
        p_desc = ""
        p_order = ""
        o_desc = ""
        o_order = ""

        # Agent A's move (only up to A's last step)
        if step < len(p_seq) and step <= p_last:
            m = p_seq[step]
            r, c = m.position.row, m.position.col
            p_order = str(m.order) if m.type != "final" else "→"

            if m.type == "piece":
                p_desc = f"{m.order}. piece→({r},{c})"
                p_pos = (r, c)
            elif m.type == "trap":
                if fog_a:
                    if a_sprung and m.position.row == a_sprung.row and \
                       m.position.col == a_sprung.col:
                        p_desc = f"{m.order}. trap@({r},{c})"
                    else:
                        p_desc = f"{m.order}. trap placed"
                else:
                    p_desc = f"{m.order}. trap@({r},{c})"
            elif m.type == "final":
                p_desc = "→GOAL"

        # Agent B's move (only up to B's last step)
        if step < len(o_seq) and step <= o_last:
            m = o_seq[step]
            rot = _rotate_position(m.position.row, m.position.col, size)
            o_order = str(m.order) if m.type != "final" else "→"

            if m.type == "piece":
                o_desc = f"{m.order}. piece→({rot.row},{rot.col})"
                o_pos = (rot.row, rot.col)
            elif m.type == "trap":
                if fog_b:
                    if b_sprung and m.position.row == b_sprung.row and \
                       m.position.col == b_sprung.col:
                        o_desc = f"{m.order}. trap@({rot.row},{rot.col})"
                    else:
                        o_desc = f"{m.order}. trap placed"
                else:
                    o_desc = f"{m.order}. trap@({rot.row},{rot.col})"
            elif m.type == "final":
                o_desc = "→GOAL"

        if p_desc or o_desc:
            lines.append(f"    A: {p_desc or '---':24s}  B: {o_desc or '---'}")

    events = []
    if result.collision:
        events.append(f"COLLISION at {p_pos}")
    if sd.playerHitTrap:
        tp = sd.playerTrapPosition
        events.append(f"A hit trap at ({tp.row},{tp.col})" if tp else "A hit trap")
    if sd.opponentHitTrap:
        tp = sd.opponentTrapPosition
        events.append(f"B hit trap at ({tp.row},{tp.col})" if tp else "B hit trap")

    if events:
        lines.append(f"  Events: {', '.join(events)}")

    return "\n".join(lines)


def render_round_summary(
    round_num: int,
    result: RoundResult,
    running_a: int,
    running_b: int,
) -> str:
    """Render a compact round summary with scores."""
    sd = result.simulationDetails
    winner_str = {
        "player": "Agent A wins",
        "opponent": "Agent B wins",
        "tie": "Tie",
    }[result.winner]

    lines = [
        f"  ╔══ Round {round_num} Result ══╗",
        f"  ║  A: {result.playerPoints} pts ({sd.playerMoves} moves{', trapped' if sd.playerHitTrap else ', goal' if result.playerFinalPosition.row == -1 else ''})",
        f"  ║  B: {result.opponentPoints} pts ({sd.opponentMoves} moves{', trapped' if sd.opponentHitTrap else ', goal' if result.opponentFinalPosition.row == -1 else ''})",
        f"  ║  → {winner_str}",
        f"  ║  Running: A {running_a} - B {running_b}",
        f"  ╚{'═' * 24}╝",
    ]
    return "\n".join(lines)


# ── Human board selection ─────────────────────────────────────────────────────

DEFAULT_BOARD_LIBRARY = "my_boards.json"


def _board_summary(board: Board) -> str:
    """One-line summary of a board."""
    n_pieces = sum(1 for m in board.sequence if m.type == "piece")
    n_traps = sum(1 for m in board.sequence if m.type == "trap")
    # Find starting column
    for m in board.sequence:
        if m.type == "piece":
            start_col = m.position.col
            break
    else:
        start_col = "?"
    return f"{n_pieces} pieces, {n_traps} traps, start col {start_col}"


def _save_board_to_library(board: Board, library_path: str):
    """Append a board to a JSON library file."""
    path = Path(library_path)
    boards_data = []

    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                boards_data = data
            elif isinstance(data, dict) and "boards" in data:
                boards_data = data["boards"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Convert board to dict
    seq = []
    for m in board.sequence:
        seq.append({
            "position": {"row": m.position.row, "col": m.position.col},
            "type": m.type,
            "order": m.order,
        })
    grid = [list(row) for row in board.grid]
    board_dict = {"boardSize": board.boardSize, "grid": grid, "sequence": seq}

    boards_data.append(board_dict)
    with open(path, "w") as f:
        json.dump(boards_data, f, indent=2)

    print(f"  Board saved to {library_path} ({len(boards_data)} total)")


def get_human_board(board_size: int, round_num: int, library_path: str) -> Board:
    """Prompt human to build interactively, pick from library, or save.

    Options:
        1) Build interactively (uses existing interactive_builder)
        2) Pick from library
        3) Reuse last board
    """
    print(f"\n  ── Your board (Round {round_num + 1}) ──")
    print("    1) Build interactively")
    print("    2) Pick from library")

    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "2":
        # Load from library
        lpath = Path(library_path)
        if not lpath.exists():
            # Try boards directory
            alt = Path(f"boards/size{board_size}")
            if alt.exists():
                files = sorted(alt.glob("*.json"))
                if files:
                    print(f"\n  No {library_path} found. Available board pools:")
                    for i, f in enumerate(files):
                        try:
                            boards = load_boards_from_json(str(f))
                            print(f"    {i}: {f.name} ({len(boards)} boards)")
                        except Exception:
                            print(f"    {i}: {f.name} (error loading)")
                    fidx = input("  Pick file # (or Enter for first): ").strip()
                    try:
                        lpath = files[int(fidx)] if fidx else files[0]
                    except (ValueError, IndexError):
                        lpath = files[0]

        if not lpath.exists():
            print(f"  No library found at {lpath}")
            print("  Falling back to interactive builder...")
            return _do_interactive_build(board_size, round_num, library_path)

        try:
            boards = load_boards_from_json(str(lpath))
        except Exception as e:
            print(f"  Error loading library: {e}")
            return _do_interactive_build(board_size, round_num, library_path)

        # Filter to matching board size
        boards = [b for b in boards if b.boardSize == board_size]
        if not boards:
            print(f"  No size-{board_size} boards in {lpath}")
            return _do_interactive_build(board_size, round_num, library_path)

        print(f"\n  Boards from {lpath} ({len(boards)} size-{board_size}):\n")
        for i, b in enumerate(boards):
            print(f"    [{i}] {_board_summary(b)}")
            print(_render_board_state(b, current_position=None, title=f"  Board {i}"))
            print()

        idx = input(f"  Select board # [0]: ").strip()
        try:
            board = boards[int(idx) if idx else 0]
        except (ValueError, IndexError):
            board = boards[0]

        print(f"  Selected: {_board_summary(board)}")
        return board

    # Default: build interactively
    return _do_interactive_build(board_size, round_num, library_path)


def _do_interactive_build(board_size: int, round_num: int, library_path: str) -> Board:
    """Build interactively, then offer to save."""
    board = _build_interactive(size=board_size)
    if board is None:
        # User quit — return a simple straight-path board as fallback
        print("  Build cancelled — using straight path col 0")
        d = build_simple_board(board_size, 0)
        seq = []
        for m in d["sequence"]:
            seq.append(BoardMove(
                position=Position(row=m["position"]["row"], col=m["position"]["col"]),
                type=m["type"],
                order=m["order"],
            ))
        grid = tuple(tuple(r) for r in d["grid"])
        return Board(boardSize=d["boardSize"], grid=grid, sequence=tuple(seq))

    # Offer to save
    save = input("  Save this board to library? (y/N): ").strip().lower()
    if save == "y":
        path = input(f"  File [{library_path}]: ").strip() or library_path
        _save_board_to_library(board, path)

    return board


# ── Agent management ─────────────────────────────────────────────────────────

class Agent:
    """Wraps either a scripted agent, an RL model, or a human player."""

    def __init__(self, spec: str, board_size: int, board_library: str = DEFAULT_BOARD_LIBRARY):
        self.spec = spec
        self.board_size = board_size
        self.board_library = board_library
        self.model = None
        self.uses_masks = False
        self.is_scripted = spec.startswith("scripted_")
        self.is_human = spec == "human"
        self.scripted_level = 0
        self.label = spec

        if self.is_human:
            self.label = "You"
        elif self.is_scripted:
            self.scripted_level = int(spec.split("_")[1])
            self.label = f"Scripted L{self.scripted_level}"
        else:
            # Check if it's a path or a skill level name
            model_path = self._resolve_model_path(spec, board_size)
            print(f"Loading model: {model_path}")
            self.model, self.uses_masks = load_agent(model_path)
            self.label = spec.capitalize()

    def _resolve_model_path(self, spec: str, board_size: int) -> str:
        """Resolve a skill level name or path to a model file."""
        if Path(spec).exists():
            return spec

        # Try standard model directory
        candidates = [
            f"models/size{board_size}/stage4/{spec}.zip",
            f"models/size{board_size}/stage3/{spec}.zip",
        ]
        for c in candidates:
            if Path(c).exists():
                return c

        raise FileNotFoundError(
            f"Could not find model for '{spec}' at size {board_size}. "
            f"Tried: {candidates}"
        )

    def build_board(
        self,
        round_num: int,
        my_score: float,
        opp_score: float,
        opponent_history_grids: np.ndarray,
        opponent_pools: list[str],
        round_scores: list[dict],
        round_history: list | None = None,
    ) -> Board:
        """Build a board for the given round context."""
        if self.is_human:
            return get_human_board(self.board_size, round_num, self.board_library)
        elif self.is_scripted:
            board_dict = scripted_board(
                self.scripted_level,
                self.board_size,
                round_num,
                round_scores,
                round_history=round_history,
            )
            # Convert dict to Board object
            return self._dict_to_board(board_dict)
        else:
            board, attempts = build_board_for_round(
                model=self.model,
                uses_masks=self.uses_masks,
                board_size=self.board_size,
                round_num=round_num,
                agent_score=my_score,
                opponent_score=opp_score,
                opponent_history_grids=opponent_history_grids,
                opponent_pools=opponent_pools,
                deterministic=False,  # Allow variation for interesting games
                use_fog=True,
            )
            if attempts > 1:
                print(f"  (took {attempts} attempts to build valid board)")
            return board

    def _dict_to_board(self, d: dict) -> Board:
        """Convert a scripted_board dict to a Board dataclass."""
        seq = []
        for m in d["sequence"]:
            seq.append(BoardMove(
                position=Position(row=m["position"]["row"], col=m["position"]["col"]),
                type=m["type"],
                order=m["order"],
            ))
        grid = tuple(tuple(r) for r in d["grid"])
        return Board(boardSize=d["boardSize"], grid=grid, sequence=tuple(seq))


# ── Session recording ────────────────────────────────────────────────────────

class Session:
    """Records game data and notes for a lab session."""

    def __init__(self, agent_a_spec: str, agent_b_spec: str, board_size: int):
        self.start_time = datetime.now()
        self.board_size = board_size
        self.agent_a = agent_a_spec
        self.agent_b = agent_b_spec
        self.rounds: list[dict] = []
        self.notes: list[dict] = []
        self.game_notes: str = ""

    def record_round(self, round_num: int, result: RoundResult, running_a: int, running_b: int):
        """Record round data."""
        sd = result.simulationDetails
        self.rounds.append({
            "round": round_num,
            "winner": result.winner,
            "a_points": result.playerPoints,
            "b_points": result.opponentPoints,
            "running_a": running_a,
            "running_b": running_b,
            "a_moves": sd.playerMoves,
            "b_moves": sd.opponentMoves,
            "a_hit_trap": sd.playerHitTrap,
            "b_hit_trap": sd.opponentHitTrap,
            "collision": result.collision,
            "a_trap_count": sum(1 for m in result.playerBoard.sequence if m.type == "trap"),
            "b_trap_count": sum(1 for m in result.opponentBoard.sequence if m.type == "trap"),
            "a_path": [(m.position.row, m.position.col) for m in result.playerBoard.sequence if m.type == "piece"],
            "b_path": [(m.position.row, m.position.col) for m in result.opponentBoard.sequence if m.type == "piece"],
            "a_traps": [(m.position.row, m.position.col) for m in result.playerBoard.sequence if m.type == "trap"],
            "b_traps": [(m.position.row, m.position.col) for m in result.opponentBoard.sequence if m.type == "trap"],
        })

    def record_note(self, round_num: int, note: str):
        """Record a note for a specific round."""
        self.notes.append({
            "round": round_num,
            "note": note,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self, output_dir: str = "lab_sessions"):
        """Save session to JSON and markdown."""
        os.makedirs(output_dir, exist_ok=True)
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")
        # Sanitize agent specs for filenames (paths contain slashes)
        safe_a = self.agent_a.replace("/", "_").replace("\\", "_")
        safe_b = self.agent_b.replace("/", "_").replace("\\", "_")
        basename = f"{ts}_size{self.board_size}_{safe_a}_vs_{safe_b}"

        # JSON data
        data = {
            "start_time": self.start_time.isoformat(),
            "board_size": self.board_size,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "rounds": self.rounds,
            "notes": self.notes,
            "game_notes": self.game_notes,
        }
        json_path = os.path.join(output_dir, f"{basename}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        # Markdown summary
        md_path = os.path.join(output_dir, f"{basename}.md")
        with open(md_path, "w") as f:
            f.write(f"# Commentary Lab: {self.agent_a} vs {self.agent_b}\n\n")
            f.write(f"**Board size**: {self.board_size}x{self.board_size}\n")
            f.write(f"**Date**: {self.start_time.strftime('%Y-%m-%d %H:%M')}\n\n")

            for rd in self.rounds:
                f.write(f"## Round {rd['round']}\n\n")
                f.write(f"- **A**: {rd['a_points']} pts ({rd['a_moves']} moves, {rd['a_trap_count']} traps)")
                if rd['a_hit_trap']:
                    f.write(" — TRAPPED")
                f.write("\n")
                f.write(f"- **B**: {rd['b_points']} pts ({rd['b_moves']} moves, {rd['b_trap_count']} traps)")
                if rd['b_hit_trap']:
                    f.write(" — TRAPPED")
                f.write("\n")
                f.write(f"- **Winner**: {rd['winner']}  |  Running: A {rd['running_a']} - B {rd['running_b']}\n")

                # Notes for this round
                round_notes = [n for n in self.notes if n["round"] == rd["round"]]
                if round_notes:
                    f.write("\n**Notes:**\n")
                    for n in round_notes:
                        f.write(f"> {n['note']}\n\n")
                f.write("\n")

            if self.game_notes:
                f.write(f"## Game Notes\n\n{self.game_notes}\n")

        print(f"\n  Session saved:")
        print(f"    {json_path}")
        print(f"    {md_path}")


# ── Main game loop ───────────────────────────────────────────────────────────

def run_game(agent_a: Agent, agent_b: Agent, board_size: int, session: Session):
    """Run a 5-round game between two agents with interactive commentary."""
    opponent_pools = discover_opponent_pools(board_size)
    if not opponent_pools:
        print(f"Warning: No opponent pools found for size {board_size}. RL agents may have issues.")

    # Track opponent history as numpy grids (from each agent's perspective)
    a_sees_b_history = np.zeros((5, board_size, board_size, 2), dtype=np.int32)
    b_sees_a_history = np.zeros((5, board_size, board_size, 2), dtype=np.int32)

    running_a = 0
    running_b = 0
    round_scores_a: list[dict] = []  # from A's perspective: {agent: a_pts, opponent: b_pts}
    round_scores_b: list[dict] = []  # from B's perspective: {agent: b_pts, opponent: a_pts}
    results: list[RoundResult] = []

    print(f"\n{'═' * 60}")
    print(f"  {agent_a.label} (A)  vs  {agent_b.label} (B)")
    print(f"  Board size: {board_size}x{board_size}  |  5 rounds")
    print(f"{'═' * 60}")

    for round_num in range(5):
        print(f"\n{'─' * 60}")
        print(f"  ROUND {round_num + 1}")
        print(f"{'─' * 60}\n")

        # Build boards — human players build first (blind), then AI
        has_human = agent_a.is_human or agent_b.is_human

        if has_human:
            # Human builds blind, then AI builds
            if agent_a.is_human:
                board_a = agent_a.build_board(
                    round_num=round_num, my_score=running_a, opp_score=running_b,
                    opponent_history_grids=a_sees_b_history, opponent_pools=opponent_pools,
                    round_scores=round_scores_a,
                )
                print("\n  Opponent is thinking...")
                board_b = agent_b.build_board(
                    round_num=round_num, my_score=running_b, opp_score=running_a,
                    opponent_history_grids=b_sees_a_history, opponent_pools=opponent_pools,
                    round_scores=round_scores_b,
                )
            else:
                print("  Opponent is thinking...")
                board_a = agent_a.build_board(
                    round_num=round_num, my_score=running_a, opp_score=running_b,
                    opponent_history_grids=a_sees_b_history, opponent_pools=opponent_pools,
                    round_scores=round_scores_a,
                )
                board_b = agent_b.build_board(
                    round_num=round_num, my_score=running_b, opp_score=running_a,
                    opponent_history_grids=b_sees_a_history, opponent_pools=opponent_pools,
                    round_scores=round_scores_b,
                )
        else:
            print("  Building boards...")
            board_a = agent_a.build_board(
                round_num=round_num, my_score=running_a, opp_score=running_b,
                opponent_history_grids=a_sees_b_history, opponent_pools=opponent_pools,
                round_scores=round_scores_a,
            )
            board_b = agent_b.build_board(
                round_num=round_num, my_score=running_b, opp_score=running_a,
                opponent_history_grids=b_sees_a_history, opponent_pools=opponent_pools,
                round_scores=round_scores_b,
            )

        if not is_board_playable(board_a):
            print(f"  ⚠ Agent A produced invalid board!")
        if not is_board_playable(board_b):
            print(f"  ⚠ Agent B produced invalid board!")

        # Simulate first — fog of war needs the result to know visibility
        result = simulate_round(round_num + 1, board_a, board_b, silent=True)
        results.append(result)
        sd = result.simulationDetails

        # Display boards with fog of war
        # Rule: human sees own board fully, everything else is fogged
        fog_a = not agent_a.is_human
        fog_b = not agent_b.is_human

        # Trap positions from simulation are in rotated coords (the piece
        # traverses the opponent's board rotated 180°). Un-rotate them back
        # to the board's native coords for the comparison in render_board_styled.
        a_sprung = None  # trap on A's board that B hit
        if fog_a and sd.opponentTrapPosition:
            tp = sd.opponentTrapPosition
            a_sprung = _rotate_position(tp.row, tp.col, board_size)

        b_sprung = None  # trap on B's board that A hit
        if fog_b and sd.playerTrapPosition:
            tp = sd.playerTrapPosition
            b_sprung = _rotate_position(tp.row, tp.col, board_size)

        print()
        print(render_board_styled(
            board_a, f"Agent A ({agent_a.label})",
            fog=fog_a,
            sprung_trap_pos=a_sprung,
        ))
        print()
        print(render_board_styled(
            board_b, f"Agent B ({agent_b.label})", rotated=True,
            fog=fog_b,
            sprung_trap_pos=b_sprung,
        ))
        print()

        # Update scores
        running_a += result.playerPoints
        running_b += result.opponentPoints
        round_scores_a.append({"agent": result.playerPoints, "opponent": result.opponentPoints})
        round_scores_b.append({"agent": result.opponentPoints, "opponent": result.playerPoints})

        # Update opponent history grids
        if round_num < 5:
            a_sees_b_history[round_num] = encode_board_for_agent(board_b, board_size)
            b_sees_a_history[round_num] = encode_board_for_agent(board_a, board_size)

        # Display simulation trace (fog matches board fog per side)
        print(render_simulation_trace(result, fog_a=fog_a, fog_b=fog_b))
        print()
        print(render_round_summary(round_num + 1, result, running_a, running_b))
        print()

        # Record
        session.record_round(round_num + 1, result, running_a, running_b)

        # Collect notes
        print("  Notes (Enter to skip, multi-line ok, blank line to finish):")
        note_lines = []
        while True:
            try:
                line = input("  > ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line == "" and not note_lines:
                break  # skip
            if line == "" and note_lines:
                break  # done with note
            note_lines.append(line)

        if note_lines:
            note = "\n".join(note_lines)
            session.record_note(round_num + 1, note)
            print(f"  ✓ Note saved for round {round_num + 1}")

    # Game over
    print(f"\n{'═' * 60}")
    if running_a > running_b:
        print(f"  GAME OVER — Agent A ({agent_a.label}) wins {running_a}-{running_b}")
    elif running_b > running_a:
        print(f"  GAME OVER — Agent B ({agent_b.label}) wins {running_b}-{running_a}")
    else:
        print(f"  GAME OVER — Tie {running_a}-{running_b}")
    print(f"{'═' * 60}")

    # Round-by-round recap
    print("\n  Recap:")
    for i, rd in enumerate(session.rounds):
        marker = "A" if rd["winner"] == "player" else "B" if rd["winner"] == "opponent" else "="
        print(f"    R{rd['round']}: A {rd['a_points']} - B {rd['b_points']}  [{marker}]  (running: {rd['running_a']}-{rd['running_b']})")

    # Final game notes
    print("\n  Any overall game notes? (Enter to skip, blank line to finish):")
    game_note_lines = []
    while True:
        try:
            line = input("  > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line == "" and not game_note_lines:
            break
        if line == "" and game_note_lines:
            break
        game_note_lines.append(line)

    if game_note_lines:
        session.game_notes = "\n".join(game_note_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Commentary Lab — observe games between agents and collect notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/commentary_lab.py --size 3 --agent-a scripted_1 --agent-b scripted_3
  python examples/commentary_lab.py --size 3 --agent-a human --agent-b scripted_3
  python examples/commentary_lab.py --size 3 --agent-a human --agent-b expert
  python examples/commentary_lab.py --size 4 --agent-a intermediate --agent-b scripted_5
        """,
    )
    parser.add_argument("--size", type=int, default=3, help="Board size (default: 3)")
    parser.add_argument("--agent-a", required=True, help="Agent A: skill level, scripted_N, 'human', or model path")
    parser.add_argument("--agent-b", required=True, help="Agent B: skill level, scripted_N, 'human', or model path")
    parser.add_argument("--output-dir", default="lab_sessions", help="Output directory (default: lab_sessions)")
    parser.add_argument("--board-library", default=DEFAULT_BOARD_LIBRARY, help=f"Board library JSON for human mode (default: {DEFAULT_BOARD_LIBRARY})")

    args = parser.parse_args()

    # Create agents
    print(f"\nSetting up agents for {args.size}x{args.size} board...")
    agent_a = Agent(args.agent_a, args.size, board_library=args.board_library)
    agent_b = Agent(args.agent_b, args.size, board_library=args.board_library)

    # Create session
    session = Session(args.agent_a, args.agent_b, args.size)

    # Run
    try:
        run_game(agent_a, agent_b, args.size, session)
    except KeyboardInterrupt:
        print("\n\n  Game interrupted.")

    # Save
    session.save(args.output_dir)


if __name__ == "__main__":
    main()
