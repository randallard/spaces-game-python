"""
Scripted AI agents for size-2 boards.

Five difficulty levels with deterministic board-building strategies,
bypassing the RL model pipeline entirely.
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RoundHistoryEntry

def build_simple_board(board_size: int, column: int) -> dict:
    """Build a straight-path board with no traps.

    Piece walks straight up the given column from row (board_size-1) to row 0,
    then a final move at row -1.
    """
    grid = [["empty"] * board_size for _ in range(board_size)]
    sequence = []
    order = 1

    for row in range(board_size - 1, -1, -1):
        grid[row][column] = "piece"
        sequence.append({
            "position": {"row": row, "col": column},
            "type": "piece",
            "order": order,
        })
        order += 1

    # Final/goal move
    sequence.append({
        "position": {"row": -1, "col": column},
        "type": "final",
        "order": order,
    })

    return {
        "sequence": sequence,
        "boardSize": board_size,
        "grid": grid,
    }


def build_trap_board(board_size: int, column: int) -> dict:
    """Build a straight-path board with an adjacent trap at the starting row.

    Trap is placed in row (board_size-1) at the opposite column from the piece.
    """
    trap_col = 1 - column if board_size == 2 else (column + 1) % board_size

    grid = [["empty"] * board_size for _ in range(board_size)]
    sequence = []
    order = 1

    # First piece at starting position
    start_row = board_size - 1
    grid[start_row][column] = "piece"
    sequence.append({
        "position": {"row": start_row, "col": column},
        "type": "piece",
        "order": order,
    })
    order += 1

    # Trap adjacent to start
    grid[start_row][trap_col] = "trap"
    sequence.append({
        "position": {"row": start_row, "col": trap_col},
        "type": "trap",
        "order": order,
    })
    order += 1

    # Continue piece path upward
    for row in range(start_row - 1, -1, -1):
        grid[row][column] = "piece"
        sequence.append({
            "position": {"row": row, "col": column},
            "type": "piece",
            "order": order,
        })
        order += 1

    # Final/goal move
    sequence.append({
        "position": {"row": -1, "col": column},
        "type": "final",
        "order": order,
    })

    return {
        "sequence": sequence,
        "boardSize": board_size,
        "grid": grid,
    }


def _lost_last_round(round_scores: list[dict]) -> bool:
    """Check if the agent lost the most recent round."""
    if not round_scores:
        return False
    last = round_scores[-1]
    return last.get("agent", 0) < last.get("opponent", 0)


def _lost_last_two_rounds(round_scores: list[dict]) -> bool:
    """Check if the agent lost the last two rounds in a row."""
    if len(round_scores) < 2:
        return False
    for score in round_scores[-2:]:
        if score.get("agent", 0) >= score.get("opponent", 0):
            return False
    return True


def pick_column_level1(board_size: int, round_num: int) -> int:
    """Level 1: Start column 0, switch once at round 2 (0-indexed).

    Columns: 0, 0, 1, 1, 1 across 5 rounds.
    Simple and predictable — the easiest opponent.
    """
    return 0 if round_num < 2 else 1


def pick_column_level2(board_size: int, round_num: int, round_scores: list[dict]) -> int:
    """Level 2/3: Start column 0, switch if lost last round."""
    if round_num == 0:
        return 0

    # Determine previous column by replaying decisions
    col = 0
    for i in range(1, round_num + 1):
        scores_so_far = round_scores[:i]
        if _lost_last_round(scores_so_far):
            col = 1 - col
    return col


def encode_board_compact(board_dict: dict) -> str:
    """Encode a board dict to compact format: '{size}|{moves}'.

    Produces strings parseable by decode_starting_column().
    """
    size = board_dict["boardSize"]
    moves = []
    for m in board_dict["sequence"]:
        pos = m["position"]
        mtype = m["type"]
        if pos["row"] == -1:  # Goal/final
            moves.append(f"G{pos['col']}f")
        else:
            cell = pos["row"] * size + pos["col"]
            type_char = "p" if mtype == "piece" else "t" if mtype == "trap" else "f"
            moves.append(f"{cell}{type_char}")
    return f"{size}|{''.join(moves)}"


def decode_starting_column(encoded: str, board_size: int | None = None) -> int:
    """Extract the first piece's column from a compact-encoded board.

    Format: "{size}|{moves}" where moves are sequences of {cell_index}{type_char}.
    Type chars: p=piece, t=trap, f=final, G=goal.

    Args:
        encoded: Compact-encoded board string like "2|2p2t0pG0f".
        board_size: Override board size (uses encoded size if None).

    Returns:
        Column index of the first piece placement.
    """
    parts = encoded.split("|", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid compact board encoding: {encoded!r}")

    size = board_size if board_size is not None else int(parts[0])
    moves_str = parts[1]

    # Parse moves: read digits then a type character
    i = 0
    while i < len(moves_str):
        # Read digits (cell index)
        num_start = i
        while i < len(moves_str) and moves_str[i].isdigit():
            i += 1

        if i >= len(moves_str):
            break

        type_char = moves_str[i]
        i += 1

        if type_char == "p" and num_start < i - 1:
            cell_index = int(moves_str[num_start : i - 1])
            return cell_index % size

    raise ValueError(f"No piece move found in compact board: {encoded!r}")


def build_supermove_board(board_size: int, column: int) -> dict:
    """Build a supermove board: piece walks straight up with a trap on the starting cell.

    The trap is placed on the same cell as the first piece placement (N-1, col).
    The piece moves off that cell first, then the trap is placed — an opponent
    following this path hits the trap immediately.
    """
    grid = [["empty"] * board_size for _ in range(board_size)]
    sequence = []
    order = 1

    start_row = board_size - 1

    # First piece at starting position
    sequence.append({
        "position": {"row": start_row, "col": column},
        "type": "piece",
        "order": order,
    })
    order += 1

    # Trap on the same starting cell (piece has moved off)
    grid[start_row][column] = "trap"
    sequence.append({
        "position": {"row": start_row, "col": column},
        "type": "trap",
        "order": order,
    })
    order += 1

    # Continue piece path upward
    for row in range(start_row - 1, -1, -1):
        grid[row][column] = "piece"
        sequence.append({
            "position": {"row": row, "col": column},
            "type": "piece",
            "order": order,
        })
        order += 1

    # Final/goal move
    sequence.append({
        "position": {"row": -1, "col": column},
        "type": "final",
        "order": order,
    })

    return {
        "sequence": sequence,
        "boardSize": board_size,
        "grid": grid,
    }


def _two_consecutive_ties(round_history: list["RoundHistoryEntry"]) -> bool:
    """Check if the last two rounds were ties."""
    if len(round_history) < 2:
        return False
    return (
        round_history[-1].agent_score == round_history[-1].opponent_score
        and round_history[-2].agent_score == round_history[-2].opponent_score
    )


def scripted_board(
    level: int,
    board_size: int,
    round_num: int,
    round_scores: list[dict],
    round_history: list["RoundHistoryEntry"] | None = None,
) -> dict:
    """Main dispatcher: return a board dict for the given scripted level.

    Args:
        level: Difficulty level (1-5)
        board_size: Board grid dimension
        round_num: Current round number (0-indexed)
        round_scores: Per-round scores [{agent: float, opponent: float}, ...]
        round_history: Rich per-round history (needed for level 5)

    Returns:
        Board dict with sequence, boardSize, and grid.
    """
    if level == 1:
        col = pick_column_level1(board_size, round_num)
        return build_simple_board(board_size, col)

    elif level == 2:
        col = pick_column_level2(board_size, round_num, round_scores)
        return build_simple_board(board_size, col)

    elif level == 3:
        col = pick_column_level2(board_size, round_num, round_scores)
        return build_trap_board(board_size, col)

    elif level == 4:
        # Always switch column from previous round
        if round_num == 0:
            col = 0
        else:
            # Replay column decisions to find previous column
            prev_col = 0
            for i in range(1, round_num):
                prev_col = 1 - prev_col
            col = 1 - prev_col

        # Escape hatch: drop trap after losing 2 in a row → go simple
        if _lost_last_two_rounds(round_scores):
            return build_simple_board(board_size, col)
        else:
            return build_trap_board(board_size, col)

    elif level == 5:
        rh = round_history or []

        if round_num == 0:
            # Round 0: pick random column, play trap board
            col = random.randint(0, 1)
            return build_trap_board(board_size, col)

        # Derive starting column from round 0's board
        if rh and rh[0].agent_board:
            col = decode_starting_column(rh[0].agent_board)
        else:
            # Fallback: no history available, pick 0
            col = 0

        # Replay state: track column and whether last board was supermove
        # Walk through round history to reconstruct decisions
        current_col = col
        last_was_supermove = False

        for i in range(1, round_num):
            if i >= len(rh):
                break

            past_rh = rh[:i]

            if last_was_supermove:
                # After supermove: switch to trap from opposite column
                current_col = 1 - current_col
                last_was_supermove = False
            elif _two_consecutive_ties(past_rh):
                # Trigger supermove
                # Check score from two rounds ago
                two_ago = past_rh[-2]
                if two_ago.agent_score == 0:
                    # Agent scored 0 → switch column, then supermove
                    current_col = 1 - current_col
                # else: supermove from same column
                last_was_supermove = True
            # else: keep playing trap from same column

        # Now decide for current round
        if last_was_supermove:
            # After supermove: switch to trap from opposite column
            current_col = 1 - current_col
            return build_trap_board(board_size, current_col)
        elif _two_consecutive_ties(rh[:round_num]):
            # Trigger supermove
            two_ago = rh[round_num - 2]
            if two_ago.agent_score == 0:
                current_col = 1 - current_col
            return build_supermove_board(board_size, current_col)
        else:
            return build_trap_board(board_size, current_col)

    else:
        raise ValueError(f"Unknown scripted level: {level}")
