"""
Scripted AI agents for size-2 boards.

Four difficulty levels with deterministic board-building strategies,
bypassing the RL model pipeline entirely.
"""

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


def scripted_board(level: int, board_size: int, round_num: int, round_scores: list[dict]) -> dict:
    """Main dispatcher: return a board dict for the given scripted level.

    Args:
        level: Difficulty level (1-4)
        board_size: Board grid dimension
        round_num: Current round number (0-indexed)
        round_scores: Per-round scores [{agent: float, opponent: float}, ...]

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

    else:
        raise ValueError(f"Unknown scripted level: {level}")
