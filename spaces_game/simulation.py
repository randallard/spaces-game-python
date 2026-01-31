"""
Game simulation engine for Spaces Game.

Rules:
- Players follow their board's sequence step-by-step
- +1 point for each forward move (toward goal)
- +1 point for reaching goal (Final cell)
- -1 point for collision (both players lose 1 point, min 0)
- -1 point for hitting opponent's trap (min 0)
- Round ends on: collision, goal reached, trap hit, or sequences complete
"""

from typing import Optional

from .types import Board, RoundResult, Position, SimulationDetails


def _rotate_position(row: int, col: int, size: int) -> Position:
    """Rotate position 180 degrees for opponent's perspective."""
    return Position(
        row=size - 1 - row,
        col=size - 1 - col,
    )


def _position_key(row: int, col: int) -> str:
    """Get position key for map lookup."""
    return f"{row},{col}"


def simulate_round(
    round_num: int,
    player_board: Board,
    opponent_board: Board,
    silent: bool = False
) -> RoundResult:
    """
    Simulate a complete round between player and opponent.

    Args:
        round_num: Round number (1-5 for round-by-round, 1-10 for deck mode)
        player_board: Player's selected board
        opponent_board: Opponent's selected board
        silent: If True, suppress console output

    Returns:
        Complete round result with winner and details

    Example:
        >>> result = simulate_round(1, player_board, opponent_board)
        >>> print(f"Player: {result.playerPoints} points")
    """
    size = len(player_board.grid)

    if not silent:
        print(f"[Game Simulation] Round {round_num}")

    # Initialize game state - positions start as None (not set)
    player_score = 0
    opponent_score = 0
    player_position: Optional[Position] = None
    opponent_position: Optional[Position] = None
    player_round_ended = False
    opponent_round_ended = False
    player_goal_reached = False
    opponent_goal_reached = False
    player_hit_trap = False
    opponent_hit_trap = False
    player_moves = 0
    opponent_moves = 0
    player_last_step = -1  # Last sequence step executed by player
    opponent_last_step = -1  # Last sequence step executed by opponent
    player_trap_position: Optional[Position] = None
    opponent_trap_position: Optional[Position] = None

    # Track trap placements
    player_traps: dict[str, int] = {}  # position key -> step number
    opponent_traps: dict[str, int] = {}

    max_steps = max(len(player_board.sequence), len(opponent_board.sequence))

    # Step-by-step simulation
    for step in range(max_steps):

        # Process player's move
        if not player_round_ended and step < len(player_board.sequence):
            move = player_board.sequence[step]
            # Only access grid for valid positions (not for final moves at row -1)
            cell_content = None
            if move.position.row >= 0:
                cell_content = player_board.grid[move.position.row][move.position.col]

            if move.type == 'piece' or cell_content == 'piece':
                # Check for forward movement before updating position
                if player_position is not None and player_position.row > move.position.row:
                    player_score += 1
                player_position = move.position
                player_moves += 1
                player_last_step = step
            elif move.type == 'trap' or cell_content == 'trap':
                player_traps[_position_key(move.position.row, move.position.col)] = step
                player_last_step = step
            elif move.type == 'final' or cell_content == 'final':
                player_goal_reached = True
                player_position = move.position  # Update position to goal (off-board)
                player_score += 1
                player_round_ended = True
                player_last_step = step

        # Process opponent's move (with rotation)
        if not opponent_round_ended and step < len(opponent_board.sequence):
            move = opponent_board.sequence[step]
            rotated = _rotate_position(move.position.row, move.position.col, size)
            # Only access grid for valid positions (not for final moves at row -1)
            cell_content = None
            if move.position.row >= 0:
                cell_content = opponent_board.grid[move.position.row][move.position.col]

            if move.type == 'piece' or cell_content == 'piece':
                # Check for forward movement before updating position
                if opponent_position is not None and opponent_position.row < rotated.row:
                    opponent_score += 1
                opponent_position = rotated
                opponent_moves += 1
                opponent_last_step = step
            elif move.type == 'trap' or cell_content == 'trap':
                opponent_traps[_position_key(rotated.row, rotated.col)] = step
                opponent_last_step = step
            elif move.type == 'final' or cell_content == 'final':
                opponent_goal_reached = True
                opponent_position = rotated  # Update position to goal (off-board)
                opponent_score += 1
                opponent_round_ended = True
                opponent_last_step = step

        # Check for collision (only if both positions are set and neither has left the board)
        if (
            not player_goal_reached and
            not opponent_goal_reached and
            player_position is not None and
            opponent_position is not None and
            player_position.row == opponent_position.row and
            player_position.col == opponent_position.col
        ):
            # Both players lose 1 point (minimum 0)
            if player_score > 0:
                player_score -= 1
            if opponent_score > 0:
                opponent_score -= 1
            break  # End round immediately

        # Check if player hit opponent's trap (only if player has a position)
        if not player_round_ended and player_position is not None:
            trap_step = opponent_traps.get(
                _position_key(player_position.row, player_position.col)
            )
            if trap_step is not None and trap_step <= step:
                if player_score > 0:
                    player_score -= 1
                player_hit_trap = True
                player_trap_position = Position(row=player_position.row, col=player_position.col)
                player_round_ended = True

        # Check if opponent hit player's trap (only if opponent has a position)
        if not opponent_round_ended and opponent_position is not None:
            trap_step = player_traps.get(
                _position_key(opponent_position.row, opponent_position.col)
            )
            if trap_step is not None and trap_step <= step:
                if opponent_score > 0:
                    opponent_score -= 1
                opponent_hit_trap = True
                opponent_trap_position = Position(row=opponent_position.row, col=opponent_position.col)
                opponent_round_ended = True

        # Stop if both players have ended their round
        if player_round_ended and opponent_round_ended:
            break

        # Stop if either player has reached their goal
        if player_goal_reached or opponent_goal_reached:
            break

    # Determine winner based on final scores
    if player_score > opponent_score:
        winner = 'player'
    elif opponent_score > player_score:
        winner = 'opponent'
    else:
        winner = 'tie'

    # Use default starting positions if never moved
    final_player_position = player_position or Position(row=size - 1, col=0)
    final_opponent_position = opponent_position or Position(row=0, col=size - 1)

    # Check for collision (both at same position)
    collision = (
        final_player_position.row == final_opponent_position.row and
        final_player_position.col == final_opponent_position.col
    )

    if not silent:
        player_status = ' (trapped)' if player_hit_trap else ' (goal)' if player_goal_reached else ''
        opponent_status = ' (trapped)' if opponent_hit_trap else ' (goal)' if opponent_goal_reached else ''
        print(
            f"[Game Simulation] Result: {winner} | "
            f"Player: {player_score}pts{player_status} | "
            f"Opponent: {opponent_score}pts{opponent_status}"
        )

    # Build simulation details
    simulation_details = SimulationDetails(
        playerMoves=player_moves,
        opponentMoves=opponent_moves,
        playerHitTrap=player_hit_trap,
        opponentHitTrap=opponent_hit_trap,
        playerLastStep=player_last_step,
        opponentLastStep=opponent_last_step,
        playerTrapPosition=player_trap_position,
        opponentTrapPosition=opponent_trap_position,
    )

    return RoundResult(
        round=round_num,
        winner=winner,
        playerBoard=player_board,
        opponentBoard=opponent_board,
        playerFinalPosition=final_player_position,
        opponentFinalPosition=final_opponent_position,
        playerPoints=player_score,
        opponentPoints=opponent_score,
        collision=collision,
        simulationDetails=simulation_details,
    )


def simulate_multiple_rounds(
    player_boards: list[Board],
    opponent_boards: list[Board],
    silent: bool = False
) -> list[RoundResult]:
    """
    Simulate multiple rounds in sequence.

    Args:
        player_boards: Player's boards for each round
        opponent_boards: Opponent's boards for each round
        silent: If True, suppress console output

    Returns:
        List of round results

    Example:
        >>> # 5 rounds for round-by-round mode
        >>> results = simulate_multiple_rounds(player_boards, opponent_boards)
        >>>
        >>> # 10 rounds for deck mode
        >>> deck_results = simulate_multiple_rounds(player_deck, opponent_deck)
    """
    if len(player_boards) != len(opponent_boards):
        raise ValueError('Player and opponent must have same number of boards')

    num_rounds = len(player_boards)

    if not silent:
        print(f"[Game Simulation] Starting {num_rounds} rounds")

    results: list[RoundResult] = []

    for round_num in range(1, num_rounds + 1):
        player_board = player_boards[round_num - 1]
        opponent_board = opponent_boards[round_num - 1]

        result = simulate_round(round_num, player_board, opponent_board, silent=silent)
        results.append(result)

    total_player_score = sum(r.playerPoints for r in results)
    total_opponent_score = sum(r.opponentPoints for r in results)

    if not silent:
        print(f"[Game Simulation] Complete: Player {total_player_score} - Opponent {total_opponent_score}")

    return results
