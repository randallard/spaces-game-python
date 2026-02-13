"""
CLI for Spaces Game operations.

Provides commands for testing, validation, playing games, and more.
"""

import sys
import json
from typing import Optional
import click

from .board_loader import load_boards_from_json, BoardPool
from .simulation import simulate_round
from .validation import is_board_playable
from .types import Board, Position, RoundResult
from .interactive_builder import build_board_interactive


@click.group()
@click.version_option(version="1.0.0", prog_name="spaces-game")
def cli():
    """Spaces Game Python CLI - Test, validate, and play."""
    pass


def _render_board_fog_of_war(result: RoundResult, title: str = "Opponent (Fog of War)") -> str:
    """
    Render opponent board with fog of war: show pieces and only traps that player hit.

    Args:
        result: Round result containing both boards and simulation details
        title: Title to display

    Returns:
        Rendered board as string
    """
    board = result.opponentBoard
    size = board.boardSize
    lines = []

    # Determine which trap the player hit (if any)
    player_hit_trap_at: Optional[tuple[int, int]] = None
    if result.simulationDetails.playerHitTrap:
        # Player's final position needs to be translated to opponent's board coordinates
        # Boards are rotated 180 degrees from each other
        # Player (0,0) = Opponent (size-1, size-1)
        opp_row = size - 1 - result.playerFinalPosition.row
        opp_col = size - 1 - result.playerFinalPosition.col
        player_hit_trap_at = (opp_row, opp_col)

    # Track pieces and traps at each position
    position_contents: dict[tuple[int, int], tuple[list[int], list[int]]] = {}

    for move in board.sequence:
        if move.position.row >= 0:  # Skip final moves
            # Rotate coordinates for display
            row = size - 1 - move.position.row
            col = size - 1 - move.position.col
            pos = (row, col)

            if pos not in position_contents:
                position_contents[pos] = ([], [])

            pieces, traps = position_contents[pos]

            if move.type == 'piece':
                pieces.append(move.order)
            elif move.type == 'trap':
                # Only show trap if player hit it
                orig_pos = (move.position.row, move.position.col)
                if player_hit_trap_at and orig_pos == player_hit_trap_at:
                    traps.append(move.order)
                # Otherwise hide the trap (don't add it)

    lines.append(f"{title}\n")

    # Top border
    lines.append("┌" + "─────────┬" * (size - 1) + "─────────┐")

    for row_idx in range(size):
        row_items = []
        for col_idx in range(size):
            pos = (row_idx, col_idx)
            pieces, traps = position_contents.get(pos, ([], []))

            if pieces and traps:
                # Supermove: both piece and trap (trap was hit)
                piece_str = click.style(f"{pieces[0]}●", fg="blue")
                trap_str = click.style(f"{traps[0]}X", fg="red")
                content = f"{piece_str},{trap_str}"
                visible_len = len(str(pieces[0])) + 1 + 1 + len(str(traps[0])) + 1
                padding = 9 - 1 - visible_len
                cell = f" {content}{' ' * padding}"
            elif pieces:
                # Just piece
                num_str = click.style(f"{pieces[0]}●", fg="blue")
                visible_len = len(str(pieces[0])) + 1
                padding = 9 - 2 - visible_len
                cell = f"  {num_str}{' ' * padding}"
            elif traps:
                # Just trap (player hit it)
                num_str = click.style(f"{traps[0]}X", fg="red")
                visible_len = len(str(traps[0])) + 1
                padding = 9 - 2 - visible_len
                cell = f"  {num_str}{' ' * padding}"
            else:
                # Empty
                cell = "         "

            row_items.append(cell)

        lines.append("│" + "│".join(row_items) + "│")

        # Middle border
        if row_idx < size - 1:
            lines.append("├" + "─────────┼" * (size - 1) + "─────────┤")

    # Bottom border
    lines.append("└" + "─────────┴" * (size - 1) + "─────────┘")

    return "\n".join(lines)


def _render_board(board: Board, title: str = "Board", rotate: bool = False, max_step: Optional[int] = None) -> str:
    """Render a board as ASCII art with numbered pieces and traps.

    Args:
        board: Board to render
        title: Title to display above the board
        rotate: If True, rotate the board 180 degrees (for opponent view)
        max_step: If provided, only show moves up to this step (for fog of war)
    """
    size = board.boardSize
    lines = []

    # Track pieces and traps at each position
    # Maps position to (piece_nums, trap_nums)
    position_contents: dict[tuple[int, int], tuple[list[int], list[int]]] = {}

    for move in board.sequence:
        # Skip moves beyond max_step (fog of war)
        if max_step is not None and move.order > max_step:
            continue
        if move.position.row >= 0:  # Skip final moves
            if rotate:
                # Rotate 180 degrees
                row = size - 1 - move.position.row
                col = size - 1 - move.position.col
            else:
                row = move.position.row
                col = move.position.col

            pos = (row, col)
            if pos not in position_contents:
                position_contents[pos] = ([], [])

            pieces, traps = position_contents[pos]
            if move.type == 'piece':
                pieces.append(move.order)
            elif move.type == 'trap':
                traps.append(move.order)

    lines.append(f"{title}\n")

    # Top border (wider cells to match TypeScript)
    lines.append("┌" + "─────────┬" * (size - 1) + "─────────┐")

    for row_idx in range(size):
        row_items = []
        for col_idx in range(size):
            pos = (row_idx, col_idx)
            pieces, traps = position_contents.get(pos, ([], []))

            if pieces and traps:
                # Supermove: both piece and trap (e.g., " 1●,3X   ")
                # Format: 1 leading space + content + trailing spaces to fill 9 chars
                piece_str = click.style(f"{pieces[0]}●", fg="blue")
                trap_str = click.style(f"{traps[0]}X", fg="red")
                content = f"{piece_str},{trap_str}"
                # Visible length: num + ● + , + num + X
                visible_len = len(str(pieces[0])) + 1 + 1 + len(str(traps[0])) + 1
                padding = 9 - 1 - visible_len  # 9 total - 1 leading space - content
                cell = f" {content}{' ' * padding}"
            elif pieces:
                # Just piece (e.g., "  5●     ")
                # Format: 2 leading spaces + content + trailing spaces to fill 9 chars
                num_str = click.style(f"{pieces[0]}●", fg="blue")
                visible_len = len(str(pieces[0])) + 1  # number + ●
                padding = 9 - 2 - visible_len  # 9 total - 2 leading spaces - content
                cell = f"  {num_str}{' ' * padding}"
            elif traps:
                # Just trap (e.g., "  2X     ")
                # Format: 2 leading spaces + content + trailing spaces to fill 9 chars
                num_str = click.style(f"{traps[0]}X", fg="red")
                visible_len = len(str(traps[0])) + 1  # number + X
                padding = 9 - 2 - visible_len  # 9 total - 2 leading spaces - content
                cell = f"  {num_str}{' ' * padding}"
            else:
                # Empty - 9 spaces
                cell = "         "

            row_items.append(cell)

        lines.append("│" + "│".join(row_items) + "│")

        # Middle border
        if row_idx < size - 1:
            lines.append("├" + "─────────┼" * (size - 1) + "─────────┤")

    # Bottom border
    lines.append("└" + "─────────┴" * (size - 1) + "─────────┘")

    return "\n".join(lines)


def _render_result_details(result: RoundResult, fog_of_war: bool = False) -> None:
    """Render detailed simulation results similar to TypeScript output.

    Args:
        result: Round result to render
        fog_of_war: If True, hide opponent traps that player didn't hit
    """
    click.echo("\n" + click.style("🎮 Simulation Results", bold=True) + "\n")

    # Determine what to hide based on fog of war
    hidden_trap_position = None

    if fog_of_war:
        # Find the trap position the player hit (if any)
        player_hit_trap_at = None
        if result.simulationDetails.playerHitTrap:
            # Player's final position is where they hit the trap
            player_hit_trap_at = result.playerFinalPosition

        # We'll pass this info to the render function
        # For now, we'll handle it by filtering the opponent board sequence

    # Show boards side by side (opponent board rotated 180 degrees)
    player_lines = _render_board(result.playerBoard, "Player").split('\n')
    opponent_title = "Opponent" + (" (Fog of War)" if fog_of_war else "")

    if fog_of_war:
        # Create filtered opponent board showing only: pieces and traps player hit
        opponent_lines = _render_board_fog_of_war(result, opponent_title).split('\n')
    else:
        opponent_lines = _render_board(result.opponentBoard, opponent_title, rotate=True).split('\n')

    max_lines = max(len(player_lines), len(opponent_lines))

    # Calculate visible width (without ANSI codes) for alignment
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

    def visible_width(s: str) -> int:
        """Get visible width of string without ANSI color codes."""
        return len(ansi_escape.sub('', s))

    player_width = max(visible_width(line) for line in player_lines)

    for i in range(max_lines):
        player_line = player_lines[i] if i < len(player_lines) else ""
        opponent_line = opponent_lines[i] if i < len(opponent_lines) else ""

        # Pad player line to align opponent board
        padding_needed = player_width - visible_width(player_line)
        click.echo(f"{player_line}{' ' * padding_needed}    {opponent_line}")

    # Technical explanation
    click.echo("\n" + click.style("📋 Technical Explanation", bold=True) + "\n")

    size = result.playerBoard.boardSize

    # Helper to rotate opponent coordinates for display
    def rotate_pos(row: int, col: int) -> tuple[int, int]:
        """Rotate opponent position 180 degrees for display."""
        return (size - 1 - row, size - 1 - col)

    # Starting positions
    if result.playerBoard.sequence:
        start = result.playerBoard.sequence[0].position
        click.echo(f"Player starts with piece at ({start.row}, {start.col})")
    if result.opponentBoard.sequence:
        # Opponent positions shown from their rotated perspective
        start = result.opponentBoard.sequence[0].position
        rot_row, rot_col = rotate_pos(start.row, start.col)
        click.echo(f"Opponent starts with piece at ({rot_row}, {rot_col})")

    click.echo()

    # Trace through the simulation step-by-step
    player_steps = result.simulationDetails.playerLastStep + 1
    opponent_steps = result.simulationDetails.opponentLastStep + 1

    player_score = 0
    opponent_score = 0
    player_pos: Optional[Position] = None
    opponent_pos: Optional[Position] = None
    player_best_row: Optional[int] = None
    opponent_best_row: Optional[int] = None

    max_steps = max(player_steps, opponent_steps)

    for step in range(max_steps):
        # Player move
        if step < len(result.playerBoard.sequence) and step <= result.simulationDetails.playerLastStep:
            move = result.playerBoard.sequence[step]
            if move.type == 'piece':
                old_pos = player_pos
                player_pos = move.position
                click.echo(f"Player moves to ({player_pos.row}, {player_pos.col})")
                if old_pos and (player_best_row is None or player_pos.row < player_best_row):
                    player_score += 1
                    player_best_row = player_pos.row
                    click.echo(f"  Player +1 point (forward movement)")
                if player_best_row is None:
                    player_best_row = player_pos.row
            elif move.type == 'trap':
                click.echo(f"Player places trap at ({move.position.row}, {move.position.col})")
            elif move.type == 'final':
                click.echo(f"Player reaches the goal!")
                player_score += 1
                click.echo(f"  Player +1 point (goal reached)")

        # Opponent move (with rotated coordinates for display)
        if step < len(result.opponentBoard.sequence) and step <= result.simulationDetails.opponentLastStep:
            move = result.opponentBoard.sequence[step]
            rot_row, rot_col = rotate_pos(move.position.row, move.position.col)

            if move.type == 'piece':
                old_pos = opponent_pos
                opponent_pos = move.position
                click.echo(f"Opponent moves to ({rot_row}, {rot_col})")
                if old_pos and (opponent_best_row is None or rot_row > opponent_best_row):
                    opponent_score += 1
                    opponent_best_row = rot_row
                    click.echo(f"  Opponent +1 point (forward movement)")
                if opponent_best_row is None:
                    opponent_best_row = rot_row
            elif move.type == 'trap':
                click.echo(f"Opponent places trap at ({rot_row}, {rot_col})")
            elif move.type == 'final':
                click.echo(f"Opponent reaches the goal!")
                opponent_score += 1
                click.echo(f"  Opponent +1 point (goal reached)")

    # Check for collision
    if result.collision:
        click.echo(click.style("\n💥 COLLISION! Both players lose 1 point", fg="yellow"))

    # Check for trap hits
    if result.simulationDetails.playerHitTrap:
        click.echo(click.style("Player hit opponent's trap! -1 point", fg="red"))
    if result.simulationDetails.opponentHitTrap:
        click.echo(click.style("Opponent hit player's trap! -1 point", fg="red"))

    # Round end reason
    click.echo()
    if player_pos and player_pos.row == -1:
        click.echo("Round ends - Player reached the goal!")
    elif opponent_pos and opponent_pos.row == -1:
        click.echo("Round ends - Opponent reached the goal!")
    elif result.collision:
        click.echo("Round ends - Collision!")
    elif result.simulationDetails.playerHitTrap or result.simulationDetails.opponentHitTrap:
        click.echo("Round ends - Trap hit!")
    else:
        click.echo("Round ends - Sequences complete")

    # Winner
    click.echo()
    if result.winner == 'player':
        click.echo(click.style("🏆 PLAYER WINS", fg="green", bold=True))
    elif result.winner == 'opponent':
        click.echo(click.style("🏆 OPPONENT WINS", fg="red", bold=True))
    else:
        click.echo(click.style("🤝 TIE", fg="yellow", bold=True))

    # Final scores and positions
    click.echo(f"\nScores:")
    click.echo(f"  Player:   {result.playerPoints} points")
    click.echo(f"  Opponent: {result.opponentPoints} points")

    click.echo(f"\nFinal Positions:")
    click.echo(f"  Player:   row {result.playerFinalPosition.row}, col {result.playerFinalPosition.col}")
    # Show opponent position rotated
    opp_final_row, opp_final_col = rotate_pos(result.opponentFinalPosition.row, result.opponentFinalPosition.col)
    click.echo(f"  Opponent: row {opp_final_row}, col {opp_final_col}")


@cli.command()
@click.argument("board_file", type=click.Path(exists=True), required=False)
@click.option("--player-index", default=0, type=int, help="Player board index in file")
@click.option("--opponent-index", default=1, type=int, help="Opponent board index in file")
@click.option("-i", "--interactive", is_flag=True, help="Interactive board builder mode")
@click.option("--size", type=int, help="Board size for interactive mode (2-100)")
@click.option("--start-col", type=int, help="Starting column for interactive mode")
@click.option("--show-all", is_flag=True, help="Show all opponent traps (not just ones you hit)")
@click.option("--save-to", type=click.Path(), help="File to save board to in interactive mode (appends if exists)")
def test(board_file: Optional[str], player_index: int, opponent_index: int, interactive: bool, size: Optional[int], start_col: Optional[int], show_all: bool, save_to: Optional[str]):
    """
    Run a single simulation test with detailed output.

    Similar to TypeScript CLI 'test' command - shows boards, step-by-step
    moves, and complete results.

    Use --interactive to build a board interactively.
    """
    # Interactive mode
    if interactive:
        click.echo(click.style("\n🎮 Interactive Board Builder", bold=True))
        click.echo("Build your board step by step!\n")

        board = build_board_interactive(size=size, start_col=start_col)

        if board is None:
            click.echo(click.style("\n✗ Board building cancelled", fg="yellow"))
            sys.exit(0)

        click.echo(click.style("\n✅ Board created successfully!", fg="green", bold=True))

        # Optionally test against a random opponent
        if board_file:
            try:
                boards = load_boards_from_json(board_file)
                import random
                opponent_board = random.choice(boards)

                click.echo(f"\nTesting against random opponent from {board_file}...\n")
                result = simulate_round(1, board, opponent_board, silent=True)
                _render_result_details(result, fog_of_war=not show_all)
            except Exception as e:
                click.echo(click.style(f"\n⚠️  Could not test against opponent: {e}", fg="yellow"))

        # Show board JSON
        click.echo(click.style("\n📋 Board JSON:", bold=True))
        import json
        board_dict = {
            "boardSize": board.boardSize,
            "grid": [[cell for cell in row] for row in board.grid],
            "sequence": [
                {
                    "position": {"row": move.position.row, "col": move.position.col},
                    "type": move.type,
                    "order": move.order
                }
                for move in board.sequence
            ]
        }
        click.echo(json.dumps(board_dict, indent=2))
        click.echo()

        # Save board
        import os
        if save_to:
            should_save = True
            save_path = save_to
        else:
            should_save = click.confirm('Would you like to save this board?', default=True)
            save_path = None
            if should_save:
                default_filename = f"board_size_{board.boardSize}.json"
                save_path = click.prompt('Save to file', default=default_filename, type=str)

        if should_save and save_path:

            try:
                # Check if file exists and load existing boards
                existing_boards = []
                if os.path.exists(save_path):
                    if save_to or click.confirm(f'File {save_path} already exists. Append to it?', default=True):
                        try:
                            with open(save_path, 'r') as f:
                                existing_data = json.load(f)
                                if isinstance(existing_data, list):
                                    existing_boards = existing_data
                                else:
                                    click.echo(click.style('⚠️  Existing file is not a board array. Creating new file.', fg='yellow'))
                        except Exception as e:
                            click.echo(click.style(f'⚠️  Could not read existing file: {e}. Creating new file.', fg='yellow'))
                    else:
                        click.echo('Cancelled save.')
                        return

                # Check for duplicate boards
                def boards_equal(board1: dict, board2: dict) -> bool:
                    """Check if two boards are identical."""
                    if board1['boardSize'] != board2['boardSize']:
                        return False

                    # Compare sequences (the authoritative source)
                    seq1 = board1['sequence']
                    seq2 = board2['sequence']

                    if len(seq1) != len(seq2):
                        return False

                    for move1, move2 in zip(seq1, seq2):
                        if (move1['position']['row'] != move2['position']['row'] or
                            move1['position']['col'] != move2['position']['col'] or
                            move1['type'] != move2['type'] or
                            move1['order'] != move2['order']):
                            return False

                    return True

                # Check if this board already exists
                duplicate_index = None
                for idx, existing_board in enumerate(existing_boards):
                    if boards_equal(board_dict, existing_board):
                        duplicate_index = idx
                        break

                if duplicate_index is not None:
                    click.echo(click.style(
                        f'\n⚠️  This board already exists in {save_path} at index {duplicate_index}',
                        fg='yellow',
                        bold=True
                    ))
                    click.echo('Board was not saved (duplicate detected).')
                    return

                # Append new board and save
                existing_boards.append(board_dict)

                with open(save_path, 'w') as f:
                    json.dump(existing_boards, f, indent=2)

                click.echo(click.style(f'\n✅ Board saved to {save_path} ({len(existing_boards)} total boards)', fg='green', bold=True))
            except Exception as e:
                click.echo(click.style(f'\n✗ Error saving board: {e}', fg='red'))

        return

    # Regular test mode
    if not board_file:
        click.echo(click.style("✗ Error: board_file is required when not using --interactive mode", fg="red"))
        sys.exit(1)
    try:
        boards = load_boards_from_json(board_file)

        if player_index >= len(boards):
            click.echo(click.style(f"✗ Player index {player_index} out of range (0-{len(boards)-1})", fg="red"))
            sys.exit(1)

        if opponent_index >= len(boards):
            click.echo(click.style(f"✗ Opponent index {opponent_index} out of range (0-{len(boards)-1})", fg="red"))
            sys.exit(1)

        player_board = boards[player_index]
        opponent_board = boards[opponent_index]

        click.echo("\nRunning simulation...\n")

        result = simulate_round(1, player_board, opponent_board, silent=True)

        _render_result_details(result, fog_of_war=not show_all)

        click.echo()

    except Exception as e:
        import traceback
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        if "--verbose" in sys.argv:
            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option(
    "--test-file",
    type=click.Path(exists=True),
    default="tests/fixtures/session-2026-01-30T13-28-47-534Z.json",
    help="Path to TypeScript test session JSON file",
)
def test_parity(test_file: str):
    """
    Run parity tests against TypeScript implementation.

    Verifies that Python simulation produces identical results to TypeScript
    for all test cases in the session file.
    """
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Running Parity Tests")
    click.echo(f"{'=' * 60}")
    click.echo(f"Test Session: {test_file}\n")

    try:
        with open(test_file, "r") as f:
            session_data = json.load(f)

        session_name = session_data.get("sessionName", "Unnamed Session")
        tests = session_data.get("tests", [])

        click.echo(f"Session: {session_name}")
        click.echo(f"Loaded {len(tests)} test cases\n")

        passed = 0
        failed = 0
        failures = []

        for idx, test_case in enumerate(tests, start=1):
            player_board_data = test_case.get("playerBoard")
            opponent_board_data = test_case.get("opponentBoard")
            expected_result = test_case.get("result")

            # Convert to Board objects
            from .board_loader import load_board_from_dict
            player_board = load_board_from_dict(player_board_data)
            opponent_board = load_board_from_dict(opponent_board_data)

            # Run simulation
            result = simulate_round(1, player_board, opponent_board, silent=True)

            # Compare results
            match = (
                result.winner == expected_result["winner"]
                and result.playerPoints == expected_result["playerScore"]
                and result.opponentPoints == expected_result["opponentScore"]
                and result.collision == expected_result["collision"]
                and result.playerFinalPosition.row == expected_result["playerFinalPosition"]["row"]
                and result.playerFinalPosition.col == expected_result["playerFinalPosition"]["col"]
                and result.opponentFinalPosition.row == expected_result["opponentFinalPosition"]["row"]
                and result.opponentFinalPosition.col == expected_result["opponentFinalPosition"]["col"]
            )

            if match:
                passed += 1
            else:
                failed += 1
                failures.append((idx, test_case, result, expected_result))

            # Progress indicator
            click.echo(f"\r  Progress: {idx}/{len(tests)} tests  ({passed} passed, {failed} failed)", nl=False)

        click.echo("\n")

        # Show failures if any
        if failures:
            click.echo(click.style("\n⚠ Failed Tests:", fg="yellow", bold=True))
            for test_num, test_case, actual, expected in failures:
                click.echo(f"\nTest #{test_num}:")
                if test_case.get("notes"):
                    click.echo(f"  Notes: {test_case['notes']}")
                click.echo(f"  Expected: {expected['winner']} wins, scores {expected['playerPoints']}-{expected['opponentPoints']}")
                click.echo(f"  Actual:   {actual.winner} wins, scores {actual.playerPoints}-{actual.opponentPoints}")

        # Summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"PARITY TEST RESULTS")
        click.echo(f"{'=' * 60}")
        click.echo(f"Total Tests: {len(tests)}")
        click.echo(f"Passed:      {passed} ({100 * passed / len(tests):.1f}%)")
        click.echo(f"Failed:      {failed} ({100 * failed / len(tests):.1f}%)")
        click.echo(f"{'=' * 60}\n")

        if failed == 0:
            click.echo(click.style("✓ All parity tests passed!", fg="green", bold=True))
        else:
            click.echo(click.style(f"✗ {failed} test(s) failed", fg="red", bold=True))
            sys.exit(1)

    except Exception as e:
        import traceback
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("board_file", type=click.Path(exists=True))
@click.option("--verbose", is_flag=True, help="Show details for each board")
def validate(board_file: str, verbose: bool):
    """
    Validate all boards in a JSON file.

    Checks if boards are structurally valid and playable.
    """
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Validating Boards")
    click.echo(f"{'=' * 60}")
    click.echo(f"File: {board_file}\n")

    try:
        boards = load_boards_from_json(board_file)
        click.echo(f"Loaded {len(boards)} boards\n")

        valid_count = 0
        playable_count = 0
        invalid_boards = []

        for idx, board in enumerate(boards):
            is_playable = is_board_playable(board)

            if is_playable:
                valid_count += 1
                playable_count += 1
                if verbose:
                    click.echo(click.style(f"✓ Board {idx}: Valid and playable", fg="green"))
            else:
                valid_count += 1  # Structurally valid
                error_msg = "Not playable (see validation rules)"
                if verbose:
                    click.echo(click.style(f"⚠ Board {idx}: Valid but not playable - {error_msg}", fg="yellow"))
                invalid_boards.append((idx, error_msg))

        # Summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"VALIDATION SUMMARY")
        click.echo(f"{'=' * 60}")
        click.echo(f"Total Boards:    {len(boards)}")
        click.echo(f"Valid:           {valid_count} ({100 * valid_count / len(boards):.1f}%)")
        click.echo(f"Playable:        {playable_count} ({100 * playable_count / len(boards):.1f}%)")
        click.echo(f"Invalid:         {len(boards) - valid_count}")
        click.echo(f"{'=' * 60}\n")

        if invalid_boards and not verbose:
            click.echo(click.style(f"⚠ {len(invalid_boards)} board(s) not playable (use --verbose for details)", fg="yellow"))

        if len(boards) == playable_count:
            click.echo(click.style("✓ All boards are valid and playable!", fg="green", bold=True))
        else:
            click.echo(click.style(f"⚠ {len(boards) - playable_count} board(s) not playable", fg="yellow", bold=True))

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.argument("board_file", type=click.Path(exists=True))
@click.option(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds to simulate (default: 5)",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option("--verbose", is_flag=True, help="Show detailed output for each round")
@click.option("--player-board", type=int, help="Use specific board index for player (from file)")
@click.option("--opponent-board", type=int, help="Use specific board index for opponent (from file)")
def play(board_file: str, rounds: int, seed: Optional[int], verbose: bool, player_board: Optional[int], opponent_board: Optional[int]):
    """
    Play a game by simulating rounds with board selections.

    By default, randomly samples boards for each round. Use --player-board and
    --opponent-board to specify exact boards by index.

    Examples:
      Play with random boards:
        spaces-game play data/boards_size_3.json --rounds 5

      Play specific boards against each other:
        spaces-game play data/boards_size_3.json --player-board 0 --opponent-board 5 --rounds 1

      Use one specific board, random opponents:
        spaces-game play data/boards_size_3.json --player-board 10 --rounds 5
    """
    import random
    import numpy as np

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Playing Spaces Game")
    click.echo(f"{'=' * 60}")
    click.echo(f"Board File: {board_file}")
    click.echo(f"Rounds:     {rounds}")
    if seed is not None:
        click.echo(f"Seed:       {seed}")
    click.echo(f"{'=' * 60}\n")

    try:
        # Load all boards from file
        all_boards = load_boards_from_json(board_file)

        # Validate specific board indices if provided
        if player_board is not None:
            if player_board < 0 or player_board >= len(all_boards):
                click.echo(click.style(f"✗ Player board index {player_board} out of range (0-{len(all_boards)-1})", fg="red"))
                sys.exit(1)
            click.echo(f"Player using board #{player_board} from file")

        if opponent_board is not None:
            if opponent_board < 0 or opponent_board >= len(all_boards):
                click.echo(click.style(f"✗ Opponent board index {opponent_board} out of range (0-{len(all_boards)-1})", fg="red"))
                sys.exit(1)
            click.echo(f"Opponent using board #{opponent_board} from file")

        # Set up board sources
        if player_board is not None and opponent_board is not None:
            # Both boards specified - use same boards for all rounds
            click.echo()
            player_boards = [all_boards[player_board]] * rounds
            opponent_boards = [all_boards[opponent_board]] * rounds
        elif player_board is not None:
            # Player board fixed, opponent random
            click.echo(f"Opponents will be randomly selected from pool\n")
            pool = BoardPool(board_file)
            player_boards = [all_boards[player_board]] * rounds
            opponent_boards = pool.sample(rounds)
        elif opponent_board is not None:
            # Opponent board fixed, player random
            click.echo(f"Player boards will be randomly selected from pool\n")
            pool = BoardPool(board_file)
            player_boards = pool.sample(rounds)
            opponent_boards = [all_boards[opponent_board]] * rounds
        else:
            # Both random - use deck system
            click.echo()
            pool = BoardPool(board_file)
            deck_size = min(10, len(all_boards))
            player_deck = pool.sample(deck_size)
            opponent_deck = pool.sample(deck_size)

            player_boards = []
            opponent_boards = []
            for _ in range(rounds):
                player_boards.append(random.choice(player_deck))
                opponent_boards.append(random.choice(opponent_deck))

        player_score = 0
        opponent_score = 0

        for round_num in range(1, rounds + 1):
            player_board_for_round = player_boards[round_num - 1]
            opponent_board_for_round = opponent_boards[round_num - 1]

            # Simulate
            result = simulate_round(round_num, player_board_for_round, opponent_board_for_round, silent=True)

            player_score += result.playerPoints
            opponent_score += result.opponentPoints

            if verbose:
                click.echo(f"\n{'=' * 60}")
                click.echo(f"ROUND {round_num}")
                click.echo(f"{'=' * 60}")
                _render_result_details(result)
                click.echo(f"\n{'=' * 60}\n")
            else:
                # Display round result
                winner_symbol = {
                    'player': '→',
                    'opponent': '←',
                    'tie': '='
                }[result.winner]

                winner_color = {
                    'player': 'green',
                    'opponent': 'red',
                    'tie': 'yellow'
                }[result.winner]

                click.echo(
                    f"Round {round_num}: "
                    f"Player {result.playerPoints:2d} {click.style(winner_symbol, fg=winner_color)} "
                    f"{result.opponentPoints:2d} Opponent  "
                    f"(Total: {player_score:3d} - {opponent_score:3d})"
                )

        # Final result
        click.echo(f"\n{'=' * 60}")
        click.echo(f"GAME RESULT")
        click.echo(f"{'=' * 60}")
        click.echo(f"Player Score:    {player_score}")
        click.echo(f"Opponent Score:  {opponent_score}")

        if player_score > opponent_score:
            click.echo(click.style("\nPlayer WINS!", fg="green", bold=True))
        elif player_score < opponent_score:
            click.echo(click.style("\nOpponent WINS!", fg="red", bold=True))
        else:
            click.echo(click.style("\nTIE GAME!", fg="yellow", bold=True))

        click.echo(f"{'=' * 60}\n")

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.argument("board_file", type=click.Path(exists=True))
@click.option("--limit", type=int, help="Limit number of boards to display")
@click.option("--index", type=int, help="Show only board at specific index")
def show(board_file: str, limit: Optional[int], index: Optional[int]):
    """
    Display boards from a file with grid visualizations.

    Shows each board with its visual grid, sequence info, and index.
    """
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Boards in File")
    click.echo(f"{'=' * 60}")
    click.echo(f"File: {board_file}\n")

    try:
        boards = load_boards_from_json(board_file)
        click.echo(f"Total boards in file: {len(boards)}\n")

        # Filter by index if specified
        if index is not None:
            if index < 0 or index >= len(boards):
                click.echo(click.style(f"✗ Index {index} out of range (0-{len(boards)-1})", fg="red"))
                sys.exit(1)
            boards_to_show = [(index, boards[index])]
        else:
            # Apply limit if specified
            if limit and limit < len(boards):
                click.echo(f"Showing first {limit} boards (use --limit to show more)\n")
                boards_to_show = list(enumerate(boards[:limit]))
            else:
                boards_to_show = list(enumerate(boards))

        # Display each board
        for idx, board in boards_to_show:
            click.echo(click.style(f"{'=' * 60}", fg="cyan"))
            click.echo(click.style(f"Board #{idx}", bold=True, fg="cyan"))
            click.echo(click.style(f"{'=' * 60}", fg="cyan"))
            click.echo(f"Size: {board.boardSize}x{board.boardSize}")
            click.echo(f"Sequence length: {len(board.sequence)} moves")

            # Count pieces and traps
            pieces = sum(1 for m in board.sequence if m.type == 'piece')
            traps = sum(1 for m in board.sequence if m.type == 'trap')
            click.echo(f"Pieces: {pieces}, Traps: {traps}")

            # Render the board
            click.echo(_render_board(board, f"\nBoard #{idx} Grid"))
            click.echo()

            # Show sequence details if there are few boards
            if len(boards_to_show) <= 5:
                click.echo(click.style("Move Sequence:", bold=True))
                for move in board.sequence:
                    if move.type == 'final':
                        click.echo(f"  {move.order}. {click.style('GOAL', fg='green', bold=True)} at row {move.position.row}")
                    elif move.type == 'piece':
                        click.echo(f"  {move.order}. {click.style('Move', fg='blue')} to ({move.position.row}, {move.position.col})")
                    elif move.type == 'trap':
                        click.echo(f"  {move.order}. {click.style('Trap', fg='red')} at ({move.position.row}, {move.position.col})")
                click.echo()

        if not index and limit and limit < len(boards):
            click.echo(click.style(f"\n... {len(boards) - limit} more boards not shown", fg="yellow"))
            click.echo(f"Use --limit {len(boards)} to show all, or --index N to show a specific board\n")

    except Exception as e:
        import traceback
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        if "--verbose" in sys.argv:
            click.echo(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.argument("board_file", type=click.Path(exists=True))
def stats(board_file: str):
    """
    Display statistics about a board pool.

    Shows size distribution, complexity metrics, and other stats.
    """
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Board Pool Statistics")
    click.echo(f"{'=' * 60}")
    click.echo(f"File: {board_file}\n")

    try:
        boards = load_boards_from_json(board_file)
        click.echo(f"Loaded {len(boards)} boards\n")

        # Size distribution
        size_counts: dict[int, int] = {}
        for board in boards:
            size_counts[board.boardSize] = size_counts.get(board.boardSize, 0) + 1

        click.echo(f"{'=' * 60}")
        click.echo(f"SIZE DISTRIBUTION")
        click.echo(f"{'=' * 60}")

        max_count = max(size_counts.values())
        for size in sorted(size_counts.keys()):
            count = size_counts[size]
            pct = 100 * count / len(boards)
            bar_len = int(50 * count / max_count)
            bar = "█" * bar_len
            click.echo(f"Size {size}:    {count} ({pct:.1f}%) {bar}")

        # Sequence length stats
        seq_lengths = [len(board.sequence) for board in boards]
        avg_len = sum(seq_lengths) / len(seq_lengths)
        min_len = min(seq_lengths)
        max_len = max(seq_lengths)
        import statistics
        std_len = statistics.stdev(seq_lengths) if len(seq_lengths) > 1 else 0

        click.echo(f"\n{'=' * 60}")
        click.echo(f"SEQUENCE LENGTH STATS")
        click.echo(f"{'=' * 60}")
        click.echo(f"Average: {avg_len:.1f} ± {std_len:.1f}")
        click.echo(f"Range:   {min_len} - {max_len}")

        # Playability
        playable_count = sum(1 for board in boards if is_board_playable(board)[0])

        click.echo(f"\n{'=' * 60}")
        click.echo(f"PLAYABILITY")
        click.echo(f"{'=' * 60}")
        click.echo(f"Playable:     {playable_count} ({100 * playable_count / len(boards):.1f}%)")
        click.echo(f"Not Playable: {len(boards) - playable_count} ({100 * (len(boards) - playable_count) / len(boards):.1f}%)")
        click.echo(f"{'=' * 60}\n")

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option("--size", type=int, required=True, help="Board size (2-5)")
@click.option("--limit", type=int, default=1000, help="Maximum number of boards to generate")
@click.option("--output", type=click.Path(), required=True, help="Output JSON file path")
@click.option(
    "--engine-path",
    type=click.Path(exists=True),
    default="../spaces-game-engine",
    help="Path to spaces-game-engine directory",
)
def generate_boards(size: int, limit: int, output: str, engine_path: str):
    """
    Generate boards using TypeScript CLI (convenience wrapper).

    This wraps the TypeScript board generator for convenience.
    """
    import subprocess
    import os

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Generating Boards")
    click.echo(f"{'=' * 60}")
    click.echo(f"Size:        {size}")
    click.echo(f"Limit:       {limit}")
    click.echo(f"Output:      {output}")
    click.echo(f"Engine Path: {engine_path}")
    click.echo(f"{'=' * 60}\n")

    try:
        # Check if engine path exists
        if not os.path.exists(engine_path):
            click.echo(click.style(f"✗ Engine path not found: {engine_path}", fg="red"))
            sys.exit(1)

        # Run TypeScript CLI
        cmd = [
            "npm", "run", "cli", "--",
            "generate-boards",
            "--size", str(size),
            "--limit", str(limit),
            "--output", output,
        ]

        click.echo(f"Running: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            cwd=engine_path,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            click.echo(result.stdout)
            click.echo(click.style(f"\n✓ Successfully generated boards to {output}", fg="green", bold=True))
        else:
            click.echo(result.stdout)
            click.echo(result.stderr)
            click.echo(click.style(f"\n✗ Board generation failed", fg="red", bold=True))
            sys.exit(1)

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        sys.exit(1)


if __name__ == "__main__":
    cli()
