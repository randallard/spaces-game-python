"""
Interactive board builder for creating boards step-by-step.

Matches the TypeScript CLI interactive mode functionality.
"""

from dataclasses import dataclass
from typing import Optional
import click

from .types import Board, BoardMove, Position, CellContent


@dataclass
class BuilderState:
    """State for the interactive board builder."""

    board_size: int
    starting_col: int
    sequence: list[BoardMove]
    current_position: Optional[Position]
    step_count: int
    trap_positions: set[str]
    visited_cells: set[str]
    supermove_active: bool


def _position_key(row: int, col: int) -> str:
    """Generate unique key for position."""
    return f"{row},{col}"


def _create_board_from_sequence(sequence: list[BoardMove], board_size: int) -> Board:
    """
    Create a Board object from a sequence of moves.

    Args:
        sequence: List of moves
        board_size: Size of the board

    Returns:
        Board object with grid filled based on sequence
    """
    # Initialize empty grid
    grid: list[list[CellContent]] = [
        ['empty' for _ in range(board_size)]
        for _ in range(board_size)
    ]

    # Fill grid based on sequence
    for move in sequence:
        row = move.position.row
        col = move.position.col

        # Skip final moves (off-board at row -1)
        if move.type == 'final':
            continue

        # Place pieces and traps (traps override pieces)
        if move.type == 'piece':
            if grid[row][col] == 'empty':
                grid[row][col] = 'piece'
        elif move.type == 'trap':
            grid[row][col] = 'trap'

    # Convert to immutable tuples
    immutable_grid = tuple(tuple(row) for row in grid)
    immutable_sequence = tuple(sequence)

    return Board(
        boardSize=board_size,
        grid=immutable_grid,
        sequence=immutable_sequence
    )


def _get_current_position(sequence: list[BoardMove]) -> Optional[Position]:
    """Get the current piece position from sequence."""
    for move in reversed(sequence):
        if move.type == 'piece':
            return move.position
    return None


def _render_board_state(
    board: Board,
    current_position: Optional[Position],
    title: str = "Current Board"
) -> str:
    """
    Render the board with current position highlighted.

    Args:
        board: Board to render
        current_position: Current piece position (highlighted)
        title: Title to display

    Returns:
        Rendered board as string
    """
    size = board.boardSize
    lines = []

    # Track pieces and traps at each position
    position_contents: dict[tuple[int, int], tuple[list[int], list[int]]] = {}

    for move in board.sequence:
        if move.position.row >= 0:  # Skip final moves
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

    lines.append(click.style(f"\n{title}\n", bold=True))

    # Top border
    lines.append("┌" + "─────────┬" * (size - 1) + "─────────┐")

    for row_idx in range(size):
        row_items = []
        for col_idx in range(size):
            pos = (row_idx, col_idx)
            pieces, traps = position_contents.get(pos, ([], []))

            # Check if this is the current position
            is_current = (current_position and
                         current_position.row == row_idx and
                         current_position.col == col_idx)

            if pieces and traps:
                # Supermove: both piece and trap
                piece_str = click.style(f"{pieces[0]}●", fg="blue", bold=is_current)
                trap_str = click.style(f"{traps[0]}X", fg="red")
                content = f"{piece_str},{trap_str}"
                visible_len = len(str(pieces[0])) + 1 + 1 + len(str(traps[0])) + 1
                padding = 9 - 1 - visible_len
                cell = f" {content}{' ' * padding}"
            elif pieces:
                # Just piece
                num_str = click.style(f"{pieces[0]}●", fg="blue", bold=is_current)
                visible_len = len(str(pieces[0])) + 1
                padding = 9 - 2 - visible_len
                cell = f"  {num_str}{' ' * padding}"
            elif traps:
                # Just trap
                num_str = click.style(f"{traps[0]}X", fg="red")
                visible_len = len(str(traps[0])) + 1
                padding = 9 - 2 - visible_len
                cell = f"  {num_str}{' ' * padding}"
            else:
                # Empty or highlight current position
                if is_current:
                    cell = click.style("    ◉    ", fg="cyan", bold=True)
                else:
                    cell = "         "

            row_items.append(cell)

        lines.append("│" + "│".join(row_items) + "│")

        # Middle border
        if row_idx < size - 1:
            lines.append("├" + "─────────┼" * (size - 1) + "─────────┤")

    # Bottom border
    lines.append("└" + "─────────┴" * (size - 1) + "─────────┘")

    return "\n".join(lines)


def _show_help() -> None:
    """Show help message for interactive builder."""
    click.echo(click.style('\n📖 Interactive Board Builder Help\n', bold=True))

    click.echo(click.style('Movement Commands:', bold=True))
    click.echo('  move <direction>    Move piece (abbreviation: m)')
    click.echo('  trap <direction>    Place trap (abbreviation: t)')
    click.echo('  trap here           Place trap at current position - supermove! (t h)')
    click.echo('  supermove <dir>     Trap here AND move in direction (abbreviation: s)')
    click.echo('  Directions: up/down/left/right (u/d/l/r)')
    click.echo('  Examples: "move left", "m l", "trap right", "t r"')
    click.echo('            "trap here", "t h", "supermove up", "s u"')

    click.echo(click.style('\nCoordinate Entry:', bold=True))
    click.echo('  <row>,<col>,<type>  Direct coordinate entry')
    click.echo('  Types: piece/trap (p/t)')
    click.echo('  Example: "1,2,piece" or "1,2,p"')

    click.echo(click.style('\nSpecial Commands:', bold=True))
    click.echo('  finish (f)          Auto-complete straight path to goal')
    click.echo('  undo (u)            Remove last move')
    click.echo('  restart (r/reset)   Start over from beginning')
    click.echo('  help (h)            Show this help message')
    click.echo('  quit (q)            Exit without saving')

    click.echo(click.style('\nGame Rules:', bold=True))
    click.echo('  • Pieces move orthogonally only (up/down/left/right)')
    click.echo('  • Traps must be adjacent to piece or at piece position (supermove)')
    click.echo('  • Pieces cannot move into traps')
    click.echo('  • After supermove, piece MUST move on next step')
    click.echo('  • Goal is reached at row -1 (top edge)')
    click.echo()


def _parse_direction(direction: str) -> Optional[tuple[int, int]]:
    """Parse direction string to (row_delta, col_delta)."""
    normalized = direction.lower().strip()

    if normalized in ('up', 'u'):
        return (-1, 0)
    elif normalized in ('down', 'd'):
        return (1, 0)
    elif normalized in ('left', 'l'):
        return (0, -1)
    elif normalized in ('right', 'r'):
        return (0, 1)

    return None


def _parse_coordinates(input_str: str) -> Optional[tuple[int, int, str]]:
    """Parse coordinate input: '1,1,piece' or '1,1,p'."""
    parts = input_str.split(',')
    if len(parts) != 3:
        return None

    try:
        row = int(parts[0].strip())
        col = int(parts[1].strip())
        type_str = parts[2].strip().lower()

        if type_str in ('piece', 'p'):
            return (row, col, 'piece')
        elif type_str in ('trap', 't'):
            return (row, col, 'trap')
    except ValueError:
        pass

    return None


def _parse_command(input_str: str) -> dict:
    """
    Parse user command.

    Returns dict with 'type' and additional fields depending on command.
    """
    trimmed = input_str.strip().lower()

    # Special commands
    if trimmed in ('finish', 'f'):
        return {'type': 'finish'}
    if trimmed in ('undo', 'u'):
        return {'type': 'undo'}
    if trimmed in ('restart', 'reset', 'r'):
        return {'type': 'restart'}
    if trimmed in ('help', 'h'):
        return {'type': 'help'}
    if trimmed in ('quit', 'q', 'exit'):
        return {'type': 'quit'}

    # Coordinate input: "1,1,piece"
    if ',' in trimmed:
        coords = _parse_coordinates(trimmed)
        if coords:
            row, col, move_type = coords
            return {
                'type': 'coord',
                'row': row,
                'col': col,
                'move_type': move_type
            }

    # Command with direction
    parts = trimmed.split()
    if len(parts) == 2:
        cmd, dir_str = parts

        # Move command
        if cmd in ('move', 'm'):
            direction = _parse_direction(dir_str)
            if direction:
                return {'type': 'move', 'direction': direction}

        # Trap command
        if cmd in ('trap', 't'):
            if dir_str in ('here', 'h'):
                return {'type': 'supermove'}

            direction = _parse_direction(dir_str)
            if direction:
                return {'type': 'trap', 'direction': direction}

        # Supermove and move
        if cmd in ('supermove', 's'):
            direction = _parse_direction(dir_str)
            if direction:
                return {'type': 'supermove-and-move', 'direction': direction}

    return {'type': 'invalid'}


def build_board_interactive(size: Optional[int] = None, start_col: Optional[int] = None) -> Optional[Board]:
    """
    Build a board interactively with step-by-step prompts.

    Args:
        size: Board size (will prompt if not provided)
        start_col: Starting column (will prompt if not provided)

    Returns:
        Completed Board or None if cancelled
    """
    # Prompt for board size if not provided
    if size is None:
        while True:
            size_input = click.prompt('Board size (2-100)', default='3', type=str)
            try:
                size = int(size_input)
                if size < 2:
                    click.echo(click.style('Please enter a number 2 or greater', fg='red'))
                    continue
                if size > 100:
                    click.echo(click.style('Please enter a number no greater than 100', fg='red'))
                    continue
                break
            except ValueError:
                click.echo(click.style('Please enter a valid number', fg='red'))

    # Prompt for starting column if not provided
    if start_col is None:
        while True:
            col_input = click.prompt(f'Starting column (0-{size-1}, 0 is left)', default='0', type=str)
            try:
                start_col = int(col_input)
                if start_col < 0 or start_col >= size:
                    click.echo(click.style(f'Please enter a number between 0 and {size-1}', fg='red'))
                    continue
                break
            except ValueError:
                click.echo(click.style('Please enter a valid number', fg='red'))

    # Initialize state
    def init_state() -> BuilderState:
        return BuilderState(
            board_size=size,
            starting_col=start_col,
            sequence=[
                BoardMove(
                    position=Position(row=size - 1, col=start_col),
                    type='piece',
                    order=1
                )
            ],
            current_position=Position(row=size - 1, col=start_col),
            step_count=1,
            trap_positions=set(),
            visited_cells={_position_key(size - 1, start_col)},
            supermove_active=False
        )

    state = init_state()

    # Show initial board
    board = _create_board_from_sequence(state.sequence, state.board_size)
    click.echo(_render_board_state(board, state.current_position, "🎮 Starting Board"))
    click.echo(click.style('\nType "help" for commands, "finish" when done\n', fg='cyan'))

    # Main command loop
    while True:
        try:
            command_input = click.prompt('Command', type=str)
        except (KeyboardInterrupt, EOFError):
            click.echo('\n\nExiting...')
            return None

        command = _parse_command(command_input)

        if command['type'] == 'invalid':
            click.echo(click.style('❌ Invalid command. Type "help" for available commands.\n', fg='red'))
            continue

        if command['type'] == 'help':
            _show_help()
            continue

        if command['type'] == 'quit':
            click.echo('Exiting without saving...')
            return None

        if command['type'] == 'restart':
            if click.confirm('Are you sure you want to restart?', default=False):
                state = init_state()
                board = _create_board_from_sequence(state.sequence, state.board_size)
                click.echo(click.style('\n🔄 Restarting...\n', fg='yellow'))
                click.echo(_render_board_state(board, state.current_position))
                click.echo()
            continue

        if command['type'] == 'undo':
            if len(state.sequence) <= 1:
                click.echo(click.style('⚠️  Cannot undo - already at starting position\n', fg='yellow'))
                continue

            # Remove last move
            last_move = state.sequence.pop()
            state.step_count -= 1

            # Update trap positions
            if last_move.type == 'trap':
                key = _position_key(last_move.position.row, last_move.position.col)
                state.trap_positions.discard(key)

            # Update visited cells if it was a piece move
            if last_move.type == 'piece':
                key = _position_key(last_move.position.row, last_move.position.col)
                state.visited_cells.discard(key)

            # Update current position
            state.current_position = _get_current_position(state.sequence)

            # Clear supermove state
            state.supermove_active = False

            board = _create_board_from_sequence(state.sequence, state.board_size)
            click.echo(click.style('↩️  Last move undone\n', fg='yellow'))
            click.echo(_render_board_state(board, state.current_position))
            click.echo()
            continue

        if command['type'] == 'finish':
            if not state.current_position:
                click.echo(click.style('❌ No current position\n', fg='red'))
                continue

            row = state.current_position.row
            col = state.current_position.col

            # Check for traps in forward path
            trap_in_path = False
            for r in range(row - 1, -1, -1):
                if _position_key(r, col) in state.trap_positions:
                    trap_in_path = True
                    break

            if trap_in_path:
                click.echo(click.style('❌ Cannot finish - trap in forward path\n', fg='red'))
                continue

            # Add moves to goal
            for r in range(row - 1, -1, -1):
                state.step_count += 1
                state.sequence.append(BoardMove(
                    position=Position(row=r, col=col),
                    type='piece',
                    order=state.step_count
                ))

            # Add final move
            state.step_count += 1
            state.sequence.append(BoardMove(
                position=Position(row=-1, col=col),
                type='final',
                order=state.step_count
            ))

            state.current_position = Position(row=-1, col=col)

            board = _create_board_from_sequence(state.sequence, state.board_size)
            click.echo(click.style('✅ Auto-completed path to goal!\n', fg='green'))
            click.echo(_render_board_state(board, None, "Completed Board"))
            click.echo()

            # Validate
            from .validation import is_board_playable
            if not is_board_playable(board):
                click.echo(click.style('❌ Board validation failed\n', fg='red'))
                continue

            if click.confirm('Confirm board?', default=True):
                return board
            else:
                # Undo finish moves
                click.echo(click.style('\n↩️  Returning to building...\n', fg='yellow'))
                while state.sequence and state.sequence[-1].type in ('final', 'piece'):
                    last = state.sequence[-1]
                    if last.position.row == state.board_size - 1:
                        break
                    state.sequence.pop()
                    state.step_count -= 1

                state.current_position = _get_current_position(state.sequence)
                board = _create_board_from_sequence(state.sequence, state.board_size)
                click.echo(_render_board_state(board, state.current_position))
                click.echo()
            continue

        # Handle supermove-and-move
        if command['type'] == 'supermove-and-move':
            if not state.current_position:
                click.echo(click.style('❌ No current position\n', fg='red'))
                continue

            row_delta, col_delta = command['direction']
            final_row = state.current_position.row + row_delta
            final_col = state.current_position.col + col_delta

            # Validate move
            if final_row < 0 or final_row >= state.board_size or final_col < 0 or final_col >= state.board_size:
                dir_name = {(-1, 0): 'up', (1, 0): 'down', (0, -1): 'left', (0, 1): 'right'}[command['direction']]
                click.echo(click.style(f'❌ Cannot supermove {dir_name} - would move off board\n', fg='red'))
                continue

            cell_key = _position_key(final_row, final_col)
            if cell_key in state.visited_cells:
                click.echo(click.style('❌ Cannot move to a previously visited cell\n', fg='red'))
                continue

            if cell_key in state.trap_positions:
                click.echo(click.style('❌ Cannot move into a trap\n', fg='red'))
                continue

            # Place trap at current position
            state.step_count += 1
            state.sequence.append(BoardMove(
                position=state.current_position,
                type='trap',
                order=state.step_count
            ))
            state.trap_positions.add(_position_key(state.current_position.row, state.current_position.col))

            # Move piece
            state.step_count += 1
            state.sequence.append(BoardMove(
                position=Position(row=final_row, col=final_col),
                type='piece',
                order=state.step_count
            ))
            state.current_position = Position(row=final_row, col=final_col)
            state.visited_cells.add(cell_key)
            state.supermove_active = False

            board = _create_board_from_sequence(state.sequence, state.board_size)
            click.echo(_render_board_state(board, state.current_position))
            click.echo()
            continue

        # Handle regular supermove (trap at current position)
        if command['type'] == 'supermove':
            if not state.current_position:
                click.echo(click.style('❌ No current position\n', fg='red'))
                continue

            # Place trap at current position
            state.step_count += 1
            state.sequence.append(BoardMove(
                position=state.current_position,
                type='trap',
                order=state.step_count
            ))
            state.trap_positions.add(_position_key(state.current_position.row, state.current_position.col))
            state.supermove_active = True

            board = _create_board_from_sequence(state.sequence, state.board_size)
            click.echo(_render_board_state(board, state.current_position))
            click.echo(click.style('⚠️  Supermove! Must move piece on next turn\n', fg='yellow'))
            continue

        # Handle move command
        if command['type'] == 'move':
            if not state.current_position:
                click.echo(click.style('❌ No current position\n', fg='red'))
                continue

            row_delta, col_delta = command['direction']
            new_row = state.current_position.row + row_delta
            new_col = state.current_position.col + col_delta

            # Validate move
            if new_row < 0 or new_row >= state.board_size or new_col < 0 or new_col >= state.board_size:
                click.echo(click.style('❌ Cannot move off board\n', fg='red'))
                continue

            cell_key = _position_key(new_row, new_col)
            if cell_key in state.visited_cells:
                click.echo(click.style('❌ Cannot move to a previously visited cell\n', fg='red'))
                continue

            if cell_key in state.trap_positions:
                click.echo(click.style('❌ Cannot move into a trap\n', fg='red'))
                continue

            # Move piece
            state.step_count += 1
            state.sequence.append(BoardMove(
                position=Position(row=new_row, col=new_col),
                type='piece',
                order=state.step_count
            ))
            state.current_position = Position(row=new_row, col=new_col)
            state.visited_cells.add(cell_key)
            state.supermove_active = False

            board = _create_board_from_sequence(state.sequence, state.board_size)
            click.echo(_render_board_state(board, state.current_position))
            click.echo()
            continue

        # Handle trap command (adjacent)
        if command['type'] == 'trap':
            if not state.current_position:
                click.echo(click.style('❌ No current position\n', fg='red'))
                continue

            if state.supermove_active:
                click.echo(click.style('❌ Must move piece after supermove\n', fg='red'))
                continue

            row_delta, col_delta = command['direction']
            trap_row = state.current_position.row + row_delta
            trap_col = state.current_position.col + col_delta

            # Validate trap position
            if trap_row < 0 or trap_row >= state.board_size or trap_col < 0 or trap_col >= state.board_size:
                click.echo(click.style('❌ Cannot place trap off board\n', fg='red'))
                continue

            cell_key = _position_key(trap_row, trap_col)
            if cell_key in state.trap_positions:
                click.echo(click.style('❌ Trap already exists at that position\n', fg='red'))
                continue

            if cell_key in state.visited_cells:
                click.echo(click.style('❌ Cannot place trap on visited cell\n', fg='red'))
                continue

            # Place trap
            state.step_count += 1
            state.sequence.append(BoardMove(
                position=Position(row=trap_row, col=trap_col),
                type='trap',
                order=state.step_count
            ))
            state.trap_positions.add(cell_key)

            board = _create_board_from_sequence(state.sequence, state.board_size)
            click.echo(_render_board_state(board, state.current_position))
            click.echo()
            continue

        # Handle coordinate entry
        if command['type'] == 'coord':
            click.echo(click.style('⚠️  Direct coordinate entry not yet implemented. Use move/trap commands.\n', fg='yellow'))
            continue

    return None
