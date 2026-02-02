"""
Command-line interface for Spaces Game.

Provides commands for board generation, validation, simulation, and testing.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Optional

import click

from .types import Board
from .board_loader import load_boards_from_json, BoardPool
from .validation import validate_board, is_board_playable
from .simulation import simulate_round


@click.group()
@click.version_option(version="0.1.0", prog_name="spaces-game")
def cli():
    """Spaces Game - Python CLI for board generation, validation, and simulation."""
    pass


@cli.command()
@click.option(
    "--size",
    type=int,
    required=True,
    help="Board size (2-5)",
)
@click.option(
    "--limit",
    type=int,
    default=1000,
    help="Maximum number of boards to generate (default: 1000)",
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output JSON file path",
)
@click.option(
    "--engine-path",
    type=click.Path(exists=True),
    default="../spaces-game-engine",
    help="Path to spaces-game-engine directory (default: ../spaces-game-engine)",
)
def generate_boards(size: int, limit: int, output: str, engine_path: str):
    """
    Generate boards using the TypeScript CLI.

    This command wraps the TypeScript board generator for convenience.
    """
    if not 2 <= size <= 5:
        click.echo(click.style("Error: Board size must be between 2 and 5", fg="red"))
        sys.exit(1)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Generating Boards")
    click.echo(f"{'=' * 60}")
    click.echo(f"Size:   {size}")
    click.echo(f"Limit:  {limit}")
    click.echo(f"Output: {output}")
    click.echo(f"{'=' * 60}\n")

    # Construct command
    engine_path = Path(engine_path).resolve()
    output_path = Path(output).resolve()

    cmd = [
        "npm", "run", "cli", "--",
        "generate-boards",
        "--size", str(size),
        "--limit", str(limit),
        "--output", str(output_path),
    ]

    # Run TypeScript CLI
    try:
        result = subprocess.run(
            cmd,
            cwd=engine_path,
            check=True,
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)

        click.echo(click.style("\n✓ Board generation complete!", fg="green"))

    except subprocess.CalledProcessError as e:
        click.echo(click.style(f"\n✗ Board generation failed!", fg="red"))
        click.echo(e.stdout)
        click.echo(e.stderr, err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo(click.style(
            f"\n✗ Error: Could not find TypeScript engine at {engine_path}",
            fg="red"
        ))
        click.echo("Use --engine-path to specify the correct path.")
        sys.exit(1)


@cli.command()
@click.argument("board_file", type=click.Path(exists=True))
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed validation results for each board",
)
def validate(board_file: str, verbose: bool):
    """
    Validate boards in a JSON file.

    Checks if all boards are valid and playable.
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
        errors = []

        for i, board in enumerate(boards):
            result = validate_board(board)
            playable = is_board_playable(board)

            if result.valid:
                valid_count += 1
            else:
                errors.append((i, result.errors))

            if playable:
                playable_count += 1

            if verbose:
                status = "✓" if result.valid else "✗"
                playable_str = "playable" if playable else "not playable"
                click.echo(f"  Board {i:4d}: {status} {playable_str}")

        # Summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"VALIDATION SUMMARY")
        click.echo(f"{'=' * 60}")
        click.echo(f"Total Boards:    {len(boards)}")
        click.echo(f"Valid:           {valid_count} ({valid_count/len(boards)*100:.1f}%)")
        click.echo(f"Playable:        {playable_count} ({playable_count/len(boards)*100:.1f}%)")
        click.echo(f"Invalid:         {len(errors)}")

        if errors:
            click.echo(f"\n{click.style('VALIDATION ERRORS', fg='red')}:")
            for board_idx, board_errors in errors[:10]:  # Show first 10
                click.echo(f"\nBoard {board_idx}:")
                for error in board_errors:
                    click.echo(f"  - {error}")

            if len(errors) > 10:
                click.echo(f"\n... and {len(errors) - 10} more boards with errors")

        click.echo(f"{'=' * 60}\n")

        if errors:
            sys.exit(1)

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
def play(board_file: str, rounds: int, seed: Optional[int]):
    """
    Play a game by simulating rounds with random board selections.

    Demonstrates game simulation with nice terminal output.
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
        pool = BoardPool(board_file)
        deck_size = 10
        player_deck = pool.sample(deck_size)
        opponent_deck = pool.sample(deck_size)

        player_score = 0
        opponent_score = 0

        for round_num in range(1, rounds + 1):
            # Random board selection
            player_board_idx = random.randint(0, deck_size - 1)
            opponent_board_idx = random.randint(0, deck_size - 1)

            player_board = player_deck[player_board_idx]
            opponent_board = opponent_deck[opponent_board_idx]

            # Simulate
            result = simulate_round(round_num, player_board, opponent_board, silent=True)

            player_score += result.playerPoints
            opponent_score += result.opponentPoints

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
        sizes = {}
        sequence_lengths = []
        playable_count = 0

        for board in boards:
            size = board.boardSize
            sizes[size] = sizes.get(size, 0) + 1
            sequence_lengths.append(len(board.sequence))

            if is_board_playable(board):
                playable_count += 1

        # Calculate stats
        import numpy as np
        avg_seq_len = np.mean(sequence_lengths)
        std_seq_len = np.std(sequence_lengths)
        min_seq_len = np.min(sequence_lengths)
        max_seq_len = np.max(sequence_lengths)

        click.echo(f"{'=' * 60}")
        click.echo(f"SIZE DISTRIBUTION")
        click.echo(f"{'=' * 60}")
        for size in sorted(sizes.keys()):
            count = sizes[size]
            pct = count / len(boards) * 100
            bar = '█' * int(pct / 2)
            click.echo(f"Size {size}: {count:6d} ({pct:5.1f}%) {bar}")

        click.echo(f"\n{'=' * 60}")
        click.echo(f"SEQUENCE LENGTH STATS")
        click.echo(f"{'=' * 60}")
        click.echo(f"Average: {avg_seq_len:.1f} ± {std_seq_len:.1f}")
        click.echo(f"Range:   {min_seq_len} - {max_seq_len}")

        click.echo(f"\n{'=' * 60}")
        click.echo(f"PLAYABILITY")
        click.echo(f"{'=' * 60}")
        click.echo(f"Playable:     {playable_count} ({playable_count/len(boards)*100:.1f}%)")
        click.echo(f"Not Playable: {len(boards) - playable_count} ({(len(boards)-playable_count)/len(boards)*100:.1f}%)")

        click.echo(f"{'=' * 60}\n")

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option(
    "--test-file",
    type=click.Path(exists=True),
    default="tests/fixtures/session-2026-01-30T13-28-47-534Z.json",
    help="Path to TypeScript test session file",
)
def test_parity(test_file: str):
    """
    Run parity tests against TypeScript implementation.

    Verifies that Python simulation produces identical results to TypeScript.
    """
    from .board_loader import load_board_from_dict

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Running Parity Tests")
    click.echo(f"{'=' * 60}")
    click.echo(f"Test Session: {test_file}\n")

    try:
        # Load test session
        with open(test_file, 'r') as f:
            session = json.load(f)

        test_cases = session.get('tests', [])
        click.echo(f"Session: {session.get('name', 'Unknown')}")
        click.echo(f"Loaded {len(test_cases)} test cases\n")

        passed = 0
        failed = 0
        failures = []

        for i, test_case in enumerate(test_cases, 1):
            # Extract test data
            test_num = test_case.get('testNumber', i)
            player_board_dict = test_case['playerBoard']
            opponent_board_dict = test_case['opponentBoard']
            expected = test_case['result']

            # Convert dict to Board objects
            player_board = load_board_from_dict(player_board_dict)
            opponent_board = load_board_from_dict(opponent_board_dict)

            # Use round 1 for all tests (doesn't affect simulation)
            round_num = 1

            # Run simulation
            result = simulate_round(round_num, player_board, opponent_board, silent=True)

            # Compare - note field name differences
            matches = (
                result.playerPoints == expected['playerScore'] and
                result.opponentPoints == expected['opponentScore'] and
                result.winner == expected['winner']
            )

            if matches:
                passed += 1
            else:
                failed += 1
                failures.append({
                    'test': test_num,
                    'expected': expected,
                    'actual': {
                        'playerScore': result.playerPoints,
                        'opponentScore': result.opponentPoints,
                        'winner': result.winner,
                    }
                })

            # Progress indicator
            if i % 10 == 0 or i == len(test_cases):
                click.echo(f"  Progress: {i}/{len(test_cases)} tests", nl=False)
                click.echo(f"  ({passed} passed, {failed} failed)", nl=True if i == len(test_cases) else False)
                if i != len(test_cases):
                    click.echo("\r", nl=False)

        # Summary
        click.echo(f"\n{'=' * 60}")
        click.echo(f"PARITY TEST RESULTS")
        click.echo(f"{'=' * 60}")
        click.echo(f"Total Tests: {len(test_cases)}")
        click.echo(click.style(f"Passed:      {passed} ({passed/len(test_cases)*100:.1f}%)", fg="green" if passed == len(test_cases) else "yellow"))
        click.echo(click.style(f"Failed:      {failed} ({failed/len(test_cases)*100:.1f}%)", fg="red" if failed > 0 else "green"))

        if failures:
            click.echo(f"\n{click.style('FAILURES', fg='red')}:")
            for failure in failures[:5]:  # Show first 5
                click.echo(f"\nTest {failure['test']}:")
                click.echo(f"  Expected: {failure['expected']}")
                click.echo(f"  Actual:   {failure['actual']}")

            if len(failures) > 5:
                click.echo(f"\n... and {len(failures) - 5} more failures")

        click.echo(f"{'=' * 60}\n")

        if failed > 0:
            sys.exit(1)
        else:
            click.echo(click.style("✓ All parity tests passed!", fg="green", bold=True))

    except Exception as e:
        click.echo(click.style(f"\n✗ Error: {e}", fg="red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
