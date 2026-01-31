"""
Parity tests - Validate Python implementation matches TypeScript exactly.

These tests load the TypeScript test session (52 real test cases) and will
eventually run them through the Python simulation engine to verify identical results.

For now, we test that we can load and parse the test data correctly.
Once simulation.py is ported, we'll add the actual parity validation.
"""

import json
from pathlib import Path
import pytest

from spaces_game.board_loader import load_board_from_dict
from spaces_game.types import Board


# Path to TypeScript test session
TEST_SESSION_PATH = Path(__file__).parent / 'fixtures' / 'session-2026-01-30T13-28-47-534Z.json'


@pytest.fixture
def test_session():
    """Load TypeScript test session."""
    with open(TEST_SESSION_PATH) as f:
        return json.load(f)


class TestLoadTestSession:
    """Test loading TypeScript test session data."""

    def test_session_metadata(self, test_session):
        """Verify session metadata."""
        assert test_session['id'] == 'session-2026-01-30T13-28-47-534Z'
        assert test_session['name'] == 'Board Testing & Validation'
        assert 'size-tests' in test_session['tags']

    def test_session_has_tests(self, test_session):
        """Verify session contains test cases."""
        assert 'tests' in test_session
        tests = test_session['tests']

        # Should have 52 test cases
        assert len(tests) >= 52, f"Expected at least 52 tests, got {len(tests)}"

    def test_load_test_case_boards(self, test_session):
        """Verify we can load boards from test cases."""
        test_case = test_session['tests'][0]

        # Load player board
        player_board = load_board_from_dict(test_case['playerBoard'])
        assert isinstance(player_board, Board)
        assert player_board.boardSize == 2

        # Load opponent board
        opponent_board = load_board_from_dict(test_case['opponentBoard'])
        assert isinstance(opponent_board, Board)
        assert opponent_board.boardSize == 2

    def test_all_test_cases_loadable(self, test_session):
        """Verify all test cases can be loaded."""
        for i, test_case in enumerate(test_session['tests']):
            try:
                player_board = load_board_from_dict(test_case['playerBoard'])
                opponent_board = load_board_from_dict(test_case['opponentBoard'])

                assert isinstance(player_board, Board)
                assert isinstance(opponent_board, Board)
            except Exception as e:
                pytest.fail(f"Failed to load test case {i}: {e}")

    def test_result_structure(self, test_session):
        """Verify test results have expected structure."""
        test_case = test_session['tests'][0]
        result = test_case['result']

        # Check required fields
        assert 'winner' in result
        assert result['winner'] in ['player', 'opponent', 'tie']
        assert 'playerScore' in result
        assert 'opponentScore' in result
        assert 'playerFinalPosition' in result
        assert 'opponentFinalPosition' in result
        assert 'collision' in result


class TestParityFramework:
    """
    Framework for parity testing.

    These tests will be expanded once simulation.py is ported.
    For now, they demonstrate the testing approach.
    """

    def test_board_sizes_match(self, test_session):
        """Verify player and opponent boards have matching sizes."""
        for test_case in test_session['tests']:
            player_board = load_board_from_dict(test_case['playerBoard'])
            opponent_board = load_board_from_dict(test_case['opponentBoard'])

            assert player_board.boardSize == opponent_board.boardSize

    def test_simulation_parity_test_1(self, test_session):
        """
        Run test case 1 through Python simulation and compare with TypeScript result.
        """
        from spaces_game import simulate_round

        test_case = test_session['tests'][0]

        player_board = load_board_from_dict(test_case['playerBoard'])
        opponent_board = load_board_from_dict(test_case['opponentBoard'])
        expected_result = test_case['result']

        result = simulate_round(1, player_board, opponent_board, silent=True)

        assert result.winner == expected_result['winner']
        assert result.playerPoints == expected_result['playerScore']
        assert result.opponentPoints == expected_result['opponentScore']
        assert result.playerFinalPosition.row == expected_result['playerFinalPosition']['row']
        assert result.playerFinalPosition.col == expected_result['playerFinalPosition']['col']
        assert result.opponentFinalPosition.row == expected_result['opponentFinalPosition']['row']
        assert result.opponentFinalPosition.col == expected_result['opponentFinalPosition']['col']
        assert result.collision == expected_result['collision']

    def test_all_52_tests_match_typescript(self, test_session):
        """
        Run all 52 test cases and verify they match TypeScript results exactly.

        This is the main parity test - every field must match.
        """
        from spaces_game import simulate_round

        failures = []
        for i, test_case in enumerate(test_session['tests']):
            player_board = load_board_from_dict(test_case['playerBoard'])
            opponent_board = load_board_from_dict(test_case['opponentBoard'])
            expected_result = test_case['result']

            result = simulate_round(1, player_board, opponent_board, silent=True)

            # Compare all fields
            if result.winner != expected_result['winner']:
                failures.append(f"Test {i}: winner mismatch (got {result.winner}, expected {expected_result['winner']})")
            if result.playerPoints != expected_result['playerScore']:
                failures.append(f"Test {i}: playerPoints mismatch (got {result.playerPoints}, expected {expected_result['playerScore']})")
            if result.opponentPoints != expected_result['opponentScore']:
                failures.append(f"Test {i}: opponentPoints mismatch (got {result.opponentPoints}, expected {expected_result['opponentScore']})")
            if result.playerFinalPosition.row != expected_result['playerFinalPosition']['row']:
                failures.append(f"Test {i}: playerFinalPosition.row mismatch")
            if result.playerFinalPosition.col != expected_result['playerFinalPosition']['col']:
                failures.append(f"Test {i}: playerFinalPosition.col mismatch")
            if result.opponentFinalPosition.row != expected_result['opponentFinalPosition']['row']:
                failures.append(f"Test {i}: opponentFinalPosition.row mismatch")
            if result.opponentFinalPosition.col != expected_result['opponentFinalPosition']['col']:
                failures.append(f"Test {i}: opponentFinalPosition.col mismatch")
            if result.collision != expected_result['collision']:
                failures.append(f"Test {i}: collision mismatch")

        if failures:
            pytest.fail(f"Parity failures:\n" + "\n".join(failures))


class TestParityStatistics:
    """Analyze the test session to understand what we're testing."""

    def test_board_size_distribution(self, test_session):
        """Show distribution of board sizes in test cases."""
        from collections import Counter

        sizes = []
        for test_case in test_session['tests']:
            player_board = load_board_from_dict(test_case['playerBoard'])
            sizes.append(player_board.boardSize)

        distribution = Counter(sizes)
        print("\nBoard size distribution:")
        for size, count in sorted(distribution.items()):
            print(f"  Size {size}: {count} tests")

        # We should have tests for various sizes
        assert len(distribution) > 0

    def test_winner_distribution(self, test_session):
        """Show distribution of winners in test cases."""
        from collections import Counter

        winners = [test['result']['winner'] for test in test_session['tests']]
        distribution = Counter(winners)

        print("\nWinner distribution:")
        for winner, count in distribution.items():
            print(f"  {winner}: {count} tests")

        # Should have variety of outcomes
        assert len(distribution) > 0
