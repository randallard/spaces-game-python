# Spaces Game CLI

Command-line interface for Spaces Game operations.

## Installation

The CLI is automatically installed when you install the package:

```bash
pip install -e .
```

This creates the `spaces-game` command.

## Commands

### `test` - Run Single Simulation Test

Run a single simulation test with detailed output, similar to TypeScript CLI.

```bash
# Test with specific board indices
spaces-game test data/boards_size_3.json --player-index 0 --opponent-index 1

# Defaults to indices 0 and 1
spaces-game test data/boards_size_3.json
```

**Output:**
```
Running simulation...

🎮 Simulation Results

Player                 Opponent

┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
│ 3●  │     │     │    │     │     │ 1●  │
├─────┼─────┼─────┤    ├─────┼─────┼─────┤
│ 2●  │     │     │    │     │ 3●  │ 2●  │
├─────┼─────┼─────┤    ├─────┼─────┼─────┤
│ 1●  │     │     │    │     │ 4●  │     │
└─────┴─────┴─────┘    └─────┴─────┴─────┘

📋 Technical Explanation

Player starts with piece at (2, 0)
Opponent starts with piece at (0, 2)

Player moves to (2, 0)
Opponent moves to (0, 2)
Player moves to (1, 0)
  Player +1 point (forward movement)
Opponent moves to (1, 2)
  Opponent +1 point (forward movement)
Player moves to (0, 0)
  Player +1 point (forward movement)
Opponent moves to (1, 1)
Player reaches the goal!
  Player +1 point (goal reached)

Round ends - Player reached the goal!

🏆 PLAYER WINS

Scores:
  Player:   3 points
  Opponent: 2 points

Final Positions:
  Player:   row -1, col 0
  Opponent: row 2, col 1
```

### `test-parity` - Run Parity Tests

Verifies that Python simulation produces identical results to TypeScript for all 52 test cases.

```bash
# Run with default test session
spaces-game test-parity

# Run with custom test session
spaces-game test-parity --test-file path/to/session.json
```

**Output:**
```
============================================================
Running Parity Tests
============================================================
Test Session: tests/fixtures/session-2026-01-30T13-28-47-534Z.json

Session: Board Testing & Validation
Loaded 52 test cases

  Progress: 52/52 tests  (52 passed, 0 failed)

============================================================
PARITY TEST RESULTS
============================================================
Total Tests: 52
Passed:      52 (100.0%)
Failed:      0 (0.0%)
============================================================

✓ All parity tests passed!
```

### `validate` - Validate Board Files

Checks if all boards in a JSON file are valid and playable.

```bash
# Validate board file
spaces-game validate data/boards_size_3.json

# Validate with verbose output (shows each board)
spaces-game validate data/boards_size_3.json --verbose
```

**Output:**
```
============================================================
Validating Boards
============================================================
File: data/boards_size_3.json

Loaded 500 boards

============================================================
VALIDATION SUMMARY
============================================================
Total Boards:    500
Valid:           500 (100.0%)
Playable:        500 (100.0%)
Invalid:         0
============================================================
```

### `play` - Play a Game

Simulate a game with random board selections.

```bash
# Play 5 rounds
spaces-game play data/boards_size_3.json --rounds 5

# Play with seed for reproducibility
spaces-game play data/boards_size_3.json --rounds 5 --seed 42

# Play with detailed output for each round
spaces-game play data/boards_size_3.json --rounds 5 --seed 42 --verbose
```

**Output:**
```
============================================================
Playing Spaces Game
============================================================
Board File: data/boards_size_3.json
Rounds:     5
Seed:       42
============================================================

Round 1: Player  0 =  0 Opponent  (Total:   0 -   0)
Round 2: Player  0 =  0 Opponent  (Total:   0 -   0)
Round 3: Player  3 →  0 Opponent  (Total:   3 -   0)
Round 4: Player  0 ←  3 Opponent  (Total:   3 -   3)
Round 5: Player  1 ←  3 Opponent  (Total:   4 -   6)

============================================================
GAME RESULT
============================================================
Player Score:    4
Opponent Score:  6

Opponent WINS!
============================================================
```

Legend:
- `→` Player wins the round
- `←` Opponent wins the round
- `=` Tie

### `stats` - Board Pool Statistics

Display statistics about a board pool.

```bash
spaces-game stats data/boards_size_3.json
```

**Output:**
```
============================================================
Board Pool Statistics
============================================================
File: data/boards_size_3.json

Loaded 500 boards

============================================================
SIZE DISTRIBUTION
============================================================
Size 3:    500 (100.0%) ██████████████████████████████████████████████████

============================================================
SEQUENCE LENGTH STATS
============================================================
Average: 8.3 ± 1.6
Range:   4 - 12

============================================================
PLAYABILITY
============================================================
Playable:     500 (100.0%)
Not Playable: 0 (0.0%)
============================================================
```

### `generate-boards` - Generate Boards

Convenience wrapper for the TypeScript board generator.

```bash
# Generate 1000 size-3 boards
spaces-game generate-boards --size 3 --limit 1000 --output data/my_boards.json

# Generate with custom engine path
spaces-game generate-boards \
  --size 4 \
  --limit 5000 \
  --output data/size4.json \
  --engine-path /path/to/spaces-game-engine
```

**Options:**
- `--size N`: Board size (2-5, required)
- `--limit N`: Maximum boards to generate (default: 1000)
- `--output PATH`: Output JSON file path (required)
- `--engine-path PATH`: Path to spaces-game-engine directory (default: `../spaces-game-engine`)

## Usage in Scripts

The CLI can be used in shell scripts for automation:

```bash
#!/bin/bash

# Validate all board pools
for file in data/*.json; do
    echo "Validating $file..."
    spaces-game validate "$file" || exit 1
done

# Run parity tests
spaces-game test-parity || exit 1

echo "All checks passed!"
```

## Development Workflow

### After Making Changes

1. **Run parity tests** to ensure TypeScript compatibility:
   ```bash
   spaces-game test-parity
   ```

2. **Validate generated boards**:
   ```bash
   spaces-game validate data/boards_size_3.json
   ```

3. **Check board pool statistics**:
   ```bash
   spaces-game stats data/boards_size_3.json
   ```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Parity Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e .
      - run: spaces-game test-parity
      - run: spaces-game validate data/boards_size_3.json
```

## Exit Codes

- `0`: Success
- `1`: Failure (validation errors, test failures, etc.)
- `2`: Invalid arguments or command-line errors

## Help

Get help for any command:

```bash
spaces-game --help
spaces-game test-parity --help
spaces-game validate --help
spaces-game play --help
spaces-game stats --help
spaces-game generate-boards --help
```
