"""
Test the BoardBuilderEnv to validate sequential board construction.

This script tests the Stage 2 environment where agents BUILD boards
step-by-step using sequential actions (cell, type, done) rather than
selecting from a pre-made library.

Usage:
    python examples/test_board_builder_env.py
"""

from spaces_game import BoardBuilderEnv
import numpy as np


def test_basic_construction():
    """Test basic construction flow: place pieces/traps, finish board."""
    print("=" * 70)
    print("TEST 1: Basic Construction Flow")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
        max_construction_steps=10,
    )

    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successfully")
    print(f"  Initial construction step: {obs['construction_step']}")
    print(f"  Valid cells mask: {obs['valid_cells_mask']}")

    # Step 1: Place first piece in bottom row (must be valid)
    # For 2x2 board, bottom row is row 1, cells 2 and 3
    action = {
        "cell": 2,      # Bottom-left (1,0)
        "type": 0,      # Piece
        "done": 0,      # Continue
    }

    obs, reward, done, truncated, info = env.step(action)
    print(f"\n✓ Step 1: Placed piece at cell 2 (bottom-left)")
    print(f"  Construction step: {obs['construction_step']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Valid cells mask: {obs['valid_cells_mask']}")
    print(f"  Building board:\n{obs['building_board'][:, :, 0]}")  # Piece orders

    # Step 2: Place adjacent piece
    action = {
        "cell": 3,      # Bottom-right (1,1), adjacent to first piece
        "type": 0,      # Piece
        "done": 0,      # Continue
    }

    obs, reward, done, truncated, info = env.step(action)
    print(f"\n✓ Step 2: Placed piece at cell 3 (bottom-right)")
    print(f"  Construction step: {obs['construction_step']}")
    print(f"  Building board:\n{obs['building_board'][:, :, 0]}")  # Piece orders

    # Step 3: Place trap on top of piece (supermove)
    action = {
        "cell": 2,      # Same as first piece
        "type": 1,      # Trap
        "done": 0,      # Continue
    }

    obs, reward, done, truncated, info = env.step(action)
    print(f"\n✓ Step 3: Placed trap at cell 2 (supermove)")
    print(f"  Construction step: {obs['construction_step']}")
    print(f"  Piece grid:\n{obs['building_board'][:, :, 0]}")
    print(f"  Trap grid:\n{obs['building_board'][:, :, 1]}")

    # Step 4: Finish board
    action = {
        "cell": 0,      # Doesn't matter
        "type": 0,      # Doesn't matter
        "done": 1,      # FINISH
    }

    obs, reward, done, truncated, info = env.step(action)
    print(f"\n✓ Step 4: Finished board")
    print(f"  Round: {info['round']}")
    print(f"  Agent score: {info['agent_total_score']}")
    print(f"  Opponent score: {info['opponent_total_score']}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Episode done: {done}")

    if not done:
        print(f"\n✓ Round 1 complete, construction reset for round 2")
        print(f"  New construction step: {obs['construction_step']}")

    env.close()
    print("\n" + "=" * 70)


def test_action_masking():
    """Test that action masking correctly identifies valid cells."""
    print("=" * 70)
    print("TEST 2: Action Masking")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    # Initially: only bottom row should be valid
    print(f"Initial valid cells: {obs['valid_cells_mask']}")
    print(f"Expected: [0, 0, 1, 1] (only cells 2,3 in bottom row)")

    assert obs['valid_cells_mask'][0] == 0, "Top-left should be invalid initially"
    assert obs['valid_cells_mask'][1] == 0, "Top-right should be invalid initially"
    assert obs['valid_cells_mask'][2] == 1, "Bottom-left should be valid (bottom row)"
    assert obs['valid_cells_mask'][3] == 1, "Bottom-right should be valid (bottom row)"
    print("✓ Initial masking correct")

    # Place piece at bottom-left
    action = {"cell": 2, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)

    # Now: adjacent cells should be valid (top-left, bottom-right)
    print(f"\nAfter placing at cell 2, valid cells: {obs['valid_cells_mask']}")
    print(f"Expected: [1, 0, 1, 1] (cell 0 now adjacent)")

    assert obs['valid_cells_mask'][0] == 1, "Top-left should be valid (adjacent)"
    assert obs['valid_cells_mask'][2] == 1, "Bottom-left should be valid (can place trap)"
    assert obs['valid_cells_mask'][3] == 1, "Bottom-right should be valid (adjacent)"
    print("✓ Adjacency masking correct")

    # Place piece at top-left
    action = {"cell": 0, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)

    # Now: top-right also becomes valid (adjacent to top-left)
    print(f"\nAfter placing at cell 0, valid cells: {obs['valid_cells_mask']}")
    print(f"Expected: [1, 1, 1, 1] (all cells now reachable)")

    assert obs['valid_cells_mask'][1] == 1, "Top-right should be valid (adjacent to 0)"
    print("✓ Full adjacency masking correct")

    env.close()
    print("\n" + "=" * 70)


def test_validity_checking():
    """Test validity rules: first piece in bottom row, adjacency."""
    print("=" * 70)
    print("TEST 3: Validity Checking")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    # Test 1: Try to place in top row (should fail)
    print("Test 1: Attempting invalid placement (top row, not bottom row)")
    action = {"cell": 0, "type": 0, "done": 0}  # Top-left
    obs, reward, done, truncated, info = env.step(action)

    print(f"  Reward: {reward:.2f}")
    print(f"  Expected: Negative (invalid placement penalty)")
    assert reward < 0, "Invalid placement should give negative reward"
    print("✓ Invalid first placement correctly penalized")

    # Episode should have been force-finished due to invalid board
    if done:
        print("✓ Episode terminated after invalid board (expected)")
        print(f"  'invalid_board' flag: {info.get('invalid_board', False)}")

    # Test 2: Valid construction sequence
    obs, info = env.reset(seed=43)
    print("\nTest 2: Valid construction sequence")

    # Place in bottom row
    action = {"cell": 2, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step 1 (bottom-left): reward={reward:.2f}")
    assert reward >= 0, "Valid placement should give non-negative reward"

    # Place adjacent piece
    action = {"cell": 3, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step 2 (adjacent): reward={reward:.2f}")
    assert reward >= 0, "Valid adjacent placement should give non-negative reward"

    # Finish board
    action = {"cell": 0, "type": 0, "done": 1}
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Finish board: reward={reward:.2f}")
    print(f"  Round completed: {info['round']}")
    print("✓ Valid construction accepted")

    env.close()
    print("\n" + "=" * 70)


def test_board_conversion():
    """Test construction sequence converts to valid Board object."""
    print("=" * 70)
    print("TEST 4: Board Conversion")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    # Build a simple board: two pieces, one trap
    print("Building board: 2 pieces, 1 trap")

    # Piece at (1,0)
    action = {"cell": 2, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Placed piece at (1,0)")

    # Piece at (1,1)
    action = {"cell": 3, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Placed piece at (1,1)")

    # Trap at (1,0) - supermove
    action = {"cell": 2, "type": 1, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Placed trap at (1,0)")

    print(f"\n  Construction sequence length: {len(env.construction_sequence)}")
    print(f"  Piece count: {info['piece_count']}")
    print(f"  Trap count: {info['trap_count']}")

    # Finish and convert to board
    action = {"cell": 0, "type": 0, "done": 1}
    obs, reward, done, truncated, info = env.step(action)

    print(f"\n✓ Board finished and played")
    print(f"  Agent scored: {info['agent_total_score']}")
    print(f"  Opponent scored: {info['opponent_total_score']}")
    print(f"  Reward: {reward:.2f}")

    # If board was invalid, we'd see negative reward and invalid_board flag
    if info.get('invalid_board'):
        print("  ⚠ Board was marked invalid (no path to goal?)")
    else:
        print("  ✓ Board converted successfully and was playable")

    env.close()
    print("\n" + "=" * 70)


def test_full_episode():
    """Test complete 5-round episode of building and playing."""
    print("=" * 70)
    print("TEST 5: Full Episode (5 Rounds)")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
        max_construction_steps=10,
    )

    obs, info = env.reset(seed=42)

    episode_reward = 0
    round_count = 0
    done = False

    while not done:
        round_count += 1
        print(f"\nRound {round_count}:")

        # Build a simple valid board (same pattern each round for testing)
        # Piece at (1,0), piece at (1,1), trap at (1,0)

        construction_steps = 0
        building = True

        while building:
            if construction_steps == 0:
                action = {"cell": 2, "type": 0, "done": 0}  # Piece at (1,0)
            elif construction_steps == 1:
                action = {"cell": 3, "type": 0, "done": 0}  # Piece at (1,1)
            elif construction_steps == 2:
                action = {"cell": 2, "type": 1, "done": 1}  # Trap at (1,0), FINISH
            else:
                action = {"cell": 0, "type": 0, "done": 1}  # Should never reach here

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            construction_steps += 1

            # Check if we finished building this round
            if action["done"] == 1 or done:
                building = False

        print(f"  Constructed in {construction_steps} steps")
        print(f"  Agent: {info['agent_total_score']}, Opponent: {info['opponent_total_score']}")
        print(f"  Round reward: {reward:.2f}")

        if done:
            print(f"\n✓ Episode complete after {round_count} rounds")
            break

    print(f"\nFinal Results:")
    print(f"  Total rounds: {round_count}")
    print(f"  Agent total score: {info['agent_total_score']}")
    print(f"  Opponent total score: {info['opponent_total_score']}")
    print(f"  Episode reward: {episode_reward:.2f}")

    if info['agent_total_score'] > info['opponent_total_score']:
        print(f"  🏆 Agent WON!")
    elif info['agent_total_score'] < info['opponent_total_score']:
        print(f"  ❌ Agent LOST")
    else:
        print(f"  🤝 TIE")

    env.close()
    print("\n" + "=" * 70)


def test_invalid_board_penalty():
    """Test that invalid boards receive appropriate penalty."""
    print("=" * 70)
    print("TEST 6: Invalid Board Penalty")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    # Test 1: Place piece and immediately finish (no path to goal)
    print("Test 1: Insufficient construction (single piece, no path)")

    action = {"cell": 2, "type": 0, "done": 0}  # Piece at (1,0)
    obs, reward, done, truncated, info = env.step(action)

    action = {"cell": 0, "type": 0, "done": 1}  # Finish immediately
    obs, reward, done, truncated, info = env.step(action)

    print(f"  Reward: {reward:.2f}")
    print(f"  Expected: Large negative penalty")
    print(f"  Invalid board flag: {info.get('invalid_board', False)}")

    assert reward < -40, "Invalid board should receive large penalty"
    assert info.get('invalid_board', False), "Invalid board flag should be set"

    # Episode should terminate after invalid board
    assert done or info['round'] >= 6, "Episode should end after invalid board"

    print("✓ Invalid board correctly penalized and episode terminated")

    env.close()
    print("\n" + "=" * 70)


def test_supermove():
    """Test that supermove (trap on piece) works correctly."""
    print("=" * 70)
    print("TEST 7: Supermove (Trap on Piece)")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    # Place piece at (1,0)
    action = {"cell": 2, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step 1: Placed piece at cell 2 (order=1)")
    print(f"  Piece grid:\n{obs['building_board'][:, :, 0]}")

    # Place trap on same cell (supermove - valid because piece was placed first)
    action = {"cell": 2, "type": 1, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nStep 2: Placed trap at cell 2 (order=2, supermove)")
    print(f"  Piece grid:\n{obs['building_board'][:, :, 0]}")
    print(f"  Trap grid:\n{obs['building_board'][:, :, 1]}")
    print(f"  Reward: {reward:.2f}")

    assert obs['building_board'][1, 0, 0] > 0, "Piece should still be at (1,0)"
    assert obs['building_board'][1, 0, 1] > 0, "Trap should now be at (1,0)"
    assert obs['building_board'][1, 0, 0] < obs['building_board'][1, 0, 1], "Piece order should be less than trap order"
    assert reward >= 0, "Supermove should be valid"

    print("✓ Supermove (trap on piece) works correctly")
    print("✓ Order validation: piece order (1) < trap order (2)")

    # Try to place another piece on same cell (should fail)
    action = {"cell": 2, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nStep 3: Attempted to place another piece at cell 2")
    print(f"  Reward: {reward:.2f}")
    print(f"  Expected: Negative (cell already has piece)")

    assert reward < 0, "Placing piece on existing piece should fail"
    print("✓ Duplicate piece placement correctly rejected")

    env.close()
    print("\n" + "=" * 70)


def test_invalid_supermove_order():
    """Test that placing piece after trap is rejected (order validation)."""
    print("=" * 70)
    print("TEST 8: Invalid Supermove Order (Piece After Trap)")
    print("=" * 70)

    env = BoardBuilderEnv(
        board_size=2,
        opponent_library_path="new_boards_2.json",
        opponent_strategy="fixed_0",
        show_opponent_board=True,
    )

    obs, info = env.reset(seed=42)

    # Place piece at (1,0) first
    action = {"cell": 2, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step 1: Placed piece at cell 2 (order=1)")

    # Place piece at adjacent cell (1,1) to expand valid cells
    action = {"cell": 3, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step 2: Placed piece at cell 3 (order=2)")

    # Place trap at (0,0) - now valid because adjacent to piece at (1,0)
    action = {"cell": 0, "type": 1, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step 3: Placed trap at cell 0 (order=3)")
    print(f"  Trap grid:\n{obs['building_board'][:, :, 1]}")

    # Try to place piece on cell 0 (where trap already exists) - should FAIL
    action = {"cell": 0, "type": 0, "done": 0}
    obs, reward, done, truncated, info = env.step(action)
    print(f"\nStep 4: Attempted to place piece at cell 0 (where trap exists)")
    print(f"  Reward: {reward:.2f}")
    print(f"  Expected: Negative (cannot place piece after trap)")

    assert reward < 0, "Placing piece after trap should fail"
    print("✓ Invalid order correctly rejected (piece cannot be placed after trap)")

    env.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BOARD BUILDER ENVIRONMENT TESTS")
    print("Testing Stage 2: Sequential Board Construction")
    print("=" * 70 + "\n")

    try:
        test_basic_construction()
        test_action_masking()
        test_validity_checking()
        test_board_conversion()
        test_full_episode()
        test_invalid_board_penalty()
        test_supermove()
        test_invalid_supermove_order()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        print("\nBoardBuilderEnv is working correctly!")
        print("Ready to proceed with training script.")
        print("=" * 70 + "\n")

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70 + "\n")
