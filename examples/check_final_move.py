"""
Check how final moves are positioned in the board library.
"""

from spaces_game import BoardPool

# Load boards
pool = BoardPool("new_boards_2.json", cache=True)
boards = pool.get_all_boards()

print("=" * 70)
print("FINAL MOVE POSITIONS IN BOARD LIBRARY")
print("=" * 70)

for i, board in enumerate(boards[:5]):  # Check first 5 boards
    print(f"\nBoard {i}:")
    print(f"  Board size: {board.boardSize}")
    print(f"  Sequence length: {len(board.sequence)}")

    # Find the final move
    for move in board.sequence:
        if move.type == "final":
            print(f"  Final move: position=({move.position.row}, {move.position.col}), order={move.order}")
            break

    # Show last few moves
    print(f"  Last 3 moves:")
    for move in board.sequence[-3:]:
        print(f"    {move.order}. {move.type} at ({move.position.row}, {move.position.col})")

print("\n" + "=" * 70)
