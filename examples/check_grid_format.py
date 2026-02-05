"""
Check the grid format used in valid boards.
"""

from spaces_game import BoardPool

# Load boards
pool = BoardPool("new_boards_2.json", cache=True)
boards = pool.get_all_boards()

print("=" * 70)
print("GRID FORMAT IN VALID BOARDS")
print("=" * 70)

board = boards[0]
print(f"\nBoard 0:")
print(f"  Board size: {board.boardSize}")
print(f"  Grid:")
for i, row in enumerate(board.grid):
    print(f"    Row {i}: {row}")

print(f"\n  Sequence:")
for move in board.sequence:
    print(f"    {move.order}. {move.type} at ({move.position.row}, {move.position.col})")

print("\n" + "=" * 70)
