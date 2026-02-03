"""
Inspect what observations the agent receives in perfect information mode.

This is a debugging/validation tool to visualize exactly what data
the neural network sees when making decisions.

Usage:
    python examples/inspect_observation.py
"""

import numpy as np
from spaces_game import SpacesGameEnv


def print_board_grid(board):
    """Pretty print a board's grid."""
    print(f"  Board {board.boardSize}x{board.boardSize}:")
    for row in board.grid:
        print(f"    {' | '.join(f'{cell:6s}' for cell in row)}")


def print_board_sequence(board):
    """Print a board's move sequence."""
    print(f"  Sequence ({len(board.sequence)} moves):")
    for move in board.sequence:
        print(f"    {move.order}. {move.type:6s} at ({move.position.row:2d}, {move.position.col:2d})")


def print_encoded_board(encoded, board_idx):
    """Print encoded board tensor in human-readable format."""
    print(f"\n  Encoded Board {board_idx}:")
    rows, cols, channels = encoded.shape

    for r in range(rows):
        for c in range(cols):
            cell = encoded[r, c]
            has_piece = int(cell[0])
            piece_order = int(cell[1])
            has_trap = int(cell[2])
            trap_order = int(cell[3])

            cell_str = f"({r},{c}): "
            if has_piece:
                cell_str += f"piece#{piece_order}"
            if has_trap:
                if has_piece:
                    cell_str += f" + trap#{trap_order}"
                else:
                    cell_str += f"trap#{trap_order}"
            if not has_piece and not has_trap:
                cell_str += "empty"

            print(f"    {cell_str}")


def inspect_partial_observability():
    """Show what agent sees in partial observability mode."""
    print("=" * 80)
    print("PARTIAL OBSERVABILITY MODE")
    print("=" * 80)

    env = SpacesGameEnv(
        board_pool_path="new_boards_2.json",
        deck_size=8,
        opponent_strategy="random",
        perfect_information=False,
    )

    obs, info = env.reset(seed=42)

    print("\nObservation keys:", list(obs.keys()))
    print("\nObservation contents:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}")
            if value.size <= 10:  # Only print small arrays
                print(f"    {' ' * 20}  values={value}")
        else:
            print(f"  {key:20s}: {value}")

    print("\n" + "-" * 80)
    print("Agent's deck (agent can see own boards):")
    for i, board in enumerate(env.agent_deck):
        print(f"\nBoard {i}:")
        print_board_grid(board)

    print("\n" + "-" * 80)
    print("Opponent's deck (agent CANNOT see in partial mode):")
    print("  → Agent only sees opponent board INDICES in history")
    print("  → No information about what boards actually look like")


def inspect_perfect_information():
    """Show what agent sees in perfect information mode."""
    print("\n\n" + "=" * 80)
    print("PERFECT INFORMATION MODE")
    print("=" * 80)

    env = SpacesGameEnv(
        board_pool_path="new_boards_2.json",
        deck_size=8,
        opponent_strategy="random",
        perfect_information=True,
    )

    obs, info = env.reset(seed=42)

    print("\nObservation keys:", list(obs.keys()))
    print("\nObservation contents:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}")
            if key != "opponent_deck" and value.size <= 10:
                print(f"    {' ' * 20}  values={value}")
        else:
            print(f"  {key:20s}: {value}")

    print("\n" + "-" * 80)
    print("Agent's deck (same as partial mode):")
    for i, board in enumerate(env.agent_deck):
        print(f"\nBoard {i}:")
        print_board_grid(board)

    print("\n" + "-" * 80)
    print("Opponent's deck (NOW VISIBLE via encoded tensor):")
    print(f"\nRaw tensor shape: {obs['opponent_deck'].shape}")
    print(f"  → {obs['opponent_deck'].shape[0]} boards")
    print(f"  → {obs['opponent_deck'].shape[1]}x{obs['opponent_deck'].shape[2]} grid")
    print(f"  → {obs['opponent_deck'].shape[3]} channels (has_piece, piece_order, has_trap, trap_order)")

    print("\n" + "-" * 80)
    print("Decoding opponent's boards from tensor:")

    for i, board in enumerate(env.opponent_deck):
        print(f"\n--- Opponent Board {i} ---")
        print("Actual board:")
        print_board_grid(board)
        print_board_sequence(board)

        print("\nEncoded as tensor:")
        print_encoded_board(obs['opponent_deck'][i], i)

    print("\n" + "-" * 80)
    print("What this means for the agent:")
    print("  ✓ Agent's neural network receives a 4D tensor: (8, 2, 2, 4)")
    print("  ✓ For each opponent board, it knows:")
    print("    - Which cells have pieces (and their sequence order)")
    print("    - Which cells have traps (and their sequence order)")
    print("  ✓ Agent can reason about optimal counters:")
    print("    - 'Opponent board 0 has trap at (1,0), I should use a right-column board'")
    print("    - 'Opponent board 3 takes 4 steps, I need 5+ to win'")


def inspect_during_game():
    """Show how observations change during gameplay."""
    print("\n\n" + "=" * 80)
    print("OBSERVATIONS DURING GAMEPLAY")
    print("=" * 80)

    env = SpacesGameEnv(
        board_pool_path="new_boards_2.json",
        deck_size=8,
        opponent_strategy="random",
        perfect_information=True,
    )

    obs, info = env.reset(seed=42)

    print("\n--- Round 1 Start ---")
    print(f"Round:          {obs['round']}")
    print(f"Score diff:     {obs['score_diff'][0]}")
    print(f"First picker:   {obs['first_picker']} (0=agent, 1=opponent)")
    print(f"Agent history:  {obs['agent_history']}")
    print(f"Opponent hist:  {obs['opponent_history']}")
    print(f"Opponent deck:  shape={obs['opponent_deck'].shape}")

    # Play round 1
    print("\nAgent selects board 0, Opponent selects board 1")
    obs, reward, done, truncated, info = env.step(0)

    print("\n--- Round 2 Start ---")
    print(f"Round:          {obs['round']}")
    print(f"Score diff:     {obs['score_diff'][0]}")
    print(f"Agent score:    {obs['agent_score'][0]}")
    print(f"Opponent score: {obs['opponent_score'][0]}")
    print(f"First picker:   {obs['first_picker']} (0=agent, 1=opponent)")
    print(f"Agent history:  {obs['agent_history']}")
    print(f"Opponent hist:  {obs['opponent_history']}")
    print(f"Round 1 reward: {reward}")

    print("\n→ Agent now knows:")
    print(f"  - We played board 0 (history shows [0, -1, -1, -1, -1])")
    print(f"  - Opponent played board 1 (history shows [1, -1, -1, -1, -1])")
    print(f"  - Current score differential: {obs['score_diff'][0]}")
    print(f"  - Opponent has boards [0,2,3,4,5,6,7] remaining")
    print(f"  - Opponent's full remaining decks still visible in opponent_deck tensor")


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "OBSERVATION INSPECTION TOOL" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")

    inspect_partial_observability()
    inspect_perfect_information()
    inspect_during_game()

    print("\n\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. PARTIAL OBSERVABILITY:
   - Agent sees opponent board INDICES only (e.g., "they played board 3")
   - No information about what those boards contain
   - Must learn patterns from outcomes only

2. PERFECT INFORMATION:
   - Agent sees full opponent deck encoded as 4D tensor
   - Each board → (2, 2, 4) tensor with piece/trap positions and orders
   - Can directly reason about board matchups

3. TEMPORAL EVOLUTION:
   - History arrays track which boards have been played
   - Score differential accumulates over rounds
   - First picker alternates each round (odd=agent, even=opponent)

4. NEURAL NETWORK INPUT:
   - Perfect info adds ~128 floats to observation (8 boards × 2×2 grid × 4 features)
   - Must learn to process spatial patterns (trap locations, path lengths)
   - Feature engineering done via grid encoding (not just raw sequences)
    """)

    print("=" * 80)
    print("Now try: python examples/test_board_selection.py models/MODEL.zip")
    print("=" * 80)


if __name__ == "__main__":
    main()
