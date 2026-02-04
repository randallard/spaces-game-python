"""
Reverse Curriculum Board Builder Environment.

Stage 2 with intelligent curriculum: Learn to build boards backward (goal → start)
while leveraging Stage 1's board selection knowledge.

Key Innovation:
- Uses frozen Stage 1 policy to select optimal base board
- Removes last N moves based on curriculum phase
- Agent learns to complete the board to beat opponent
- Adaptive progression: advance phase when mastery achieved

Curriculum Phases:
- Phase 0: Only place goal move (trivial, ~100% win rate expected)
- Phase 1: Place last piece/trap + goal (easy)
- Phase 2: Place last 2 moves + goal (medium)
- ...
- Phase N: Full board construction from scratch (hard)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from .types import Board, BoardMove, Position
from .board_loader import BoardPool
from .simulation import simulate_round
from .validation import is_board_playable


class ReverseCurriculumBuilderEnv(gym.Env):
    """
    Gymnasium environment for reverse curriculum board building.

    Leverages Stage 1 (BoardConstructionEnv) trained policy to select
    base boards, then trains agent to complete partial boards.

    Training Flow:
    1. Show opponent board
    2. Use Stage 1 policy (frozen) to select counter board
    3. Remove last N moves from that board (based on curriculum phase)
    4. Agent completes the remaining moves
    5. Reward based on: validity + win against opponent

    Curriculum Phases (0-indexed):
    - 0: Remove goal only (place goal) - ~1 move
    - 1: Remove last move + goal - ~2 moves
    - 2: Remove last 2 moves + goal - ~3 moves
    - ...
    - Max: Full construction (all moves removed)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        board_size: int = 2,
        board_library_path: str = "new_boards_2.json",
        stage1_model_path: Optional[str] = None,
        curriculum_phase: int = 0,
        opponent_strategy: str = "random",
        show_opponent_board: bool = True,
        max_construction_steps: int = 20,
    ):
        """
        Initialize reverse curriculum builder environment.

        Args:
            board_size: Board size (2 for 2x2, 3 for 3x3)
            board_library_path: Path to board library JSON
            stage1_model_path: Path to trained Stage 1 model (.zip)
                If None, will randomly select base boards (no Stage 1)
            curriculum_phase: Current phase (0=easiest, higher=harder)
                Phase N means remove last N+1 moves (including goal)
            opponent_strategy: Opponent selection ("random", "fixed_N", etc.)
            show_opponent_board: If True, agent sees opponent board
            max_construction_steps: Maximum steps for construction
        """
        super().__init__()

        self.board_size = board_size
        self.curriculum_phase = curriculum_phase
        self.opponent_strategy = opponent_strategy
        self.show_opponent_board = show_opponent_board
        self.max_construction_steps = max_construction_steps

        # Load board library
        self.board_pool = BoardPool(board_library_path, cache=True)
        self.library = self.board_pool.get_all_boards()

        if len(self.library) == 0:
            raise ValueError(f"Board library is empty: {board_library_path}")

        # Load Stage 1 model if provided
        self.stage1_model = None
        self._stage1_loaded = False
        if stage1_model_path and Path(stage1_model_path).exists():
            from stable_baselines3 import PPO
            self.stage1_model = PPO.load(stage1_model_path)
            self._stage1_loaded = True
            # Only print in non-parallel contexts (rank 0 or single env)
            # Suppress for parallel envs to avoid cluttering output
        else:
            if stage1_model_path:
                print(f"⚠ Stage 1 model not found: {stage1_model_path}")
            print("  Using random base board selection")

        # Action space: same as BoardBuilderEnv
        # [cell, type, done]
        self.action_space = spaces.MultiDiscrete([
            board_size * board_size,  # cell
            2,                         # type (0=piece, 1=trap)
            2,                         # done (0=continue, 1=finish)
        ])

        # Observation space
        board_shape = (board_size, board_size, 2)
        self.observation_space = spaces.Dict({
            "opponent_board": spaces.Box(
                low=0, high=board_size * board_size,
                shape=board_shape, dtype=np.int32
            ),
            "building_board": spaces.Box(
                low=0, high=board_size * board_size,
                shape=board_shape, dtype=np.int32
            ),
            "construction_step": spaces.Discrete(max_construction_steps + 1),
            "remaining_moves": spaces.Box(
                low=0, high=max_construction_steps, shape=(1,), dtype=np.int32
            ),
            "valid_cells_mask": spaces.MultiBinary(board_size * board_size),
            "curriculum_phase": spaces.Discrete(20),  # Support up to phase 19
        })

        # Episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0

        # Construction state
        self.opponent_board: Optional[Board] = None
        self.base_board: Optional[Board] = None
        self.target_sequence: List[BoardMove] = []  # Moves agent needs to place
        self.construction_step = 0
        self.building_grid = np.zeros((board_size, board_size, 2), dtype=np.int32)
        self.placed_moves: List[Dict[str, Any]] = []

    def _select_opponent_board(self) -> Board:
        """Select opponent's board based on strategy."""
        if self.opponent_strategy == "random":
            return self.library[np.random.randint(len(self.library))]
        elif self.opponent_strategy.startswith("fixed_"):
            idx = int(self.opponent_strategy.split("_")[1])
            return self.library[idx % len(self.library)]
        else:
            return self.library[0]

    def _encode_board_for_stage1(self, board: Board) -> np.ndarray:
        """
        Encode a board for Stage 1 model input.

        Stage 1 expects a 4-channel representation:
        - Channel 0: has_piece (0 or 1)
        - Channel 1: piece_order (0 if no piece, 1-N for sequence order)
        - Channel 2: has_trap (0 or 1)
        - Channel 3: trap_order (0 if no trap, 1-N for sequence order)

        Returns:
            Grid of shape (board_size, board_size, 4) with dtype float32
        """
        grid = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)

        for move in board.sequence:
            if move.type == "final":
                continue  # Skip goal moves

            row, col = move.position.row, move.position.col
            if row < 0 or col < 0:  # Skip invalid positions
                continue

            if move.type == "piece":
                grid[row, col, 0] = 1.0  # has_piece
                grid[row, col, 1] = float(move.order)  # piece_order
            elif move.type == "trap":
                grid[row, col, 2] = 1.0  # has_trap
                grid[row, col, 3] = float(move.order)  # trap_order

        return grid

    def _select_base_board_with_stage1(self, opponent_board: Board) -> Board:
        """
        Use Stage 1 frozen policy to select optimal base board.

        If Stage 1 model not available, select randomly.
        """
        if self.stage1_model is None:
            # Fallback: random selection
            return self.library[np.random.randint(len(self.library))]

        # Create Stage 1 observation with correct format
        opponent_grid = self._encode_board_for_stage1(opponent_board)

        # Stage 1's observation space (must match training format)
        stage1_obs = {
            "round": 0,  # int, matches Discrete(5)
            "score_diff": np.array([0.0], dtype=np.float32),
            "agent_score": np.array([0.0], dtype=np.float32),
            "opponent_score": np.array([0.0], dtype=np.float32),
            "agent_history": np.array([-1, -1, -1, -1, -1], dtype=np.int32),
            "opponent_history": np.array([-1, -1, -1, -1, -1], dtype=np.int32),
            "opponent_board": opponent_grid,  # (board_size, board_size, 4) float32
        }

        # Get Stage 1 prediction (frozen, deterministic)
        try:
            action, _states = self.stage1_model.predict(stage1_obs, deterministic=True)
            board_idx = int(action) % len(self.library)
            return self.library[board_idx]
        except Exception as e:
            print(f"⚠ Stage 1 prediction failed: {e}, using random selection")
            return self.library[np.random.randint(len(self.library))]

    def _remove_last_n_moves(self, board: Board, n: int) -> Tuple[List[BoardMove], List[BoardMove]]:
        """
        Remove last N moves from board sequence.

        Args:
            board: Complete board
            n: Number of moves to remove (curriculum phase + 1)

        Returns:
            (partial_sequence, target_sequence)
            - partial_sequence: Moves to pre-populate board with
            - target_sequence: Moves agent needs to place
        """
        sequence = list(board.sequence)

        # Always remove goal (it's always last)
        # Then remove additional moves based on phase
        moves_to_remove = min(n + 1, len(sequence))  # +1 for goal

        partial_sequence = sequence[:-moves_to_remove] if moves_to_remove > 0 else sequence
        target_sequence = sequence[-moves_to_remove:] if moves_to_remove > 0 else []

        return partial_sequence, target_sequence

    def _board_to_grid(self, board: Board) -> np.ndarray:
        """Convert Board to grid representation."""
        grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)

        for move in board.sequence:
            if move.type == "final":
                continue

            row, col = move.position.row, move.position.col
            order = move.order

            if move.type == "piece":
                grid[row, col, 0] = order
            elif move.type == "trap":
                grid[row, col, 1] = order

        return grid

    def _sequence_to_grid(self, sequence: List[BoardMove]) -> np.ndarray:
        """Convert move sequence to grid."""
        grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)

        for move in sequence:
            if move.type == "goal":
                continue

            row, col = move.position.row, move.position.col
            order = move.order

            if move.type == "piece":
                grid[row, col, 0] = order
            elif move.type == "trap":
                grid[row, col, 1] = order

        return grid

    def _get_valid_cells_mask(self) -> np.ndarray:
        """Get mask of valid placement cells."""
        # For now, all cells valid (validation happens in step)
        return np.ones(self.board_size * self.board_size, dtype=np.int8)

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        opponent_grid = self._board_to_grid(self.opponent_board)

        return {
            "opponent_board": opponent_grid,
            "building_board": self.building_grid.copy(),
            "construction_step": self.construction_step,
            "remaining_moves": np.array([len(self.target_sequence) - len(self.placed_moves)], dtype=np.int32),
            "valid_cells_mask": self._get_valid_cells_mask(),
            "curriculum_phase": self.curriculum_phase,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0

        # Select opponent board
        self.opponent_board = self._select_opponent_board()

        # Use Stage 1 to select base board
        self.base_board = self._select_base_board_with_stage1(self.opponent_board)

        # Remove last N moves based on curriculum phase
        partial_seq, target_seq = self._remove_last_n_moves(
            self.base_board,
            self.curriculum_phase
        )

        self.target_sequence = target_seq

        # Pre-populate building grid with partial sequence
        self.building_grid = self._sequence_to_grid(partial_seq)
        self.construction_step = len(partial_seq)
        self.placed_moves = []

        obs = self._get_observation()
        info = {
            "round": self.current_round + 1,
            "curriculum_phase": self.curriculum_phase,
            "moves_to_place": len(self.target_sequence),
            "base_board_idx": self.library.index(self.base_board) if self.base_board in self.library else -1,
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one construction step."""
        cell = action[0] % (self.board_size * self.board_size)
        piece_or_trap = action[1] % 2
        done = action[2] > 0

        row = cell // self.board_size
        col = cell % self.board_size

        reward = 0.0

        # If not done, place the move
        if not done and len(self.placed_moves) < len(self.target_sequence):
            order = self.construction_step + 1

            # Determine move type
            move_type = "piece" if piece_or_trap == 0 else "trap"

            # Check if this is the final/goal move
            if len(self.placed_moves) == len(self.target_sequence) - 1:
                # Last move should be final/goal
                if self.target_sequence[-1].type == "final":
                    move_type = "final"

            # Place the move
            if move_type == "piece":
                self.building_grid[row, col, 0] = order
            elif move_type == "trap":
                self.building_grid[row, col, 1] = order
            # Final move doesn't update grid (no physical position)

            # Record the move
            if move_type == "final":
                # Final move has no physical position
                self.placed_moves.append({
                    "row": -1,
                    "col": -1,
                    "type": move_type,
                    "order": order,
                })
            else:
                self.placed_moves.append({
                    "row": row,
                    "col": col,
                    "type": move_type,
                    "order": order,
                })

            self.construction_step += 1

            # Small reward for making progress
            reward += 0.1

            # Check if all moves placed
            if len(self.placed_moves) >= len(self.target_sequence):
                done = True

        # If done or all moves placed, evaluate the board
        if done or len(self.placed_moves) >= len(self.target_sequence):
            agent_board = self._construct_board_from_state()

            # Check validity
            is_valid = is_board_playable(agent_board)

            if not is_valid:
                reward += -50.0  # Heavy penalty for invalid
                agent_score = 0
                opponent_score = 0
            else:
                # Play the round
                result = simulate_round(
                    self.current_round,
                    agent_board,
                    self.opponent_board,
                    size=self.board_size,
                )

                agent_score = result.playerPoints
                opponent_score = result.opponentPoints
                score_diff = agent_score - opponent_score

                # Reward based on performance
                reward += float(score_diff)

                if agent_score > opponent_score:
                    reward += 20.0  # Win bonus
                elif agent_score == opponent_score:
                    reward += 5.0   # Tie bonus

            # Update totals
            self.agent_total_score += agent_score
            self.opponent_total_score += opponent_score
            self.current_round += 1

            # Episode done (single round for now)
            terminated = True

            obs = self._get_observation()
            info = {
                "round": self.current_round,
                "agent_score": agent_score,
                "opponent_score": opponent_score,
                "agent_total_score": self.agent_total_score,
                "opponent_total_score": self.opponent_total_score,
                "valid_board": is_valid,
                "curriculum_phase": self.curriculum_phase,
                "moves_placed": len(self.placed_moves),
                "moves_required": len(self.target_sequence),
            }

            return obs, reward, terminated, False, info

        # Continue construction
        obs = self._get_observation()
        info = {
            "round": self.current_round + 1,
            "construction_step": self.construction_step,
            "moves_placed": len(self.placed_moves),
            "moves_required": len(self.target_sequence),
        }

        return obs, reward, False, False, info

    def _construct_board_from_state(self) -> Board:
        """Construct Board object from current building state."""
        # Combine partial sequence with placed moves
        all_moves = []

        # Add all moves from building grid
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece_order = self.building_grid[row, col, 0]
                trap_order = self.building_grid[row, col, 1]

                if piece_order > 0:
                    all_moves.append(BoardMove(
                        position=Position(row=row, col=col),
                        type="piece",
                        order=int(piece_order)
                    ))
                if trap_order > 0:
                    all_moves.append(BoardMove(
                        position=Position(row=row, col=col),
                        type="trap",
                        order=int(trap_order)
                    ))

        # Add placed moves (including final/goal if applicable)
        for move in self.placed_moves:
            if move["type"] == "final":
                all_moves.append(BoardMove(
                    position=Position(row=-1, col=-1),
                    type="final",
                    order=move["order"]
                ))

        # Sort by order
        all_moves.sort(key=lambda m: m.order)

        # Ensure final is last
        if not any(m.type == "final" for m in all_moves):
            all_moves.append(BoardMove(
                position=Position(row=-1, col=-1),
                type="final",
                order=len(all_moves) + 1
            ))

        # Build string grid
        str_grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        for move in all_moves:
            if move.type == "final":
                continue
            row, col = move.position.row, move.position.col
            if move.type == "piece":
                if str_grid[row][col] == "T":
                    str_grid[row][col] = "B"
                else:
                    str_grid[row][col] = "P"
            elif move.type == "trap":
                if str_grid[row][col] == "P":
                    str_grid[row][col] = "B"
                else:
                    str_grid[row][col] = "T"

        return Board(
            boardSize=self.board_size,
            grid=tuple(tuple(row) for row in str_grid),
            sequence=tuple(all_moves)
        )

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
