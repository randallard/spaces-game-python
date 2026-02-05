"""
Reverse Curriculum Board Builder Environment.

Stage 2 with intelligent curriculum: Learn to build boards backward (goal -> start)
while leveraging Stage 1's board selection knowledge.

Key Innovation:
- Uses frozen Stage 1 policy to select optimal base board
- Removes last N moves based on curriculum phase
- Agent learns to complete the board to beat opponent
- Adaptive progression: advance phase when mastery achieved
- Action masking ensures only valid placements are attempted

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
from .validation import is_board_playable, _is_adjacent_orthogonal


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
        else:
            if stage1_model_path:
                print(f"Stage 1 model not found: {stage1_model_path}")
            print("  Using random base board selection")

        # Action space: [cell, type, done]
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
            "curriculum_phase": spaces.Discrete(20),
        })

        # Episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0

        # Construction state
        self.opponent_board: Optional[Board] = None
        self.base_board: Optional[Board] = None
        self.target_sequence: List[BoardMove] = []
        self.construction_step = 0
        self.building_grid = np.zeros((board_size, board_size, 2), dtype=np.int32)
        self.placed_moves: List[Dict[str, Any]] = []

        # Game state tracking (reconstructed from partial sequence)
        self.current_piece_position: Optional[Position] = None
        self.trap_positions: set = set()  # "row,col" strings
        self.supermove_active: bool = False
        self.supermove_position: Optional[Position] = None
        self.construction_sequence: List[Dict[str, Any]] = []
        self.steps_taken = 0  # Total steps including invalid, for truncation

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
        """Encode a board for Stage 1 model input (4-channel grid)."""
        grid = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)

        for move in board.sequence:
            if move.type == "final":
                continue
            row, col = move.position.row, move.position.col
            if row < 0 or col < 0:
                continue
            if move.type == "piece":
                grid[row, col, 0] = 1.0
                grid[row, col, 1] = float(move.order)
            elif move.type == "trap":
                grid[row, col, 2] = 1.0
                grid[row, col, 3] = float(move.order)

        return grid

    def _select_base_board_with_stage1(self, opponent_board: Board) -> Board:
        """Use Stage 1 frozen policy to select optimal base board."""
        if self.stage1_model is None:
            return self.library[np.random.randint(len(self.library))]

        opponent_grid = self._encode_board_for_stage1(opponent_board)
        stage1_obs = {
            "round": 0,
            "score_diff": np.array([0.0], dtype=np.float32),
            "agent_score": np.array([0.0], dtype=np.float32),
            "opponent_score": np.array([0.0], dtype=np.float32),
            "agent_history": np.array([-1, -1, -1, -1, -1], dtype=np.int32),
            "opponent_history": np.array([-1, -1, -1, -1, -1], dtype=np.int32),
            "opponent_board": opponent_grid,
        }

        try:
            action, _states = self.stage1_model.predict(stage1_obs, deterministic=True)
            board_idx = int(action) % len(self.library)
            return self.library[board_idx]
        except Exception:
            return self.library[np.random.randint(len(self.library))]

    def _remove_last_n_moves(self, board: Board, n: int) -> Tuple[List[BoardMove], List[BoardMove]]:
        """Remove last N moves from board sequence."""
        sequence = list(board.sequence)
        moves_to_remove = min(n + 1, len(sequence))  # +1 for goal
        partial_sequence = sequence[:-moves_to_remove] if moves_to_remove > 0 else sequence
        target_sequence = sequence[-moves_to_remove:] if moves_to_remove > 0 else []
        return partial_sequence, target_sequence

    def _reconstruct_game_state_from_partial(self, partial_sequence: List[BoardMove]) -> None:
        """
        Reconstruct game state from partial sequence.
        Mirrors validation.py's is_board_playable() logic.
        """
        self.current_piece_position = None
        self.trap_positions = set()
        self.supermove_active = False
        self.supermove_position = None
        self.construction_sequence = []

        for move in partial_sequence:
            if move.type == "final":
                continue

            entry = {
                "row": move.position.row,
                "col": move.position.col,
                "type": move.type,
                "order": move.order,
            }
            self.construction_sequence.append(entry)

            if move.type == "piece":
                if self.supermove_active:
                    self.supermove_active = False
                    self.supermove_position = None
                self.current_piece_position = move.position

            elif move.type == "trap":
                pos_key = f"{move.position.row},{move.position.col}"
                self.trap_positions.add(pos_key)
                if (self.current_piece_position is not None and
                    move.position.row == self.current_piece_position.row and
                    move.position.col == self.current_piece_position.col):
                    self.supermove_active = True
                    self.supermove_position = move.position

    def _is_valid_placement(self, row: int, col: int, move_type: str) -> bool:
        """
        Check if placement at (row, col) of given type is valid.

        Rules from validation.py:
        - Piece: adjacent to current piece, not into trap, resolve supermove
        - Trap: adjacent to piece or at piece (supermove), not after active supermove
        """
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False

        if move_type == "piece":
            if self.current_piece_position is None:
                # First piece: must be in bottom row
                return row == self.board_size - 1

            # Must be adjacent to current piece
            if not _is_adjacent_orthogonal(
                self.current_piece_position,
                Position(row=row, col=col)
            ):
                return False

            # Cannot move into a trap
            if f"{row},{col}" in self.trap_positions:
                return False

            # After supermove: must move to different position
            if self.supermove_active:
                if (row == self.supermove_position.row and
                    col == self.supermove_position.col):
                    return False

            return True

        elif move_type == "trap":
            if self.current_piece_position is None:
                return False  # Cannot place trap before piece

            if self.supermove_active:
                return False  # Must move piece first after supermove

            # No duplicate trap
            if f"{row},{col}" in self.trap_positions:
                return False

            # Same position as piece (supermove) or adjacent
            same_pos = (row == self.current_piece_position.row and
                        col == self.current_piece_position.col)

            if same_pos:
                return True  # Supermove

            return _is_adjacent_orthogonal(
                self.current_piece_position,
                Position(row=row, col=col)
            )

        return False

    def _get_valid_cells_mask(self) -> np.ndarray:
        """Get mask of valid cells (valid for either piece or trap)."""
        mask = np.zeros(self.board_size * self.board_size, dtype=np.int8)

        for row in range(self.board_size):
            for col in range(self.board_size):
                cell_idx = row * self.board_size + col
                if (self._is_valid_placement(row, col, "piece") or
                    self._is_valid_placement(row, col, "trap")):
                    mask[cell_idx] = 1

        # Fallback: if no valid cells, allow all to prevent MaskablePPO crash
        if mask.sum() == 0:
            mask[:] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        """
        Return action mask for MaskablePPO with MultiDiscrete action space.

        For MultiDiscrete([board_size*board_size, 2, 2]):
        Returns concatenated: [cell_mask (N), type_mask (2), done_mask (2)]
        """
        n_cells = self.board_size * self.board_size
        moves_remaining = len(self.target_sequence) - len(self.placed_moves)

        # If episode is effectively done (no moves left), allow anything
        # (step will handle termination)
        if moves_remaining <= 0:
            cell_mask = np.ones(n_cells, dtype=np.int8)
            type_mask = np.ones(2, dtype=np.int8)
            done_mask = np.array([0, 1], dtype=np.int8)  # must finish
            return np.concatenate([cell_mask, type_mask, done_mask])

        # Check if only the final/goal move remains
        only_goal_left = (moves_remaining == 1 and
                          len(self.target_sequence) > 0 and
                          self.target_sequence[-1].type == "final")

        if only_goal_left:
            # Goal is auto-placed; agent should signal done
            cell_mask = np.ones(n_cells, dtype=np.int8)
            type_mask = np.ones(2, dtype=np.int8)
            done_mask = np.array([1, 1], dtype=np.int8)  # either is fine
            return np.concatenate([cell_mask, type_mask, done_mask])

        # --- Check for deadlock (no valid moves at all) ---
        has_valid_piece = any(
            self._is_valid_placement(r, c, "piece")
            for r in range(self.board_size)
            for c in range(self.board_size)
        )
        has_valid_trap = (
            not self.supermove_active and
            self.current_piece_position is not None and
            any(
                self._is_valid_placement(r, c, "trap")
                for r in range(self.board_size)
                for c in range(self.board_size)
            )
        )

        if not has_valid_piece and not has_valid_trap:
            # Deadlock — force finish to prevent infinite episode
            cell_mask = np.ones(n_cells, dtype=np.int8)
            type_mask = np.ones(2, dtype=np.int8)
            done_mask = np.array([0, 1], dtype=np.int8)  # must finish
            return np.concatenate([cell_mask, type_mask, done_mask])

        # --- Cell mask ---
        cell_mask = self._get_valid_cells_mask()

        # --- Type mask [piece, trap] ---
        type_mask = np.zeros(2, dtype=np.int8)
        if self.supermove_active or self.current_piece_position is None:
            type_mask[0] = 1  # must place piece
            type_mask[1] = 0
        else:
            type_mask[0] = 1 if has_valid_piece else 0
            type_mask[1] = 1 if has_valid_trap else 0

        # --- Done mask [continue, finish] ---
        done_mask = np.array([1, 1], dtype=np.int8)  # allow both by default

        return np.concatenate([cell_mask, type_mask, done_mask])

    def _board_to_grid(self, board: Board) -> np.ndarray:
        """Convert Board to grid representation."""
        grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)
        for move in board.sequence:
            if move.type == "final":
                continue
            row, col = move.position.row, move.position.col
            if move.type == "piece":
                grid[row, col, 0] = move.order
            elif move.type == "trap":
                grid[row, col, 1] = move.order
        return grid

    def _get_last_move_column(self) -> int:
        """Get the column of the last non-final move."""
        if self.construction_sequence:
            for step in reversed(self.construction_sequence):
                if step["type"] != "final":
                    return step["col"]
        return 0

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        opponent_grid = self._board_to_grid(self.opponent_board)

        return {
            "opponent_board": opponent_grid,
            "building_board": self.building_grid.copy(),
            "construction_step": self.construction_step,
            "remaining_moves": np.array(
                [len(self.target_sequence) - len(self.placed_moves)],
                dtype=np.int32
            ),
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

        # Reconstruct game state from partial sequence
        self._reconstruct_game_state_from_partial(partial_seq)

        # Build grid from construction_sequence (single source of truth)
        self.building_grid = np.zeros(
            (self.board_size, self.board_size, 2), dtype=np.int32
        )
        for step in self.construction_sequence:
            row, col = step["row"], step["col"]
            order = step["order"]
            if step["type"] == "piece":
                self.building_grid[row, col, 0] = order
            elif step["type"] == "trap":
                self.building_grid[row, col, 1] = order

        self.construction_step = len(self.construction_sequence)
        self.placed_moves = []
        self.steps_taken = 0

        obs = self._get_observation()
        info = {
            "round": self.current_round + 1,
            "curriculum_phase": self.curriculum_phase,
            "moves_to_place": len(self.target_sequence),
            "base_board_idx": (self.library.index(self.base_board)
                               if self.base_board in self.library else -1),
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one construction step."""
        self.steps_taken += 1

        # Truncate if too many steps (prevents infinite episodes)
        if self.steps_taken >= self.max_construction_steps:
            agent_board = self._construct_board_from_state()
            is_valid = is_board_playable(agent_board)
            reward = -10.0 if not is_valid else 0.0
            obs = self._get_observation()
            info = {
                "round": self.current_round + 1,
                "valid_board": is_valid,
                "curriculum_phase": self.curriculum_phase,
                "moves_placed": len(self.placed_moves),
                "moves_required": len(self.target_sequence),
                "truncated": True,
            }
            return obs, reward, False, True, info  # truncated=True

        cell = int(action[0]) % (self.board_size * self.board_size)
        piece_or_trap = int(action[1]) % 2
        done_flag = int(action[2]) > 0

        row = cell // self.board_size
        col = cell % self.board_size

        reward = 0.0
        moves_remaining = len(self.target_sequence) - len(self.placed_moves)

        # Check if only goal move remains — auto-place it
        only_goal_left = (moves_remaining == 1 and
                          len(self.target_sequence) > 0 and
                          self.target_sequence[-1].type == "final")

        if only_goal_left:
            # Auto-place goal move
            last_col = self._get_last_move_column()
            order = self.construction_step + 1
            self.construction_sequence.append({
                "row": -1, "col": last_col, "type": "final", "order": order,
            })
            self.placed_moves.append({
                "row": -1, "col": last_col, "type": "final", "order": order,
            })
            self.construction_step += 1
            reward += 0.1
            # Fall through to evaluation below

        # If agent signals done or all moves placed, evaluate
        elif done_flag or moves_remaining <= 0:
            pass  # Fall through to evaluation

        # Otherwise, place a move
        elif moves_remaining > 0:
            order = self.construction_step + 1
            move_type = "piece" if piece_or_trap == 0 else "trap"

            # Check if this should be the goal move
            target_idx = len(self.placed_moves)
            if (target_idx < len(self.target_sequence) and
                self.target_sequence[target_idx].type == "final"):
                # This target move is a goal — auto-place and evaluate
                last_col = self._get_last_move_column()
                self.construction_sequence.append({
                    "row": -1, "col": last_col, "type": "final", "order": order,
                })
                self.placed_moves.append({
                    "row": -1, "col": last_col, "type": "final", "order": order,
                })
                self.construction_step += 1
                reward += 0.1
                # Fall through to evaluation

            elif self._is_valid_placement(row, col, move_type):
                # Valid placement — update state
                self.construction_sequence.append({
                    "row": row, "col": col, "type": move_type, "order": order,
                })
                self.placed_moves.append({
                    "row": row, "col": col, "type": move_type, "order": order,
                })

                # Update grid
                if move_type == "piece":
                    self.building_grid[row, col, 0] = order
                elif move_type == "trap":
                    self.building_grid[row, col, 1] = order

                # Update game state
                if move_type == "piece":
                    if self.supermove_active:
                        self.supermove_active = False
                        self.supermove_position = None
                    self.current_piece_position = Position(row=row, col=col)

                    # Small shaping rewards — just enough to guide placement
                    reward += 0.1
                    if row == 0:
                        reward += 0.3  # Top row — can reach goal

                elif move_type == "trap":
                    pos_key = f"{row},{col}"
                    self.trap_positions.add(pos_key)
                    if (self.current_piece_position is not None and
                        row == self.current_piece_position.row and
                        col == self.current_piece_position.col):
                        self.supermove_active = True
                        self.supermove_position = Position(row=row, col=col)
                        reward += 0.2  # Supermove bonus
                    else:
                        reward += 0.1  # Regular trap

                self.construction_step += 1

                # Check if more moves needed (excluding auto-goal)
                moves_remaining = len(self.target_sequence) - len(self.placed_moves)
                if moves_remaining > 0:
                    # Not done yet — continue building
                    obs = self._get_observation()
                    info = {
                        "round": self.current_round + 1,
                        "construction_step": self.construction_step,
                        "moves_placed": len(self.placed_moves),
                        "moves_required": len(self.target_sequence),
                    }
                    return obs, reward, False, False, info

                # All moves placed — fall through to evaluation

            else:
                # Invalid placement — penalty but don't terminate
                reward += -2.0
                obs = self._get_observation()
                info = {
                    "round": self.current_round + 1,
                    "construction_step": self.construction_step,
                    "moves_placed": len(self.placed_moves),
                    "moves_required": len(self.target_sequence),
                    "invalid_placement": True,
                }
                return obs, reward, False, False, info

        # --- Evaluate the constructed board ---
        agent_board = self._construct_board_from_state()
        is_valid = is_board_playable(agent_board)

        if not is_valid:
            reward += -50.0
            agent_score = 0
            opponent_score = 0
        else:
            result = simulate_round(
                self.current_round,
                agent_board,
                self.opponent_board,
                silent=True,
            )
            agent_score = result.playerPoints
            opponent_score = result.opponentPoints
            score_diff = agent_score - opponent_score
            reward += float(score_diff) * 2.0  # Amplify score differential

            if agent_score > opponent_score:
                reward += 25.0
            elif agent_score == opponent_score:
                reward += 5.0
            else:
                reward += -15.0  # Losing must hurt

        self.agent_total_score += agent_score
        self.opponent_total_score += opponent_score
        self.current_round += 1

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

        return obs, reward, True, False, info

    def _construct_board_from_state(self) -> Board:
        """Construct Board from construction_sequence (single source of truth)."""
        if len(self.construction_sequence) == 0:
            return Board(
                boardSize=self.board_size,
                grid=tuple(
                    tuple("empty" for _ in range(self.board_size))
                    for _ in range(self.board_size)
                ),
                sequence=(BoardMove(Position(row=-1, col=0), "final", 1),)
            )

        sequence = []
        for step in self.construction_sequence:
            sequence.append(BoardMove(
                position=Position(row=step["row"], col=step["col"]),
                type=step["type"],
                order=step["order"]
            ))

        # Ensure final/goal move exists
        if not any(m.type == "final" for m in sequence):
            last_col = self._get_last_move_column()
            sequence.append(BoardMove(
                position=Position(row=-1, col=last_col),
                type="final",
                order=len(sequence) + 1
            ))

        # Build string grid for validation compatibility
        str_grid = [
            ["empty" for _ in range(self.board_size)]
            for _ in range(self.board_size)
        ]
        for move in sequence:
            if move.type == "final":
                continue
            row, col = move.position.row, move.position.col
            if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
                continue
            if move.type == "piece":
                if str_grid[row][col] != "trap":
                    str_grid[row][col] = "piece"
            elif move.type == "trap":
                str_grid[row][col] = "trap"

        return Board(
            boardSize=self.board_size,
            grid=tuple(tuple(r) for r in str_grid),
            sequence=tuple(sequence)
        )

    def set_curriculum_phase(self, phase: int) -> None:
        """Set curriculum phase. Callable via SubprocVecEnv.env_method()."""
        self.curriculum_phase = phase

    def render(self):
        """Render environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
