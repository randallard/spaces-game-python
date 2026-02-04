"""
Board Modifier Environment (Stage 1.5).

An intermediate curriculum stage between board selection (Stage 1)
and full construction (Stage 2).

Agent workflow:
1. Select a base board from the opponent library
2. Make 0-3 modifications (add/remove pieces or traps)
3. Play the modified board against the opponent's board

This provides a more tractable learning task than full construction
while still allowing strategic adaptation to opponent boards.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces

from .types import Board
from .board_loader import BoardPool
from .simulation import simulate_round
from .validation import is_board_playable


class BoardModifierEnv(gym.Env):
    """
    Gymnasium environment for learning board modification strategy.

    Observation space:
        - opponent_board: (board_size, board_size, 2) - opponent's pieces and traps
        - base_board: (board_size, board_size, 2) - current base board being modified
        - modification_step: scalar - which modification step (0 to max_modifications)
        - modifications_made: scalar - count of modifications so far

    Action space (MultiDiscrete):
        During base selection phase:
            [base_board_index, 0, 0, 0]  # Only first element used

        During modification phase:
            [modify_type, cell, piece_or_trap, done]
            - modify_type: 0=add, 1=remove
            - cell: 0 to board_size²-1
            - piece_or_trap: 0=piece, 1=trap
            - done: 0=continue modifications, 1=finish and play
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        board_size: int = 2,
        opponent_library_path: str = "new_boards_2.json",
        base_library_path: Optional[str] = None,
        opponent_strategy: str = "random",
        show_opponent_board: bool = True,
        max_modifications: int = 3,
        max_rounds: int = 5,
    ):
        """
        Initialize Board Modifier Environment.

        Args:
            board_size: Size of the board (2 for 2x2, 3 for 3x3, etc.)
            opponent_library_path: JSON file with opponent board library
            base_library_path: JSON file for base boards (defaults to opponent_library_path)
            opponent_strategy: How opponent selects boards:
                - "random": Random from library each round
                - "greedy": Select best board against agent's previous board
                - "fixed": Use first board from library
                - "fixed_N": Use board at index N
            show_opponent_board: If True, agent observes opponent's board (perfect info)
            max_modifications: Maximum number of modifications allowed (0-3 typical)
            max_rounds: Number of rounds per episode
        """
        super().__init__()

        self.board_size = board_size
        self.opponent_strategy = opponent_strategy
        self.show_opponent_board = show_opponent_board
        self.max_modifications = max_modifications
        self.max_rounds = max_rounds

        # Load board libraries
        self.opponent_pool = BoardPool(opponent_library_path, cache=True)
        self.opponent_library = self.opponent_pool.get_all_boards()

        if base_library_path is None:
            self.base_pool = self.opponent_pool
            self.base_library = self.opponent_library
        else:
            self.base_pool = BoardPool(base_library_path, cache=True)
            self.base_library = self.base_pool.get_all_boards()

        if len(self.opponent_library) == 0:
            raise ValueError(f"Opponent library is empty: {opponent_library_path}")
        if len(self.base_library) == 0:
            raise ValueError(f"Base library is empty: {base_library_path or opponent_library_path}")

        # Action space: [base_or_modify, cell, type, done]
        # Phase 1 (base selection): only first element used (0 to base_library_size-1)
        # Phase 2 (modification): all elements used
        self.action_space = spaces.MultiDiscrete([
            len(self.base_library),  # base_board_index (or modify_type in phase 2)
            board_size * board_size,  # cell
            2,                         # piece_or_trap
            2,                         # done
        ])

        # Observation space
        board_shape = (board_size, board_size, 2)
        self.observation_space = spaces.Dict({
            "opponent_board": spaces.Box(
                low=0, high=board_size * board_size,
                shape=board_shape, dtype=np.int32
            ),
            "base_board": spaces.Box(
                low=0, high=board_size * board_size,
                shape=board_shape, dtype=np.int32
            ),
            "modification_step": spaces.Discrete(max_modifications + 1),
            "modifications_made": spaces.Box(
                low=0, high=max_modifications, shape=(1,), dtype=np.int32
            ),
            "phase": spaces.Discrete(2),  # 0=base_selection, 1=modification
            "valid_cells_mask": spaces.MultiBinary(board_size * board_size),
        })

        # Episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0

        # Modification state
        self.phase = 0  # 0=base_selection, 1=modification
        self.base_board: Optional[Board] = None
        self.working_board: Optional[np.ndarray] = None
        self.opponent_board: Optional[Board] = None
        self.modification_step = 0
        self.modifications_made = 0

    def _select_opponent_board(self) -> Board:
        """Select opponent's board based on strategy."""
        if self.opponent_strategy == "random":
            return self.opponent_library[np.random.randint(len(self.opponent_library))]
        elif self.opponent_strategy == "greedy":
            # TODO: Implement greedy selection (select board that beats agent's previous)
            return self.opponent_library[np.random.randint(len(self.opponent_library))]
        elif self.opponent_strategy == "fixed":
            return self.opponent_library[0]
        elif self.opponent_strategy.startswith("fixed_"):
            idx = int(self.opponent_strategy.split("_")[1])
            return self.opponent_library[idx % len(self.opponent_library)]
        else:
            raise ValueError(f"Unknown opponent strategy: {self.opponent_strategy}")

    def _board_to_grid(self, board: Board) -> np.ndarray:
        """Convert Board object to grid representation."""
        grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)

        for move in board.sequence:
            # Skip goal moves
            if move.type == "goal":
                continue

            row, col = move.position.row, move.position.col
            order = move.order

            if move.type == "piece":
                grid[row, col, 0] = order
            elif move.type == "trap":
                grid[row, col, 1] = order

        return grid

    def _grid_to_board(self, grid: np.ndarray) -> Board:
        """Convert grid representation to Board object."""
        from .types import BoardMove, Position

        # Collect all moves (pieces and traps)
        moves = []

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece_order = grid[row, col, 0]
                trap_order = grid[row, col, 1]

                if piece_order > 0:
                    moves.append(BoardMove(
                        position=Position(row=row, col=col),
                        type="piece",
                        order=int(piece_order)
                    ))
                if trap_order > 0:
                    moves.append(BoardMove(
                        position=Position(row=row, col=col),
                        type="trap",
                        order=int(trap_order)
                    ))

        # Sort by order
        moves.sort(key=lambda m: m.order)

        # Add goal move
        moves.append(BoardMove(
            position=Position(row=-1, col=-1),
            type="goal",
            order=len(moves) + 1
        ))

        # Build grid for Board (string representation)
        str_grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        for move in moves:
            if move.type == "goal":
                continue
            row, col = move.position.row, move.position.col
            if move.type == "piece":
                if str_grid[row][col] == "T":
                    str_grid[row][col] = "B"  # Both
                else:
                    str_grid[row][col] = "P"
            elif move.type == "trap":
                if str_grid[row][col] == "P":
                    str_grid[row][col] = "B"  # Both
                else:
                    str_grid[row][col] = "T"

        return Board(
            boardSize=self.board_size,
            grid=tuple(tuple(row) for row in str_grid),
            sequence=tuple(moves)
        )

    def _get_valid_cells_mask(self) -> np.ndarray:
        """Get mask of cells that can be modified."""
        # All cells are valid for modification
        return np.ones(self.board_size * self.board_size, dtype=np.int8)

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        opponent_grid = self._board_to_grid(self.opponent_board)

        if self.phase == 0:
            # Base selection phase - no base board yet
            base_grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)
        else:
            # Modification phase - show working board
            base_grid = self.working_board.copy()

        return {
            "opponent_board": opponent_grid,
            "base_board": base_grid,
            "modification_step": self.modification_step,
            "modifications_made": np.array([self.modifications_made], dtype=np.int32),
            "phase": self.phase,
            "valid_cells_mask": self._get_valid_cells_mask(),
        }

    def _is_valid_board(self, board: Board) -> bool:
        """Check if board is valid according to game rules."""
        return is_board_playable(board)

    def _play_round(self) -> Tuple[int, int, bool]:
        """
        Play one round with current boards.

        Returns:
            (agent_score, opponent_score, is_valid)
        """
        agent_board = self._grid_to_board(self.working_board)

        # Check validity
        if not self._is_valid_board(agent_board):
            return 0, 0, False

        # Play the round
        result = simulate_round(
            self.current_round,
            agent_board,
            self.opponent_board,
            size=self.board_size,
        )

        return result.playerPoints, result.opponentPoints, True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0

        # Start new round
        self.opponent_board = self._select_opponent_board()
        self.phase = 0  # Start with base selection
        self.base_board = None
        self.working_board = None
        self.modification_step = 0
        self.modifications_made = 0

        obs = self._get_observation()
        info = {
            "round": self.current_round + 1,
            "phase": "base_selection",
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.

        Args:
            action: [base_or_modify, cell, type, done]
                Phase 0: base_or_modify = base_board_index (others ignored)
                Phase 1: base_or_modify = modify_type (0=add, 1=remove)
        """
        if self.phase == 0:
            # Phase 0: Base selection
            base_idx = action[0] % len(self.base_library)
            self.base_board = self.base_library[base_idx]
            self.working_board = self._board_to_grid(self.base_board)
            self.phase = 1
            self.modification_step = 0
            self.modifications_made = 0

            obs = self._get_observation()
            info = {
                "round": self.current_round + 1,
                "phase": "modification",
                "base_board_selected": base_idx,
            }

            # Small reward for selecting base
            return obs, 0.1, False, False, info

        else:
            # Phase 1: Modification
            modify_type = action[0] % 2  # 0=add, 1=remove
            cell = action[1] % (self.board_size * self.board_size)
            piece_or_trap = action[2] % 2
            done = action[3] > 0

            row = cell // self.board_size
            col = cell % self.board_size

            reward = 0.0

            # Apply modification if not done
            if not done and self.modifications_made < self.max_modifications:
                if modify_type == 0:  # Add
                    # Find next available order number
                    if piece_or_trap == 0:  # Piece
                        max_order = np.max(self.working_board[:, :, 0])
                        self.working_board[row, col, 0] = max_order + 1
                    else:  # Trap
                        max_order = np.max(self.working_board[:, :, 1])
                        self.working_board[row, col, 1] = max_order + 1

                    reward += 0.05  # Small reward for making modification

                else:  # Remove
                    if piece_or_trap == 0:
                        self.working_board[row, col, 0] = 0
                    else:
                        self.working_board[row, col, 1] = 0

                    reward += 0.05

                self.modifications_made += 1
                self.modification_step += 1

                # Check if max modifications reached
                if self.modifications_made >= self.max_modifications:
                    done = True

            # If done or max modifications, play the round
            if done or self.modifications_made >= self.max_modifications:
                agent_score, opponent_score, is_valid = self._play_round()

                # Calculate reward
                if not is_valid:
                    reward += -50.0  # Heavy penalty for invalid board
                else:
                    score_diff = agent_score - opponent_score
                    reward += float(score_diff)

                    if agent_score > opponent_score:
                        reward += 10.0  # Win bonus
                    elif agent_score == opponent_score:
                        reward += 5.0  # Tie bonus (better than nothing)

                # Update episode scores
                self.agent_total_score += agent_score
                self.opponent_total_score += opponent_score
                self.current_round += 1

                # Check if episode is done
                episode_done = self.current_round >= self.max_rounds

                obs = self._get_observation()
                info = {
                    "round": self.current_round,
                    "agent_score": agent_score,
                    "opponent_score": opponent_score,
                    "agent_total_score": self.agent_total_score,
                    "opponent_total_score": self.opponent_total_score,
                    "valid_board": is_valid,
                    "modifications_made": self.modifications_made,
                }

                if episode_done:
                    # Episode complete
                    return obs, reward, True, False, info
                else:
                    # Start next round
                    self.opponent_board = self._select_opponent_board()
                    self.phase = 0
                    self.base_board = None
                    self.working_board = None
                    self.modification_step = 0
                    self.modifications_made = 0

                    obs = self._get_observation()
                    info["round"] = self.current_round + 1
                    info["phase"] = "base_selection"

                    return obs, reward, False, False, info

            else:
                # Continue modifications
                obs = self._get_observation()
                info = {
                    "round": self.current_round + 1,
                    "phase": "modification",
                    "modification_step": self.modification_step,
                    "modifications_made": self.modifications_made,
                }

                return obs, reward, False, False, info

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
