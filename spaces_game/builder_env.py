"""
Board Builder Environment for Spaces Game.

Stage 2 of progressive curriculum: Agent learns to BUILD boards from scratch
using sequential construction (place pieces/traps step-by-step) rather than
selecting from a pre-made library.

Key differences from construction_env.py:
- Agent constructs board sequentially (multiple steps per board)
- Action masking ensures only valid placements
- Focus on learning construction rules and validity constraints
"""

import random
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .types import Board, BoardMove, Position
from .board_loader import BoardPool
from .simulation import simulate_round
from .validation import is_board_playable


class BoardBuilderEnv(gym.Env):
    """
    Gymnasium environment for sequential board construction.

    Agent builds a board step-by-step by placing pieces and traps, then
    plays against an opponent board to learn what makes a good board.

    Game Structure:
    - 5 rounds per game
    - Each round:
      1. Agent builds board sequentially (3-8 steps)
      2. Opponent selects board from library
      3. Boards revealed and simulated
      4. Score differential as reward
    - Winner determined by total score after 5 rounds

    Construction Rules:
    - First piece must be in bottom row
    - Each subsequent piece/trap must be adjacent to existing pieces
    - Supermove: Can place trap on same cell as piece
    - Must reach goal (valid path exists)

    Action Space:
    - cell: Which cell to place in (0 to board_size²-1)
    - type: 0=piece, 1=trap
    - done: 0=continue building, 1=finish board

    Observation Space:
    - Current board being built (partial)
    - Opponent's board (if show_opponent_board=True)
    - Round number, score differential
    - Valid action mask (which cells can be placed)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        board_size: int = 2,
        opponent_library_path: str = "new_boards_2.json",
        opponent_strategy: str = "random",
        render_mode: Optional[str] = None,
        show_opponent_board: bool = True,
        max_construction_steps: int = 10,
    ):
        """
        Initialize board builder environment.

        Args:
            board_size: Size of board (2 for 2x2, 3 for 3x3, etc.)
            opponent_library_path: Path to opponent's board library
            opponent_strategy: Opponent strategy ("random", "greedy", "fixed_N")
            render_mode: Rendering mode ("human", "ansi", or None)
            show_opponent_board: If True, agent sees opponent's board before building
            max_construction_steps: Maximum steps to build a board
        """
        super().__init__()

        self.board_size = board_size
        self.opponent_strategy = opponent_strategy
        self.render_mode = render_mode
        self.show_opponent_board = show_opponent_board
        self.max_construction_steps = max_construction_steps

        # Load opponent board library
        self.opponent_pool = BoardPool(opponent_library_path, cache=True)
        self.opponent_library = self.opponent_pool.get_all_boards()

        # Action space: cell to place (flattened), type (piece/trap), done flag
        self.action_space = spaces.Dict({
            "cell": spaces.Discrete(board_size * board_size),
            "type": spaces.Discrete(2),  # 0=piece, 1=trap
            "done": spaces.Discrete(2),  # 0=continue, 1=finish
        })

        # Observation space
        obs_space_dict = {
            # Current round (0-4, representing rounds 1-5)
            "round": spaces.Discrete(5),

            # Score differential (agent - opponent)
            "score_diff": spaces.Box(low=-500, high=500, shape=(1,), dtype=np.float32),

            # Construction state
            "construction_step": spaces.Discrete(max_construction_steps + 1),
            "building_board": spaces.Box(
                low=0, high=max_construction_steps,
                shape=(board_size, board_size, 2),  # [piece_order, trap_order]
                dtype=np.float32
            ),

            # Valid action mask (which cells can be placed)
            "valid_cells_mask": spaces.MultiBinary(board_size * board_size),
        }

        # Add opponent board if showing it
        if self.show_opponent_board:
            obs_space_dict["opponent_board"] = spaces.Box(
                low=0, high=20,
                shape=(board_size, board_size, 4),
                dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_space_dict)

        # Episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.round_results: List[Any] = []
        self.current_opponent_board: Optional[Board] = None

        # Construction state
        self.construction_step = 0
        self.building_grid = np.zeros((board_size, board_size, 2), dtype=np.int32)  # [piece_order, trap_order]
        self.construction_sequence: List[Dict[str, Any]] = []
        self.piece_count = 0
        self.trap_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        # Reset episode state
        self.current_round = 1
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.round_results = []

        # Pre-select opponent's board for round 1 if showing it
        if self.show_opponent_board:
            self.current_opponent_board = self._select_opponent_board()

        # Reset construction state
        self._reset_construction()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _reset_construction(self):
        """Reset construction state for new board."""
        self.construction_step = 0
        self.building_grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)
        self.construction_sequence = []
        self.piece_count = 0
        self.trap_count = 0

    def step(
        self,
        action: Dict[str, int]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one construction step or finish board and play round.

        Args:
            action: Dict with 'cell', 'type', 'done'

        Returns:
            observation, reward, terminated, truncated, info
        """
        cell_idx = action["cell"]
        piece_or_trap = action["type"]  # 0=piece, 1=trap
        done_building = action["done"]  # 0=continue, 1=finish

        # If done flag set, finish board and play round
        if done_building == 1 or self.construction_step >= self.max_construction_steps:
            return self._finish_board_and_play()

        # Otherwise, place piece/trap
        row, col = divmod(cell_idx, self.board_size)

        # Validate placement (should be caught by masking, but check anyway)
        if not self._is_valid_placement(row, col):
            # Invalid placement: negative reward, force finish
            return self._finish_board_and_play(invalid_penalty=-10.0)

        # Place piece or trap
        order = self.construction_step + 1

        if piece_or_trap == 0:  # Piece
            # Check if cell already has piece
            if self.building_grid[row, col, 0] > 0:
                # Already has piece, invalid
                return self._finish_board_and_play(invalid_penalty=-10.0)

            # Check if cell has trap (piece cannot be placed after trap)
            if self.building_grid[row, col, 1] > 0:
                # Trap already there, cannot place piece after trap
                return self._finish_board_and_play(invalid_penalty=-10.0)

            self.building_grid[row, col, 0] = order
            self.piece_count += 1
            self.construction_sequence.append({
                "row": row, "col": col, "type": "piece", "order": order
            })

        else:  # Trap
            # Check if cell already has trap
            if self.building_grid[row, col, 1] > 0:
                # Already has trap, invalid
                return self._finish_board_and_play(invalid_penalty=-10.0)

            # Supermove: trap on piece is OK (piece was placed first, has lower order)
            # But validate that if there's a piece, it was placed BEFORE this trap
            if self.building_grid[row, col, 0] > 0:
                piece_order = self.building_grid[row, col, 0]
                if piece_order >= order:
                    # Piece order should be LESS than trap order (piece placed first)
                    return self._finish_board_and_play(invalid_penalty=-10.0)

            self.building_grid[row, col, 1] = order
            self.trap_count += 1
            self.construction_sequence.append({
                "row": row, "col": col, "type": "trap", "order": order
            })

        self.construction_step += 1

        # Intermediate reward for valid placement
        intermediate_reward = 0.1

        observation = self._get_observation()
        info = self._get_info()

        # Episode not done yet (still building)
        return observation, intermediate_reward, False, False, info

    def _is_valid_placement(self, row: int, col: int) -> bool:
        """
        Check if placement at (row, col) is valid.

        Rules:
        - First piece must be in bottom row
        - Subsequent pieces/traps must be adjacent to existing pieces
        - Can place on same cell as existing piece (supermove: trap on piece)
        """
        # First piece: must be in bottom row
        if len(self.construction_sequence) == 0:
            return row == self.board_size - 1

        # Check if cell itself has a piece (valid for supermove)
        if self.building_grid[row, col, 0] > 0:
            return True

        # Otherwise, must be adjacent to existing piece
        has_adjacent_piece = False
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                if self.building_grid[nr, nc, 0] > 0:  # Has piece
                    has_adjacent_piece = True
                    break

        return has_adjacent_piece

    def _get_valid_cells_mask(self) -> np.ndarray:
        """Get mask of valid cells for placement."""
        mask = np.zeros(self.board_size * self.board_size, dtype=np.int8)

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self._is_valid_placement(row, col):
                    cell_idx = row * self.board_size + col
                    mask[cell_idx] = 1

        return mask

    def _finish_board_and_play(self, invalid_penalty: float = 0.0) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Finish board construction and play round against opponent.

        Args:
            invalid_penalty: Penalty for invalid board construction

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Convert construction to Board object
        agent_board = self._construction_to_board()

        # Check if board is valid (playable)
        if agent_board is None or not is_board_playable(agent_board):
            # Invalid board: large penalty, end episode
            reward = invalid_penalty - 50.0
            self.current_round = 6  # Force episode end
            observation = self._get_observation()
            info = self._get_info()
            info["invalid_board"] = True
            return observation, reward, True, False, info

        # Get opponent's board
        if self.show_opponent_board and self.current_opponent_board is not None:
            opponent_board = self.current_opponent_board
        else:
            opponent_board = self._select_opponent_board()

        # Simulate round
        result = simulate_round(
            self.current_round,
            agent_board,
            opponent_board,
            silent=True
        )
        self.round_results.append(result)

        # Update scores
        self.agent_total_score += result.playerPoints
        self.opponent_total_score += result.opponentPoints

        # Calculate reward (score differential for this round)
        round_reward = float(result.playerPoints - result.opponentPoints) + invalid_penalty

        # Check if episode is done
        terminated = (self.current_round >= 5)

        # Add episode-end bonus/penalty
        if terminated:
            if self.agent_total_score > self.opponent_total_score:
                round_reward += 100.0  # Win bonus
            elif self.agent_total_score < self.opponent_total_score:
                round_reward -= 100.0  # Loss penalty

        # Move to next round
        self.current_round += 1

        # Reset construction for next round
        if not terminated:
            self._reset_construction()
            if self.show_opponent_board:
                self.current_opponent_board = self._select_opponent_board()

        observation = self._get_observation()
        info = self._get_info()

        return observation, round_reward, terminated, False, info

    def _construction_to_board(self) -> Optional[Board]:
        """Convert current construction state to Board object."""
        if len(self.construction_sequence) == 0:
            return None

        # Build grid
        grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Build sequence
        sequence = []
        for step in self.construction_sequence:
            row, col, step_type, order = step["row"], step["col"], step["type"], step["order"]

            move = BoardMove(
                position=Position(row=row, col=col),
                type=step_type,
                order=order
            )
            sequence.append(move)

            # Update grid representation (for debugging)
            if step_type == "piece":
                grid[row][col] = "P" if grid[row][col] == "." else "B"  # B = both piece+trap
            elif step_type == "trap":
                grid[row][col] = "T" if grid[row][col] == "." else "B"

        # Add goal move (always last)
        goal_move = BoardMove(
            position=Position(row=-1, col=-1),
            type="goal",
            order=len(sequence) + 1
        )
        sequence.append(goal_move)

        board = Board(
            boardSize=self.board_size,
            grid=tuple(tuple(row) for row in grid),
            sequence=tuple(sequence)
        )

        return board

    def _select_opponent_board(self) -> Board:
        """Select opponent's board based on strategy."""
        if self.opponent_strategy == "random":
            return random.choice(self.opponent_library)
        elif self.opponent_strategy == "greedy":
            return max(self.opponent_library, key=lambda b: len(b.sequence))
        elif self.opponent_strategy.startswith("fixed_"):
            try:
                idx = int(self.opponent_strategy.split("_")[1])
                return self.opponent_library[idx]
            except (ValueError, IndexError):
                return random.choice(self.opponent_library)
        else:
            return random.choice(self.opponent_library)

    def _encode_board_as_grid(self, board: Board) -> np.ndarray:
        """Encode board as grid for observation."""
        grid = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)

        for move in board.sequence:
            if move.position.row < 0:  # Skip goal
                continue

            row, col = move.position.row, move.position.col

            if move.type == 'piece':
                grid[row, col, 0] = 1.0  # has_piece
                grid[row, col, 1] = float(move.order)  # piece_order
            elif move.type == 'trap':
                grid[row, col, 2] = 1.0  # has_trap
                grid[row, col, 3] = float(move.order)  # trap_order

        return grid

    def _get_observation(self) -> Dict[str, Any]:
        """Build observation dict."""
        obs = {
            "round": self.current_round - 1,  # 0-indexed
            "score_diff": np.array([self.agent_total_score - self.opponent_total_score], dtype=np.float32),
            "construction_step": self.construction_step,
            "building_board": self.building_grid.astype(np.float32),
            "valid_cells_mask": self._get_valid_cells_mask(),
        }

        # Add opponent board if showing it
        if self.show_opponent_board and self.current_opponent_board is not None:
            obs["opponent_board"] = self._encode_board_as_grid(self.current_opponent_board)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict."""
        return {
            "round": self.current_round,
            "agent_total_score": self.agent_total_score,
            "opponent_total_score": self.opponent_total_score,
            "construction_step": self.construction_step,
            "piece_count": self.piece_count,
            "trap_count": self.trap_count,
        }

    def render(self) -> Optional[str]:
        """Render environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render as ASCII."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Board Builder - Round {self.current_round}/5")
        lines.append("=" * 60)
        lines.append(f"Score: Agent {self.agent_total_score} - {self.opponent_total_score} Opponent")
        lines.append(f"Construction Step: {self.construction_step}/{self.max_construction_steps}")
        lines.append("")

        # Show building board
        lines.append("Building Board:")
        for row in range(self.board_size):
            row_str = "  "
            for col in range(self.board_size):
                piece_order = int(self.building_grid[row, col, 0])
                trap_order = int(self.building_grid[row, col, 1])

                if piece_order > 0 and trap_order > 0:
                    cell = f"P{piece_order}T{trap_order}"
                elif piece_order > 0:
                    cell = f"P{piece_order}"
                elif trap_order > 0:
                    cell = f"T{trap_order}"
                else:
                    cell = "."

                row_str += f"{cell:6s} "
            lines.append(row_str)

        lines.append("=" * 60)
        return "\n".join(lines)
