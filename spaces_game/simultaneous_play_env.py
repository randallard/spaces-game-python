"""
Simultaneous Play Environment (Stage 3).

5-round game where both players build boards simultaneously (no peeking),
then full boards are revealed after simulation. Agent must learn to
adapt across rounds based on what the opponent played.

Progressive opponent curriculum:
- Phase 0: Simple boards only
- Phase 1: One-trap boards
- Phase 2: Simple + one-trap mixed
- Phase 3: Supermove boards
- Phase 4: All board types mixed
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from .types import Board, BoardMove, Position
from .board_loader import load_boards_from_json
from .simulation import simulate_round, _rotate_position
from .validation import is_board_playable, _is_adjacent_orthogonal


# Default phase mapping: which pools are active at each phase
DEFAULT_PHASE_MAP = {
    0: [0],           # pool[0] only (e.g. simple)
    1: [1],           # pool[1] only (e.g. one_trap)
    2: [0, 1],        # mixed
    3: [2],           # pool[2] only (e.g. super_move)
    4: [0, 1, 2, 3],  # all pools
}


class SimultaneousPlayEnv(gym.Env):
    """
    Gymnasium environment for 5-round simultaneous board construction.

    Each round:
    1. Agent constructs a board (blind - can't see opponent's current board)
    2. Opponent picks from their pool
    3. Simulation runs
    4. Opponent's full board is revealed in opponent_history
    5. Scores update, next round begins

    Episode ends after 5 rounds with game-level win/loss bonus.
    """

    metadata = {"render_modes": ["human"]}

    ROUNDS_PER_GAME = 5

    def __init__(
        self,
        board_size: int = 2,
        opponent_pools: Optional[List[str]] = None,
        opponent_phase: int = 0,
        phase_map: Optional[Dict[int, List[int]]] = None,
        max_construction_steps: int = 20,
    ):
        super().__init__()

        self.board_size = board_size
        self.opponent_phase = opponent_phase
        self.max_construction_steps = max_construction_steps
        self.phase_map = phase_map or DEFAULT_PHASE_MAP

        # Load opponent board pools
        if opponent_pools is None:
            opponent_pools = [f"boards/size{board_size}/simple.json"]

        self.opponent_pool_paths = opponent_pools
        self.opponent_pools: List[List[Board]] = []
        for path in opponent_pools:
            boards = load_boards_from_json(path)
            if len(boards) == 0:
                raise ValueError(f"Empty board pool: {path}")
            self.opponent_pools.append(boards)

        # Action space: [cell, type, done]
        self.action_space = spaces.MultiDiscrete([
            board_size * board_size,  # cell
            2,                         # type (0=piece, 1=trap)
            2,                         # done (0=continue, 1=finish)
        ])

        # Observation space
        board_shape = (board_size, board_size, 2)
        self.observation_space = spaces.Dict({
            # Construction state (resets each round)
            "building_board": spaces.Box(
                low=0, high=board_size * board_size,
                shape=board_shape, dtype=np.int32,
            ),
            "construction_step": spaces.Discrete(max_construction_steps + 1),
            "valid_cells_mask": spaces.MultiBinary(board_size * board_size),

            # Game state (persists across rounds)
            "round": spaces.Discrete(self.ROUNDS_PER_GAME),
            "score_diff": spaces.Box(
                low=-50, high=50, shape=(1,), dtype=np.float32,
            ),
            "agent_score": spaces.Box(
                low=0, high=50, shape=(1,), dtype=np.float32,
            ),
            "opponent_score": spaces.Box(
                low=0, high=50, shape=(1,), dtype=np.float32,
            ),

            # Full reveal history - opponent's boards from previous rounds
            "opponent_history": spaces.Box(
                low=0, high=20,
                shape=(self.ROUNDS_PER_GAME, board_size, board_size, 2),
                dtype=np.int32,
            ),
        })

        # Persistent game state (across rounds within episode)
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.opponent_history_grids = np.zeros(
            (self.ROUNDS_PER_GAME, board_size, board_size, 2), dtype=np.int32,
        )

        # Construction state (resets each round)
        self._init_construction_state()

    def _init_construction_state(self):
        """Reset construction state for a new round."""
        self.building_grid = np.zeros(
            (self.board_size, self.board_size, 2), dtype=np.int32,
        )
        self.construction_step = 0
        self.steps_taken = 0
        self.current_piece_position: Optional[Position] = None
        self.trap_positions: set = set()
        self.supermove_active: bool = False
        self.supermove_position: Optional[Position] = None
        self.construction_sequence: List[Dict[str, Any]] = []

    # --- Opponent selection ---

    def _get_active_pools(self) -> List[List[Board]]:
        """Get opponent pools active at current phase."""
        pool_indices = self.phase_map.get(
            self.opponent_phase,
            list(range(len(self.opponent_pools))),  # fallback: all pools
        )
        # Clamp to available pools
        return [
            self.opponent_pools[i]
            for i in pool_indices
            if i < len(self.opponent_pools)
        ]

    def _select_opponent_board(self) -> Board:
        """Select opponent board from active pools."""
        active = self._get_active_pools()
        if not active:
            active = self.opponent_pools  # fallback
        # Pick a random pool, then a random board from it
        pool = active[np.random.randint(len(active))]
        return pool[np.random.randint(len(pool))]

    # --- Board encoding ---

    def _encode_opponent_board(self, board: Board) -> np.ndarray:
        """Encode opponent's full board rotated to agent's frame."""
        grid = np.zeros((self.board_size, self.board_size, 2), dtype=np.int32)
        for move in board.sequence:
            if move.type == "final":
                continue
            rot = _rotate_position(
                move.position.row, move.position.col, self.board_size,
            )
            if rot.row < 0 or rot.row >= self.board_size:
                continue
            if rot.col < 0 or rot.col >= self.board_size:
                continue
            channel = 0 if move.type == "piece" else 1
            grid[rot.row][rot.col][channel] = move.order
        return grid

    # --- Construction logic (adapted from ReverseCurriculumBuilderEnv) ---

    def _is_valid_placement(self, row: int, col: int, move_type: str) -> bool:
        """Check if placement at (row, col) of given type is valid."""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False

        if move_type == "piece":
            if self.current_piece_position is None:
                return row == self.board_size - 1  # First piece: bottom row

            if not _is_adjacent_orthogonal(
                self.current_piece_position, Position(row=row, col=col),
            ):
                return False

            if f"{row},{col}" in self.trap_positions:
                return False

            if self.supermove_active:
                if (row == self.supermove_position.row and
                        col == self.supermove_position.col):
                    return False

            return True

        elif move_type == "trap":
            if self.current_piece_position is None:
                return False

            if self.supermove_active:
                return False

            if f"{row},{col}" in self.trap_positions:
                return False

            same_pos = (row == self.current_piece_position.row and
                        col == self.current_piece_position.col)
            if same_pos:
                return True

            return _is_adjacent_orthogonal(
                self.current_piece_position, Position(row=row, col=col),
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

        if mask.sum() == 0:
            mask[:] = 1  # Fallback to prevent MaskablePPO crash

        return mask

    def action_masks(self) -> np.ndarray:
        """Return action mask for MaskablePPO with MultiDiscrete action space."""
        n_cells = self.board_size * self.board_size

        # Check for deadlock
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
            # Deadlock - force finish
            cell_mask = np.ones(n_cells, dtype=np.int8)
            type_mask = np.ones(2, dtype=np.int8)
            done_mask = np.array([0, 1], dtype=np.int8)
            return np.concatenate([cell_mask, type_mask, done_mask])

        # Cell mask
        cell_mask = self._get_valid_cells_mask()

        # Type mask [piece, trap]
        type_mask = np.zeros(2, dtype=np.int8)
        if self.supermove_active or self.current_piece_position is None:
            type_mask[0] = 1  # must place piece
        else:
            type_mask[0] = 1 if has_valid_piece else 0
            type_mask[1] = 1 if has_valid_trap else 0

        # Done mask [continue, finish]
        can_finish = (
            self.current_piece_position is not None
            and self.current_piece_position.row == 0
            and not self.supermove_active
        )
        done_mask = np.array([1, 1 if can_finish else 0], dtype=np.int8)

        return np.concatenate([cell_mask, type_mask, done_mask])

    def _get_last_move_column(self) -> int:
        """Get the column of the last non-final move."""
        if self.construction_sequence:
            for step in reversed(self.construction_sequence):
                if step["type"] != "final":
                    return step["col"]
        return 0

    def _construct_board_from_state(self) -> Board:
        """Construct Board from construction_sequence."""
        if len(self.construction_sequence) == 0:
            return Board(
                boardSize=self.board_size,
                grid=tuple(
                    tuple("empty" for _ in range(self.board_size))
                    for _ in range(self.board_size)
                ),
                sequence=(BoardMove(Position(row=-1, col=0), "final", 1),),
            )

        sequence = []
        for step in self.construction_sequence:
            sequence.append(BoardMove(
                position=Position(row=step["row"], col=step["col"]),
                type=step["type"],
                order=step["order"],
            ))

        # Ensure final/goal move exists
        if not any(m.type == "final" for m in sequence):
            last_col = self._get_last_move_column()
            sequence.append(BoardMove(
                position=Position(row=-1, col=last_col),
                type="final",
                order=len(sequence) + 1,
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
            sequence=tuple(sequence),
        )

    # --- Observation ---

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            "building_board": self.building_grid.copy(),
            "construction_step": self.construction_step,
            "valid_cells_mask": self._get_valid_cells_mask(),
            "round": self.current_round,
            "score_diff": np.array(
                [self.agent_total_score - self.opponent_total_score],
                dtype=np.float32,
            ),
            "agent_score": np.array(
                [self.agent_total_score], dtype=np.float32,
            ),
            "opponent_score": np.array(
                [self.opponent_total_score], dtype=np.float32,
            ),
            "opponent_history": self.opponent_history_grids.copy(),
        }

    # --- Gym API ---

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new 5-round game."""
        super().reset(seed=seed)

        # Reset game state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.opponent_history_grids = np.zeros(
            (self.ROUNDS_PER_GAME, self.board_size, self.board_size, 2),
            dtype=np.int32,
        )

        # Reset construction state for round 0
        self._init_construction_state()

        obs = self._get_observation()
        info = {
            "round": 0,
            "opponent_phase": self.opponent_phase,
        }
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one construction step or round transition."""
        self.steps_taken += 1

        # Truncate round if too many steps
        if self.steps_taken >= self.max_construction_steps:
            return self._finish_round(forced_invalid=True)

        cell = int(action[0]) % (self.board_size * self.board_size)
        piece_or_trap = int(action[1]) % 2
        done_flag = int(action[2]) > 0

        row = cell // self.board_size
        col = cell % self.board_size
        move_type = "piece" if piece_or_trap == 0 else "trap"

        reward = 0.0

        # Agent signals done - finish this round
        if done_flag:
            return self._finish_round()

        # Try to place a move
        order = self.construction_step + 1

        if self._is_valid_placement(row, col, move_type):
            # Valid placement - update state
            self.construction_sequence.append({
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
                reward += 0.1
                if row == 0:
                    reward += 0.3  # Top row - can reach goal

            elif move_type == "trap":
                self.trap_positions.add(f"{row},{col}")
                if (self.current_piece_position is not None and
                        row == self.current_piece_position.row and
                        col == self.current_piece_position.col):
                    self.supermove_active = True
                    self.supermove_position = Position(row=row, col=col)
                    reward += 0.2  # Supermove bonus
                else:
                    reward += 0.1

            self.construction_step += 1

            obs = self._get_observation()
            info = {
                "round": self.current_round,
                "construction_step": self.construction_step,
            }
            return obs, reward, False, False, info

        else:
            # Invalid placement
            reward = -2.0
            obs = self._get_observation()
            info = {
                "round": self.current_round,
                "construction_step": self.construction_step,
                "invalid_placement": True,
            }
            return obs, reward, False, False, info

    def _finish_round(
        self, forced_invalid: bool = False,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Finish the current round: validate, simulate, update state."""
        reward = 0.0

        # Build agent's board
        agent_board = self._construct_board_from_state()
        is_valid = not forced_invalid and is_board_playable(agent_board)

        # Select opponent's board
        opponent_board = self._select_opponent_board()

        if not is_valid:
            # Invalid board penalty - opponent wins round by default
            reward += -20.0
            agent_round_score = 0
            opponent_round_score = 5  # Default win for opponent
        else:
            # Simulate
            result = simulate_round(
                self.current_round + 1,
                agent_board,
                opponent_board,
                silent=True,
            )
            agent_round_score = result.playerPoints
            opponent_round_score = result.opponentPoints
            score_diff = agent_round_score - opponent_round_score

            reward += float(score_diff) * 5.0
            if agent_round_score > opponent_round_score:
                reward += 10.0
            elif agent_round_score == opponent_round_score:
                reward += 0.0  # Tie is neutral per round
            else:
                reward += -5.0

        # Update game state
        self.agent_total_score += agent_round_score
        self.opponent_total_score += opponent_round_score

        # Record opponent's board in history (full reveal)
        self.opponent_history_grids[self.current_round] = (
            self._encode_opponent_board(opponent_board)
        )

        # Advance round
        self.current_round += 1
        terminated = self.current_round >= self.ROUNDS_PER_GAME

        # Episode-end bonus
        if terminated:
            if self.agent_total_score > self.opponent_total_score:
                reward += 50.0
            elif self.agent_total_score < self.opponent_total_score:
                reward += -50.0

        # Reset construction state for next round (if not terminated)
        if not terminated:
            self._init_construction_state()

        obs = self._get_observation()
        info = {
            "round": self.current_round,
            "agent_round_score": agent_round_score,
            "opponent_round_score": opponent_round_score,
            "agent_total_score": self.agent_total_score,
            "opponent_total_score": self.opponent_total_score,
            "valid_board": is_valid,
            "opponent_phase": self.opponent_phase,
        }

        if terminated:
            info["game_winner"] = (
                "agent" if self.agent_total_score > self.opponent_total_score
                else "opponent" if self.agent_total_score < self.opponent_total_score
                else "tie"
            )

        return obs, reward, terminated, False, info

    # --- Phase control ---

    def set_opponent_phase(self, phase: int) -> None:
        """Set opponent phase. Callable via SubprocVecEnv.env_method()."""
        self.opponent_phase = phase

    def render(self):
        pass

    def close(self):
        pass
