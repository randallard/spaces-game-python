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

import collections
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

    IMPORTANT - Agent no-revisit rule:
    The agent's action masks prevent revisiting cells (piece_visited_positions).
    This is an agent optimization, NOT a game rule — human players are allowed to
    revisit cells (it's just a bad strategy since scoring is first-visit only).

    If you are integrating this agent into an app or API, you MUST track
    piece_visited_positions when driving construction manually (outside of
    env.step()). See play_against_agent.py _agent_build_board_blind() for
    the reference implementation.
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
        board_library_path: Optional[str] = None,
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

        # board_library_path accepted for backward compatibility but ignored
        # (strict masking makes construction scaffolding unnecessary)

        # Self-play support
        self._opponent_model = None
        self.use_self_play = False

        # Action space: flat Discrete
        # [0..n_cells-1] = piece at cell i
        # [n_cells..2*n_cells-1] = trap at cell i
        # [2*n_cells] = finish
        n_cells = board_size * board_size
        self.action_space = spaces.Discrete(2 * n_cells + 1)

        # Observation space
        board_shape = (board_size, board_size, 2)
        self.observation_space = spaces.Dict({
            # Construction state (resets each round)
            "building_board": spaces.Box(
                low=0, high=board_size * board_size,
                shape=board_shape, dtype=np.int32,
            ),
            "construction_step": spaces.Discrete(max_construction_steps + 1),

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
        self.piece_visited_positions: set = set()
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

    def _can_reach_all_rows(
        self,
        extra_trap: Optional[Tuple[int, int]] = None,
        hypothetical_pos: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """BFS reachability check: can the piece visit all required rows AND reach row 0?

        The piece must:
        1. Be able to visit all rows 0..board_size-1 (via already-visited + reachable)
        2. Be able to reach a cell at row 0 (to finish the board)

        Args:
            extra_trap: hypothetical trap position to add (for trap placement checks)
            hypothetical_pos: hypothetical piece position (for piece move checks)

        Returns:
            True if both conditions are satisfiable.
        """
        size = self.board_size

        # Determine starting position
        if hypothetical_pos is not None:
            start_row, start_col = hypothetical_pos
        elif self.current_piece_position is not None:
            start_row = self.current_piece_position.row
            start_col = self.current_piece_position.col
        else:
            return False  # No piece placed yet

        # Rows already visited by piece
        visited_rows = set()
        for pos_key in self.piece_visited_positions:
            r, c = pos_key.split(",")
            visited_rows.add(int(r))

        # If hypothetical_pos, add its row too
        if hypothetical_pos is not None:
            visited_rows.add(hypothetical_pos[0])

        required_rows = set(range(size))
        all_rows_visited = visited_rows >= required_rows

        # Build set of blocked cells (traps + visited piece positions)
        traps = set(self.trap_positions)
        if extra_trap is not None:
            traps.add(f"{extra_trap[0]},{extra_trap[1]}")

        visited_cells = set(self.piece_visited_positions)
        if hypothetical_pos is not None:
            visited_cells.add(f"{hypothetical_pos[0]},{hypothetical_pos[1]}")

        # If all rows visited AND piece is at row 0, no BFS needed
        if all_rows_visited and start_row == 0:
            return True

        # BFS from start position through unvisited, non-trap cells
        queue = collections.deque()
        queue.append((start_row, start_col))
        bfs_visited = {(start_row, start_col)}
        reachable_rows = set(visited_rows)
        reachable_rows.add(start_row)
        can_reach_row0 = start_row == 0

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (0, -1), (0, 1)]:  # forward-only (no backward)
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= size or nc < 0 or nc >= size:
                    continue
                if (nr, nc) in bfs_visited:
                    continue
                cell_key = f"{nr},{nc}"
                if cell_key in traps:
                    continue
                if cell_key in visited_cells:
                    continue
                bfs_visited.add((nr, nc))
                reachable_rows.add(nr)
                if nr == 0:
                    can_reach_row0 = True
                queue.append((nr, nc))

        return reachable_rows >= required_rows and can_reach_row0

    def _is_valid_placement(self, row: int, col: int, move_type: str) -> bool:
        """Check if placement at (row, col) of given type is valid."""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False

        if move_type == "piece":
            if self.current_piece_position is None:
                return row == self.board_size - 1  # First piece: bottom row

            # Forward-only: cannot move to a higher row index
            if row > self.current_piece_position.row:
                return False

            if f"{row},{col}" in self.piece_visited_positions:
                return False  # No revisiting cells

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

            # Level 2: BFS reachability - can piece still reach all rows?
            if not self._can_reach_all_rows(hypothetical_pos=(row, col)):
                return False

            return True

        elif move_type == "trap":
            if self.current_piece_position is None:
                return False

            # Forward-only: traps cannot be placed behind the piece
            if row > self.current_piece_position.row:
                return False

            if len(self.trap_positions) >= self.board_size - 1:
                return False  # Trap limit: max board_size - 1 traps

            if self.supermove_active:
                return False

            if f"{row},{col}" in self.trap_positions:
                return False

            same_pos = (row == self.current_piece_position.row and
                        col == self.current_piece_position.col)
            if same_pos:
                # Supermove trap: check that at least one adjacent escape cell
                # exists AND from that escape cell all rows remain reachable
                has_escape = False
                for dr, dc in [(-1, 0), (0, -1), (0, 1)]:  # forward-only escapes
                    nr, nc = row + dr, col + dc
                    if nr < 0 or nr >= self.board_size or nc < 0 or nc >= self.board_size:
                        continue
                    cell_key = f"{nr},{nc}"
                    if cell_key in self.trap_positions:
                        continue
                    if cell_key in self.piece_visited_positions:
                        continue
                    # Check reachability from this escape cell with the new trap
                    if self._can_reach_all_rows(
                        extra_trap=(row, col), hypothetical_pos=(nr, nc),
                    ):
                        has_escape = True
                        break
                return has_escape

            if not _is_adjacent_orthogonal(
                self.current_piece_position, Position(row=row, col=col),
            ):
                return False

            # Level 2: BFS reachability - does this trap block all paths?
            if not self._can_reach_all_rows(extra_trap=(row, col)):
                return False

            return True

        return False

    def action_masks(self) -> np.ndarray:
        """Return flat action mask for MaskablePPO with Discrete action space.

        Layout: [piece_cell_0..piece_cell_n-1, trap_cell_0..trap_cell_n-1, finish]
        """
        n_cells = self.board_size * self.board_size
        mask = np.zeros(2 * n_cells + 1, dtype=np.int8)

        has_any_valid = False
        for r in range(self.board_size):
            for c in range(self.board_size):
                idx = r * self.board_size + c
                if self._is_valid_placement(r, c, "piece"):
                    mask[idx] = 1
                    has_any_valid = True
                if self._is_valid_placement(r, c, "trap"):
                    mask[n_cells + idx] = 1
                    has_any_valid = True

        # Finish action
        visited_rows = set()
        for pos_key in self.piece_visited_positions:
            row_str, _ = pos_key.split(",")
            visited_rows.add(int(row_str))
        all_rows_visited = visited_rows >= set(range(self.board_size))
        can_finish = (
            self.current_piece_position is not None
            and self.current_piece_position.row == 0
            and not self.supermove_active
            and all_rows_visited
        )

        if can_finish:
            mask[2 * n_cells] = 1

        # Deadlock: force finish if no valid placements
        if not has_any_valid:
            mask[2 * n_cells] = 1

        return mask

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
            # Goal column must match piece's current column
            goal_col = (
                self.current_piece_position.col
                if self.current_piece_position is not None
                else self._get_last_move_column()
            )
            sequence.append(BoardMove(
                position=Position(row=-1, col=goal_col),
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
            "construction_step": min(self.construction_step, self.max_construction_steps),
            "round": min(self.current_round, self.ROUNDS_PER_GAME - 1),
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

        # Track agent's boards for opponent history in self-play
        self._agent_boards_this_game: List[Board] = []

        # Reset construction state for round 0
        self._init_construction_state()

        obs = self._get_observation()
        info = {
            "round": 0,
            "opponent_phase": self.opponent_phase,
        }
        return obs, info

    def step(
        self, action,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one construction step or round transition."""
        self.steps_taken += 1

        # Truncate round if too many steps
        if self.steps_taken >= self.max_construction_steps:
            return self._finish_round(forced_invalid=True)

        action = int(action)
        n_cells = self.board_size * self.board_size

        # Finish action
        if action >= 2 * n_cells:
            return self._finish_round()

        # Decode cell and type
        cell = action % n_cells
        move_type = "piece" if action < n_cells else "trap"
        row = cell // self.board_size
        col = cell % self.board_size

        reward = 0.0

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
                    reward += 0.2  # Supermove landing
                self.piece_visited_positions.add(f"{row},{col}")
                self.current_piece_position = Position(row=row, col=col)
                reward += 0.1  # Piece placement
                if row == 0:
                    reward += 0.3  # Reached row 0 (goal row)

            elif move_type == "trap":
                self.trap_positions.add(f"{row},{col}")
                if (self.current_piece_position is not None and
                        row == self.current_piece_position.row and
                        col == self.current_piece_position.col):
                    self.supermove_active = True
                    self.supermove_position = Position(row=row, col=col)
                reward += 0.1  # Trap placement

            self.construction_step += 1

            obs = self._get_observation()
            info = {
                "round": self.current_round,
                "construction_step": self.construction_step,
            }
            return obs, reward, False, False, info

        else:
            # Should not occur with flat masking, but handle gracefully.
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

        # Track agent's board for self-play opponent history
        if is_valid:
            self._agent_boards_this_game.append(agent_board)

        # Select opponent's board (self-play or JSON pool)
        opponent_board = None
        if self.use_self_play and self._opponent_model is not None:
            opponent_board = self._build_opponent_board_from_model()
        if opponent_board is None:
            opponent_board = self._select_opponent_board()

        if not is_valid:
            # Invalid board fallback (shouldn't happen with strict masking)
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
            elif agent_round_score < opponent_round_score:
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

    # --- Self-play opponent model support ---

    def set_opponent_model(self, model_path: str) -> None:
        """Load opponent model from path. Callable via SubprocVecEnv.env_method().

        Takes a string path (not model object) for SubprocVecEnv pickle compatibility.
        """
        from sb3_contrib import MaskablePPO
        self._opponent_model = MaskablePPO.load(model_path)
        self.use_self_play = True

    def clear_opponent_model(self) -> None:
        """Revert to JSON pool opponents."""
        self._opponent_model = None
        self.use_self_play = False

    def _build_opponent_board_from_model(self) -> Optional[Board]:
        """Build opponent board using the loaded opponent model.

        Mirrors the manual construction pattern from inference_server/inference.py.
        The opponent sees swapped scores (it thinks it's the agent).
        Returns None if the model produces an invalid board.
        """
        if self._opponent_model is None:
            return None

        max_steps = self.max_construction_steps

        # Create a temporary env for opponent construction
        opp_env = SimultaneousPlayEnv(
            board_size=self.board_size,
            opponent_pools=self.opponent_pool_paths,
            max_construction_steps=max_steps,
            phase_map=self.phase_map,
        )
        opp_env.reset(seed=np.random.randint(100000))

        # Set opponent context: swapped scores, agent's boards as opponent history
        opp_env.current_round = self.current_round
        opp_env.agent_total_score = self.opponent_total_score  # swapped
        opp_env.opponent_total_score = self.agent_total_score  # swapped

        # Encode agent's previous boards into opponent's history (rotated)
        if hasattr(self, '_agent_boards_this_game'):
            for i, board in enumerate(self._agent_boards_this_game):
                if i < self.ROUNDS_PER_GAME:
                    opp_env.opponent_history_grids[i] = (
                        opp_env._encode_opponent_board(board)
                    )

        obs = opp_env._get_observation()

        for _ in range(max_steps):
            masks = opp_env.action_masks()
            action, _ = self._opponent_model.predict(
                obs, deterministic=False, action_masks=masks,
            )

            # Decode flat action
            act = int(action)
            opp_n_cells = self.board_size * self.board_size
            if act >= 2 * opp_n_cells:
                break  # finish

            cell = act % opp_n_cells
            move_type = "piece" if act < opp_n_cells else "trap"
            row = cell // self.board_size
            col = cell % self.board_size
            order = opp_env.construction_step + 1

            if opp_env._is_valid_placement(row, col, move_type):
                opp_env.construction_sequence.append({
                    "row": row, "col": col, "type": move_type, "order": order,
                })
                if move_type == "piece":
                    opp_env.building_grid[row, col, 0] = order
                    if opp_env.supermove_active:
                        opp_env.supermove_active = False
                        opp_env.supermove_position = None
                    opp_env.piece_visited_positions.add(f"{row},{col}")
                    opp_env.current_piece_position = Position(row=row, col=col)
                elif move_type == "trap":
                    opp_env.building_grid[row, col, 1] = order
                    opp_env.trap_positions.add(f"{row},{col}")
                    if (opp_env.current_piece_position is not None and
                            row == opp_env.current_piece_position.row and
                            col == opp_env.current_piece_position.col):
                        opp_env.supermove_active = True
                        opp_env.supermove_position = Position(row=row, col=col)
                opp_env.construction_step += 1

            obs = opp_env._get_observation()

        board = opp_env._construct_board_from_state()
        opp_env.close()

        if is_board_playable(board):
            return board
        return None

    def render(self):
        pass

    def close(self):
        pass
