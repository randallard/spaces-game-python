"""
Board Construction Environment for Spaces Game.

Stage 1 of progressive curriculum: Agent learns to select counter-boards
from a library of valid boards. Unlike deck selection mode, boards can be
reused across rounds (no resource management).

Key differences from gym_env.py:
- No deck tracking or "remaining boards" concept
- All boards always available for selection
- Focus on matchup learning, not resource allocation
- Simplified game flow for faster learning
"""

import random
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .types import Board, RoundResult
from .board_loader import BoardPool
from .simulation import simulate_round


class BoardConstructionEnv(gym.Env):
    """
    Gymnasium environment for board construction training.

    Training Mode: Agent sees opponent's board, selects counter from library.

    Game Structure:
    - 5 rounds per game
    - Each round: agent selects board from library (can reuse boards)
    - Opponent selects from their library (can reuse boards)
    - Boards revealed simultaneously and simulated
    - Winner determined by total score after 5 rounds

    Observation Space:
    - Round number (1-5)
    - Score differential
    - Opponent's board (encoded as grid)
    - Round history (which boards were played)

    Action Space:
    - Discrete(N): Select one board from library (N boards available)

    Reward:
    - Per round: Score differential (agent_score - opponent_score)
    - Episode end: +100 for win, -100 for loss, 0 for tie

    Episode:
    - 5 rounds
    - Terminates after round 5
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        board_library_path: str = "new_boards_2.json",
        opponent_strategy: str = "random",
        render_mode: Optional[str] = None,
        show_opponent_board: bool = True,
    ):
        """
        Initialize board construction environment.

        Args:
            board_library_path: Path to JSON file with valid boards
            opponent_strategy: Opponent strategy ("random", "greedy", or "fixed")
            render_mode: Rendering mode ("human", "ansi", or None)
            show_opponent_board: If True, agent sees opponent's board before selecting
        """
        super().__init__()

        self.opponent_strategy = opponent_strategy
        self.render_mode = render_mode
        self.show_opponent_board = show_opponent_board

        # Load board library (all valid boards agent can choose from)
        self.board_pool = BoardPool(board_library_path, cache=True)
        self.agent_library = self.board_pool.get_all_boards()
        self.opponent_library = self.board_pool.get_all_boards()

        # Get board size from first board
        sample_board = self.agent_library[0]
        self.board_size = sample_board.boardSize
        self.library_size = len(self.agent_library)

        # Action space: select one board from library
        self.action_space = spaces.Discrete(self.library_size)

        # Observation space
        obs_space_dict = {
            # Current round (0-4, representing rounds 1-5)
            "round": spaces.Discrete(5),

            # Score differential (agent - opponent)
            "score_diff": spaces.Box(low=-500, high=500, shape=(1,), dtype=np.float32),

            # Total scores
            "agent_score": spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
            "opponent_score": spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),

            # Board selection history (NOT used boards, just history)
            "agent_history": spaces.Box(low=-1, high=self.library_size-1, shape=(5,), dtype=np.int32),
            "opponent_history": spaces.Box(low=-1, high=self.library_size-1, shape=(5,), dtype=np.int32),
        }

        # Add opponent board observation if enabled
        if self.show_opponent_board:
            obs_space_dict["opponent_board"] = spaces.Box(
                low=0,
                high=20,
                shape=(self.board_size, self.board_size, 4),
                dtype=np.float32
            )

        self.observation_space = spaces.Dict(obs_space_dict)

        # Episode state
        self.current_round = 0
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.agent_history: list[int] = []
        self.opponent_history: list[int] = []
        self.round_results: list[RoundResult] = []
        self.current_opponent_board: Optional[Board] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)

        # Seed Python's random for opponent
        if seed is not None:
            random.seed(seed)

        # Reset episode state
        self.current_round = 1
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.agent_history = []
        self.opponent_history = []
        self.round_results = []

        # Pre-select opponent's board for round 1 if showing it
        if self.show_opponent_board:
            self.current_opponent_board = self._opponent_select_board()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one round of the game.

        Args:
            action: Board index to select from library

        Returns:
            observation: New observation
            reward: Reward for this round
            terminated: Whether episode ended
            truncated: Whether episode was truncated (always False)
            info: Additional info
        """
        if not 0 <= action < self.library_size:
            raise ValueError(f"Invalid action {action}, must be in [0, {self.library_size})")

        if self.current_round > 5:
            raise RuntimeError("Episode already terminated")

        # Agent selects board from library
        agent_board = self.agent_library[action]
        self.agent_history.append(action)

        # Get opponent's board (already selected if showing, else select now)
        if self.show_opponent_board and self.current_opponent_board is not None:
            opponent_board = self.current_opponent_board
            opponent_action = self.opponent_library.index(opponent_board)
        else:
            opponent_board = self._opponent_select_board()
            opponent_action = self.opponent_library.index(opponent_board)

        self.opponent_history.append(opponent_action)

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
        round_reward = float(result.playerPoints - result.opponentPoints)

        # Check if episode is done
        terminated = (self.current_round >= 5)

        # Add episode-end bonus/penalty
        if terminated:
            if self.agent_total_score > self.opponent_total_score:
                round_reward += 100.0  # Win bonus
            elif self.agent_total_score < self.opponent_total_score:
                round_reward -= 100.0  # Loss penalty
            # Tie: no bonus

        # Move to next round
        self.current_round += 1

        # Pre-select opponent's board for next round if showing it
        if not terminated and self.show_opponent_board:
            self.current_opponent_board = self._opponent_select_board()

        observation = self._get_observation()
        info = self._get_info()

        return observation, round_reward, terminated, False, info

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _encode_board_as_grid(self, board: Board) -> np.ndarray:
        """
        Encode a board as a grid for neural network input.

        Args:
            board: Board to encode

        Returns:
            Grid of shape (board_size, board_size, 4) with:
            - Channel 0: has_piece (0 or 1)
            - Channel 1: piece_order (0 if no piece, 1-N for sequence order)
            - Channel 2: has_trap (0 or 1)
            - Channel 3: trap_order (0 if no trap, 1-N for sequence order)
        """
        grid = np.zeros((board.boardSize, board.boardSize, 4), dtype=np.float32)

        for move in board.sequence:
            row, col = move.position.row, move.position.col

            # Skip goal position (row -1)
            if row < 0:
                continue

            if move.type == 'piece':
                grid[row, col, 0] = 1.0  # has_piece
                grid[row, col, 1] = float(move.order)  # piece_order
            elif move.type == 'trap':
                grid[row, col, 2] = 1.0  # has_trap
                grid[row, col, 3] = float(move.order)  # trap_order

        return grid

    def _get_observation(self) -> Dict[str, Any]:
        """Build observation dict."""
        # Pad history arrays with -1 for unplayed rounds
        agent_hist = np.array(self.agent_history + [-1] * (5 - len(self.agent_history)), dtype=np.int32)
        opponent_hist = np.array(self.opponent_history + [-1] * (5 - len(self.opponent_history)), dtype=np.int32)

        obs = {
            "round": self.current_round - 1,  # 0-indexed for SB3 (rounds 1-5 become 0-4)
            "score_diff": np.array([self.agent_total_score - self.opponent_total_score], dtype=np.float32),
            "agent_score": np.array([self.agent_total_score], dtype=np.float32),
            "opponent_score": np.array([self.opponent_total_score], dtype=np.float32),
            "agent_history": agent_hist,
            "opponent_history": opponent_hist,
        }

        # Add opponent board if showing it
        if self.show_opponent_board and self.current_opponent_board is not None:
            obs["opponent_board"] = self._encode_board_as_grid(self.current_opponent_board)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict with additional details."""
        return {
            "round": self.current_round,
            "agent_total_score": self.agent_total_score,
            "opponent_total_score": self.opponent_total_score,
            "rounds_completed": len(self.round_results),
            "library_size": self.library_size,
        }

    def _opponent_select_board(self) -> Board:
        """
        Opponent board selection strategy.

        NOTE: Unlike deck selection, ALL boards are always available.
        No filtering based on history - boards can be reused!

        Returns:
            Selected board
        """
        if self.opponent_strategy == "random":
            # Random selection from entire library
            return random.choice(self.opponent_library)

        elif self.opponent_strategy == "greedy":
            # Select board with longest sequence
            return max(self.opponent_library, key=lambda b: len(b.sequence))

        elif self.opponent_strategy == "fixed":
            # Always use first board (for testing counter-play)
            return self.opponent_library[0]

        elif self.opponent_strategy.startswith("fixed_"):
            # Fixed board by index: "fixed_0", "fixed_1", etc.
            try:
                board_idx = int(self.opponent_strategy.split("_")[1])
                return self.opponent_library[board_idx]
            except (ValueError, IndexError):
                # Fall back to random if invalid
                return random.choice(self.opponent_library)

        else:
            # Default to random
            return random.choice(self.opponent_library)

    def _render_ansi(self) -> str:
        """Render game state as ASCII."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Board Construction - Round {self.current_round}/5")
        lines.append("=" * 60)
        lines.append(f"Score: Agent {self.agent_total_score} - {self.opponent_total_score} Opponent")
        lines.append(f"Differential: {self.agent_total_score - self.opponent_total_score:+d}")
        lines.append(f"Library size: {self.library_size} boards (all reusable)")
        lines.append("")

        if self.round_results:
            lines.append("Round History:")
            for i, result in enumerate(self.round_results, 1):
                winner_str = {
                    'player': 'Agent',
                    'opponent': 'Opponent',
                    'tie': 'Tie'
                }[result.winner]
                lines.append(
                    f"  Round {i}: {winner_str} "
                    f"(Agent: {result.playerPoints}, Opponent: {result.opponentPoints})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)
