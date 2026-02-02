"""
Gymnasium environment for Spaces Game.

Implements a standard RL environment for training agents to play Spaces Game.
The game is a 5-round simultaneous selection game with partial observability.
"""

import random
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .types import Board, RoundResult
from .board_loader import BoardPool
from .simulation import simulate_round


class SpacesGameEnv(gym.Env):
    """
    Gymnasium environment for Spaces Game.

    Game Structure:
    - 5 rounds per game
    - Each round: both players select a board from their deck
    - Boards are revealed simultaneously and simulated
    - Winner determined by total score after 5 rounds

    Observation Space (Partial Observability):
    - Round number (1-5)
    - Score differential
    - Agent's deck (10 boards)
    - Agent's board selection history (5 rounds)
    - Opponent's board selection history (5 rounds) - INDICES ONLY
    - Who picks first this round (alternates)

    Action Space:
    - Discrete(10): Select one of 10 boards from agent's deck

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
        board_pool_path: str = "data/boards_size_3.json",
        deck_size: int = 10,
        opponent_strategy: str = "random",
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Spaces Game environment.

        Args:
            board_pool_path: Path to JSON file with pre-generated boards
            deck_size: Number of boards in each player's deck (default 10)
            opponent_strategy: Opponent strategy ("random", "greedy", or "trained")
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()

        self.deck_size = deck_size
        self.opponent_strategy = opponent_strategy
        self.render_mode = render_mode

        # Load board pool
        self.board_pool = BoardPool(board_pool_path, cache=True)

        # Observation space: dict with multiple components
        self.observation_space = spaces.Dict({
            # Current round (1-5)
            "round": spaces.Discrete(5, start=1),

            # Score differential (agent - opponent), bounded by max possible score
            "score_diff": spaces.Box(low=-500, high=500, shape=(1,), dtype=np.float32),

            # Total scores
            "agent_score": spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
            "opponent_score": spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),

            # Who picks first this round (0=agent, 1=opponent)
            "first_picker": spaces.Discrete(2),

            # Board selection history (indices into deck, -1 = not played yet)
            "agent_history": spaces.Box(low=-1, high=deck_size-1, shape=(5,), dtype=np.int32),
            "opponent_history": spaces.Box(low=-1, high=deck_size-1, shape=(5,), dtype=np.int32),
        })

        # Action space: select one board from deck
        self.action_space = spaces.Discrete(deck_size)

        # Episode state
        self.current_round = 0
        self.agent_deck: list[Board] = []
        self.opponent_deck: list[Board] = []
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.agent_history: list[int] = []
        self.opponent_history: list[int] = []
        self.round_results: list[RoundResult] = []

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

        # Seed Python's random module for opponent strategy
        if seed is not None:
            random.seed(seed)

        # Reset episode state
        self.current_round = 1
        self.agent_total_score = 0
        self.opponent_total_score = 0
        self.agent_history = []
        self.opponent_history = []
        self.round_results = []

        # Sample new decks for both players
        self.agent_deck = self.board_pool.sample(self.deck_size)
        self.opponent_deck = self.board_pool.sample(self.deck_size)

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
            action: Board index to select (0-9)

        Returns:
            observation: New observation
            reward: Reward for this round
            terminated: Whether episode ended
            truncated: Whether episode was truncated (always False)
            info: Additional info
        """
        if not 0 <= action < self.deck_size:
            raise ValueError(f"Invalid action {action}, must be in [0, {self.deck_size})")

        if self.current_round > 5:
            raise RuntimeError("Episode already terminated")

        # Agent selects board
        agent_board = self.agent_deck[action]
        self.agent_history.append(action)

        # Opponent selects board using their strategy
        opponent_action = self._opponent_select_board()
        opponent_board = self.opponent_deck[opponent_action]
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

        observation = self._get_observation()
        info = self._get_info()

        return observation, round_reward, terminated, False, info

    def render(self) -> Optional[str]:
        """
        Render the environment.

        Returns:
            Rendered output (if render_mode is "ansi")
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _get_observation(self) -> Dict[str, Any]:
        """Build observation dict."""
        # Pad history arrays with -1 for unplayed rounds
        agent_hist = np.array(self.agent_history + [-1] * (5 - len(self.agent_history)), dtype=np.int32)
        opponent_hist = np.array(self.opponent_history + [-1] * (5 - len(self.opponent_history)), dtype=np.int32)

        # Determine who picks first (game creator picks first in odd rounds)
        first_picker = 0 if self.current_round % 2 == 1 else 1

        return {
            "round": self.current_round,
            "score_diff": np.array([self.agent_total_score - self.opponent_total_score], dtype=np.float32),
            "agent_score": np.array([self.agent_total_score], dtype=np.float32),
            "opponent_score": np.array([self.opponent_total_score], dtype=np.float32),
            "first_picker": first_picker,
            "agent_history": agent_hist,
            "opponent_history": opponent_hist,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Build info dict with additional details."""
        return {
            "round": self.current_round,
            "agent_total_score": self.agent_total_score,
            "opponent_total_score": self.opponent_total_score,
            "rounds_completed": len(self.round_results),
        }

    def _opponent_select_board(self) -> int:
        """
        Opponent board selection strategy.

        Returns:
            Board index (0-9)
        """
        if self.opponent_strategy == "random":
            # Random selection from unused boards
            used_indices = set(self.opponent_history)
            available = [i for i in range(self.deck_size) if i not in used_indices]
            if not available:
                # All boards used, can repeat (shouldn't happen in 5-round game with 10 boards)
                available = list(range(self.deck_size))
            return random.choice(available)

        elif self.opponent_strategy == "greedy":
            # Simple greedy: pick board with most moves (rough heuristic)
            used_indices = set(self.opponent_history)
            available = [i for i in range(self.deck_size) if i not in used_indices]
            if not available:
                available = list(range(self.deck_size))

            # Select board with longest sequence (more moves = potentially higher score)
            best_idx = max(available, key=lambda i: len(self.opponent_deck[i].sequence))
            return best_idx

        else:
            # Default to random
            return self._opponent_select_board.__func__(self)  # Call random strategy

    def _render_ansi(self) -> str:
        """Render game state as ASCII."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Spaces Game - Round {self.current_round}/5")
        lines.append("=" * 60)
        lines.append(f"Score: Agent {self.agent_total_score} - {self.opponent_total_score} Opponent")
        lines.append(f"Differential: {self.agent_total_score - self.opponent_total_score:+d}")
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
