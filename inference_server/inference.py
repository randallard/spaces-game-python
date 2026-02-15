"""
Core inference functions extracted from examples/play_against_agent.py.

Provides reusable model loading, board encoding, and board construction
without any CLI (click) dependencies.
"""

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from spaces_game import SimultaneousPlayEnv
from spaces_game.simulation import _rotate_position
from spaces_game.validation import is_board_playable
from spaces_game.types import Board, Position

logger = logging.getLogger(__name__)


def load_agent(model_path: str) -> Tuple[object, bool]:
    """Load a trained agent model.

    Tries MaskablePPO first (sb3_contrib), then falls back to PPO (stable_baselines3).

    Args:
        model_path: Path to the .zip model file.

    Returns:
        Tuple of (model, uses_masks) where uses_masks indicates MaskablePPO.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If the model cannot be loaded by either PPO or MaskablePPO.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        logger.info("Loaded model as MaskablePPO: %s", model_path)
        return model, True
    except Exception as e:
        logger.debug("MaskablePPO load failed (%s), trying PPO...", e)

    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        logger.info("Loaded model as PPO: %s", model_path)
        return model, False
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path}: {e}"
        ) from e


def get_model_board_size(model) -> Optional[int]:
    """Extract board size from a loaded model's observation space.

    Args:
        model: A loaded SB3 model.

    Returns:
        The board size (N for NxN grid), or None if it cannot be determined.
    """
    try:
        obs_space = model.observation_space
        if hasattr(obs_space, 'spaces') and 'building_board' in obs_space.spaces:
            return obs_space.spaces['building_board'].shape[0]
        if hasattr(obs_space, 'shape') and obs_space.shape is not None:
            return obs_space.shape[0]
    except Exception:
        pass
    return None


def is_stage3_model(model) -> bool:
    """Check if model was trained with SimultaneousPlayEnv (has opponent_history).

    Args:
        model: A loaded SB3 model.

    Returns:
        True if the model has opponent_history in its observation space.
    """
    try:
        return 'opponent_history' in model.observation_space.spaces
    except Exception:
        return False


def encode_board_for_agent(board: Board, board_size: int) -> np.ndarray:
    """Encode a Board rotated 180 degrees (opponent's perspective for agent).

    Used to populate the opponent_history observation field.

    Args:
        board: The Board to encode.
        board_size: The grid dimension (N for NxN).

    Returns:
        numpy array of shape (board_size, board_size, 2) with piece orders
        in channel 0 and trap orders in channel 1.
    """
    grid = np.zeros((board_size, board_size, 2), dtype=np.int32)
    for move in board.sequence:
        if move.type == "final":
            continue
        rot = _rotate_position(move.position.row, move.position.col, board_size)
        if rot.row < 0 or rot.row >= board_size or rot.col < 0 or rot.col >= board_size:
            continue
        channel = 0 if move.type == "piece" else 1
        grid[rot.row][rot.col][channel] = move.order
    return grid


def discover_opponent_pools(board_size: int, boards_dir: str = "boards/") -> List[str]:
    """Auto-discover opponent board pool JSON files in boards_dir/size{N}/.

    Args:
        board_size: The board size to look for pools.
        boards_dir: Base directory containing sizeN subdirectories.

    Returns:
        Sorted list of JSON file paths found.
    """
    pools = []
    base = Path(boards_dir) / f"size{board_size}"
    if base.exists():
        for f in sorted(base.glob("*.json")):
            pools.append(str(f))
    logger.debug("Discovered %d opponent pools for size %d in %s", len(pools), board_size, boards_dir)
    return pools


def build_board_for_round(
    model,
    uses_masks: bool,
    board_size: int,
    round_num: int,
    agent_score: float,
    opponent_score: float,
    opponent_history_grids: np.ndarray,
    opponent_pools: List[str],
    deterministic: bool = True,
    max_retries: int = 5,
) -> Tuple[Board, int]:
    """Have a Stage 3 agent build a board blind (simultaneous play).

    Manually drives construction using the SimultaneousPlayEnv's observation
    space and action masking, but does NOT simulate or advance rounds.

    Args:
        model: Loaded SB3 model.
        uses_masks: Whether model uses MaskablePPO action masking.
        board_size: Grid dimension (NxN).
        round_num: Current round (0-indexed).
        agent_score: Agent's cumulative score so far.
        opponent_score: Opponent's cumulative score so far.
        opponent_history_grids: numpy array of shape (5, board_size, board_size, 2)
            containing encoded opponent boards from previous rounds.
        opponent_pools: List of paths to opponent pool JSON files.
        deterministic: If True, use deterministic prediction.
        max_retries: Maximum attempts to produce a valid board.

    Returns:
        Tuple of (Board, attempts_used). Board may be invalid if all retries fail.
    """
    max_steps = board_size * 10
    env = SimultaneousPlayEnv(
        board_size=board_size,
        opponent_pools=opponent_pools,
        max_construction_steps=max_steps,
    )

    board = None
    for attempt in range(max_retries):
        seed = 42 + attempt if deterministic else random.randint(0, 100000)
        env.reset(seed=seed)

        # Override game state to match current round context
        env.current_round = round_num
        env.agent_total_score = agent_score
        env.opponent_total_score = opponent_score
        env.opponent_history_grids = opponent_history_grids.copy()

        obs = env._get_observation()

        for _ in range(max_steps):
            if uses_masks:
                masks = env.action_masks()
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            # Decode flat action
            act = int(action)
            n_cells = board_size * board_size
            if act >= 2 * n_cells:
                break  # finish

            cell = act % n_cells
            move_type = "piece" if act < n_cells else "trap"
            row = cell // board_size
            col = cell % board_size
            order = env.construction_step + 1

            if env._is_valid_placement(row, col, move_type):
                env.construction_sequence.append({
                    "row": row, "col": col, "type": move_type, "order": order,
                })
                if move_type == "piece":
                    env.building_grid[row, col, 0] = order
                    if env.supermove_active:
                        env.supermove_active = False
                        env.supermove_position = None
                    env.piece_visited_positions.add(f"{row},{col}")
                    env.current_piece_position = Position(row=row, col=col)
                elif move_type == "trap":
                    env.building_grid[row, col, 1] = order
                    env.trap_positions.add(f"{row},{col}")
                    if (env.current_piece_position is not None and
                            row == env.current_piece_position.row and
                            col == env.current_piece_position.col):
                        env.supermove_active = True
                        env.supermove_position = Position(row=row, col=col)
                env.construction_step += 1

            obs = env._get_observation()

        board = env._construct_board_from_state()
        if is_board_playable(board):
            env.close()
            if attempt > 0:
                logger.info("Valid board produced on attempt %d", attempt + 1)
            return board, attempt + 1

    env.close()
    logger.warning("No valid board after %d attempts", max_retries)
    return board, max_retries
