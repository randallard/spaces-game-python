"""
Board loading utilities.

Loads pre-generated boards from JSON files and converts them to
immutable Board dataclasses.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union
import random

from .types import Board, BoardMove, Position, CellContent


def _convert_position(pos_dict: dict[str, int]) -> Position:
    """Convert JSON position dict to Position dataclass."""
    return Position(row=pos_dict['row'], col=pos_dict['col'])


def _convert_board_move(move_dict: dict[str, Any]) -> BoardMove:
    """Convert JSON board move dict to BoardMove dataclass."""
    return BoardMove(
        position=_convert_position(move_dict['position']),
        type=move_dict['type'],
        order=move_dict['order'],
    )


def _convert_grid(grid_list: list[list[str]]) -> tuple[tuple[CellContent, ...], ...]:
    """Convert JSON grid (list of lists) to immutable tuple of tuples."""
    return tuple(
        tuple(cell for cell in row)  # type: ignore
        for row in grid_list
    )


def _convert_sequence(sequence_list: list[dict[str, Any]]) -> tuple[BoardMove, ...]:
    """Convert JSON sequence (list of move dicts) to immutable tuple of BoardMoves."""
    return tuple(_convert_board_move(move) for move in sequence_list)


def load_board_from_dict(board_dict: dict[str, Any]) -> Board:
    """
    Load a single board from a dictionary.

    Args:
        board_dict: Board data as dictionary (from JSON)

    Returns:
        Immutable Board dataclass

    Example:
        >>> data = json.load(open('board.json'))
        >>> board = load_board_from_dict(data)
    """
    return Board(
        boardSize=board_dict['boardSize'],
        grid=_convert_grid(board_dict['grid']),
        sequence=_convert_sequence(board_dict['sequence']),
    )


def load_boards_from_json(file_path: Union[str, Path]) -> list[Board]:
    """
    Load all boards from a JSON file.

    Args:
        file_path: Path to JSON file containing board(s)

    Returns:
        List of Board dataclasses

    The JSON file can be either:
    - A single board object: {"boardSize": 3, "grid": [...], "sequence": [...]}
    - An array of boards: [{"boardSize": 3, ...}, ...]
    - A collection object: {"boards": [...], "name": "...", ...}

    Example:
        >>> boards = load_boards_from_json('data/boards_size_3.json')
        >>> print(f"Loaded {len(boards)} boards")
    """
    file_path = Path(file_path)

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Handle different JSON formats
    if isinstance(data, list):
        # Array of boards
        return [load_board_from_dict(board_dict) for board_dict in data]
    elif 'boards' in data:
        # Collection object (e.g., from my-boards.json)
        return [load_board_from_dict(board_dict) for board_dict in data['boards']]
    else:
        # Single board object
        return [load_board_from_dict(data)]


def load_board_by_index(file_path: Union[str, Path], index: int) -> Board:
    """
    Load a specific board by index from a JSON file.

    Args:
        file_path: Path to JSON file
        index: Board index (0-based)

    Returns:
        Board at the specified index

    Raises:
        IndexError: If index is out of range

    Example:
        >>> board = load_board_by_index('my-boards.json', 0)
    """
    boards = load_boards_from_json(file_path)
    return boards[index]


class BoardPool:
    """
    Efficient board pool for training.

    Loads boards once and provides fast sampling.
    Optionally caches to pickle for faster subsequent loads.
    """

    def __init__(self, json_path: Union[str, Path], cache: bool = True):
        """
        Initialize board pool.

        Args:
            json_path: Path to JSON file with boards
            cache: If True, cache boards as pickle for faster loading
        """
        self.json_path = Path(json_path)
        self.cache_path = self.json_path.with_suffix('.pkl')
        self.cache_enabled = cache

        self.boards = self._load_boards()

    def _load_boards(self) -> list[Board]:
        """Load boards from cache if available, otherwise from JSON."""
        # Try loading from cache first
        if self.cache_enabled and self.cache_path.exists():
            cache_mtime = self.cache_path.stat().st_mtime
            json_mtime = self.json_path.stat().st_mtime

            # Use cache only if it's newer than JSON
            if cache_mtime > json_mtime:
                try:
                    with open(self.cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    # Fall back to JSON if cache is corrupted
                    pass

        # Load from JSON
        boards = load_boards_from_json(self.json_path)

        # Save to cache
        if self.cache_enabled:
            try:
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(boards, f)
            except Exception:
                # Silently ignore cache write failures
                pass

        return boards

    def sample(self, n: int = 1) -> list[Board]:
        """
        Sample random boards from the pool.

        Args:
            n: Number of boards to sample

        Returns:
            List of randomly sampled boards

        Example:
            >>> pool = BoardPool('data/boards_size_3.json')
            >>> opponents = pool.sample(10)
        """
        return random.sample(self.boards, min(n, len(self.boards)))

    def sample_one(self) -> Board:
        """Sample a single random board."""
        return random.choice(self.boards)

    def get(self, index: int) -> Board:
        """Get board by index."""
        return self.boards[index]

    def get_all_boards(self) -> list[Board]:
        """
        Get all boards in the pool.

        Returns:
            Complete list of all boards

        Example:
            >>> pool = BoardPool('new_boards_2.json')
            >>> all_boards = pool.get_all_boards()
            >>> print(f"Library has {len(all_boards)} boards")
        """
        return self.boards

    def __len__(self) -> int:
        """Number of boards in the pool."""
        return len(self.boards)

    def __getitem__(self, index: int) -> Board:
        """Allow indexing: pool[0]"""
        return self.boards[index]

    def __iter__(self):
        """Allow iteration: for board in pool"""
        return iter(self.boards)
