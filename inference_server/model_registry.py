"""
Model registry for managing skill-level-to-model mapping.

Handles convention-based discovery of model checkpoints and provides
a cache of loaded models keyed by (board_size, checkpoint_type).

Directory structure:
    {models_dir}/size{N}/
    ├── best_model.zip              # best trained model
    ├── difficulty/                  # skill-level checkpoints
    │   ├── beginner.zip
    │   ├── intermediate.zip
    │   ├── advanced.zip
    │   ├── advanced_plus.zip
    │   └── expert.zip
    ├── scripted_checkpoints/       # models trained against scripted agents
    └── level_advancement/          # curriculum level snapshots (3: first, mid, last)
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .inference import load_agent, get_model_board_size, is_stage3_model, discover_opponent_pools

logger = logging.getLogger(__name__)


# Skill level -> (checkpoint_type, deterministic)
SKILL_LEVEL_CONFIG: Dict[str, Tuple[str, bool]] = {
    "beginner":          ("early", False),
    "beginner_plus":     ("early", True),
    "intermediate":      ("mid", False),
    "intermediate_plus": ("mid", True),
    "advanced":          ("advanced", False),
    "advanced_plus":     ("advanced", True),
}

# Difficulty filename -> checkpoint_type mapping
DIFFICULTY_TO_CHECKPOINT: Dict[str, str] = {
    "beginner": "early",
    "intermediate": "mid",
    "advanced": "advanced",
    "advanced_plus": "advanced",
    "expert": "advanced",
}


def generate_model_id(board_size: int, category: str, label: str) -> str:
    """Generate a stable, short model ID from board_size, category, and label.

    Uses first 8 hex chars of SHA256("{board_size}:{category}:{label}").
    """
    key = f"{board_size}:{category}:{label}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class ModelRegistry:
    """Registry that loads and caches models, mapping skill levels to checkpoints.

    Convention-based discovery:
    - Models are expected in {models_dir}/size{N}/
    - difficulty/ contains skill-level checkpoints (beginner, intermediate, etc.)
    - scripted_checkpoints/ contains models trained against scripted agents
    - level_advancement/ contains curriculum level snapshots
    - best_model.zip is the best trained model
    - All models are fog-of-war (use_fog=True)
    """

    def __init__(self, models_dir: str = "inference_server/models/", boards_dir: str = "boards/"):
        self.models_dir = models_dir
        self.boards_dir = boards_dir

        # Cache: (board_size, checkpoint_type) -> (model, uses_masks)
        self._model_cache: Dict[Tuple[int, str], Tuple[object, bool]] = {}

        # Discovered model paths: (board_size, checkpoint_type) -> path
        self._model_paths: Dict[Tuple[int, str], str] = {}

        # Discovered board sizes
        self._board_sizes: List[int] = []

        # Opponent pools cache: board_size -> list of pool paths
        self._opponent_pools: Dict[int, List[str]] = {}

        # Flat indexed list of all models
        self._indexed_models: List[dict] = []

        # Reverse lookup: model_id -> index
        self._model_id_to_index: Dict[str, int] = {}

        # Cache for indexed models: index -> (model, uses_masks)
        self._indexed_model_cache: Dict[int, Tuple[object, bool]] = {}

    def discover_models(self) -> None:
        """Scan models_dir for available models and populate the path registry."""
        base = Path(self.models_dir)
        if not base.exists():
            logger.warning("Models directory does not exist: %s", self.models_dir)
            return

        for size_dir in sorted(base.glob("size*")):
            if not size_dir.is_dir():
                continue
            match = re.match(r"size(\d+)", size_dir.name)
            if not match:
                continue
            board_size = int(match.group(1))

            # Discover difficulty checkpoints for skill-level mapping
            difficulty_dir = size_dir / "difficulty"
            if difficulty_dir.is_dir():
                self._assign_difficulty_checkpoints(board_size, difficulty_dir)

            # Check if we found any models for this size
            has_models = any(
                k[0] == board_size for k in self._model_paths
            )

            # Also check for any zip files in subdirs
            if not has_models:
                has_models = any(size_dir.rglob("*.zip"))

            if has_models:
                if board_size not in self._board_sizes:
                    self._board_sizes.append(board_size)

        # Build the flat indexed list from all discovered models
        self._build_indexed_models()

        # Discover opponent pools for each board size
        for board_size in self._board_sizes:
            pools = discover_opponent_pools(board_size, self.boards_dir)
            if pools:
                self._opponent_pools[board_size] = pools
            else:
                logger.warning(
                    "No opponent pools found for size %d in %s",
                    board_size, self.boards_dir,
                )

        logger.info(
            "Discovery complete: %d board sizes, %d skill-level paths, %d indexed models",
            len(self._board_sizes), len(self._model_paths), len(self._indexed_models),
        )

    def _assign_difficulty_checkpoints(self, board_size: int, difficulty_dir: Path) -> None:
        """Assign difficulty files to checkpoint types (early, mid, advanced)."""
        for zip_file in sorted(difficulty_dir.glob("*.zip")):
            name = zip_file.stem.lower()
            checkpoint_type = DIFFICULTY_TO_CHECKPOINT.get(name)
            if checkpoint_type is None:
                continue
            # Later files overwrite earlier ones for the same checkpoint_type,
            # so expert.zip overwrites advanced.zip for "advanced" — that's fine
            self._model_paths[(board_size, checkpoint_type)] = str(zip_file)
            logger.info(
                "Size %d, %s checkpoint (%s): %s",
                board_size, checkpoint_type, name, zip_file.name,
            )

    def load_all(self) -> None:
        """Load all discovered models into cache."""
        for (board_size, checkpoint_type), path in self._model_paths.items():
            if (board_size, checkpoint_type) in self._model_cache:
                continue
            try:
                model, uses_masks = load_agent(path)
                self._model_cache[(board_size, checkpoint_type)] = (model, uses_masks)
                logger.info(
                    "Loaded model: size=%d, checkpoint=%s, path=%s, masks=%s",
                    board_size, checkpoint_type, path, uses_masks,
                )
            except Exception as e:
                logger.error(
                    "Failed to load model: size=%d, checkpoint=%s, path=%s: %s",
                    board_size, checkpoint_type, path, e,
                )

    def get_model(
        self, board_size: int, skill_level: str, agent_type: str = "fog",
    ) -> Tuple[object, bool, bool]:
        """Get a loaded model for the given board size and skill level.

        Args:
            board_size: Grid dimension (NxN).
            skill_level: One of the SKILL_LEVEL_CONFIG keys.
            agent_type: Kept for API compatibility (ignored, all models are fog).

        Returns:
            Tuple of (model, uses_masks, deterministic).

        Raises:
            ValueError: If skill_level is unknown.
            KeyError: If no model is available for the given board_size and checkpoint.
        """
        if skill_level not in SKILL_LEVEL_CONFIG:
            raise ValueError(
                f"Unknown skill level: {skill_level}. "
                f"Available: {list(SKILL_LEVEL_CONFIG.keys())}"
            )

        checkpoint_type, deterministic = SKILL_LEVEL_CONFIG[skill_level]
        cache_key = (board_size, checkpoint_type)

        if cache_key not in self._model_cache:
            # Try to load on-demand if path is known
            if cache_key in self._model_paths:
                path = self._model_paths[cache_key]
                model, uses_masks = load_agent(path)
                self._model_cache[cache_key] = (model, uses_masks)
            else:
                raise KeyError(
                    f"No model available for board_size={board_size}, "
                    f"checkpoint={checkpoint_type}. "
                    f"Available: {list(self._model_paths.keys())}"
                )

        model, uses_masks = self._model_cache[cache_key]
        return model, uses_masks, deterministic

    def get_opponent_pools(self, board_size: int) -> List[str]:
        """Get opponent pool paths for a given board size."""
        return self._opponent_pools.get(board_size, [])

    def _build_indexed_models(self) -> None:
        """Build a flat indexed list of all discovered models.

        Scans difficulty/, scripted_checkpoints/, level_advancement/,
        and best_model.zip under each size directory.
        Each entry gets a stable `model_id` (8-char hex hash).
        """
        self._indexed_models = []
        self._model_id_to_index = {}
        seen_paths: set = set()

        def _add_model(board_size: int, category: str, label: str, path: str) -> None:
            model_id = generate_model_id(board_size, category, label)
            idx = len(self._indexed_models)
            self._indexed_models.append({
                "index": idx,
                "model_id": model_id,
                "board_size": board_size,
                "category": category,
                "label": label,
                "path": path,
                "use_fog": True,
            })
            self._model_id_to_index[model_id] = idx

        base = Path(self.models_dir)
        subdirs = ["difficulty", "scripted_checkpoints", "level_advancement"]

        for size_dir in sorted(base.glob("size*")):
            if not size_dir.is_dir():
                continue
            match = re.match(r"size(\d+)", size_dir.name)
            if not match:
                continue
            board_size = int(match.group(1))

            # Add best_model.zip at the size level
            best = size_dir / "best_model.zip"
            if best.is_file():
                path_str = str(best)
                if path_str not in seen_paths:
                    seen_paths.add(path_str)
                    _add_model(board_size, "best", "best_model", path_str)

            # Scan each subdirectory
            for subdir_name in subdirs:
                subdir = size_dir / subdir_name
                if not subdir.is_dir():
                    continue
                for zip_file in sorted(subdir.glob("*.zip")):
                    path_str = str(zip_file)
                    if path_str in seen_paths:
                        continue
                    seen_paths.add(path_str)
                    _add_model(board_size, subdir_name, zip_file.stem, path_str)

        logger.info("Indexed %d total models", len(self._indexed_models))

    def get_model_by_index(self, index: int) -> Tuple[object, bool, bool]:
        """Get a model by its flat index.

        Returns:
            Tuple of (model, uses_masks, use_fog).
        """
        if index < 0 or index >= len(self._indexed_models):
            raise IndexError(
                f"Model index {index} out of range. "
                f"Available: 0-{len(self._indexed_models) - 1}"
            )

        if index not in self._indexed_model_cache:
            entry = self._indexed_models[index]
            model, uses_masks = load_agent(entry["path"])
            self._indexed_model_cache[index] = (model, uses_masks)

        model, uses_masks = self._indexed_model_cache[index]
        use_fog = self._indexed_models[index]["use_fog"]
        return model, uses_masks, use_fog

    def get_model_by_id(self, model_id: str) -> Tuple[object, bool, bool]:
        """Get a model by its stable model ID.

        Returns:
            Tuple of (model, uses_masks, use_fog).
        """
        if model_id not in self._model_id_to_index:
            raise KeyError(
                f"Unknown model_id: {model_id}. "
                f"Available IDs: {list(self._model_id_to_index.keys())}"
            )
        index = self._model_id_to_index[model_id]
        return self.get_model_by_index(index)

    def get_model_meta_by_id(self, model_id: str) -> dict:
        """Get model metadata by its stable model ID."""
        if model_id not in self._model_id_to_index:
            raise KeyError(
                f"Unknown model_id: {model_id}. "
                f"Available IDs: {list(self._model_id_to_index.keys())}"
            )
        index = self._model_id_to_index[model_id]
        return self._indexed_models[index]

    def get_indexed_models_info(self) -> List[dict]:
        """Return the flat indexed list of all discovered models."""
        return self._indexed_models

    def get_loaded_models_info(self) -> dict:
        """Get summary of all loaded models for the /info endpoint."""
        info: Dict[str, list] = {}
        for (board_size, checkpoint_type) in self._model_cache:
            key = f"size{board_size}"
            if key not in info:
                info[key] = []
            path = self._model_paths.get((board_size, checkpoint_type), "unknown")
            info[key].append({
                "checkpoint": checkpoint_type,
                "path": path,
            })
        return info

    def get_available_agent_types(self, board_size: int) -> List[str]:
        """Return list of agent types available for a given board size.

        All models are fog-of-war now.
        """
        if board_size in self._board_sizes:
            return ["fog"]
        return []

    @property
    def supported_board_sizes(self) -> List[int]:
        """Return list of board sizes with available models."""
        return sorted(self._board_sizes)
