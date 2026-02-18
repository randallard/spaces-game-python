"""
Model registry for managing skill-level-to-model mapping.

Handles convention-based discovery of model checkpoints and provides
a cache of loaded models keyed by (board_size, stage, checkpoint_type).
"""

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

# Agent type -> stage directory name
AGENT_TYPE_TO_STAGE: Dict[str, str] = {
    "standard": "stage3",
    "fog": "stage4",
}


def _extract_step_count(filename: str) -> Optional[int]:
    """Extract step count from a checkpoint filename like 'ppo_100000_steps.zip'."""
    match = re.search(r"_(\d+)_steps\.zip$", filename)
    if match:
        return int(match.group(1))
    return None


class ModelRegistry:
    """Registry that loads and caches models, mapping skill levels to checkpoints.

    Convention-based discovery:
    - Models are expected in {models_dir}/size{N}/stage{3,4}/
    - "early" checkpoint: earliest step checkpoint or files matching *_early*
    - "mid" checkpoint: mid-range step checkpoint or files matching *_mid*
    - "advanced" checkpoint: latest/best/final model
    - Fallback: if only one model exists for a size, use it for all checkpoints
    """

    def __init__(self, models_dir: str = "models/", boards_dir: str = "boards/"):
        self.models_dir = models_dir
        self.boards_dir = boards_dir

        # Cache: (board_size, stage, checkpoint_type) -> (model, uses_masks)
        self._model_cache: Dict[Tuple[int, str, str], Tuple[object, bool]] = {}

        # Discovered model paths: (board_size, stage, checkpoint_type) -> path
        self._model_paths: Dict[Tuple[int, str, str], str] = {}

        # Discovered board sizes
        self._board_sizes: List[int] = []

        # Available stages per board size
        self._stages_by_size: Dict[int, List[str]] = {}

        # Opponent pools cache: board_size -> list of pool paths
        self._opponent_pools: Dict[int, List[str]] = {}

        # Flat indexed list of all models (skill-level + level_advancement)
        self._indexed_models: List[dict] = []

        # Cache for indexed models: index -> (model, uses_masks)
        self._indexed_model_cache: Dict[int, Tuple[object, bool]] = {}

    def discover_models(self) -> None:
        """Scan models_dir for available models and populate the path registry."""
        base = Path(self.models_dir)
        if not base.exists():
            logger.warning("Models directory does not exist: %s", self.models_dir)
            return

        # Look for size{N}/stage{3,4}/ directories
        for size_dir in sorted(base.glob("size*")):
            if not size_dir.is_dir():
                continue
            match = re.match(r"size(\d+)", size_dir.name)
            if not match:
                continue
            board_size = int(match.group(1))

            stages_found = []
            for stage in ["stage3", "stage4"]:
                stage_dir = size_dir / stage
                if not stage_dir.exists():
                    logger.debug("No %s directory for size %d", stage, board_size)
                    continue

                # Gather all zip files
                zip_files = sorted(stage_dir.glob("*.zip"))
                if not zip_files:
                    logger.debug("No model files in %s", stage_dir)
                    continue

                stages_found.append(stage)
                self._assign_checkpoints(board_size, stage, zip_files)

            if stages_found:
                if board_size not in self._board_sizes:
                    self._board_sizes.append(board_size)
                self._stages_by_size[board_size] = stages_found

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
            "Discovery complete: %d board sizes, %d model paths",
            len(self._board_sizes), len(self._model_paths),
        )

    def _assign_checkpoints(self, board_size: int, stage: str, zip_files: List[Path]) -> None:
        """Assign zip files to checkpoint types (early, mid, advanced)."""
        if len(zip_files) == 0:
            return

        # If only one model, use it for all checkpoint types
        if len(zip_files) == 1:
            path_str = str(zip_files[0])
            for checkpoint_type in ("early", "mid", "advanced"):
                self._model_paths[(board_size, stage, checkpoint_type)] = path_str
            logger.info(
                "Single model for size %d %s, using for all checkpoints: %s",
                board_size, stage, zip_files[0].name,
            )
            return

        # Look for explicitly named checkpoints first
        named_map: Dict[str, Optional[Path]] = {
            "early": None,
            "mid": None,
            "advanced": None,
        }

        for f in zip_files:
            name_lower = f.name.lower()
            if any(kw in name_lower for kw in ("early", "beginner")):
                named_map["early"] = f
            elif any(kw in name_lower for kw in ("mid", "intermediate")):
                named_map["mid"] = f
            elif any(kw in name_lower for kw in ("best", "final", "advanced", "expert")):
                named_map["advanced"] = f

        # Try step-count-based assignment for any remaining gaps
        step_files: List[Tuple[int, Path]] = []
        for f in zip_files:
            step = _extract_step_count(f.name)
            if step is not None:
                step_files.append((step, f))

        step_files.sort(key=lambda x: x[0])

        if step_files:
            if named_map["early"] is None:
                named_map["early"] = step_files[0][1]
            if named_map["advanced"] is None:
                named_map["advanced"] = step_files[-1][1]
            if named_map["mid"] is None:
                mid_idx = len(step_files) // 2
                named_map["mid"] = step_files[mid_idx][1]

        # Also look for best_model.zip in a 'best' subdirectory
        for f in zip_files:
            if f.parent.name == "best" or "best" in f.name.lower():
                if named_map["advanced"] is None:
                    named_map["advanced"] = f
                break

        # Final fallback: use last file for any remaining gaps
        fallback = zip_files[-1]
        for checkpoint_type in ("early", "mid", "advanced"):
            if named_map[checkpoint_type] is None:
                named_map[checkpoint_type] = fallback

        for checkpoint_type, path in named_map.items():
            if path is not None:
                self._model_paths[(board_size, stage, checkpoint_type)] = str(path)
                logger.info(
                    "Size %d %s, %s checkpoint: %s",
                    board_size, stage, checkpoint_type, path.name,
                )

    def load_all(self) -> None:
        """Load all discovered models into cache."""
        for (board_size, stage, checkpoint_type), path in self._model_paths.items():
            if (board_size, stage, checkpoint_type) in self._model_cache:
                continue
            try:
                model, uses_masks = load_agent(path)
                self._model_cache[(board_size, stage, checkpoint_type)] = (model, uses_masks)
                logger.info(
                    "Loaded model: size=%d, stage=%s, checkpoint=%s, path=%s, masks=%s",
                    board_size, stage, checkpoint_type, path, uses_masks,
                )
            except Exception as e:
                logger.error(
                    "Failed to load model: size=%d, stage=%s, checkpoint=%s, path=%s: %s",
                    board_size, stage, checkpoint_type, path, e,
                )

    def get_model(
        self, board_size: int, skill_level: str, agent_type: str = "standard",
    ) -> Tuple[object, bool, bool]:
        """Get a loaded model for the given board size, skill level, and agent type.

        Args:
            board_size: Grid dimension (NxN).
            skill_level: One of the SKILL_LEVEL_CONFIG keys.
            agent_type: "standard" or "fog".

        Returns:
            Tuple of (model, uses_masks, deterministic).

        Raises:
            ValueError: If skill_level or agent_type is unknown.
            KeyError: If no model is available for the given board_size, stage, and checkpoint.
        """
        if skill_level not in SKILL_LEVEL_CONFIG:
            raise ValueError(
                f"Unknown skill level: {skill_level}. "
                f"Available: {list(SKILL_LEVEL_CONFIG.keys())}"
            )

        if agent_type not in AGENT_TYPE_TO_STAGE:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available: {list(AGENT_TYPE_TO_STAGE.keys())}"
            )

        stage = AGENT_TYPE_TO_STAGE[agent_type]
        checkpoint_type, deterministic = SKILL_LEVEL_CONFIG[skill_level]
        cache_key = (board_size, stage, checkpoint_type)

        if cache_key not in self._model_cache:
            # Try to load on-demand if path is known
            if cache_key in self._model_paths:
                path = self._model_paths[cache_key]
                model, uses_masks = load_agent(path)
                self._model_cache[cache_key] = (model, uses_masks)
            else:
                raise KeyError(
                    f"No model available for board_size={board_size}, "
                    f"agent_type={agent_type} (stage={stage}), "
                    f"checkpoint={checkpoint_type}. "
                    f"Available: {list(self._model_paths.keys())}"
                )

        model, uses_masks = self._model_cache[cache_key]
        return model, uses_masks, deterministic

    def get_opponent_pools(self, board_size: int) -> List[str]:
        """Get opponent pool paths for a given board size.

        Args:
            board_size: Grid dimension.

        Returns:
            List of JSON file paths for opponent pools.
        """
        return self._opponent_pools.get(board_size, [])

    def _build_indexed_models(self) -> None:
        """Build a flat indexed list of all discovered models.

        Includes skill-level checkpoints (early/mid/advanced) and
        level_advancement models from each stage directory.
        """
        self._indexed_models = []
        seen_paths: set = set()

        # First, add skill-level checkpoints (deduplicated by path)
        for (board_size, stage, checkpoint_type), path in sorted(self._model_paths.items()):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            use_fog = stage == "stage4"
            label = Path(path).stem  # e.g., "beginner", "expert", "ppo_100000_steps"
            self._indexed_models.append({
                "index": len(self._indexed_models),
                "board_size": board_size,
                "stage": stage,
                "label": label,
                "path": path,
                "use_fog": use_fog,
            })

        # Then, scan level_advancement/ subdirectories
        base = Path(self.models_dir)
        for size_dir in sorted(base.glob("size*")):
            if not size_dir.is_dir():
                continue
            match = re.match(r"size(\d+)", size_dir.name)
            if not match:
                continue
            board_size = int(match.group(1))

            for stage in ["stage3", "stage4"]:
                la_dir = size_dir / stage / "level_advancement"
                if not la_dir.is_dir():
                    continue

                for zip_file in sorted(la_dir.glob("*.zip")):
                    path_str = str(zip_file)
                    if path_str in seen_paths:
                        continue
                    seen_paths.add(path_str)
                    use_fog = stage == "stage4"
                    self._indexed_models.append({
                        "index": len(self._indexed_models),
                        "board_size": board_size,
                        "stage": stage,
                        "label": zip_file.stem,
                        "path": path_str,
                        "use_fog": use_fog,
                    })

        logger.info("Indexed %d total models", len(self._indexed_models))

    def get_model_by_index(self, index: int) -> Tuple[object, bool, bool]:
        """Get a model by its flat index.

        Args:
            index: The model index from the indexed list.

        Returns:
            Tuple of (model, uses_masks, use_fog).

        Raises:
            IndexError: If the index is out of range.
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

    def get_indexed_models_info(self) -> List[dict]:
        """Return the flat indexed list of all discovered models."""
        return self._indexed_models

    def get_loaded_models_info(self) -> dict:
        """Get summary of all loaded models for the /info endpoint.

        Returns:
            Dict mapping "size{N}" to dict of agent_type -> list of checkpoints.
        """
        info: Dict[str, Dict[str, list]] = {}
        for (board_size, stage, checkpoint_type) in self._model_cache:
            key = f"size{board_size}"
            # Map stage back to agent_type
            agent_type = "fog" if stage == "stage4" else "standard"
            if key not in info:
                info[key] = {}
            if agent_type not in info[key]:
                info[key][agent_type] = []
            path = self._model_paths.get((board_size, stage, checkpoint_type), "unknown")
            info[key][agent_type].append({
                "checkpoint": checkpoint_type,
                "path": path,
            })
        return info

    def get_available_agent_types(self, board_size: int) -> List[str]:
        """Return list of agent types available for a given board size."""
        stages = self._stages_by_size.get(board_size, [])
        agent_types = []
        if "stage3" in stages:
            agent_types.append("standard")
        if "stage4" in stages:
            agent_types.append("fog")
        return agent_types

    @property
    def supported_board_sizes(self) -> List[int]:
        """Return list of board sizes with available models."""
        return sorted(self._board_sizes)
