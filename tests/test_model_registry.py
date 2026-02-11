"""
Unit tests for the model registry.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inference_server.model_registry import (
    ModelRegistry,
    SKILL_LEVEL_CONFIG,
    _extract_step_count,
)


# ---------------------------------------------------------------------------
# _extract_step_count tests
# ---------------------------------------------------------------------------

class TestExtractStepCount:

    def test_standard_format(self):
        assert _extract_step_count("ppo_100000_steps.zip") == 100000

    def test_with_prefix(self):
        assert _extract_step_count("stage3_500000_steps.zip") == 500000

    def test_no_match(self):
        assert _extract_step_count("best_model.zip") is None

    def test_final_model(self):
        assert _extract_step_count("ppo_stage3_final.zip") is None


# ---------------------------------------------------------------------------
# SKILL_LEVEL_CONFIG tests
# ---------------------------------------------------------------------------

class TestSkillLevelConfig:

    def test_all_levels_present(self):
        expected_levels = [
            "beginner", "beginner_plus",
            "intermediate", "intermediate_plus",
            "advanced", "advanced_plus",
        ]
        for level in expected_levels:
            assert level in SKILL_LEVEL_CONFIG

    def test_beginner_is_stochastic(self):
        checkpoint, deterministic = SKILL_LEVEL_CONFIG["beginner"]
        assert checkpoint == "early"
        assert deterministic is False

    def test_beginner_plus_is_deterministic(self):
        checkpoint, deterministic = SKILL_LEVEL_CONFIG["beginner_plus"]
        assert checkpoint == "early"
        assert deterministic is True

    def test_intermediate_is_stochastic(self):
        checkpoint, deterministic = SKILL_LEVEL_CONFIG["intermediate"]
        assert checkpoint == "mid"
        assert deterministic is False

    def test_advanced_plus_is_deterministic(self):
        checkpoint, deterministic = SKILL_LEVEL_CONFIG["advanced_plus"]
        assert checkpoint == "advanced"
        assert deterministic is True


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------

class TestModelRegistryDiscovery:

    def test_discover_single_model(self):
        """Test that a single model is used for all checkpoint types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create models/size2/stage3/model.zip
            stage3_dir = Path(tmpdir) / "size2" / "stage3"
            stage3_dir.mkdir(parents=True)
            (stage3_dir / "model.zip").write_bytes(b"fake")

            # Create boards/size2/simple.json
            boards_dir = Path(tmpdir) / "boards"
            size2_boards = boards_dir / "size2"
            size2_boards.mkdir(parents=True)
            (size2_boards / "simple.json").write_text("[]")

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(boards_dir),
            )
            registry.discover_models()

            assert 2 in registry.supported_board_sizes
            # Single model should be assigned to all three checkpoints
            assert (2, "early") in registry._model_paths
            assert (2, "mid") in registry._model_paths
            assert (2, "advanced") in registry._model_paths

    def test_discover_multiple_step_models(self):
        """Test assignment of early/mid/advanced from step checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = Path(tmpdir) / "size2" / "stage3"
            stage3_dir.mkdir(parents=True)

            (stage3_dir / "ppo_100000_steps.zip").write_bytes(b"fake")
            (stage3_dir / "ppo_500000_steps.zip").write_bytes(b"fake")
            (stage3_dir / "ppo_1000000_steps.zip").write_bytes(b"fake")

            boards_dir = Path(tmpdir) / "boards"
            size2_boards = boards_dir / "size2"
            size2_boards.mkdir(parents=True)
            (size2_boards / "simple.json").write_text("[]")

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(boards_dir),
            )
            registry.discover_models()

            assert 2 in registry.supported_board_sizes
            # Early should be lowest step
            assert "100000" in registry._model_paths[(2, "early")]
            # Advanced should be highest step
            assert "1000000" in registry._model_paths[(2, "advanced")]

    def test_discover_named_checkpoints(self):
        """Test that explicitly named files (early, mid, best) are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = Path(tmpdir) / "size3" / "stage3"
            stage3_dir.mkdir(parents=True)

            (stage3_dir / "model_early.zip").write_bytes(b"fake")
            (stage3_dir / "model_mid.zip").write_bytes(b"fake")
            (stage3_dir / "best_model.zip").write_bytes(b"fake")

            boards_dir = Path(tmpdir) / "boards"

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(boards_dir),
            )
            registry.discover_models()

            assert 3 in registry.supported_board_sizes
            assert "early" in registry._model_paths[(3, "early")]
            assert "mid" in registry._model_paths[(3, "mid")]
            assert "best" in registry._model_paths[(3, "advanced")]

    def test_discover_alternative_named_checkpoints(self):
        """Test that beginner/intermediate/expert names are recognized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = Path(tmpdir) / "size3" / "stage3"
            stage3_dir.mkdir(parents=True)

            (stage3_dir / "beginner.zip").write_bytes(b"fake")
            (stage3_dir / "intermediate.zip").write_bytes(b"fake")
            (stage3_dir / "expert.zip").write_bytes(b"fake")

            boards_dir = Path(tmpdir) / "boards"

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(boards_dir),
            )
            registry.discover_models()

            assert 3 in registry.supported_board_sizes
            assert "beginner" in registry._model_paths[(3, "early")]
            assert "intermediate" in registry._model_paths[(3, "mid")]
            assert "expert" in registry._model_paths[(3, "advanced")]

    def test_discover_no_models_dir(self):
        """Test graceful handling when models dir doesn't exist."""
        registry = ModelRegistry(
            models_dir="/nonexistent/models",
            boards_dir="/nonexistent/boards",
        )
        registry.discover_models()

        assert registry.supported_board_sizes == []

    def test_discover_empty_stage3_dir(self):
        """Test graceful handling when stage3 directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = Path(tmpdir) / "size2" / "stage3"
            stage3_dir.mkdir(parents=True)
            # No zip files

            registry = ModelRegistry(models_dir=str(Path(tmpdir)), boards_dir=str(tmpdir))
            registry.discover_models()

            assert registry.supported_board_sizes == []

    def test_opponent_pools_discovered(self):
        """Test that opponent pools are found during discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage3_dir = Path(tmpdir) / "models" / "size2" / "stage3"
            stage3_dir.mkdir(parents=True)
            (stage3_dir / "model.zip").write_bytes(b"fake")

            boards_dir = Path(tmpdir) / "boards"
            size2_boards = boards_dir / "size2"
            size2_boards.mkdir(parents=True)
            (size2_boards / "simple.json").write_text("[]")
            (size2_boards / "one_trap.json").write_text("[]")

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir) / "models"),
                boards_dir=str(boards_dir),
            )
            registry.discover_models()

            pools = registry.get_opponent_pools(2)
            assert len(pools) == 2


class TestModelRegistryGetModel:

    def _make_registry_with_mock_model(self):
        """Create a registry with a mock model already cached."""
        registry = ModelRegistry()
        mock_model = MagicMock()
        registry._model_cache[(2, "early")] = (mock_model, True)
        registry._model_cache[(2, "mid")] = (mock_model, True)
        registry._model_cache[(2, "advanced")] = (mock_model, True)
        registry._board_sizes = [2]
        return registry, mock_model

    def test_get_model_beginner(self):
        """Test getting a beginner model returns stochastic early checkpoint."""
        registry, mock_model = self._make_registry_with_mock_model()

        model, uses_masks, deterministic = registry.get_model(2, "beginner")
        assert model is mock_model
        assert deterministic is False

    def test_get_model_advanced_plus(self):
        """Test getting an advanced_plus model returns deterministic advanced."""
        registry, mock_model = self._make_registry_with_mock_model()

        model, uses_masks, deterministic = registry.get_model(2, "advanced_plus")
        assert model is mock_model
        assert deterministic is True

    def test_get_model_invalid_skill(self):
        """Test that invalid skill level raises ValueError."""
        registry, _ = self._make_registry_with_mock_model()

        with pytest.raises(ValueError, match="Unknown skill level"):
            registry.get_model(2, "grandmaster")

    def test_get_model_missing_board_size(self):
        """Test that missing board size raises KeyError."""
        registry, _ = self._make_registry_with_mock_model()

        with pytest.raises(KeyError, match="No model available"):
            registry.get_model(5, "beginner")

    def test_get_loaded_models_info(self):
        """Test the info endpoint data format."""
        registry, _ = self._make_registry_with_mock_model()
        registry._model_paths[(2, "early")] = "models/size2/stage3/early.zip"
        registry._model_paths[(2, "mid")] = "models/size2/stage3/mid.zip"
        registry._model_paths[(2, "advanced")] = "models/size2/stage3/best.zip"

        info = registry.get_loaded_models_info()
        assert "size2" in info
        checkpoints = [entry["checkpoint"] for entry in info["size2"]]
        assert "early" in checkpoints
        assert "mid" in checkpoints
        assert "advanced" in checkpoints
