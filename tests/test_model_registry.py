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
    generate_model_id,
)


# ---------------------------------------------------------------------------
# generate_model_id tests
# ---------------------------------------------------------------------------

class TestGenerateModelId:

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        id1 = generate_model_id(3, "difficulty", "my_model")
        id2 = generate_model_id(3, "difficulty", "my_model")
        assert id1 == id2

    def test_length(self):
        """Model ID should be exactly 8 hex characters."""
        model_id = generate_model_id(3, "difficulty", "my_model")
        assert len(model_id) == 8
        assert all(c in "0123456789abcdef" for c in model_id)

    def test_uniqueness_different_sizes(self):
        """Different board sizes produce different IDs."""
        id1 = generate_model_id(2, "difficulty", "my_model")
        id2 = generate_model_id(3, "difficulty", "my_model")
        assert id1 != id2

    def test_uniqueness_different_categories(self):
        """Different categories produce different IDs."""
        id1 = generate_model_id(3, "difficulty", "my_model")
        id2 = generate_model_id(3, "scripted_checkpoints", "my_model")
        assert id1 != id2

    def test_uniqueness_different_labels(self):
        """Different labels produce different IDs."""
        id1 = generate_model_id(3, "difficulty", "model_a")
        id2 = generate_model_id(3, "difficulty", "model_b")
        assert id1 != id2


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

    def test_discover_difficulty_checkpoints(self):
        """Test that difficulty files map to early/mid/advanced checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diff_dir = Path(tmpdir) / "size3" / "difficulty"
            diff_dir.mkdir(parents=True)
            (diff_dir / "beginner.zip").write_bytes(b"fake")
            (diff_dir / "intermediate.zip").write_bytes(b"fake")
            (diff_dir / "expert.zip").write_bytes(b"fake")

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

    def test_discover_empty_size_dir(self):
        """Test graceful handling when size directory has no subdirs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size_dir = Path(tmpdir) / "size2"
            size_dir.mkdir(parents=True)

            registry = ModelRegistry(models_dir=str(Path(tmpdir)), boards_dir=str(tmpdir))
            registry.discover_models()

            assert registry.supported_board_sizes == []

    def test_opponent_pools_discovered(self):
        """Test that opponent pools are found during discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diff_dir = Path(tmpdir) / "models" / "size2" / "difficulty"
            diff_dir.mkdir(parents=True)
            (diff_dir / "beginner.zip").write_bytes(b"fake")

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

    def test_discover_all_subdirs(self):
        """Test that difficulty, scripted_checkpoints, level_advancement, and best_model are indexed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            size_dir = Path(tmpdir) / "size4"

            diff_dir = size_dir / "difficulty"
            diff_dir.mkdir(parents=True)
            (diff_dir / "beginner.zip").write_bytes(b"fake")
            (diff_dir / "expert.zip").write_bytes(b"fake")

            sc_dir = size_dir / "scripted_checkpoints"
            sc_dir.mkdir(parents=True)
            (sc_dir / "cleared_level1.zip").write_bytes(b"fake")
            (sc_dir / "cleared_level3.zip").write_bytes(b"fake")

            la_dir = size_dir / "level_advancement"
            la_dir.mkdir(parents=True)
            (la_dir / "level0_first.zip").write_bytes(b"fake")
            (la_dir / "level5_mid.zip").write_bytes(b"fake")
            (la_dir / "level10_last.zip").write_bytes(b"fake")

            (size_dir / "best_model.zip").write_bytes(b"fake")

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(tmpdir),
            )
            registry.discover_models()

            assert 4 in registry.supported_board_sizes
            models = registry.get_indexed_models_info()
            # 1 best + 2 difficulty + 2 scripted + 3 level_advancement = 8
            assert len(models) == 8

            categories = {m["category"] for m in models}
            assert categories == {"best", "difficulty", "scripted_checkpoints", "level_advancement"}

    def test_all_models_use_fog(self):
        """All indexed models should have use_fog=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diff_dir = Path(tmpdir) / "size2" / "difficulty"
            diff_dir.mkdir(parents=True)
            (diff_dir / "beginner.zip").write_bytes(b"fake")

            registry = ModelRegistry(models_dir=str(Path(tmpdir)), boards_dir=str(tmpdir))
            registry.discover_models()

            for m in registry.get_indexed_models_info():
                assert m["use_fog"] is True


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

    def test_get_model_by_id_happy_path(self):
        """Test looking up a model by its stable ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diff_dir = Path(tmpdir) / "size2" / "difficulty"
            diff_dir.mkdir(parents=True)
            (diff_dir / "beginner.zip").write_bytes(b"fake")

            boards_dir = Path(tmpdir) / "boards"
            size2_boards = boards_dir / "size2"
            size2_boards.mkdir(parents=True)
            (size2_boards / "simple.json").write_text("[]")

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(boards_dir),
            )
            registry.discover_models()

            models = registry.get_indexed_models_info()
            assert len(models) > 0
            model_id = models[0]["model_id"]
            assert len(model_id) == 8

            meta = registry.get_model_meta_by_id(model_id)
            assert meta["board_size"] == 2
            assert meta["model_id"] == model_id

    def test_get_model_by_id_unknown(self):
        """Test that unknown model_id raises KeyError."""
        registry = ModelRegistry()
        registry._build_indexed_models()

        with pytest.raises(KeyError, match="Unknown model_id"):
            registry.get_model_by_id("deadbeef")

    def test_get_model_meta_by_id_unknown(self):
        """Test that unknown model_id raises KeyError for meta."""
        registry = ModelRegistry()
        registry._build_indexed_models()

        with pytest.raises(KeyError, match="Unknown model_id"):
            registry.get_model_meta_by_id("deadbeef")

    def test_indexed_models_have_model_id(self):
        """Test that all indexed models include a model_id field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diff_dir = Path(tmpdir) / "size3" / "difficulty"
            diff_dir.mkdir(parents=True)
            (diff_dir / "beginner.zip").write_bytes(b"fake")
            (diff_dir / "intermediate.zip").write_bytes(b"fake")
            (diff_dir / "expert.zip").write_bytes(b"fake")

            registry = ModelRegistry(
                models_dir=str(Path(tmpdir)),
                boards_dir=str(tmpdir),
            )
            registry.discover_models()

            models = registry.get_indexed_models_info()
            assert len(models) == 3
            ids = set()
            for m in models:
                assert "model_id" in m
                assert len(m["model_id"]) == 8
                ids.add(m["model_id"])
            assert len(ids) == 3

    def test_get_loaded_models_info(self):
        """Test the info endpoint data format."""
        registry, _ = self._make_registry_with_mock_model()
        registry._model_paths[(2, "early")] = "inference_server/models/size2/difficulty/beginner.zip"
        registry._model_paths[(2, "mid")] = "inference_server/models/size2/difficulty/intermediate.zip"
        registry._model_paths[(2, "advanced")] = "inference_server/models/size2/difficulty/expert.zip"

        info = registry.get_loaded_models_info()
        assert "size2" in info
        checkpoints = [entry["checkpoint"] for entry in info["size2"]]
        assert "early" in checkpoints
        assert "mid" in checkpoints
        assert "advanced" in checkpoints

    def test_get_available_agent_types(self):
        """All sizes return fog only."""
        registry, _ = self._make_registry_with_mock_model()
        assert registry.get_available_agent_types(2) == ["fog"]
        assert registry.get_available_agent_types(99) == []
