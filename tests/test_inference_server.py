"""
Tests for the FastAPI inference server endpoints.

Uses TestClient with mocked model registry to avoid requiring trained models.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from spaces_game.types import Board, BoardMove, Position


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_valid_board(board_size: int = 2) -> Board:
    """Create a simple valid board for testing."""
    # Simple 2x2 board: piece at (1,0), piece at (0,0), final at (-1,0)
    sequence = (
        BoardMove(Position(row=1, col=0), "piece", 1),
        BoardMove(Position(row=0, col=0), "piece", 2),
        BoardMove(Position(row=-1, col=0), "final", 3),
    )
    grid = (
        ("piece", "empty"),
        ("piece", "empty"),
    )
    return Board(boardSize=board_size, grid=grid, sequence=sequence)


def _create_mock_registry(board_sizes=None, indexed_models=None):
    """Create a mock ModelRegistry.

    Args:
        board_sizes: List of supported board sizes.
        indexed_models: Optional list of indexed model dicts for GET /models.
    """
    if board_sizes is None:
        board_sizes = [2]
    if indexed_models is None:
        indexed_models = [
            {"index": 0, "board_size": 2, "category": "difficulty", "label": "beginner", "path": "inference_server/models/size2/difficulty/beginner.zip", "use_fog": True},
            {"index": 1, "board_size": 2, "category": "difficulty", "label": "expert", "path": "inference_server/models/size2/difficulty/expert.zip", "use_fog": True},
        ]

    mock_registry = MagicMock()
    mock_registry.supported_board_sizes = board_sizes

    # Build loaded_models_info (all fog now)
    loaded_models = {}
    for s in board_sizes:
        key = f"size{s}"
        loaded_models[key] = [
            {"checkpoint": "advanced", "path": f"inference_server/models/size{s}/difficulty/expert.zip"}
        ]
    mock_registry.get_loaded_models_info.return_value = loaded_models
    mock_registry.get_indexed_models_info.return_value = indexed_models

    def _get_available_agent_types(board_size):
        return ["fog"] if board_size in board_sizes else []
    mock_registry.get_available_agent_types.side_effect = _get_available_agent_types

    # Mock model that looks like a Stage 3/4 model (has opponent_history)
    mock_model = MagicMock()
    mock_model.observation_space = MagicMock()
    mock_model.observation_space.spaces = {
        "building_board": MagicMock(shape=(2, 2, 2)),
        "opponent_history": MagicMock(),
    }

    mock_registry.get_model.return_value = (mock_model, True, True)
    mock_registry.get_model_by_index.return_value = (mock_model, True, False)
    mock_registry.get_opponent_pools.return_value = ["boards/size2/simple.json"]

    return mock_registry, mock_model


@pytest.fixture
def mock_registry():
    """Provide a mock registry and patch it into the main module."""
    registry, model = _create_mock_registry()
    return registry, model


@pytest.fixture
def client(mock_registry):
    """Create a TestClient with mocked registry."""
    registry, model = mock_registry

    from inference_server.main import app
    import inference_server.main as main_mod
    with TestClient(app, raise_server_exceptions=False) as c:
        # Patch AFTER entering TestClient so lifespan doesn't overwrite
        main_mod.registry = registry
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Info endpoint
# ---------------------------------------------------------------------------

class TestInfoEndpoint:

    def test_info_returns_model_info(self, client):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "loaded_models" in data
        assert "supported_board_sizes" in data
        assert 2 in data["supported_board_sizes"]

    def test_info_model_details(self, client):
        response = client.get("/info")
        data = response.json()
        assert "size2" in data["loaded_models"]
        models = data["loaded_models"]["size2"]
        assert len(models) >= 1
        assert models[0]["checkpoint"] == "advanced"


# ---------------------------------------------------------------------------
# Construct-board endpoint
# ---------------------------------------------------------------------------

class TestConstructBoardEndpoint:

    def test_construct_board_valid_request(self, mock_registry):
        """Test a valid board construction request with mocked inference."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)), \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "agent_score": 0,
                    "opponent_score": 0,
                    "opponent_history": [],
                    "skill_level": "advanced",
                })

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["board"] is not None
            assert data["board"]["boardSize"] == 2
            assert len(data["board"]["sequence"]) == 3
            assert data["board"]["grid"] is not None
            assert data["model_info"]["skill_level"] == "advanced"
            assert data["model_info"]["agent_type"] == "fog"

    def test_construct_board_invalid_skill_level(self, mock_registry):
        """Test with an invalid skill level returns 422."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "skill_level": "grandmaster",
            })

        # Pydantic validation error - 422
        assert response.status_code == 422

    def test_construct_board_unsupported_board_size(self, mock_registry):
        """Test with unsupported board size returns 400."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 5,
                "round_num": 0,
                "skill_level": "beginner",
            })

        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]

    def test_construct_board_no_models_loaded(self):
        """Test when no models are loaded returns 503."""
        registry, _ = _create_mock_registry(board_sizes=[])
        registry.supported_board_sizes = []

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "skill_level": "beginner",
            })

        assert response.status_code == 503
        assert "No models" in response.json()["detail"]

    def test_construct_board_with_opponent_history(self, mock_registry):
        """Test with opponent history data."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)) as mock_build, \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 1,
                    "agent_score": 3,
                    "opponent_score": 2,
                    "opponent_history": [
                        {
                            "sequence": [
                                {"row": 1, "col": 0, "type": "piece", "order": 1},
                                {"row": 0, "col": 0, "type": "piece", "order": 2},
                                {"row": -1, "col": 0, "type": "final", "order": 3},
                            ]
                        }
                    ],
                    "skill_level": "intermediate",
                })

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

            # Verify build_board_for_round was called with correct parameters
            call_args = mock_build.call_args
            assert call_args.kwargs["round_num"] == 1
            assert call_args.kwargs["agent_score"] == 3
            assert call_args.kwargs["opponent_score"] == 2

    def test_construct_board_response_format(self, mock_registry):
        """Test the response board format matches frontend expectations."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)), \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "skill_level": "advanced_plus",
                })

            data = response.json()
            board = data["board"]

            # Check sequence format
            for move in board["sequence"]:
                assert "position" in move
                assert "row" in move["position"]
                assert "col" in move["position"]
                assert "type" in move
                assert "order" in move

            # Check grid format
            assert len(board["grid"]) == 2
            assert len(board["grid"][0]) == 2

    def test_construct_board_default_uses_fog(self, mock_registry):
        """Test that omitting agent_type defaults to fog (all models are fog now)."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)) as mock_build, \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                })

            assert response.status_code == 200
            call_args = mock_build.call_args
            assert call_args.kwargs["use_fog"] is True
            data = response.json()
            assert data["model_info"]["agent_type"] == "fog"

    def test_construct_board_invalid_agent_type(self, mock_registry):
        """Test with invalid agent_type returns 422."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "agent_type": "invisible",
            })

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestTestFailSkillLevel:
    """Tests for the test_fail skill level that always returns invalid boards."""

    def test_returns_200_with_valid_false(self, mock_registry):
        """test_fail should return HTTP 200 with valid=False."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "skill_level": "test_fail",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["attempts_used"] == 3
        assert data["model_info"]["skill_level"] == "test_fail"

    def test_does_not_call_get_model(self, mock_registry):
        """test_fail should short-circuit without looking up a model."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "skill_level": "test_fail",
            })

        assert response.status_code == 200
        registry.get_model.assert_not_called()

    def test_works_with_no_models_loaded(self):
        """test_fail should work even when no models are loaded."""
        registry, _ = _create_mock_registry(board_sizes=[])
        registry.supported_board_sizes = []

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 4,
                "round_num": 0,
                "skill_level": "test_fail",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_response_board_has_correct_structure(self, mock_registry):
        """test_fail board should have sequence, boardSize, and grid."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 3,
                "round_num": 0,
                "skill_level": "test_fail",
            })

        data = response.json()
        board = data["board"]
        assert "sequence" in board
        assert "boardSize" in board
        assert "grid" in board
        assert board["boardSize"] == 3
        assert len(board["grid"]) == 3
        assert len(board["grid"][0]) == 3


class TestCORS:

    def test_cors_headers_present(self, client):
        """Test that CORS headers are set on responses."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_allows_post(self, client):
        """Test that CORS allows POST method."""
        response = client.options(
            "/construct-board",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert "POST" in response.headers.get("access-control-allow-methods", "")


# ---------------------------------------------------------------------------
# GET /models endpoint
# ---------------------------------------------------------------------------

class TestModelsEndpoint:

    def test_models_returns_indexed_list(self, mock_registry):
        """GET /models returns the flat indexed list."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.get("/models")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["index"] == 0
        assert data[0]["board_size"] == 2
        assert data[0]["label"] == "beginner"
        assert data[1]["index"] == 1

    def test_models_includes_fog_info(self):
        """GET /models includes use_fog field (all models are fog now)."""
        indexed = [
            {"index": 0, "board_size": 3, "category": "difficulty", "label": "expert", "path": "p", "use_fog": True},
            {"index": 1, "board_size": 3, "category": "level_advancement", "label": "level0_before_50k", "path": "p2", "use_fog": True},
        ]
        registry, _ = _create_mock_registry(board_sizes=[3], indexed_models=indexed)

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.get("/models")

        data = response.json()
        assert data[0]["use_fog"] is True
        assert data[1]["use_fog"] is True


# ---------------------------------------------------------------------------
# model_index in construct-board
# ---------------------------------------------------------------------------

class TestModelIndexSelection:

    def test_model_index_selects_model(self, mock_registry):
        """model_index bypasses skill_level and uses get_model_by_index."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)), \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "model_index": 0,
                })

            assert response.status_code == 200
            data = response.json()
            assert data["model_info"]["model_index"] == 0
            assert data["model_info"]["label"] == "beginner"
            # get_model_by_index should be called, not get_model
            registry.get_model_by_index.assert_called_once_with(0)
            registry.get_model.assert_not_called()

    def test_model_index_out_of_range_returns_404(self, mock_registry):
        """model_index out of range returns 404."""
        registry, _ = mock_registry
        registry.get_model_by_index.side_effect = IndexError("Model index 99 out of range. Available: 0-1")

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "model_index": 99,
            })

        assert response.status_code == 404
        assert "out of range" in response.json()["detail"]

    def test_model_index_board_size_mismatch_returns_400(self, mock_registry):
        """model_index with wrong board_size returns 400."""
        registry, model = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            # Model at index 0 is board_size=2, but request says board_size=3
            response = client.post("/construct-board", json={
                "board_size": 3,
                "round_num": 0,
                "model_index": 0,
            })

        assert response.status_code == 400
        assert "does not match" in response.json()["detail"]

    def test_without_model_index_uses_skill_level(self, mock_registry):
        """Request without model_index still uses skill_level flow."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)), \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "skill_level": "advanced",
                })

            assert response.status_code == 200
            data = response.json()
            assert "model_index" not in data["model_info"]
            registry.get_model.assert_called_once()
            registry.get_model_by_index.assert_not_called()

    def test_model_index_with_temperature(self, mock_registry):
        """model_index with temperature passes temperature to build."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)) as mock_build, \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "model_index": 0,
                    "temperature": 0.8,
                })

            assert response.status_code == 200
            call_args = mock_build.call_args
            assert call_args.kwargs["temperature"] == 0.8
            data = response.json()
            assert data["model_info"]["temperature"] == 0.8

    def test_model_index_fog_model_passes_use_fog(self, mock_registry):
        """model_index pointing to a fog model passes use_fog=True."""
        indexed = [
            {"index": 0, "board_size": 2, "category": "level_advancement", "label": "level0_before_50k", "path": "p", "use_fog": True},
        ]
        registry, model = _create_mock_registry(board_sizes=[2], indexed_models=indexed)
        registry.get_model_by_index.return_value = (model, True, True)  # use_fog=True
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)) as mock_build, \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "model_index": 0,
                })

            assert response.status_code == 200
            call_args = mock_build.call_args
            assert call_args.kwargs["use_fog"] is True


# ---------------------------------------------------------------------------
# Temperature field
# ---------------------------------------------------------------------------

class TestTemperature:

    def test_temperature_passed_with_skill_level(self, mock_registry):
        """Temperature works with skill_level flow too."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)) as mock_build, \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "skill_level": "advanced",
                    "temperature": 1.5,
                })

            assert response.status_code == 200
            call_args = mock_build.call_args
            assert call_args.kwargs["temperature"] == 1.5
            data = response.json()
            assert data["model_info"]["temperature"] == 1.5

    def test_no_temperature_passes_none(self, mock_registry):
        """Omitting temperature passes None (uses deterministic from skill_level)."""
        registry, model = mock_registry
        valid_board = _make_valid_board(2)

        with patch("inference_server.main.is_stage3_model", return_value=True), \
             patch("inference_server.main.build_board_for_round", return_value=(valid_board, 1)) as mock_build, \
             patch("inference_server.main.get_model_board_size", return_value=2):

            from inference_server.main import app
            import inference_server.main as main_mod
            with TestClient(app, raise_server_exceptions=False) as client:
                main_mod.registry = registry
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                })

            assert response.status_code == 200
            call_args = mock_build.call_args
            assert call_args.kwargs["temperature"] is None
            data = response.json()
            assert "temperature" not in data["model_info"]

    def test_temperature_out_of_range_returns_422(self, mock_registry):
        """Temperature > 2.0 is rejected by validation."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "temperature": 3.0,
            })

        assert response.status_code == 422

    def test_temperature_negative_returns_422(self, mock_registry):
        """Negative temperature is rejected by validation."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "temperature": -0.5,
            })

        assert response.status_code == 422


class TestScriptedModelIdOverride:
    """Test that scripted agent IDs passed via model_id work correctly.

    This covers the case where an opponent (e.g. Pip/beginner) has a scripted
    agent assigned via modelAssignments, which sends model_id='scripted_1'
    with a non-scripted skill_level like 'beginner'.
    """

    def test_scripted_model_id_with_non_scripted_skill_level(self, mock_registry):
        """model_id='scripted_1' should route to scripted handler even when skill_level='beginner'."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "skill_level": "beginner",
                "model_id": "scripted_1",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["model_info"]["scripted"] is True
        assert data["model_info"]["skill_level"] == "scripted_1"

    def test_scripted_model_id_all_levels(self, mock_registry):
        """All scripted levels (1-5) work when passed as model_id."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            for level in range(1, 6):
                response = client.post("/construct-board", json={
                    "board_size": 2,
                    "round_num": 0,
                    "skill_level": "advanced",
                    "model_id": f"scripted_{level}",
                })
                assert response.status_code == 200
                data = response.json()
                assert data["valid"] is True
                assert data["model_info"]["scripted"] is True

    def test_scripted_skill_level_still_works(self, mock_registry):
        """When skill_level is already scripted, it should work without model_id."""
        registry, _ = mock_registry

        from inference_server.main import app
        import inference_server.main as main_mod
        with TestClient(app, raise_server_exceptions=False) as client:
            main_mod.registry = registry
            response = client.post("/construct-board", json={
                "board_size": 2,
                "round_num": 0,
                "skill_level": "scripted_1",
            })

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["model_info"]["scripted"] is True
