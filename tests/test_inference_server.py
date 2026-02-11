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


def _create_mock_registry(board_sizes=None):
    """Create a mock ModelRegistry."""
    if board_sizes is None:
        board_sizes = [2]

    mock_registry = MagicMock()
    mock_registry.supported_board_sizes = board_sizes
    mock_registry.get_loaded_models_info.return_value = {
        f"size{s}": [{"checkpoint": "advanced", "path": f"models/size{s}/stage3/model.zip"}]
        for s in board_sizes
    }

    # Mock model that looks like a Stage 3 model
    mock_model = MagicMock()
    mock_model.observation_space = MagicMock()
    mock_model.observation_space.spaces = {
        "building_board": MagicMock(shape=(2, 2, 2)),
        "opponent_history": MagicMock(),
    }

    mock_registry.get_model.return_value = (mock_model, True, True)
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
             patch("inference_server.main.build_board_for_round", return_value=valid_board), \
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
             patch("inference_server.main.build_board_for_round", return_value=valid_board) as mock_build, \
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
             patch("inference_server.main.build_board_for_round", return_value=valid_board), \
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


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

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
