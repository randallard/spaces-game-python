"""
FastAPI inference server for Spaces Game AI agent.

Provides endpoints to construct boards using trained RL models
at various skill levels.
"""

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from spaces_game.types import Board, BoardMove, Position

from . import config
from .scripted_agents import scripted_board
from .inference import (
    build_board_for_round,
    encode_board_for_agent,
    is_stage3_model,
    get_model_board_size,
)
from .model_registry import ModelRegistry, SKILL_LEVEL_CONFIG
from .models import (
    AgentType,
    ConstructBoardRequest,
    ConstructBoardResponse,
    HealthResponse,
    InfoResponse,
)

logger = logging.getLogger(__name__)

# Module-level registry, initialized during startup
registry: ModelRegistry = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global registry
    logger.info("Starting inference server...")
    logger.info("Models dir: %s", config.INFERENCE_MODELS_DIR)
    logger.info("Boards dir: %s", config.INFERENCE_BOARDS_DIR)

    registry = ModelRegistry(
        models_dir=config.INFERENCE_MODELS_DIR,
        boards_dir=config.INFERENCE_BOARDS_DIR,
    )
    registry.discover_models()
    registry.load_all()

    if registry.supported_board_sizes:
        logger.info(
            "Ready! Supported board sizes: %s",
            registry.supported_board_sizes,
        )
    else:
        logger.warning(
            "No models found in %s. Server will start but /construct-board "
            "will return errors until models are available.",
            config.INFERENCE_MODELS_DIR,
        )

    yield

    logger.info("Shutting down inference server.")


app = FastAPI(
    title="Spaces Game Inference Server",
    description="AI agent inference for the Spaces Game board construction",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.INFERENCE_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse)
async def info():
    """Get server information and loaded models."""
    return InfoResponse(
        loaded_models=registry.get_loaded_models_info(),
        supported_board_sizes=registry.supported_board_sizes,
    )


@app.get("/models")
async def list_models():
    """Return a flat indexed list of all available models."""
    return registry.get_indexed_models_info()


def _convert_opponent_history_to_grids(
    opponent_history: list,
    board_size: int,
) -> np.ndarray:
    """Convert frontend opponent_history format to numpy grids.

    Each entry in opponent_history is an OpponentBoardHistory with a sequence
    of moves. We encode each one into a (board_size, board_size, 2) grid
    using the same rotation logic as the training environment.

    Args:
        opponent_history: List of OpponentBoardHistory from the request.
        board_size: Grid dimension.

    Returns:
        numpy array of shape (5, board_size, board_size, 2).
    """
    grids = np.zeros((5, board_size, board_size, 2), dtype=np.int32)

    for round_idx, board_hist in enumerate(opponent_history):
        if round_idx >= 5:
            break
        if not board_hist.sequence:
            continue

        # Reconstruct a Board object so we can use encode_board_for_agent
        sequence = []
        str_grid = [
            ["empty" for _ in range(board_size)]
            for _ in range(board_size)
        ]

        for move_dict in board_hist.sequence:
            pos = Position(row=move_dict.row, col=move_dict.col)
            move = BoardMove(
                position=pos,
                type=move_dict.type,
                order=move_dict.order,
            )
            sequence.append(move)

            if move_dict.type in ("piece", "trap") and 0 <= move_dict.row < board_size and 0 <= move_dict.col < board_size:
                if move_dict.type == "piece":
                    if str_grid[move_dict.row][move_dict.col] != "trap":
                        str_grid[move_dict.row][move_dict.col] = "piece"
                elif move_dict.type == "trap":
                    str_grid[move_dict.row][move_dict.col] = "trap"

        # Add a final move if not present
        if not any(m.type == "final" for m in sequence):
            last_col = 0
            for m in reversed(sequence):
                if m.type != "final":
                    last_col = m.position.col
                    break
            sequence.append(BoardMove(
                position=Position(row=-1, col=last_col),
                type="final",
                order=len(sequence) + 1,
            ))

        board = Board(
            boardSize=board_size,
            grid=tuple(tuple(r) for r in str_grid),
            sequence=tuple(sequence),
        )
        grids[round_idx] = encode_board_for_agent(board, board_size)

    return grids


def _board_to_response_dict(board: Board) -> dict:
    """Convert a Board object to the response dict format.

    Returns:
        Dict with sequence, boardSize, and grid.
    """
    sequence = []
    for move in board.sequence:
        sequence.append({
            "position": {"row": move.position.row, "col": move.position.col},
            "type": move.type,
            "order": move.order,
        })

    # Convert grid tuples to lists for JSON serialization
    grid = [list(row) for row in board.grid]

    return {
        "sequence": sequence,
        "boardSize": board.boardSize,
        "grid": grid,
    }


def _build_test_fail_board(board_size: int) -> dict:
    """Build a structurally valid but unplayable board for test_fail skill level.

    Returns a board dict with a single piece in the bottom-left corner,
    no final move, and no complete path — guaranteed to fail validation.
    """
    grid = [["empty"] * board_size for _ in range(board_size)]
    grid[board_size - 1][0] = "piece"

    return {
        "sequence": [
            {"position": {"row": board_size - 1, "col": 0}, "type": "piece", "order": 1},
        ],
        "boardSize": board_size,
        "grid": grid,
    }


@app.post("/construct-board", response_model=ConstructBoardResponse)
async def construct_board(request: ConstructBoardRequest):
    """Construct a board using the AI agent at the specified skill level."""
    board_size = request.board_size
    skill_level = request.skill_level.value

    # test_fail: short-circuit with an invalid board (no model needed)
    if skill_level == "test_fail":
        return ConstructBoardResponse(
            board=_build_test_fail_board(board_size),
            valid=False,
            attempts_used=3,
            model_info={"skill_level": "test_fail"},
        )

    # Resolve scripted agent ID from either skill_level or model_id override
    scripted_id = None
    if skill_level.startswith("scripted_"):
        scripted_id = skill_level
    elif request.model_id is not None and request.model_id.startswith("scripted_"):
        scripted_id = request.model_id

    # Scripted agents: no model needed, deterministic board construction
    if scripted_id is not None:
        level = int(scripted_id.split("_")[1])
        board_dict = scripted_board(
            level, board_size, request.round_num, request.round_scores,
            round_history=request.round_history,
        )

        # Validate using the game engine
        from spaces_game.validation import is_board_playable

        seq = []
        for m in board_dict["sequence"]:
            seq.append(BoardMove(
                position=Position(row=m["position"]["row"], col=m["position"]["col"]),
                type=m["type"],
                order=m["order"],
            ))
        grid_tuples = tuple(tuple(r) for r in board_dict["grid"])
        board_obj = Board(
            boardSize=board_size,
            grid=grid_tuples,
            sequence=tuple(seq),
        )
        valid = is_board_playable(board_obj)

        return ConstructBoardResponse(
            board=board_dict,
            valid=valid,
            attempts_used=1,
            model_info={"skill_level": scripted_id, "scripted": True},
        )

    if request.model_id is not None:
        try:
            model, uses_masks, use_fog = registry.get_model_by_id(request.model_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        model_meta = registry.get_model_meta_by_id(request.model_id)
        agent_type = "fog" if use_fog else "standard"
        deterministic = True

        # Validate board_size matches the model
        if model_meta["board_size"] != board_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Board size {board_size} does not match model's board size "
                    f"{model_meta['board_size']} (model_id {request.model_id})."
                ),
            )
    elif request.model_index is not None:
        try:
            model, uses_masks, use_fog = registry.get_model_by_index(request.model_index)
        except IndexError as e:
            raise HTTPException(status_code=404, detail=str(e))

        model_meta = registry.get_indexed_models_info()[request.model_index]
        agent_type = "fog" if use_fog else "standard"
        deterministic = True

        # Validate board_size matches the model
        if model_meta["board_size"] != board_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Board size {board_size} does not match model's board size "
                    f"{model_meta['board_size']} (index {request.model_index})."
                ),
            )
    else:
        # Validate board size is supported
        if board_size not in registry.supported_board_sizes:
            if not registry.supported_board_sizes:
                raise HTTPException(
                    status_code=503,
                    detail="No models are loaded. Train models first and restart the server.",
                )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Board size {board_size} not supported. "
                    f"Available sizes: {registry.supported_board_sizes}"
                ),
            )

        # Get model for skill level and agent type
        agent_type = request.agent_type.value
        use_fog = agent_type == "fog"
        try:
            model, uses_masks, deterministic = registry.get_model(
                board_size, skill_level, agent_type=agent_type,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    # Verify it's a simultaneous play model (both stage3 and stage4 have opponent_history)
    if not is_stage3_model(model):
        raise HTTPException(
            status_code=500,
            detail="Loaded model is not a simultaneous play model.",
        )

    # Get opponent pools
    opponent_pools = registry.get_opponent_pools(board_size)
    if not opponent_pools:
        raise HTTPException(
            status_code=500,
            detail=f"No opponent pools found for board size {board_size}.",
        )

    # Convert opponent history to numpy grids
    try:
        opponent_history_grids = _convert_opponent_history_to_grids(
            request.opponent_history, board_size,
        )
    except Exception as e:
        logger.error("Failed to convert opponent history: %s", e)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid opponent_history format: {e}",
        )

    # Build the board
    try:
        board, attempts_used = build_board_for_round(
            model=model,
            uses_masks=uses_masks,
            board_size=board_size,
            round_num=request.round_num,
            agent_score=request.agent_score,
            opponent_score=request.opponent_score,
            opponent_history_grids=opponent_history_grids,
            opponent_pools=opponent_pools,
            deterministic=deterministic,
            use_fog=use_fog,
            temperature=request.temperature,
        )
    except ValueError as e:
        if "observation shape" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Board size mismatch: model was trained on a different size "
                    f"than {board_size}."
                ),
            )
        logger.error("Board construction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Board construction failed: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error during board construction: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Board construction failed unexpectedly: {e}",
        )

    # Validate and build response
    from spaces_game.validation import is_board_playable
    valid = is_board_playable(board)

    model_board_size = get_model_board_size(model)
    model_info = {
        "skill_level": skill_level,
        "agent_type": agent_type,
        "deterministic": deterministic,
        "uses_masks": uses_masks,
        "model_board_size": model_board_size,
    }
    if request.temperature is not None:
        model_info["temperature"] = request.temperature
    if request.model_id is not None:
        model_meta = registry.get_model_meta_by_id(request.model_id)
        model_info["model_id"] = request.model_id
        model_info["label"] = model_meta["label"]
    elif request.model_index is not None:
        model_meta = registry.get_indexed_models_info()[request.model_index]
        model_info["model_index"] = request.model_index
        model_info["label"] = model_meta["label"]

    return ConstructBoardResponse(
        board=_board_to_response_dict(board),
        valid=valid,
        attempts_used=attempts_used,
        model_info=model_info,
    )


def create_app() -> FastAPI:
    """Factory function for creating the app (useful for testing)."""
    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "inference_server.main:app",
        host=config.INFERENCE_HOST,
        port=config.INFERENCE_PORT,
        reload=False,
    )
