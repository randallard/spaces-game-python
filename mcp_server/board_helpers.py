"""
Board conversion helpers for the MCP server.

Wraps existing board_loader and types to convert between
JSON dicts and frozen dataclasses.
"""

from typing import Any

from spaces_game.board_loader import load_board_from_dict
from spaces_game.types import Board, RoundResult, Position


def board_from_json(board_dict: dict[str, Any]) -> Board:
    """
    Convert a JSON-style board dict to a frozen Board dataclass.

    Delegates entirely to the existing load_board_from_dict which handles
    all list→tuple conversions for frozen dataclasses.

    Args:
        board_dict: Board data with keys: boardSize, grid, sequence

    Returns:
        Immutable Board dataclass

    Raises:
        KeyError: If required fields are missing
        ValueError: If board size is invalid
    """
    return load_board_from_dict(board_dict)


def _position_to_dict(pos: Position) -> dict[str, int]:
    """Convert a Position to a JSON-safe dict."""
    return {"row": pos.row, "col": pos.col}


def round_result_to_dict(result: RoundResult) -> dict[str, Any]:
    """
    Serialize a RoundResult frozen dataclass to a JSON-safe dict.

    Includes the key fields an LLM would need to understand the round outcome.
    Omits the full board data (playerBoard/opponentBoard) to keep responses concise.

    Args:
        result: Simulation result from simulate_round()

    Returns:
        JSON-serializable dict
    """
    details = result.simulationDetails
    return {
        "round": result.round,
        "winner": result.winner,
        "playerPoints": result.playerPoints,
        "opponentPoints": result.opponentPoints,
        "collision": result.collision,
        "playerFinalPosition": _position_to_dict(result.playerFinalPosition),
        "opponentFinalPosition": _position_to_dict(result.opponentFinalPosition),
        "simulationDetails": {
            "playerMoves": details.playerMoves,
            "opponentMoves": details.opponentMoves,
            "playerHitTrap": details.playerHitTrap,
            "opponentHitTrap": details.opponentHitTrap,
            "playerLastStep": details.playerLastStep,
            "opponentLastStep": details.opponentLastStep,
            "playerTrapPosition": (
                _position_to_dict(details.playerTrapPosition)
                if details.playerTrapPosition
                else None
            ),
            "opponentTrapPosition": (
                _position_to_dict(details.opponentTrapPosition)
                if details.opponentTrapPosition
                else None
            ),
        },
    }
