"""
Spaces Game MCP Server.

Exposes game knowledge, board validation, and round simulation
as MCP resources and tools for Claude Desktop.

Run with:
    python -m mcp_server.main
"""

import json
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from spaces_game.validation import validate_board
from spaces_game.simulation import simulate_round

from .config import KNOWLEDGE_BASE_PATH
from .board_helpers import board_from_json, round_result_to_dict

mcp = FastMCP("spaces-game")


# ── Resource ──────────────────────────────────────────────────────────


@mcp.resource("knowledge://game-rules")
def game_rules_resource() -> str:
    """Complete Spaces Game knowledge base — rules, strategy, and technical details."""
    path = Path(KNOWLEDGE_BASE_PATH)
    if not path.exists():
        return f"Knowledge base not found at {path}"
    return path.read_text(encoding="utf-8")


# ── Tools ─────────────────────────────────────────────────────────────


@mcp.tool()
def validate_board_tool(board: dict) -> str:
    """
    Validate a Spaces Game board.

    Args:
        board: Board dict with keys: boardSize (int), grid (list[list[str]]),
               sequence (list[dict] with position, type, order)

    Returns:
        JSON string with valid (bool), errors (list[str]), boardSize (int),
        sequenceLength (int)
    """
    try:
        parsed = board_from_json(board)
    except (KeyError, ValueError, TypeError) as e:
        return json.dumps({
            "valid": False,
            "errors": [f"Failed to parse board: {e}"],
        })

    result = validate_board(parsed)
    return json.dumps({
        "valid": result.valid,
        "errors": result.errors,
        "boardSize": parsed.boardSize,
        "sequenceLength": len(parsed.sequence),
    })


@mcp.tool()
def simulate_round_tool(
    round_num: int,
    player_board: dict,
    opponent_board: dict,
) -> str:
    """
    Simulate a single round between two boards.

    Args:
        round_num: Round number (1-5)
        player_board: Player's board dict (boardSize, grid, sequence)
        opponent_board: Opponent's board dict (boardSize, grid, sequence)

    Returns:
        JSON string with winner, points, collision, simulation details
    """
    try:
        parsed_player = board_from_json(player_board)
    except (KeyError, ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid player board: {e}"})

    try:
        parsed_opponent = board_from_json(opponent_board)
    except (KeyError, ValueError, TypeError) as e:
        return json.dumps({"error": f"Invalid opponent board: {e}"})

    try:
        result = simulate_round(
            round_num, parsed_player, parsed_opponent, silent=True
        )
    except Exception as e:
        return json.dumps({"error": f"Simulation failed: {e}"})

    return json.dumps(round_result_to_dict(result))


@mcp.tool()
def get_game_rules(topic: str) -> str:
    """
    Search the Spaces Game knowledge base by topic.

    Args:
        topic: Keyword or phrase to search for (e.g. "scoring", "traps",
               "fog of war", "supermove", "collision")

    Returns:
        Matching sections from the knowledge base, or a message if no match found
    """
    path = Path(KNOWLEDGE_BASE_PATH)
    if not path.exists():
        return json.dumps({"error": f"Knowledge base not found at {path}"})

    content = path.read_text(encoding="utf-8")

    # Split into sections by ## headings
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    # Search for topic (case-insensitive) in section heading and content
    pattern = re.compile(re.escape(topic), re.IGNORECASE)
    matches = [s.strip() for s in sections if pattern.search(s)]

    if not matches:
        # List available section headings to help the caller refine
        headings = [
            line.strip()
            for line in content.splitlines()
            if line.startswith("## ")
        ]
        return json.dumps({
            "found": False,
            "message": f"No sections matched '{topic}'.",
            "availableSections": headings,
        })

    return json.dumps({
        "found": True,
        "matchCount": len(matches),
        "sections": matches,
    })


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
