"""
Configuration for the MCP server.

All settings are configurable via environment variables.
"""

import os
from pathlib import Path

# Knowledge base path — defaults to GAME_KNOWLEDGE.md at the repo root
# (two levels up from this file: mcp_server/ -> spaces-game-python/ -> spaces-game-node/)
_default_knowledge_path = str(
    Path(__file__).resolve().parent.parent.parent / "GAME_KNOWLEDGE.md"
)

KNOWLEDGE_BASE_PATH = os.environ.get("MCP_KNOWLEDGE_PATH", _default_knowledge_path)
