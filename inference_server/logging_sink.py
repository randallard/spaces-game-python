"""
Async logging sink for game events.

Posts construct-board events to Supabase for research data collection.
Errors are always caught and logged — the sink never interrupts inference.
"""

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LoggingSink:
    """Fire-and-forget async logger that posts events to a Supabase table."""

    def __init__(self, supabase_url: str, supabase_key: str, table: str = "game_events") -> None:
        self._url = f"{supabase_url.rstrip('/')}/rest/v1/{table}"
        self._headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }

    async def _post(self, event: dict[str, Any]) -> None:
        """POST a single event. Errors are caught and never re-raised."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self._url, json=event, headers=self._headers)
                if resp.status_code not in (200, 201):
                    logger.warning("Logging sink: unexpected status %s", resp.status_code)
        except Exception as exc:
            logger.warning("Logging sink: failed to post event: %s", exc)

    def log(self, event: dict[str, Any]) -> None:
        """Schedule an async POST as a fire-and-forget task.

        Must be called from within a running asyncio event loop (i.e. inside
        a FastAPI request handler). The event is posted in the background
        without blocking the response.
        """
        asyncio.create_task(self._post(event))
