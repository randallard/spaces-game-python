"""
Tests for LoggingSink.

Verifies URL construction, header format, successful posting,
error handling, and fire-and-forget task scheduling.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_server.logging_sink import LoggingSink


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sink():
    return LoggingSink("https://abc.supabase.co", "test-anon-key", "game_events")


# ---------------------------------------------------------------------------
# URL and header construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_url_includes_table(self, sink):
        assert sink._url == "https://abc.supabase.co/rest/v1/game_events"

    def test_url_strips_trailing_slash(self):
        s = LoggingSink("https://abc.supabase.co/", "key", "game_events")
        assert s._url == "https://abc.supabase.co/rest/v1/game_events"

    def test_default_table_is_game_events(self):
        s = LoggingSink("https://abc.supabase.co", "key")
        assert "game_events" in s._url

    def test_custom_table(self):
        s = LoggingSink("https://abc.supabase.co", "key", "research_events")
        assert s._url.endswith("/rest/v1/research_events")

    def test_apikey_header(self, sink):
        assert sink._headers["apikey"] == "test-anon-key"

    def test_authorization_header(self, sink):
        assert sink._headers["Authorization"] == "Bearer test-anon-key"

    def test_prefer_minimal(self, sink):
        assert sink._headers["Prefer"] == "return=minimal"

    def test_content_type_json(self, sink):
        assert sink._headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# _post: successful submission
# ---------------------------------------------------------------------------

def _make_mock_client(status_code: int = 201):
    mock_response = MagicMock()
    mock_response.status_code = status_code

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


class TestPost:

    def test_post_sends_to_correct_url(self, sink):
        mock_client, _ = _make_mock_client(201)
        with patch("httpx.AsyncClient", return_value=mock_client):
            asyncio.run(sink._post({"session_id": "abc"}))
        mock_client.post.assert_called_once()
        call_url = mock_client.post.call_args[0][0]
        assert call_url == sink._url

    def test_post_sends_event_as_json(self, sink):
        mock_client, _ = _make_mock_client(201)
        event = {"session_id": "abc", "board_size": 3, "round_num": 0}
        with patch("httpx.AsyncClient", return_value=mock_client):
            asyncio.run(sink._post(event))
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"] == event

    def test_post_sends_correct_headers(self, sink):
        mock_client, _ = _make_mock_client(201)
        with patch("httpx.AsyncClient", return_value=mock_client):
            asyncio.run(sink._post({"session_id": "abc"}))
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["headers"] == sink._headers

    def test_post_accepts_200_status(self, sink):
        mock_client, _ = _make_mock_client(200)
        with patch("httpx.AsyncClient", return_value=mock_client):
            # Should not raise or warn
            asyncio.run(sink._post({"session_id": "abc"}))

    def test_post_logs_warning_on_unexpected_status(self, sink, caplog):
        mock_client, _ = _make_mock_client(500)
        with patch("httpx.AsyncClient", return_value=mock_client):
            with caplog.at_level(logging.WARNING, logger="inference_server.logging_sink"):
                asyncio.run(sink._post({"session_id": "abc"}))
        assert "500" in caplog.text

    def test_post_swallows_network_error(self, sink):
        mock_client, _ = _make_mock_client()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            # Should not raise
            asyncio.run(sink._post({"session_id": "abc"}))

    def test_post_logs_warning_on_network_error(self, sink, caplog):
        mock_client, _ = _make_mock_client()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        with patch("httpx.AsyncClient", return_value=mock_client):
            with caplog.at_level(logging.WARNING, logger="inference_server.logging_sink"):
                asyncio.run(sink._post({"session_id": "abc"}))
        assert "Connection refused" in caplog.text

    def test_post_uses_5_second_timeout(self, sink):
        mock_client, _ = _make_mock_client(201)
        with patch("httpx.AsyncClient") as MockClient:
            MockClient.return_value = mock_client
            asyncio.run(sink._post({"session_id": "abc"}))
        MockClient.assert_called_once_with(timeout=5.0)


# ---------------------------------------------------------------------------
# log: fire-and-forget scheduling
# ---------------------------------------------------------------------------

class TestLog:

    def test_log_schedules_post(self, sink):
        """log() fires the event without blocking."""
        posted = []

        async def fake_post(event):
            posted.append(event)

        sink._post = fake_post

        async def run():
            sink.log({"session_id": "x", "board_size": 2})
            await asyncio.sleep(0)  # yield to let the task execute

        asyncio.run(run())
        assert posted == [{"session_id": "x", "board_size": 2}]

    def test_log_does_not_block(self, sink):
        """log() returns synchronously even if the post is slow."""
        slow_started = []
        slow_finished = []

        async def slow_post(event):
            slow_started.append(event)
            await asyncio.sleep(0.1)
            slow_finished.append(event)

        sink._post = slow_post

        async def run():
            sink.log({"session_id": "x"})
            # log() returned immediately; slow_post hasn't finished
            assert slow_finished == []
            await asyncio.sleep(0.2)
            assert slow_finished == [{"session_id": "x"}]

        asyncio.run(run())
