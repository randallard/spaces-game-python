"""Tests for DiscordNotifierCallback."""

import json
import time
from unittest.mock import patch, MagicMock, PropertyMock
import pytest

from spaces_game.callbacks.discord_notifier import (
    DiscordNotifierCallback,
    _format_steps,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_callback(
    check_in_minutes=30,
    use_fog=False,
    self_play=False,
    phase_callback=None,
    self_play_callback=None,
):
    """Create a DiscordNotifierCallback with mocked webhook delivery."""
    cb = DiscordNotifierCallback(
        webhook_url="https://discord.com/api/webhooks/test/token",
        board_size=3,
        total_timesteps=1_000_000,
        n_envs=4,
        use_fog=use_fog,
        self_play=self_play,
        check_in_minutes=check_in_minutes,
        phase_callback=phase_callback,
        self_play_callback=self_play_callback,
        verbose=0,
    )
    # Mock the webhook sender so no HTTP calls happen
    cb._send_webhook = MagicMock()
    return cb


def _make_phase_callback(current_phase=0, max_phase=6, phase_history=None):
    """Create a mock OpponentProgressionCallback."""
    mock = MagicMock()
    mock.current_phase = current_phase
    mock.max_phase = max_phase
    mock.phase_history = phase_history or []
    return mock


def _make_self_play_callback(window_level=0, in_recovery=False, snapshot_paths=None, sp_win_rate=None):
    """Create a mock SelfPlayCurriculumCallback."""
    mock = MagicMock()
    mock.window_level = window_level
    mock._in_recovery = in_recovery
    mock.snapshot_paths = snapshot_paths or []
    mock._sp_win_rate = sp_win_rate
    return mock


# ── format_steps ─────────────────────────────────────────────────────

class TestFormatSteps:
    def test_millions(self):
        assert _format_steps(1_000_000) == "1M"
        assert _format_steps(1_500_000) == "1.5M"
        assert _format_steps(7_500_000) == "7.5M"

    def test_thousands(self):
        assert _format_steps(1_000) == "1K"
        assert _format_steps(250_000) == "250K"  # 250.0K rounds to 250K
        assert _format_steps(50_000) == "50K"

    def test_small(self):
        assert _format_steps(500) == "500"
        assert _format_steps(0) == "0"


# ── Training label ───────────────────────────────────────────────────

class TestTrainingLabel:
    def test_basic(self):
        cb = _make_callback()
        assert cb._training_label() == "Size 3"

    def test_fog(self):
        cb = _make_callback(use_fog=True)
        assert cb._training_label() == "Size 3 Fog"

    def test_self_play(self):
        cb = _make_callback(self_play=True)
        assert cb._training_label() == "Size 3 Self-Play"

    def test_fog_and_self_play(self):
        cb = _make_callback(use_fog=True, self_play=True)
        assert cb._training_label() == "Size 3 Fog+Self-Play"


# ── Edge detection: phase milestones ─────────────────────────────────

class TestPhaseMilestoneEdgeDetection:
    def test_fires_once_per_transition(self):
        """Phase change should trigger exactly one notification, not on every step."""
        phase_cb = _make_phase_callback(current_phase=0, max_phase=6)
        cb = _make_callback(phase_callback=phase_cb)
        cb._prev_phase = 0
        # Simulate adding num_timesteps attribute
        type(cb).num_timesteps = PropertyMock(return_value=100_000)

        # No change — no notification
        cb._check_phase_milestone()
        cb._send_webhook.assert_not_called()

        # Phase advances
        phase_cb.current_phase = 1
        phase_cb.phase_history = [{"game_win_rate": 0.75, "valid_rate": 0.98}]
        cb._check_phase_milestone()
        assert cb._send_webhook.call_count == 1
        embed = cb._send_webhook.call_args[0][0]
        assert "1" in embed["title"]

        # Same phase on next step — no additional notification
        cb._check_phase_milestone()
        assert cb._send_webhook.call_count == 1  # Still 1

    def test_all_phases_cleared(self):
        """Final phase should trigger 'All Phases Cleared' message."""
        phase_cb = _make_phase_callback(current_phase=5, max_phase=6)
        phase_cb.phase_history = [{"game_win_rate": 0.80, "valid_rate": 1.0}]
        cb = _make_callback(phase_callback=phase_cb)
        cb._prev_phase = 5
        type(cb).num_timesteps = PropertyMock(return_value=500_000)

        # Advance to final phase
        phase_cb.current_phase = 6
        cb._check_phase_milestone()
        embed = cb._send_webhook.call_args[0][0]
        assert "All Phases Cleared" in embed["title"]

    def test_no_notification_without_phase_callback(self):
        """Should silently do nothing if no phase_callback provided."""
        cb = _make_callback(phase_callback=None)
        cb._check_phase_milestone()
        cb._send_webhook.assert_not_called()


# ── Edge detection: self-play milestones ─────────────────────────────

class TestSelfPlayMilestoneEdgeDetection:
    def test_window_level_advance(self):
        sp_cb = _make_self_play_callback(window_level=0)
        cb = _make_callback(self_play=True, self_play_callback=sp_cb)
        cb._prev_window_level = 0
        cb._prev_in_recovery = False
        type(cb).num_timesteps = PropertyMock(return_value=200_000)

        # No change
        cb._check_self_play_milestones()
        cb._send_webhook.assert_not_called()

        # Level advances
        sp_cb.window_level = 1
        cb._check_self_play_milestones()
        assert cb._send_webhook.call_count == 1
        embed = cb._send_webhook.call_args[0][0]
        assert "Advanced" in embed["title"]

    def test_window_level_backtrack(self):
        sp_cb = _make_self_play_callback(window_level=2)
        cb = _make_callback(self_play=True, self_play_callback=sp_cb)
        cb._prev_window_level = 2
        cb._prev_in_recovery = False
        type(cb).num_timesteps = PropertyMock(return_value=300_000)

        sp_cb.window_level = 1
        cb._check_self_play_milestones()
        embed = cb._send_webhook.call_args[0][0]
        assert "Backtracked" in embed["title"]

    def test_recovery_entered(self):
        sp_cb = _make_self_play_callback(window_level=0, in_recovery=False)
        phase_cb = _make_phase_callback()
        phase_cb.phase_history = [{"game_win_rate": 0.48, "valid_rate": 1.0}]
        cb = _make_callback(self_play=True, self_play_callback=sp_cb, phase_callback=phase_cb)
        cb._prev_window_level = 0
        cb._prev_in_recovery = False
        type(cb).num_timesteps = PropertyMock(return_value=400_000)

        sp_cb._in_recovery = True
        cb._check_self_play_milestones()
        embed = cb._send_webhook.call_args[0][0]
        assert "Entered Pool Recovery" in embed["title"]
        assert cb.COLOR_DANGER == embed["color"]

    def test_recovery_exited(self):
        sp_cb = _make_self_play_callback(window_level=0, in_recovery=True)
        phase_cb = _make_phase_callback()
        phase_cb.phase_history = [{"game_win_rate": 0.72, "valid_rate": 1.0}]
        cb = _make_callback(self_play=True, self_play_callback=sp_cb, phase_callback=phase_cb)
        cb._prev_window_level = 0
        cb._prev_in_recovery = True
        type(cb).num_timesteps = PropertyMock(return_value=500_000)

        sp_cb._in_recovery = False
        cb._check_self_play_milestones()
        embed = cb._send_webhook.call_args[0][0]
        assert "Exited Pool Recovery" in embed["title"]
        assert cb.COLOR_SUCCESS == embed["color"]

    def test_no_notification_without_self_play_callback(self):
        cb = _make_callback(self_play_callback=None)
        cb._check_self_play_milestones()
        cb._send_webhook.assert_not_called()


# ── Commentary heuristics ────────────────────────────────────────────

class TestCommentaryHeuristics:
    def test_win_rate_climbing(self):
        phase_cb = _make_phase_callback(phase_history=[
            {"game_win_rate": 0.50, "valid_rate": 1.0},
            {"game_win_rate": 0.55, "valid_rate": 1.0},
            {"game_win_rate": 0.60, "valid_rate": 1.0},
            {"game_win_rate": 0.65, "valid_rate": 1.0},
            {"game_win_rate": 0.70, "valid_rate": 1.0},
        ])
        cb = _make_callback(phase_callback=phase_cb)
        commentary = cb._analyze_trends()
        assert "climbing" in commentary.lower() or "progressing" in commentary.lower()

    def test_win_rate_declining(self):
        phase_cb = _make_phase_callback(phase_history=[
            {"game_win_rate": 0.70, "valid_rate": 1.0},
            {"game_win_rate": 0.65, "valid_rate": 1.0},
            {"game_win_rate": 0.60, "valid_rate": 1.0},
            {"game_win_rate": 0.55, "valid_rate": 1.0},
            {"game_win_rate": 0.50, "valid_rate": 1.0},
        ])
        cb = _make_callback(phase_callback=phase_cb)
        commentary = cb._analyze_trends()
        assert "declining" in commentary.lower() or "collapse" in commentary.lower()

    def test_win_rate_plateau(self):
        history = [{"game_win_rate": 0.75, "valid_rate": 1.0} for _ in range(12)]
        phase_cb = _make_phase_callback(phase_history=history)
        cb = _make_callback(phase_callback=phase_cb)
        commentary = cb._analyze_trends()
        assert "plateau" in commentary.lower()

    def test_low_valid_rate_warning(self):
        phase_cb = _make_phase_callback(phase_history=[
            {"game_win_rate": 0.70, "valid_rate": 0.90},
            {"game_win_rate": 0.70, "valid_rate": 0.90},
            {"game_win_rate": 0.70, "valid_rate": 0.90},
        ])
        cb = _make_callback(phase_callback=phase_cb)
        commentary = cb._analyze_trends()
        assert "valid rate" in commentary.lower()

    def test_self_play_volatility(self):
        phase_cb = _make_phase_callback(phase_history=[
            {"game_win_rate": 0.50, "valid_rate": 1.0},
            {"game_win_rate": 0.80, "valid_rate": 1.0},
            {"game_win_rate": 0.55, "valid_rate": 1.0},
            {"game_win_rate": 0.75, "valid_rate": 1.0},
            {"game_win_rate": 0.60, "valid_rate": 1.0},
        ])
        sp_cb = _make_self_play_callback()
        cb = _make_callback(phase_callback=phase_cb, self_play_callback=sp_cb)
        commentary = cb._analyze_trends()
        assert "volatile" in commentary.lower()

    def test_no_data(self):
        phase_cb = _make_phase_callback(phase_history=[])
        cb = _make_callback(phase_callback=phase_cb)
        commentary = cb._analyze_trends()
        assert "no evaluation data" in commentary.lower()

    def test_recovery_commentary(self):
        phase_cb = _make_phase_callback(phase_history=[
            {"game_win_rate": 0.50, "valid_rate": 1.0},
        ])
        sp_cb = _make_self_play_callback(in_recovery=True)
        cb = _make_callback(phase_callback=phase_cb, self_play_callback=sp_cb)
        commentary = cb._analyze_trends()
        assert "recovery" in commentary.lower()


# ── Check-in timing ─────────────────────────────────────────────────

class TestCheckInTiming:
    def test_check_in_fires_after_interval(self):
        phase_cb = _make_phase_callback(phase_history=[
            {"game_win_rate": 0.70, "valid_rate": 1.0},
        ])
        cb = _make_callback(check_in_minutes=1, phase_callback=phase_cb)
        cb._last_check_in_time = time.time() - 120  # 2 minutes ago
        cb._training_start_time = time.time() - 300
        type(cb).num_timesteps = PropertyMock(return_value=50_000)

        cb._on_step()
        # Should have sent a check-in
        assert cb._send_webhook.call_count >= 1
        # Verify it was a check-in embed
        embeds = [call[0][0] for call in cb._send_webhook.call_args_list]
        assert any("Check-In" in e["title"] for e in embeds)

    def test_no_check_in_before_interval(self):
        cb = _make_callback(check_in_minutes=30)
        cb._last_check_in_time = time.time()  # Just now
        cb._training_start_time = time.time()

        cb._on_step()
        cb._send_webhook.assert_not_called()


# ── Embed formatting ────────────────────────────────────────────────

class TestEmbedFormatting:
    def test_make_embed_basic(self):
        cb = _make_callback()
        embed = cb._make_embed(title="Test", description="Desc", color=0x123456)
        assert embed["title"] == "Test"
        assert embed["description"] == "Desc"
        assert embed["color"] == 0x123456
        assert "fields" not in embed

    def test_make_embed_with_fields(self):
        cb = _make_callback()
        fields = [{"name": "A", "value": "1", "inline": True}]
        embed = cb._make_embed(title="Test", fields=fields)
        assert embed["fields"] == fields

    def test_training_end_embed(self):
        phase_cb = _make_phase_callback(current_phase=6, max_phase=6)
        phase_cb.phase_history = [{"game_win_rate": 0.78, "valid_rate": 1.0}]
        sp_cb = _make_self_play_callback(window_level=3, snapshot_paths=["a", "b", "c"])
        cb = _make_callback(
            use_fog=True,
            self_play=True,
            phase_callback=phase_cb,
            self_play_callback=sp_cb,
        )
        cb._training_start_time = time.time() - 7200  # 2 hours ago
        type(cb).num_timesteps = PropertyMock(return_value=1_000_000)

        cb._on_training_end()
        assert cb._send_webhook.call_count == 1
        embed = cb._send_webhook.call_args[0][0]
        assert "Training Complete" in embed["title"]
        assert "Fog+Self-Play" in embed["title"]
        field_names = [f["name"] for f in embed["fields"]]
        assert "Steps" in field_names
        assert "Pool WR" in field_names
        assert "Self-Play Level" in field_names


# ── Webhook delivery ────────────────────────────────────────────────

class TestWebhookDelivery:
    @patch("spaces_game.callbacks.discord_notifier.urllib.request.urlopen")
    def test_sends_json_payload(self, mock_urlopen):
        """Verify the webhook sends correct JSON payload."""
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        cb = DiscordNotifierCallback(
            webhook_url="https://discord.com/api/webhooks/test/token",
            board_size=3,
            total_timesteps=1_000_000,
            verbose=0,
        )

        embed = cb._make_embed(title="Test", description="Hello")
        cb._send_webhook(embed)

        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://discord.com/api/webhooks/test/token"
        assert req.get_header("Content-type") == "application/json"
        payload = json.loads(req.data.decode("utf-8"))
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["title"] == "Test"

    @patch("spaces_game.callbacks.discord_notifier.urllib.request.urlopen")
    def test_failure_does_not_raise(self, mock_urlopen):
        """Webhook failures should be swallowed, not crash training."""
        mock_urlopen.side_effect = Exception("Connection refused")

        cb = DiscordNotifierCallback(
            webhook_url="https://discord.com/api/webhooks/test/token",
            board_size=3,
            total_timesteps=1_000_000,
            verbose=0,
        )

        embed = cb._make_embed(title="Test")
        # Should not raise
        cb._send_webhook(embed)


# ── Commentary color ─────────────────────────────────────────────────

class TestCommentaryColor:
    def test_danger_for_collapse(self):
        cb = _make_callback()
        assert cb._commentary_color("Win rate declining — watch for policy collapse") == cb.COLOR_DANGER

    def test_warning_for_plateau(self):
        cb = _make_callback()
        assert cb._commentary_color("Win rate plateaued at ~75%") == cb.COLOR_WARNING

    def test_success_for_climbing(self):
        cb = _make_callback()
        assert cb._commentary_color("Win rate climbing steadily") == cb.COLOR_SUCCESS

    def test_info_for_default(self):
        cb = _make_callback()
        assert cb._commentary_color("Training in progress.") == cb.COLOR_INFO
