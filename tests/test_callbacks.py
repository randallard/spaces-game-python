"""Tests for training callbacks."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from spaces_game.callbacks.self_play import SelfPlayCurriculumCallback


class TestSelfPlayCurriculumCallback:
    """Tests for the progressive window self-play curriculum."""

    def _make_callback(self, **kwargs):
        """Create a callback with mock dependencies."""
        defaults = dict(
            output_dir=tempfile.mkdtemp(),
            snapshot_freq=100,
            pool_size=5,
            warmup_steps=0,
            n_envs=1,
            phase_callback=None,
            advance_threshold=0.70,
            backtrack_threshold=0.55,
            min_steps_per_level=10,
            recovery_win_rate=0.70,
            verbose=0,
        )
        defaults.update(kwargs)
        return SelfPlayCurriculumCallback(**defaults)

    def _setup_callback(self, cb):
        """Wire up mock model/env/logger so _on_step works.

        BaseCallback.training_env is a property -> model.get_env()
        BaseCallback.logger is a property -> model.logger
        """
        mock_env = MagicMock()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_env
        cb.model = mock_model
        cb.num_timesteps = 0
        cb.n_calls = 0

    def test_default_snapshot_win_rate(self):
        cb = self._make_callback(advance_threshold=0.70, backtrack_threshold=0.55)
        assert cb.snapshot_win_rate == pytest.approx(0.625)

    def test_custom_snapshot_win_rate(self):
        cb = self._make_callback(snapshot_win_rate=0.80)
        assert cb.snapshot_win_rate == 0.80

    def test_starts_at_level_zero(self):
        cb = self._make_callback()
        assert cb.window_level == 0
        assert cb.max_level == 0
        assert not cb._in_recovery

    def test_warmup_skips_self_play(self):
        cb = self._make_callback(warmup_steps=1000, n_envs=4)
        self._setup_callback(cb)
        cb.n_calls = 10  # 10 * 4 = 40 < 1000
        result = cb._on_step()
        assert result is True
        assert not cb._warmup_complete

    def test_warmup_completes(self):
        cb = self._make_callback(warmup_steps=40, n_envs=4)
        self._setup_callback(cb)
        cb.n_calls = 10  # 10 * 4 = 40 >= 40
        cb._on_step()
        assert cb._warmup_complete

    def test_advance_requires_min_steps(self):
        """Should not advance even with high win rate if min steps not met."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.80}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            min_steps_per_level=100,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = False
        cb.snapshot_paths = ["s1.zip", "s2.zip"]
        cb._level_start_step = 0
        cb.n_calls = 10  # 10 * 1 = 10 < 100
        cb._check_level_transition()
        assert cb.window_level == 0

    def test_advance_with_sufficient_steps(self):
        """Should advance when win rate and min steps are both met."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.80}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            min_steps_per_level=10,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = False
        cb.snapshot_paths = ["s1.zip", "s2.zip"]
        cb._level_start_step = 0
        cb.n_calls = 100  # 100 * 1 = 100 >= 10
        cb._check_level_transition()
        assert cb.window_level == 1
        assert cb.max_level == 1

    def test_backtrack_on_low_win_rate(self):
        """Should backtrack when win rate drops below threshold after min steps."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.40}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            backtrack_threshold=0.55,
            min_steps_per_level=10,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = False
        cb.window_level = 2
        cb._level_start_step = 0
        cb.n_calls = 100  # 100 * 1 = 100 >= 10
        cb._check_level_transition()
        assert cb.window_level == 1

    def test_recovery_at_level_zero(self):
        """Should enter recovery when at level 0 and win rate is below backtrack after min steps."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.40}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            backtrack_threshold=0.55,
            min_steps_per_level=10,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = False
        cb.window_level = 0
        cb._level_start_step = 0
        cb.n_calls = 100  # 100 * 1 = 100 >= 10
        cb._check_level_transition()
        assert cb._in_recovery
        assert cb.window_level == 0

    def test_recovery_exit_on_high_win_rate(self):
        """Should exit recovery when win rate recovers."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.75}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            recovery_win_rate=0.70,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = True
        cb._check_level_transition()
        assert not cb._in_recovery
        assert cb.window_level == 0

    def test_recovery_stays_on_low_win_rate(self):
        """Should stay in recovery if win rate hasn't recovered."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.60}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            recovery_win_rate=0.70,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = True
        cb._check_level_transition()
        assert cb._in_recovery

    def test_snapshot_quality_gate(self):
        """Should skip snapshot when win rate is below quality threshold."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.50}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            snapshot_win_rate=0.60,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        initial_count = len(cb.snapshot_paths)
        cb._take_snapshot()
        assert len(cb.snapshot_paths) == initial_count

    def test_snapshot_saves_when_above_gate(self):
        """Should save snapshot when win rate passes quality threshold."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.80}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            snapshot_win_rate=0.60,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._take_snapshot()
        assert len(cb.snapshot_paths) == 1

    def test_max_level_tracks_highest(self):
        """max_level should track the highest window level ever reached."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.80}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            min_steps_per_level=0,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = False
        cb.snapshot_paths = ["s1.zip", "s2.zip", "s3.zip"]

        # Advance twice
        cb.n_calls = 100
        cb._level_start_step = 0
        cb._check_level_transition()
        assert cb.window_level == 1
        cb._level_start_step = 0
        cb._check_level_transition()
        assert cb.window_level == 2
        assert cb.max_level == 2

        # Backtrack
        phase_cb.phase_history = [{"game_win_rate": 0.40}]
        cb._check_level_transition()
        assert cb.window_level == 1
        assert cb.max_level == 2  # max_level unchanged

    def test_window_cannot_exceed_snapshot_count(self):
        """Window level should not advance beyond the number of snapshots."""
        phase_cb = MagicMock()
        phase_cb.phase_history = [{"game_win_rate": 0.90}]
        cb = self._make_callback(
            phase_callback=phase_cb,
            min_steps_per_level=0,
            warmup_steps=0,
            n_envs=1,
        )
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb._in_recovery = False
        cb.snapshot_paths = ["s1.zip"]  # Only 1 snapshot
        cb.window_level = 1  # Already at max for 1 snapshot
        cb.n_calls = 100
        cb._level_start_step = 0
        cb._check_level_transition()
        assert cb.window_level == 1  # Should not advance to 2

    def test_log_metrics(self):
        """Should log all expected metrics to TensorBoard."""
        cb = self._make_callback()
        self._setup_callback(cb)
        cb._warmup_complete = True
        cb.window_level = 2
        cb.max_level = 3
        cb._in_recovery = False
        cb.snapshot_paths = ["a.zip", "b.zip"]
        cb._log_metrics()
        cb.logger.record.assert_any_call("self_play/window_level", 2)
        cb.logger.record.assert_any_call("self_play/max_level", 3)
        cb.logger.record.assert_any_call("self_play/in_recovery", 0.0)
        cb.logger.record.assert_any_call("self_play/pool_snapshots", 2)

    def test_seed_model_copied(self):
        """Should copy seed model to snapshot dir."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"fake model")
            seed_path = f.name
        try:
            cb = self._make_callback(seed_model_path=seed_path)
            assert cb._seed_path is not None
            assert os.path.exists(cb._seed_path)
        finally:
            os.unlink(seed_path)
            if cb._seed_path and os.path.exists(cb._seed_path):
                os.unlink(cb._seed_path)
