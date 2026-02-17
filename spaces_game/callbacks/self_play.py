"""SelfPlayCurriculumCallback — progressive window self-play with backtracking."""

import os
import shutil
import numpy as np
from typing import Optional, List, TYPE_CHECKING
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .opponent_progression import OpponentProgressionCallback


class SelfPlayCurriculumCallback(BaseCallback):
    """Progressive self-play curriculum with window levels and backtracking.

    Instead of binary block scheduling (self-play OR recovery), uses graduated
    difficulty via a window of active snapshots:

    - **Window level** (0..N): controls how many snapshots are active as opponents
      - Level 0: seed model only
      - Level k: seed + snapshots[0..k-1]
    - **Advance**: pool win rate >= advance_threshold for min_steps_per_level → level + 1
    - **Backtrack**: pool win rate < backtrack_threshold → level - 1
    - **Pool recovery**: level 0 AND win rate < backtrack_threshold →
      switch to pool opponents (ratio=0.0) until win rate >= recovery_win_rate,
      then resume self-play at level 0
    - **Snapshot quality gate**: only snapshot when pool win rate >= snapshot_win_rate
    - **Seed model**: resumed model permanently in pool, never pruned

    TensorBoard self_play/ panel:
      window_level, max_level, in_recovery, pool_snapshots, pool_win_rate
    """

    # Skill tier thresholds (win rate -> tier name)
    SKILL_TIERS = [
        (0.55, "beginner"),
        (0.60, "intermediate"),
        (0.65, "advanced"),
        (0.70, "expert"),
        (0.75, "advanced_plus"),
    ]

    def __init__(
        self,
        output_dir: str,
        snapshot_freq: int = 50_000,
        pool_size: int = 10,
        warmup_steps: int = 100_000,
        n_envs: int = 4,
        phase_callback: Optional['OpponentProgressionCallback'] = None,
        advance_threshold: float = 0.70,
        backtrack_threshold: float = 0.55,
        min_steps_per_level: int = 50_000,
        recovery_win_rate: float = 0.70,
        snapshot_win_rate: Optional[float] = None,
        seed_model_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.output_dir = output_dir
        self.snapshot_freq = snapshot_freq
        self.pool_size = pool_size
        self.warmup_steps = warmup_steps
        self.n_envs = n_envs
        self.phase_callback = phase_callback
        self.advance_threshold = advance_threshold
        self.backtrack_threshold = backtrack_threshold
        self.min_steps_per_level = min_steps_per_level
        self.recovery_win_rate = recovery_win_rate
        # Quality gate: default to midpoint of backtrack and advance thresholds
        if snapshot_win_rate is not None:
            self.snapshot_win_rate = snapshot_win_rate
        else:
            self.snapshot_win_rate = (backtrack_threshold + advance_threshold) / 2.0
        self.seed_model_path = seed_model_path

        # Snapshot pool (ordered chronologically)
        self.snapshot_paths: List[str] = []
        self._warmup_complete = False
        self._last_snapshot_step = 0
        self._saved_tiers: set = set()
        self._best_win_rate = 0.0

        # Window level state
        self.window_level = 0
        self.max_level = 0  # Highest level ever reached
        self._level_start_step = 0
        self._in_recovery = False

        # Create snapshot directory
        self.snapshot_dir = os.path.join(output_dir, "opponent_pool")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Seed pool with initial converged model (never pruned)
        self._seed_path: Optional[str] = None
        if seed_model_path and os.path.exists(seed_model_path):
            self._seed_path = os.path.join(self.snapshot_dir, "seed_model.zip")
            shutil.copy2(seed_model_path, self._seed_path)
            if self.verbose >= 1:
                print(f"  SELF-PLAY: Seeded pool with {seed_model_path}")

    def _on_step(self) -> bool:
        total_steps = self.n_calls * self.n_envs

        if total_steps < self.warmup_steps:
            return True

        if not self._warmup_complete:
            self._warmup_complete = True
            self._level_start_step = self.n_calls
            self.window_level = 0
            self._in_recovery = False
            if self.verbose >= 1:
                print(f"\n  SELF-PLAY: Warmup complete at {total_steps:,} steps")
                print(f"  SELF-PLAY: Starting at window level 0 (seed only)")
            self._take_snapshot()
            self._activate_self_play()
            return True

        # Periodic snapshots
        if self.n_calls - self._last_snapshot_step >= self.snapshot_freq:
            self._take_snapshot()

        # Check level transitions
        self._check_level_transition()

        # Check for skill milestone checkpoints
        self._check_skill_milestones()

        # Log metrics to TensorBoard
        self._log_metrics()

        return True

    def _check_level_transition(self):
        """Check if we should advance, backtrack, or enter/exit recovery."""
        pool_win_rate = self._get_latest_pool_win_rate()
        steps_at_level = (self.n_calls - self._level_start_step) * self.n_envs

        if self._in_recovery:
            # In pool recovery — wait for win rate to recover
            if pool_win_rate >= self.recovery_win_rate:
                self._in_recovery = False
                self._level_start_step = self.n_calls
                self.window_level = 0
                self._activate_self_play()
                if self.verbose >= 1:
                    print(f"\n  SELF-PLAY: Recovery complete (win rate {pool_win_rate:.1%}). "
                          f"Resuming at level 0")
            return

        # Check backtrack condition (requires minimum steps to avoid noisy eval knee-jerks)
        if (pool_win_rate < self.backtrack_threshold and
                steps_at_level >= self.min_steps_per_level):
            if self.window_level > 0:
                self.window_level -= 1
                self._level_start_step = self.n_calls
                self._assign_opponents()
                if self.verbose >= 1:
                    print(f"\n  SELF-PLAY: Backtrack to level {self.window_level} "
                          f"(win rate {pool_win_rate:.1%} < {self.backtrack_threshold:.1%})")
            else:
                # Already at level 0 and still failing — enter pool recovery
                self._in_recovery = True
                self._level_start_step = self.n_calls
                self._set_ratio(0.0)
                if self.verbose >= 1:
                    print(f"\n  SELF-PLAY: Entering pool recovery "
                          f"(win rate {pool_win_rate:.1%} < {self.backtrack_threshold:.1%})")
            return

        # Check advance condition (requires minimum steps at this level)
        max_possible_level = len(self.snapshot_paths)
        if (pool_win_rate >= self.advance_threshold and
                steps_at_level >= self.min_steps_per_level and
                self.window_level < max_possible_level):
            self.window_level += 1
            self.max_level = max(self.max_level, self.window_level)
            self._level_start_step = self.n_calls
            self._assign_opponents()
            if self.verbose >= 1:
                print(f"\n  SELF-PLAY: Advanced to level {self.window_level} "
                      f"(win rate {pool_win_rate:.1%} >= {self.advance_threshold:.1%}, "
                      f"{len(self.snapshot_paths)} snapshots)")

    def _activate_self_play(self):
        """Switch from pool opponents to self-play at current window level."""
        self._set_ratio(1.0)
        self._assign_opponents()

    def _log_metrics(self):
        """Log self-play state to TensorBoard."""
        if self.logger is None:
            return
        self.logger.record("self_play/window_level", self.window_level)
        self.logger.record("self_play/max_level", self.max_level)
        self.logger.record("self_play/in_recovery", float(self._in_recovery))
        self.logger.record("self_play/pool_snapshots", len(self.snapshot_paths))
        pool_win_rate = self._get_latest_pool_win_rate()
        self.logger.record("self_play/pool_win_rate", pool_win_rate)

    def _get_latest_pool_win_rate(self) -> float:
        """Get the latest pool win rate from the phase callback."""
        if self.phase_callback is None or not self.phase_callback.phase_history:
            return 1.0  # Assume OK if no data
        return self.phase_callback.phase_history[-1].get("game_win_rate", 1.0)

    def _set_ratio(self, ratio: float):
        """Set self-play ratio on all training envs."""
        for i in range(self.n_envs):
            try:
                self.training_env.env_method(
                    "set_self_play_ratio", ratio, indices=[i],
                )
            except Exception as e:
                if self.verbose >= 1:
                    print(f"  Warning: Could not set ratio for env {i}: {e}")

    def _take_snapshot(self):
        """Save model snapshot if quality gate passes, and assign to training envs."""
        self._last_snapshot_step = self.n_calls

        # Quality gate: only snapshot if pool win rate is above threshold
        pool_win_rate = self._get_latest_pool_win_rate()
        if pool_win_rate < self.snapshot_win_rate:
            if self.verbose >= 1:
                state = "RECOVERY" if self._in_recovery else f"LEVEL {self.window_level}"
                print(f"  SELF-PLAY [{state}]: Snapshot skipped — pool win rate "
                      f"{pool_win_rate:.1%} < {self.snapshot_win_rate:.1%}")
            # Still assign opponents from existing pool
            if not self._in_recovery:
                self._assign_opponents()
            return

        # Save snapshot
        snapshot_path = os.path.join(
            self.snapshot_dir, f"snapshot_{self.n_calls}.zip",
        )
        self.model.save(snapshot_path)
        self.snapshot_paths.append(snapshot_path)

        # Prune oldest (never prune seed model)
        while len(self.snapshot_paths) > self.pool_size:
            old_path = self.snapshot_paths.pop(0)
            if old_path != self._seed_path and os.path.exists(old_path):
                os.remove(old_path)

        if self.verbose >= 1:
            state = "RECOVERY" if self._in_recovery else f"LEVEL {self.window_level}"
            print(f"  SELF-PLAY [{state}]: Snapshot saved ({len(self.snapshot_paths)} in pool, "
                  f"win rate {pool_win_rate:.1%})")

        # Assign opponents (only during active self-play, not recovery)
        if not self._in_recovery:
            self._assign_opponents()

    def _assign_opponents(self):
        """Assign opponents from the window to each training env.

        At window level k, opponents are drawn from:
        - seed model (always included if present)
        - snapshots[0..k-1] (the first k chronological snapshots)
        """
        candidates = []
        if self._seed_path:
            candidates.append(self._seed_path)

        # Add snapshots up to window level
        window_end = min(self.window_level, len(self.snapshot_paths))
        candidates.extend(self.snapshot_paths[:window_end])

        if not candidates:
            return

        for i in range(self.n_envs):
            path = candidates[np.random.randint(len(candidates))]
            try:
                self.training_env.env_method(
                    "set_opponent_model", path, indices=[i],
                )
            except Exception as e:
                if self.verbose >= 1:
                    print(f"  Warning: Could not set opponent model for env {i}: {e}")

    def _check_skill_milestones(self):
        """Check phase callback's eval win rate and save skill checkpoints."""
        if self.phase_callback is None or not self.phase_callback.phase_history:
            return

        latest = self.phase_callback.phase_history[-1]
        win_rate = latest.get("game_win_rate", 0.0)

        if win_rate <= self._best_win_rate:
            return
        self._best_win_rate = win_rate

        diff_dir = os.path.join(self.output_dir, "difficulty")
        os.makedirs(diff_dir, exist_ok=True)

        for threshold, tier_name in self.SKILL_TIERS:
            if tier_name in self._saved_tiers:
                continue
            if win_rate >= threshold:
                path = os.path.join(diff_dir, f"{tier_name}.zip")
                self.model.save(path)
                self._saved_tiers.add(tier_name)
                if self.verbose >= 1:
                    print(f"  SKILL CHECKPOINT: {tier_name} (win_rate={win_rate:.1%}) -> {path}")

    def _on_training_end(self) -> None:
        # Save final tier checkpoints from snapshot timeline
        diff_dir = os.path.join(self.output_dir, "difficulty")
        os.makedirs(diff_dir, exist_ok=True)

        # Map snapshot timeline to 6 tiers
        tier_names = ["beginner", "intermediate", "advanced", "expert", "advanced_plus", "master"]
        if self.snapshot_paths:
            n_snapshots = len(self.snapshot_paths)
            for i, tier in enumerate(tier_names):
                if tier in self._saved_tiers:
                    continue  # Already saved by milestone
                idx = min(i * n_snapshots // len(tier_names), n_snapshots - 1)
                src = self.snapshot_paths[idx]
                dst = os.path.join(diff_dir, f"{tier}_checkpoint.zip")
                shutil.copy2(src, dst)
                if self.verbose >= 1:
                    print(f"  Skill tier '{tier}' -> {dst}")

        # Always save current model as expert
        expert_path = os.path.join(diff_dir, "expert.zip")
        self.model.save(expert_path)

        if self.verbose >= 1:
            print(f"\nSelf-play: {len(self.snapshot_paths)} snapshots in pool")
            print(f"Max window level reached: {self.max_level}")
            print(f"Skill milestones saved: {sorted(self._saved_tiers)}")
