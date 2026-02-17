"""DiscordNotifierCallback — sends training milestones and check-ins to Discord."""

import json
import time
import urllib.request
from typing import Optional, List, Dict, TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .opponent_progression import OpponentProgressionCallback
    from .self_play import SelfPlayCurriculumCallback


def _format_steps(n: int) -> str:
    """Format step count as human-readable: 1500000 -> '1.5M', 250000 -> '250K'."""
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
    if n >= 1_000:
        val = n / 1_000
        return f"{val:.1f}K" if val != int(val) else f"{int(val)}K"
    return str(n)


class DiscordNotifierCallback(BaseCallback):
    """Sends training milestones and periodic check-ins to a Discord webhook.

    Milestone alerts fire on state transitions (edge detection) — phase advance,
    self-play level change, recovery enter/exit, etc.

    Periodic check-ins summarize progress with trend analysis and commentary.
    """

    # Discord embed colors
    COLOR_SUCCESS = 0x2ECC71   # Green
    COLOR_WARNING = 0xF39C12   # Orange
    COLOR_DANGER = 0xE74C3C    # Red
    COLOR_INFO = 0x3498DB      # Blue
    COLOR_PURPLE = 0x9B59B6    # Purple

    def __init__(
        self,
        webhook_url: str,
        board_size: int,
        total_timesteps: int,
        n_envs: int = 4,
        use_fog: bool = False,
        self_play: bool = False,
        check_in_minutes: int = 30,
        phase_callback: Optional['OpponentProgressionCallback'] = None,
        self_play_callback: Optional['SelfPlayCurriculumCallback'] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.webhook_url = webhook_url
        self.board_size = board_size
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.use_fog = use_fog
        self.self_play = self_play
        self.check_in_minutes = check_in_minutes
        self.phase_callback = phase_callback
        self.self_play_callback = self_play_callback

        # Edge detection — previous values
        self._prev_phase: Optional[int] = None
        self._prev_window_level: Optional[int] = None
        self._prev_in_recovery: Optional[bool] = None
        self._prev_best_model_path: Optional[str] = None

        # Periodic check-in timer
        self._last_check_in_time: float = 0.0
        self._training_start_time: float = 0.0

    def _on_training_start(self) -> None:
        self._training_start_time = time.time()
        self._last_check_in_time = time.time()

        # Initialize edge detection from current state
        if self.phase_callback is not None:
            self._prev_phase = self.phase_callback.current_phase
        if self.self_play_callback is not None:
            self._prev_window_level = self.self_play_callback.window_level
            self._prev_in_recovery = self.self_play_callback._in_recovery

        # Send training started notification
        label = self._training_label()
        embed = self._make_embed(
            title=f"Training Started — {label}",
            description=f"Target: {_format_steps(self.total_timesteps)} steps",
            color=self.COLOR_INFO,
        )
        self._send_webhook(embed)

    def _on_step(self) -> bool:
        # Check milestones (edge detection)
        self._check_phase_milestone()
        self._check_self_play_milestones()

        # Periodic check-in
        elapsed = time.time() - self._last_check_in_time
        if elapsed >= self.check_in_minutes * 60:
            self._send_check_in()
            self._last_check_in_time = time.time()

        return True

    def _on_training_end(self) -> None:
        total_steps = self.num_timesteps
        win_rate = self._latest_win_rate()
        phase = self.phase_callback.current_phase if self.phase_callback else "?"
        max_phase = self.phase_callback.max_phase if self.phase_callback else "?"

        duration = time.time() - self._training_start_time
        hours = duration / 3600

        fields = [
            {"name": "Steps", "value": _format_steps(total_steps), "inline": True},
            {"name": "Duration", "value": f"{hours:.1f}h", "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.0%}" if win_rate is not None else "N/A", "inline": True},
            {"name": "Phase", "value": f"{phase}/{max_phase}", "inline": True},
        ]

        if self.self_play_callback is not None:
            fields.append({
                "name": "Self-Play Level",
                "value": str(self.self_play_callback.window_level),
                "inline": True,
            })
            fields.append({
                "name": "Snapshots",
                "value": str(len(self.self_play_callback.snapshot_paths)),
                "inline": True,
            })

        embed = self._make_embed(
            title=f"Training Complete — {self._training_label()}",
            description=f"Finished {_format_steps(total_steps)} steps in {hours:.1f} hours",
            color=self.COLOR_SUCCESS,
            fields=fields,
        )
        self._send_webhook(embed)

    # ── Milestone detection ──────────────────────────────────────────

    def _check_phase_milestone(self):
        if self.phase_callback is None:
            return

        current_phase = self.phase_callback.current_phase
        max_phase = self.phase_callback.max_phase

        if self._prev_phase is not None and current_phase != self._prev_phase:
            win_rate = self._latest_win_rate()
            steps = self.num_timesteps

            if current_phase >= max_phase:
                # All phases cleared
                embed = self._make_embed(
                    title="All Phases Cleared!",
                    description=(
                        f"All {max_phase + 1} phases cleared at {_format_steps(steps)} steps. "
                        f"Win rate: {win_rate:.0%}" if win_rate is not None else
                        f"All {max_phase + 1} phases cleared at {_format_steps(steps)} steps."
                    ),
                    color=self.COLOR_SUCCESS,
                )
            else:
                embed = self._make_embed(
                    title=f"Phase Advanced: {self._prev_phase} → {current_phase}",
                    description=(
                        f"Advanced to phase {current_phase}/{max_phase} at "
                        f"{_format_steps(steps)} steps. "
                        f"Win rate: {win_rate:.0%}" if win_rate is not None else
                        f"Advanced to phase {current_phase}/{max_phase} at "
                        f"{_format_steps(steps)} steps."
                    ),
                    color=self.COLOR_INFO,
                )

            self._send_webhook(embed)

        self._prev_phase = current_phase

    def _check_self_play_milestones(self):
        if self.self_play_callback is None:
            return

        sp = self.self_play_callback
        steps = self.num_timesteps

        # Window level change
        if self._prev_window_level is not None and sp.window_level != self._prev_window_level:
            direction = "advanced" if sp.window_level > self._prev_window_level else "backtracked"
            embed = self._make_embed(
                title=f"Self-Play Level {direction.title()}: {self._prev_window_level} → {sp.window_level}",
                description=(
                    f"Window {direction} to level {sp.window_level} "
                    f"({sp.window_level + 1} opponents) at {_format_steps(steps)} steps"
                ),
                color=self.COLOR_INFO if direction == "advanced" else self.COLOR_WARNING,
            )
            self._send_webhook(embed)

        self._prev_window_level = sp.window_level

        # Recovery state change
        if self._prev_in_recovery is not None and sp._in_recovery != self._prev_in_recovery:
            win_rate = self._latest_win_rate()
            if sp._in_recovery:
                embed = self._make_embed(
                    title="Entered Pool Recovery",
                    description=(
                        f"Win rate dropped to {win_rate:.0%}. "
                        f"Switching to pool opponents at {_format_steps(steps)} steps."
                        if win_rate is not None else
                        f"Switching to pool opponents at {_format_steps(steps)} steps."
                    ),
                    color=self.COLOR_DANGER,
                )
            else:
                embed = self._make_embed(
                    title="Exited Pool Recovery",
                    description=(
                        f"Win rate recovered to {win_rate:.0%}. "
                        f"Resuming self-play at {_format_steps(steps)} steps."
                        if win_rate is not None else
                        f"Resuming self-play at {_format_steps(steps)} steps."
                    ),
                    color=self.COLOR_SUCCESS,
                )
            self._send_webhook(embed)

        self._prev_in_recovery = sp._in_recovery

    # ── Periodic check-in ────────────────────────────────────────────

    def _send_check_in(self):
        steps = self.num_timesteps
        progress_pct = steps / self.total_timesteps if self.total_timesteps > 0 else 0
        win_rate = self._latest_win_rate()

        fields = [
            {"name": "Progress", "value": f"{_format_steps(steps)} / {_format_steps(self.total_timesteps)} ({progress_pct:.0%})", "inline": False},
        ]

        if self.phase_callback is not None:
            phase = self.phase_callback.current_phase
            max_phase = self.phase_callback.max_phase
            fields.append({"name": "Phase", "value": f"{phase}/{max_phase}", "inline": True})

        if win_rate is not None:
            fields.append({"name": "Win Rate", "value": f"{win_rate:.0%}", "inline": True})

        valid_rate = self._latest_valid_rate()
        if valid_rate is not None:
            fields.append({"name": "Valid Rate", "value": f"{valid_rate:.0%}", "inline": True})

        if self.self_play_callback is not None:
            sp = self.self_play_callback
            sp_status = f"Level {sp.window_level}, {len(sp.snapshot_paths)} snapshots"
            if sp._in_recovery:
                sp_status += " (IN RECOVERY)"
            fields.append({"name": "Self-Play", "value": sp_status, "inline": False})

        # Commentary
        commentary = self._analyze_trends()

        elapsed = time.time() - self._training_start_time
        hours = elapsed / 3600

        embed = self._make_embed(
            title=f"Check-In — {self._training_label()}",
            description=commentary if commentary else f"Training for {hours:.1f}h",
            color=self._commentary_color(commentary),
            fields=fields,
        )
        self._send_webhook(embed)

    # ── Trend analysis ───────────────────────────────────────────────

    def _analyze_trends(self) -> str:
        """Analyze recent phase_history to generate commentary."""
        lines = []

        if self.phase_callback is None or not self.phase_callback.phase_history:
            return "No evaluation data yet."

        history = self.phase_callback.phase_history

        # Win rate trend (last 5 evals)
        recent = history[-5:]
        win_rates = [h["game_win_rate"] for h in recent]

        if len(win_rates) >= 3:
            slope = win_rates[-1] - win_rates[0]
            if slope > 0.05:
                lines.append("Win rate climbing steadily — training progressing well.")
            elif slope < -0.05:
                lines.append("Win rate declining — watch for policy collapse.")
            else:
                # Check for plateau
                if len(history) >= 10:
                    older = [h["game_win_rate"] for h in history[-10:-5]]
                    older_avg = sum(older) / len(older) if older else 0
                    recent_avg = sum(win_rates) / len(win_rates)
                    if abs(recent_avg - older_avg) < 0.03:
                        lines.append(f"Win rate plateaued at ~{recent_avg:.0%}. May be near convergence.")
                    else:
                        lines.append("Win rate stable.")
                else:
                    lines.append("Win rate stable.")

        # Valid rate check
        valid_rate = self._latest_valid_rate()
        if valid_rate is not None and valid_rate < 0.95:
            lines.append(f"Valid rate dropped to {valid_rate:.0%} — unusual with strict masking, check logs.")

        # Self-play volatility
        if self.self_play_callback is not None and len(win_rates) >= 3:
            wr_std = max(win_rates) - min(win_rates)
            if wr_std > 0.15:
                lines.append("Win rate volatile during self-play — normal during adaptation.")

        # Recovery status
        if self.self_play_callback is not None and self.self_play_callback._in_recovery:
            lines.append("In pool recovery — waiting for win rate to stabilize.")

        return " ".join(lines) if lines else "Training in progress."

    def _commentary_color(self, commentary: str) -> int:
        """Pick embed color based on commentary sentiment."""
        if not commentary:
            return self.COLOR_INFO
        if "collapse" in commentary or "declining" in commentary:
            return self.COLOR_DANGER
        if "plateau" in commentary or "recovery" in commentary or "dropped" in commentary:
            return self.COLOR_WARNING
        if "climbing" in commentary or "progressing well" in commentary:
            return self.COLOR_SUCCESS
        return self.COLOR_INFO

    # ── Helpers ──────────────────────────────────────────────────────

    def _training_label(self) -> str:
        """Generate a label like 'Size 3 Fog+Self-Play'."""
        parts = [f"Size {self.board_size}"]
        if self.use_fog:
            parts.append("Fog")
        if self.self_play:
            parts.append("Self-Play")
        if len(parts) > 1:
            return parts[0] + " " + "+".join(parts[1:])
        return parts[0]

    def _latest_win_rate(self) -> Optional[float]:
        if self.phase_callback and self.phase_callback.phase_history:
            return self.phase_callback.phase_history[-1]["game_win_rate"]
        return None

    def _latest_valid_rate(self) -> Optional[float]:
        if self.phase_callback and self.phase_callback.phase_history:
            return self.phase_callback.phase_history[-1]["valid_rate"]
        return None

    def _make_embed(
        self,
        title: str,
        description: str = "",
        color: int = COLOR_INFO,
        fields: Optional[List[Dict]] = None,
    ) -> dict:
        """Build a Discord embed object."""
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }
        if fields:
            embed["fields"] = fields
        return embed

    def _send_webhook(self, embed: dict) -> None:
        """POST an embed to the Discord webhook. Failures are logged but don't stop training."""
        payload = json.dumps({"embeds": [embed]}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if self.verbose >= 2:
                    print(f"  DISCORD: Sent '{embed.get('title', '')}' (status {resp.status})")
        except Exception as e:
            if self.verbose >= 1:
                print(f"  DISCORD: Failed to send webhook: {e}")
