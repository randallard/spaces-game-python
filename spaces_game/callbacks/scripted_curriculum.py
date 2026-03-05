"""ScriptedCurriculumCallback — advance through scripted opponent levels, then hand off to self-play."""

import os
from datetime import datetime
from typing import Optional, List, Dict, TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from .self_play import SelfPlayCurriculumCallback


class ScriptedCurriculumCallback(BaseCallback):
    """Progressive curriculum through scripted opponent levels 1-5.

    Evaluates the agent's game win rate against the current scripted level.
    When win rate exceeds the advance threshold for min_steps_per_level,
    advances to the next level. After clearing all levels, transitions
    to self-play by activating the linked SelfPlayCurriculumCallback.

    Exposes ``phase_history`` for compatibility with SelfPlayCurriculumCallback
    (which reads it for pool_win_rate).
    """

    SCRIPTED_LEVELS = [1, 2, 3, 4, 5]
    LEVEL_NAMES = {
        1: "Simple",
        2: "Reactive",
        3: "Trapper",
        4: "Adaptive",
        5: "Supermove",
    }

    def __init__(
        self,
        eval_freq: int = 2000,
        eval_episodes: int = 20,
        advance_threshold: float = 0.45,
        min_steps_per_level: int = 10_000,
        board_size: int = 2,
        opponent_pools: Optional[List[str]] = None,
        phase_map: Optional[Dict[int, List[int]]] = None,
        use_fog: bool = False,
        output_dir: str = "models",
        self_play_callback: Optional['SelfPlayCurriculumCallback'] = None,
        n_envs: int = 4,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.advance_threshold = advance_threshold
        self.min_steps_per_level = min_steps_per_level
        self.board_size = board_size
        self.use_fog = use_fog
        self.output_dir = output_dir
        self.self_play_callback = self_play_callback
        self.n_envs = n_envs

        # State
        self._current_level_idx = 0
        self._level_start_step = 0
        self._last_eval_step = 0
        self._game_win_rate = 0.0
        self._all_levels_cleared = False

        # Compatible with SelfPlayCurriculumCallback's phase_callback interface
        self.phase_history: List[Dict] = []

        # Dedicated eval env
        from ..simultaneous_play_env import SimultaneousPlayEnv
        max_construction_steps = board_size * 10
        self._eval_env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            max_construction_steps=max_construction_steps,
            phase_map=phase_map,
            use_fog=use_fog,
        )
        self._eval_env.set_scripted_opponent_level(self.SCRIPTED_LEVELS[0])

    @property
    def current_scripted_level(self) -> int:
        """Current scripted level (1-5), or 0 if all cleared."""
        if self._all_levels_cleared:
            return 0
        return self.SCRIPTED_LEVELS[self._current_level_idx]

    def _init_callback(self) -> None:
        """Set scripted level on all training envs at training start."""
        level = self.SCRIPTED_LEVELS[0]
        for i in range(self.n_envs):
            self.training_env.env_method(
                "set_scripted_opponent_level", level, indices=[i],
            )
        if self.verbose >= 1:
            print(f"  SCRIPTED: Starting at level {level} ({self.LEVEL_NAMES[level]})")

    def _on_step(self) -> bool:
        if self._all_levels_cleared:
            return True

        if self.n_calls - self._last_eval_step >= self.eval_freq:
            self._evaluate()
            self._last_eval_step = self.n_calls
            self._check_advancement()
            self._log_metrics()

        return True

    def _evaluate(self):
        """Run evaluation episodes against the current scripted level."""
        level = self.SCRIPTED_LEVELS[self._current_level_idx]
        self._eval_env.set_scripted_opponent_level(level)

        wins = 0
        for ep in range(self.eval_episodes):
            obs, info = self._eval_env.reset(seed=42 + ep)
            done = False

            while not done:
                action_masks = self._eval_env.action_masks()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=action_masks,
                )
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                done = terminated or truncated

            if info.get("game_winner") == "agent":
                wins += 1

        self._game_win_rate = wins / self.eval_episodes
        total_steps = self.n_calls * self.n_envs

        self.phase_history.append({
            "step": total_steps,
            "scripted_level": level,
            "game_win_rate": self._game_win_rate,
        })

        if self.verbose >= 1:
            print(f"  SCRIPTED: Level {level} ({self.LEVEL_NAMES[level]}) — "
                  f"WR {self._game_win_rate:.0%} "
                  f"(gate: {self.advance_threshold:.0%})")

    def _check_advancement(self):
        """Check if we should advance to the next scripted level."""
        steps_at_level = (self.n_calls - self._level_start_step) * self.n_envs

        if (self._game_win_rate >= self.advance_threshold
                and steps_at_level >= self.min_steps_per_level):

            # Save level checkpoint
            self._save_level_checkpoint()

            if self._current_level_idx < len(self.SCRIPTED_LEVELS) - 1:
                # Advance to next scripted level
                self._current_level_idx += 1
                self._level_start_step = self.n_calls
                new_level = self.SCRIPTED_LEVELS[self._current_level_idx]

                for i in range(self.n_envs):
                    self.training_env.env_method(
                        "set_scripted_opponent_level", new_level, indices=[i],
                    )
                self._eval_env.set_scripted_opponent_level(new_level)

                if self.verbose >= 1:
                    print(f"  SCRIPTED: Advanced to level {new_level} "
                          f"({self.LEVEL_NAMES[new_level]})")
            else:
                # All levels cleared — transition to self-play
                self._all_levels_cleared = True

                for i in range(self.n_envs):
                    self.training_env.env_method(
                        "clear_scripted_opponent", indices=[i],
                    )

                if self.self_play_callback is not None:
                    self.self_play_callback.warmup_steps = 0
                    if self.verbose >= 1:
                        total_steps = self.n_calls * self.n_envs
                        print(f"\n  SCRIPTED: All 5 levels cleared at {total_steps:,} steps!")
                        print(f"  SCRIPTED: Transitioning to self-play...")
                else:
                    if self.verbose >= 1:
                        print(f"\n  SCRIPTED: All 5 levels cleared! No self-play callback — training continues against pool.")

    def _save_level_checkpoint(self):
        """Save a model checkpoint when clearing a scripted level."""
        level = self.SCRIPTED_LEVELS[self._current_level_idx]
        ckpt_dir = os.path.join(self.output_dir, "scripted_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        total_steps = self.n_calls * self.n_envs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        steps_str = (f"{total_steps / 1000:.0f}k" if total_steps < 1_000_000
                     else f"{total_steps / 1_000_000:.2f}M")
        filename = f"{timestamp}_cleared_level{level}_{steps_str}.zip"
        path = os.path.join(ckpt_dir, filename)
        self.model.save(path)

        if self.verbose >= 1:
            print(f"  SCRIPTED CHECKPOINT: {filename}")

    def _log_metrics(self):
        """Log scripted curriculum metrics to TensorBoard."""
        if self.logger is None:
            return
        self.logger.record("scripted/level", self.current_scripted_level)
        self.logger.record("scripted/game_win_rate", self._game_win_rate)
        self.logger.record("scripted/all_cleared", float(self._all_levels_cleared))

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
