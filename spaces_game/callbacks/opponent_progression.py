"""OpponentProgressionCallback — advances opponent difficulty based on game win rate."""

import os
import json
import numpy as np
from typing import Optional, List, Dict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..simultaneous_play_env import SimultaneousPlayEnv
from .pool_utils import DIFFICULTY_CHECKPOINTS


class OpponentProgressionCallback(BaseCallback):
    """
    Callback to advance opponent difficulty based on game win rate.
    """

    def __init__(
        self,
        eval_freq: int = 2000,
        eval_episodes: int = 20,
        win_rate_threshold: float = 0.70,
        valid_rate_threshold: float = 0.90,
        min_steps_per_phase: int = 10000,
        max_phase: int = 4,
        board_size: int = 2,
        opponent_pools: Optional[List[str]] = None,
        eval_callback_env: Optional[DummyVecEnv] = None,
        output_dir: str = "models/stage3",
        phase_map: Optional[Dict[int, List[int]]] = None,
        start_opponent_phase: Optional[int] = None,
        use_fog: bool = False,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold
        self.valid_rate_threshold = valid_rate_threshold
        self.min_steps_per_phase = min_steps_per_phase
        self.max_phase = max_phase
        self.current_phase = start_opponent_phase if start_opponent_phase is not None else 0
        self.phase_history = []
        self.eval_callback_env = eval_callback_env
        self.output_dir = output_dir
        self._phase_start_step = 0

        # Dedicated single env for evaluation
        max_construction_steps = board_size * 10
        self._eval_env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            opponent_phase=self.current_phase,
            max_construction_steps=max_construction_steps,
            phase_map=phase_map,
            use_fog=use_fog,
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_maybe_advance()
        return True

    def _evaluate_and_maybe_advance(self):
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"OPPONENT PHASE {self.current_phase} EVALUATION ({self.eval_episodes} games)")
            print(f"{'='*70}")

        game_wins = 0
        game_losses = 0
        total_rounds_valid = 0
        total_rounds = 0
        total_reward = 0.0

        for ep in range(self.eval_episodes):
            obs, info = self._eval_env.reset(seed=42 + ep)
            done = False
            episode_reward = 0.0

            while not done:
                action_masks = self._eval_env.action_masks()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=action_masks,
                )
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Track per-round validity
                if "valid_board" in info:
                    total_rounds += 1
                    if info["valid_board"]:
                        total_rounds_valid += 1

            total_reward += episode_reward

            game_winner = info.get("game_winner", "tie")
            if game_winner == "agent":
                game_wins += 1
            elif game_winner == "opponent":
                game_losses += 1

        game_win_rate = game_wins / self.eval_episodes
        valid_rate = total_rounds_valid / max(total_rounds, 1)
        avg_reward = total_reward / self.eval_episodes

        # Log
        self.logger.record("curriculum/opponent_phase", self.current_phase)
        self.logger.record("curriculum/game_win_rate", game_win_rate)
        self.logger.record("curriculum/valid_rate", valid_rate)
        self.logger.record("curriculum/avg_reward", avg_reward)

        if self.verbose >= 1:
            ties = self.eval_episodes - game_wins - game_losses
            print(f"  Game wins:  {game_win_rate:.1%} ({game_wins}W/{game_losses}L/{ties}T)")
            print(f"  Valid rate: {valid_rate:.1%} ({total_rounds_valid}/{total_rounds} rounds)")
            print(f"  Avg reward: {avg_reward:.2f}")

        # Advance opponent phase if thresholds met
        steps_at_phase = self.n_calls - self._phase_start_step
        self._maybe_advance_opponent(game_win_rate, valid_rate, steps_at_phase)

        self.phase_history.append({
            "timestep": self.n_calls,
            "opponent_phase": self.current_phase,
            "game_win_rate": game_win_rate,
            "valid_rate": valid_rate,
            "avg_reward": avg_reward,
        })

        if self.verbose >= 1:
            print(f"{'='*70}\n")

    def _maybe_advance_opponent(self, game_win_rate: float, valid_rate: float, steps_at_phase: int):
        """Advance opponent phase based on game win rate (existing logic)."""
        if (game_win_rate >= self.win_rate_threshold and
                valid_rate >= self.valid_rate_threshold and
                steps_at_phase >= self.min_steps_per_phase and
                self.current_phase < self.max_phase):
            self.current_phase += 1
            self._phase_start_step = self.n_calls

            # Update training envs
            try:
                self.training_env.env_method(
                    "set_opponent_phase", self.current_phase,
                )
            except Exception as e:
                if self.verbose >= 1:
                    print(f"  Warning: Could not update training envs: {e}")

            # Update eval envs
            self._eval_env.set_opponent_phase(self.current_phase)
            if self.eval_callback_env is not None:
                try:
                    self.eval_callback_env.env_method(
                        "set_opponent_phase", self.current_phase,
                    )
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"  Warning: Could not update eval env: {e}")

            # Save checkpoint
            completed_phase = self.current_phase - 1
            ckpt_path = f"{self.output_dir}/phase_{completed_phase}_checkpoint.zip"
            self.model.save(ckpt_path)

            # Save named difficulty checkpoint if this phase is a milestone
            if completed_phase in DIFFICULTY_CHECKPOINTS:
                diff_name = DIFFICULTY_CHECKPOINTS[completed_phase]
                diff_dir = f"{self.output_dir}/difficulty"
                os.makedirs(diff_dir, exist_ok=True)
                diff_path = f"{diff_dir}/{diff_name}.zip"
                self.model.save(diff_path)
                if self.verbose >= 1:
                    print(f"  Difficulty checkpoint saved: {diff_path}")

            if self.verbose >= 1:
                print(f"\n  OPPONENT PHASE ADVANCED: {completed_phase} -> {self.current_phase}")
                print(f"  Checkpoint saved: {ckpt_path}")

    def _on_training_end(self) -> None:
        # Save expert difficulty checkpoint at training end
        diff_dir = f"{self.output_dir}/difficulty"
        os.makedirs(diff_dir, exist_ok=True)
        expert_path = f"{diff_dir}/expert.zip"
        self.model.save(expert_path)
        if self.verbose >= 1:
            print(f"\nExpert difficulty checkpoint saved: {expert_path}")

        history_path = f"{self.output_dir}/phase_history.json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.phase_history, f, indent=2)
        if self.verbose >= 1:
            print(f"Phase history saved to: {history_path}")
