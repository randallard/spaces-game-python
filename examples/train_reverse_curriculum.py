"""
Train an agent for reverse curriculum board building (Stage 2).

The agent learns to build boards backward (goal -> start) while leveraging
Stage 1's frozen policy for base board selection. Training progresses through
curriculum phases:
  - Phase 0: Place goal only (trivial)
  - Phase 1: Place last move + goal (easy)
  - Phase 2: Place last 2 moves + goal (medium)
  - ...
  - Phase N: Full construction from scratch (hard)

The curriculum advances automatically when the agent achieves 80%+ win rate
on the current phase. Uses MaskablePPO with action masking to ensure only
valid board placements are attempted.

Usage:
    python examples/train_reverse_curriculum.py
    python examples/train_reverse_curriculum.py --start-phase 0 --timesteps 200000

Training will save checkpoints to models/reverse_curriculum/ and log to
tensorboard. Monitor progress with:
    tensorboard --logdir logs/reverse_curriculum/
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from spaces_game import ReverseCurriculumBuilderEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    """Get action masks from the environment."""
    return env.action_masks()


class PhaseProgressionCallback(BaseCallback):
    """
    Callback to automatically advance curriculum phase based on win rate.

    Evaluates agent performance every N steps using a dedicated single
    environment (not the training vectorized env) for accurate metrics.
    Advances to next phase when win rate exceeds threshold (default 80%).
    """

    def __init__(
        self,
        eval_freq: int = 1000,
        eval_episodes: int = 20,
        win_rate_threshold: float = 0.65,
        max_phase: int = 10,
        board_size: int = 2,
        board_library_path: str = "new_boards_2.json",
        stage1_model_path: Optional[str] = None,
        eval_callback_env: Optional[DummyVecEnv] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold
        self.max_phase = max_phase
        self.current_phase = 0
        self.phase_history = []
        self.eval_callback_env = eval_callback_env

        # Dedicated single env for accurate phase evaluation
        self._eval_env = ReverseCurriculumBuilderEnv(
            board_size=board_size,
            board_library_path=board_library_path,
            stage1_model_path=stage1_model_path,
            curriculum_phase=0,
            opponent_strategy="random",
            show_opponent_board=True,
        )

    def _on_step(self) -> bool:
        """Called after every step."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_maybe_advance()
        return True

    def _evaluate_and_maybe_advance(self):
        """Evaluate current phase performance and advance if ready."""
        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"PHASE {self.current_phase} EVALUATION ({self.eval_episodes} episodes)")
            print(f"{'='*70}")

        wins = 0
        ties = 0
        valid_boards = 0
        total_reward = 0.0

        for ep in range(self.eval_episodes):
            obs, info = self._eval_env.reset(seed=42 + ep)
            done = False
            episode_reward = 0.0

            while not done:
                action_masks = self._eval_env.action_masks()
                action, _states = self.model.predict(
                    obs, deterministic=True, action_masks=action_masks
                )
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            if info.get('valid_board', False):
                valid_boards += 1
                agent_score = info.get('agent_score', 0)
                opponent_score = info.get('opponent_score', 0)
                if agent_score > opponent_score:
                    wins += 1
                elif agent_score == opponent_score:
                    ties += 1

            total_reward += episode_reward

        # Calculate metrics
        win_rate = wins / self.eval_episodes
        valid_rate = valid_boards / self.eval_episodes
        # Non-loss rate: wins + ties among valid boards (accounts for tie-only boards)
        non_loss_rate = (wins + ties) / self.eval_episodes if self.eval_episodes > 0 else 0
        avg_reward = total_reward / self.eval_episodes

        # Log metrics
        self.logger.record("curriculum/phase", self.current_phase)
        self.logger.record("curriculum/win_rate", win_rate)
        self.logger.record("curriculum/valid_rate", valid_rate)
        self.logger.record("curriculum/avg_reward", avg_reward)

        if self.verbose >= 1:
            print(f"  Win rate:   {win_rate:.1%} ({wins}/{self.eval_episodes})")
            print(f"  Valid rate: {valid_rate:.1%} ({valid_boards}/{self.eval_episodes})")
            print(f"  Avg reward: {avg_reward:.2f}")

        # Advance when win rate >= 65% AND valid rate >= 80%
        # Win rate threshold accounts for ~25% tie-only boards (2/8 size-2 boards)
        # 65% win rate = winning ~87% of beatable games
        if (win_rate >= self.win_rate_threshold and
            valid_rate >= 0.80 and
            self.current_phase < self.max_phase):
            self.current_phase += 1

            # Update training environments (works for both SubprocVecEnv and DummyVecEnv)
            try:
                self.training_env.env_method("set_curriculum_phase", self.current_phase)
            except Exception as e:
                if self.verbose >= 1:
                    print(f"  Warning: Could not update training envs: {e}")

            # Update dedicated eval env
            self._eval_env.set_curriculum_phase(self.current_phase)

            # Update EvalCallback's eval env so best_model reflects current phase
            if self.eval_callback_env is not None:
                try:
                    self.eval_callback_env.env_method("set_curriculum_phase", self.current_phase)
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"  Warning: Could not update eval env: {e}")

            # Save phase checkpoint
            phase_model_path = f"models/reverse_curriculum/phase_{self.current_phase-1}_checkpoint.zip"
            self.model.save(phase_model_path)

            if self.verbose >= 1:
                print(f"\n  PHASE ADVANCED: {self.current_phase-1} -> {self.current_phase}")
                print(f"  Checkpoint saved: {phase_model_path}")

        # Record phase history
        self.phase_history.append({
            'timestep': self.n_calls,
            'phase': self.current_phase,
            'win_rate': win_rate,
            'valid_rate': valid_rate,
            'avg_reward': avg_reward,
        })

        if self.verbose >= 1:
            print(f"{'='*70}\n")

    def _on_training_end(self) -> None:
        """Save phase progression history."""
        history_path = "models/reverse_curriculum/phase_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.phase_history, f, indent=2)

        if self.verbose >= 1:
            print(f"\nPhase history saved to: {history_path}")


def make_env(rank: int, seed: int = 0, start_phase: int = 0, stage1_model_path: Optional[str] = None):
    """Create a single environment instance with action masking."""
    def _init():
        env = ReverseCurriculumBuilderEnv(
            board_size=2,
            board_library_path="new_boards_2.json",
            stage1_model_path=stage1_model_path,
            curriculum_phase=start_phase,
            opponent_strategy="random",
            show_opponent_board=True,
            max_construction_steps=20,
        )
        env.reset(seed=seed + rank)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def train(
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    start_phase: int = 0,
    eval_freq: int = 1000,
    save_freq: int = 10_000,
    stage1_model_path: str = "models/construction/best/best_model.zip",
):
    """Train MaskablePPO agent for reverse curriculum board building."""
    print("=" * 70)
    print("REVERSE CURRICULUM BOARD BUILDING (Stage 2) - MaskablePPO")
    print("=" * 70)
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Starting phase:    {start_phase}")
    print(f"Eval frequency:    {eval_freq:,} steps")
    print(f"Save frequency:    {save_freq:,} steps")
    print(f"Stage 1 model:     {stage1_model_path}")

    # Check if Stage 1 model exists
    if not Path(stage1_model_path).exists():
        print(f"\nWARNING: Stage 1 model not found: {stage1_model_path}")
        print("   Will use random base board selection (suboptimal)")
        stage1_model_path = None
    else:
        print(f"   Stage 1 model will be loaded in each environment")

    print(f"\nCurriculum Strategy:")
    print(f"  - Phase 0: Place goal only (1 move)")
    print(f"  - Phase 1: Place last move + goal (2 moves)")
    print(f"  - Phase N: Place last N moves + goal")
    print(f"  - Auto-advance at 80%+ win rate")
    print(f"  - Action masking enforces valid placements")
    print("=" * 70)

    # Create directories
    os.makedirs("models/reverse_curriculum", exist_ok=True)
    os.makedirs("logs/reverse_curriculum", exist_ok=True)
    os.makedirs("eval/reverse_curriculum", exist_ok=True)

    # Create vectorized training environment
    print("\nCreating training environments...")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, start_phase=start_phase, stage1_model_path=stage1_model_path)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, start_phase=start_phase, stage1_model_path=stage1_model_path)
        ])

    # Create evaluation environment (single, deterministic)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(rank=1000, seed=42, start_phase=start_phase, stage1_model_path=stage1_model_path)
    ])

    # Callbacks
    phase_callback = PhaseProgressionCallback(
        eval_freq=eval_freq,
        eval_episodes=20,
        win_rate_threshold=0.65,
        max_phase=10,
        board_size=2,
        board_library_path="new_boards_2.json",
        stage1_model_path=stage1_model_path,
        eval_callback_env=eval_env,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path="models/reverse_curriculum",
        name_prefix="ppo_reverse_curriculum",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="models/reverse_curriculum/best",
        log_path="eval/reverse_curriculum",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Create MaskablePPO agent
    print("\nInitializing MaskablePPO agent...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/reverse_curriculum",
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
    )

    # Train
    print("\nStarting training...")
    print("Monitor progress with: tensorboard --logdir logs/reverse_curriculum/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[phase_callback, checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = "models/reverse_curriculum/ppo_reverse_curriculum_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: models/reverse_curriculum/best/best_model.zip")
    print(f"Final phase reached: {phase_callback.current_phase}")
    print(f"\nPhase checkpoints saved in: models/reverse_curriculum/")
    print(f"Phase history saved to: models/reverse_curriculum/phase_history.json")
    print("=" * 70)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train reverse curriculum board builder")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200,000)",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--start-phase",
        type=int,
        default=0,
        help="Starting curriculum phase (default: 0 = easiest)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Phase evaluation frequency in timesteps (default: 1,000)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency in timesteps (default: 10,000)",
    )
    parser.add_argument(
        "--stage1-model",
        type=str,
        default="models/construction/best/best_model.zip",
        help="Path to Stage 1 model (default: models/construction/best/best_model.zip)",
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        start_phase=args.start_phase,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        stage1_model_path=args.stage1_model,
    )
