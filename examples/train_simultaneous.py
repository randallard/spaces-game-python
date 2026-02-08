"""
Train an agent for simultaneous 5-round play (Stage 3).

The agent constructs a board each round without seeing the opponent's board.
After simulation, the opponent's full board is revealed. The agent must learn
to adapt across rounds based on opponent patterns.

Progressive opponent curriculum:
  - Phase 0: Simple boards only (straight paths, no traps)
  - Phase 1: One-trap boards
  - Phase 2: Simple + one-trap mixed
  - Phase 3: Supermove boards
  - Phase 4: All board types mixed

Usage:
    python examples/train_simultaneous.py --size 2
    python examples/train_simultaneous.py --size 2 --timesteps 500000
    python examples/train_simultaneous.py --size 2 --board-pools boards/size2/simple.json,boards/size2/one_trap.json

Monitor with:
    tensorboard --logdir logs/size{N}_stage3/
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from spaces_game import SimultaneousPlayEnv


def mask_fn(env: gym.Env) -> np.ndarray:
    """Get action masks from the environment."""
    return env.action_masks()


class OpponentProgressionCallback(BaseCallback):
    """
    Callback to advance opponent difficulty based on game win rate.

    Evaluates agent performance every N steps using a dedicated single
    environment. Advances opponent phase when game win rate exceeds threshold.
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
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold
        self.valid_rate_threshold = valid_rate_threshold
        self.min_steps_per_phase = min_steps_per_phase
        self.max_phase = max_phase
        self.current_phase = 0
        self.phase_history = []
        self.eval_callback_env = eval_callback_env
        self.output_dir = output_dir
        self._phase_start_step = 0

        # Dedicated single env for evaluation
        self._eval_env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            opponent_phase=0,
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
        self.logger.record("opponent/phase", self.current_phase)
        self.logger.record("opponent/game_win_rate", game_win_rate)
        self.logger.record("opponent/valid_rate", valid_rate)
        self.logger.record("opponent/avg_reward", avg_reward)

        if self.verbose >= 1:
            ties = self.eval_episodes - game_wins - game_losses
            print(f"  Game wins:  {game_win_rate:.1%} ({game_wins}W/{game_losses}L/{ties}T)")
            print(f"  Valid rate: {valid_rate:.1%} ({total_rounds_valid}/{total_rounds} rounds)")
            print(f"  Avg reward: {avg_reward:.2f}")

        # Advance phase
        steps_at_phase = self.n_calls - self._phase_start_step
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
            ckpt_path = f"{self.output_dir}/phase_{self.current_phase-1}_checkpoint.zip"
            self.model.save(ckpt_path)

            if self.verbose >= 1:
                print(f"\n  PHASE ADVANCED: {self.current_phase-1} -> {self.current_phase}")
                print(f"  Checkpoint saved: {ckpt_path}")

        self.phase_history.append({
            "timestep": self.n_calls,
            "phase": self.current_phase,
            "game_win_rate": game_win_rate,
            "valid_rate": valid_rate,
            "avg_reward": avg_reward,
        })

        if self.verbose >= 1:
            print(f"{'='*70}\n")

    def _on_training_end(self) -> None:
        history_path = f"{self.output_dir}/phase_history.json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(self.phase_history, f, indent=2)
        if self.verbose >= 1:
            print(f"\nPhase history saved to: {history_path}")


def make_env(
    rank: int,
    seed: int = 0,
    board_size: int = 2,
    opponent_pools: Optional[List[str]] = None,
    opponent_phase: int = 0,
    max_construction_steps: int = 20,
):
    """Create a single environment instance with action masking."""
    def _init():
        env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            opponent_phase=opponent_phase,
            max_construction_steps=max_construction_steps,
        )
        env.reset(seed=seed + rank)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def train(
    board_size: int = 2,
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    eval_freq: int = 2000,
    save_freq: int = 10_000,
    opponent_pools: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Train MaskablePPO agent for simultaneous 5-round play."""
    # Defaults
    if opponent_pools is None:
        base = f"boards/size{board_size}"
        opponent_pools = [f"{base}/simple.json"]
        # Auto-discover additional pools
        for name in ["one_trap", "super_move", "super_move_counter"]:
            path = f"{base}/{name}.json"
            if Path(path).exists():
                opponent_pools.append(path)

    if output_dir is None:
        output_dir = f"models/size{board_size}/stage3"

    max_construction_steps = board_size * 10
    max_phase = min(len(opponent_pools) + 1, len(DEFAULT_PHASE_MAP))

    print("=" * 70)
    print("SIMULTANEOUS 5-ROUND PLAY (Stage 3) - MaskablePPO")
    print("=" * 70)
    print(f"Board size:        {board_size}x{board_size}")
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Eval frequency:    {eval_freq:,} steps")
    print(f"Save frequency:    {save_freq:,} steps")
    print(f"Max steps/round:   {max_construction_steps}")
    print(f"Output directory:  {output_dir}")
    print(f"\nOpponent pools ({len(opponent_pools)}):")
    for i, p in enumerate(opponent_pools):
        print(f"  [{i}] {p}")
    print(f"\nProgressive opponent phases (max {max_phase}):")
    for phase in range(max_phase + 1):
        pool_indices = DEFAULT_PHASE_MAP.get(phase, list(range(len(opponent_pools))))
        active = [opponent_pools[i] for i in pool_indices if i < len(opponent_pools)]
        names = [Path(p).stem for p in active]
        print(f"  Phase {phase}: {', '.join(names)}")
    if resume_from:
        print(f"\nResuming from:     {resume_from}")
    print("=" * 70)

    # Validate pools exist
    for path in opponent_pools:
        if not Path(path).exists():
            print(f"\nERROR: Board pool not found: {path}")
            return

    # Create directories
    log_dir = f"logs/size{board_size}_stage3"
    eval_dir = f"eval/size{board_size}_stage3"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Common env kwargs
    env_kwargs = dict(
        board_size=board_size,
        opponent_pools=opponent_pools,
        opponent_phase=0,
        max_construction_steps=max_construction_steps,
    )

    # Create training environments
    print("\nCreating training environments...")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, **env_kwargs) for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([make_env(0, **env_kwargs)])

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(rank=1000, seed=42, **env_kwargs)
    ])

    # Callbacks
    phase_callback = OpponentProgressionCallback(
        eval_freq=eval_freq,
        eval_episodes=20,
        win_rate_threshold=0.70,
        valid_rate_threshold=0.90,
        min_steps_per_phase=10000,
        max_phase=max_phase,
        board_size=board_size,
        opponent_pools=opponent_pools,
        eval_callback_env=eval_env,
        output_dir=output_dir,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=output_dir,
        name_prefix="ppo_stage3",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best",
        log_path=eval_dir,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Create or load model
    if resume_from and Path(resume_from).exists():
        print(f"\nResuming from: {resume_from}")
        model = MaskablePPO.load(resume_from, env=env)
        model.learning_rate = 3e-4
        model.ent_coef = 0.1
    else:
        print("\nInitializing MaskablePPO agent...")
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048 // n_envs,  # 512 per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
        )

    # Train
    print("\nStarting training...")
    print(f"Monitor progress with: tensorboard --logdir {log_dir}/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[phase_callback, checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{output_dir}/ppo_stage3_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir}/best/best_model.zip")
    print(f"Final opponent phase: {phase_callback.current_phase}")
    print(f"\nPhase history: {output_dir}/phase_history.json")
    print("=" * 70)

    env.close()
    eval_env.close()


# Import DEFAULT_PHASE_MAP for display
from spaces_game.simultaneous_play_env import DEFAULT_PHASE_MAP


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train simultaneous 5-round play agent (Stage 3)",
    )
    parser.add_argument(
        "--size", type=int, default=2,
        help="Board size (default: 2)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training timesteps (default: 200,000)",
    )
    parser.add_argument(
        "--envs", type=int, default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=2000,
        help="Evaluation frequency in timesteps (default: 2,000)",
    )
    parser.add_argument(
        "--save-freq", type=int, default=10_000,
        help="Checkpoint save frequency (default: 10,000)",
    )
    parser.add_argument(
        "--board-pools", type=str, default=None,
        help="Comma-separated board pool JSON paths",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from existing model",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: models/size{N}/stage3)",
    )

    args = parser.parse_args()

    pools = None
    if args.board_pools:
        pools = [p.strip() for p in args.board_pools.split(",")]

    train(
        board_size=args.size,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        opponent_pools=pools,
        resume_from=args.resume,
        output_dir=args.output_dir,
    )
