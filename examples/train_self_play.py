"""
Train an agent with self-play from scratch (no pool pre-training).

Unlike train_simultaneous.py which trains pool-only first then adds self-play
via --resume, this script starts self-play immediately. This forces the model
to learn history-dependent play from the beginning — both the agent and its
self-play opponents see each other's boards and must adapt across rounds.

Only a single simple pool file is used as fallback (boards/sizeN/00_simple.json
or simple.json for legacy sizes). Fog of war is enabled by default.

Usage:
    python examples/train_self_play.py --size 3 --timesteps 10M
    python examples/train_self_play.py --size 5 --timesteps 20M --discord-webhook URL

Monitor with:
    tensorboard --logdir logs/size{N}_selfplay/
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from spaces_game import SimultaneousPlayEnv
from spaces_game.callbacks import (
    OpponentProgressionCallback,
    SelfPlayCurriculumCallback,
    DiscordNotifierCallback,
)


def mask_fn(env: gym.Env) -> np.ndarray:
    """Get action masks from the environment."""
    return env.action_masks()


def make_env(
    rank: int,
    seed: int = 0,
    board_size: int = 2,
    opponent_pools: Optional[List[str]] = None,
    opponent_phase: int = 0,
    max_construction_steps: int = 20,
    phase_map: Optional[Dict[int, List[int]]] = None,
    use_fog: bool = True,
):
    """Create a single environment instance with action masking."""
    def _init():
        env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            opponent_phase=opponent_phase,
            max_construction_steps=max_construction_steps,
            phase_map=phase_map,
            use_fog=use_fog,
        )
        env.reset(seed=seed + rank)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    return _init


def find_simple_pool(board_size: int) -> str:
    """Find the simple pool file for a given board size.

    Checks for 00_simple.json first (numbered convention), then simple.json (legacy).
    """
    pool_dir = Path(f"boards/size{board_size}")
    candidates = [
        pool_dir / "00_simple.json",
        pool_dir / "simple.json",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    print(f"ERROR: No simple pool found in {pool_dir}/")
    print(f"  Checked: {', '.join(str(c) for c in candidates)}")
    sys.exit(1)


def train(
    board_size: int = 3,
    total_timesteps: int = 10_000_000,
    n_envs: int = 4,
    eval_freq: int = 2000,
    save_freq: int = 10_000,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.05,
    n_steps: int = 2048,
    batch_size: int = 64,
    use_fog: bool = True,
    snapshot_freq: int = 50_000,
    pool_size: int = 10,
    advance_threshold: float = 0.70,
    backtrack_threshold: float = 0.55,
    min_steps_per_level: int = 50_000,
    recovery_win_rate: float = 0.70,
    snapshot_win_rate: float = 0.30,
    output_dir: Optional[str] = None,
    discord_webhook: Optional[str] = None,
    discord_check_in: int = 30,
):
    """Train MaskablePPO agent with self-play from scratch."""
    # Find single simple pool
    simple_pool = find_simple_pool(board_size)
    opponent_pools = [simple_pool]

    # Single phase: just the one pool file
    phase_map = {0: [0]}
    max_phase = 0

    fog_label = "fog" if use_fog else "nofog"
    if output_dir is None:
        output_dir = f"models/size{board_size}/selfplay"

    max_construction_steps = board_size * 10

    print("=" * 70)
    print(f"SELF-PLAY FROM SCRATCH — Size {board_size}")
    print("=" * 70)
    print(f"Board size:        {board_size}x{board_size}")
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Fog of war:        {'ENABLED' if use_fog else 'DISABLED'}")
    print(f"Eval frequency:    {eval_freq:,} steps")
    print(f"Save frequency:    {save_freq:,} steps")
    print(f"Max steps/round:   {max_construction_steps}")
    print(f"Learning rate:     {learning_rate}")
    print(f"Entropy coeff:     {ent_coef}")
    print(f"N steps (total):   {n_steps} ({n_steps // n_envs} per env)")
    print(f"Batch size:        {batch_size}")
    print(f"Output directory:  {output_dir}")
    print(f"\nFallback pool:     {simple_pool}")
    print(f"\nSelf-play (from step 0):")
    print(f"  Snapshot freq:   {snapshot_freq:,} steps")
    print(f"  Pool size:       {pool_size}")
    print(f"  Warmup steps:    0 (immediate self-play)")
    print(f"  Advance thresh:  {advance_threshold:.0%}")
    print(f"  Backtrack thresh:{backtrack_threshold:.0%}")
    print(f"  Min steps/level: {min_steps_per_level:,}")
    print(f"  Recovery WR:     {recovery_win_rate:.0%}")
    print(f"  Snapshot WR:     {snapshot_win_rate:.0%} (quality gate)")
    print("=" * 70)

    # Create directories
    log_dir = f"logs/size{board_size}_selfplay"
    eval_dir = f"eval/size{board_size}_selfplay"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Common env kwargs
    env_kwargs = dict(
        board_size=board_size,
        opponent_pools=opponent_pools,
        opponent_phase=0,
        max_construction_steps=max_construction_steps,
        phase_map=phase_map,
        use_fog=use_fog,
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
        min_steps_per_phase=10_000,
        max_phase=max_phase,
        board_size=board_size,
        opponent_pools=opponent_pools,
        eval_callback_env=eval_env,
        output_dir=output_dir,
        phase_map=phase_map,
        use_fog=use_fog,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=output_dir,
        name_prefix="ppo_selfplay",
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

    self_play_callback = SelfPlayCurriculumCallback(
        output_dir=output_dir,
        snapshot_freq=snapshot_freq,
        pool_size=pool_size,
        warmup_steps=0,
        n_envs=n_envs,
        phase_callback=phase_callback,
        advance_threshold=advance_threshold,
        backtrack_threshold=backtrack_threshold,
        min_steps_per_level=min_steps_per_level,
        recovery_win_rate=recovery_win_rate,
        snapshot_win_rate=snapshot_win_rate,
        board_size=board_size,
        opponent_pools=opponent_pools,
        phase_map=phase_map,
        use_fog=use_fog,
        sp_eval_freq=eval_freq,
        verbose=1,
    )

    callbacks = [phase_callback, checkpoint_callback, eval_callback, self_play_callback]
    phase_callback.self_play_callback = self_play_callback

    if discord_webhook:
        discord_callback = DiscordNotifierCallback(
            webhook_url=discord_webhook,
            board_size=board_size,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            use_fog=use_fog,
            self_play=True,
            check_in_minutes=discord_check_in,
            phase_callback=phase_callback,
            self_play_callback=self_play_callback,
            verbose=1,
        )
        callbacks.append(discord_callback)

    # Initialize model from scratch
    print("\nInitializing MaskablePPO agent (fresh, no pre-training)...")
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=learning_rate,
        n_steps=n_steps // n_envs,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
    )

    # Train
    print("\nStarting self-play training from scratch...")
    print(f"Monitor progress with: tensorboard --logdir {log_dir}/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = f"{output_dir}/ppo_selfplay_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir}/best/best_model.zip")
    print(f"Max self-play level: {self_play_callback.max_level}")
    print(f"Snapshots taken:     {len(self_play_callback.snapshot_paths)}")
    print("=" * 70)

    env.close()
    eval_env.close()


def human_int(value: str) -> int:
    """Parse human-readable integers: 7.5M, 200k, 50K, 10_000, etc."""
    value = value.strip().replace("_", "")
    suffixes = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if value and value[-1].lower() in suffixes:
        return int(float(value[:-1]) * suffixes[value[-1].lower()])
    return int(float(value))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train self-play agent from scratch (no pool pre-training)",
    )
    parser.add_argument(
        "--size", type=int, default=3,
        help="Board size (default: 3)",
    )
    parser.add_argument(
        "--timesteps", type=human_int, default=10_000_000,
        help="Total training timesteps (default: 10M). Supports k/M/B suffixes.",
    )
    parser.add_argument(
        "--envs", type=int, default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--eval-freq", type=human_int, default=2000,
        help="Evaluation frequency in timesteps (default: 2k)",
    )
    parser.add_argument(
        "--save-freq", type=human_int, default=10_000,
        help="Checkpoint save frequency (default: 10k)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.05,
        help="Entropy coefficient (default: 0.05)",
    )
    parser.add_argument(
        "--n-steps", type=human_int, default=2048,
        help="Total rollout steps across all envs (default: 2048)",
    )
    parser.add_argument(
        "--batch-size", type=human_int, default=64,
        help="Minibatch size (default: 64)",
    )
    parser.add_argument(
        "--no-fog", action="store_true",
        help="Disable fog of war (fog is enabled by default)",
    )
    parser.add_argument(
        "--snapshot-freq", type=human_int, default=50_000,
        help="Steps between self-play snapshots (default: 50k)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=10,
        help="Max self-play snapshots to keep (default: 10)",
    )
    parser.add_argument(
        "--advance-threshold", type=float, default=0.70,
        help="Win rate to advance window level (default: 0.70)",
    )
    parser.add_argument(
        "--backtrack-threshold", type=float, default=0.55,
        help="Win rate to back up a level (default: 0.55)",
    )
    parser.add_argument(
        "--min-steps-per-level", type=human_int, default=50_000,
        help="Minimum steps before advancing level (default: 50k)",
    )
    parser.add_argument(
        "--recovery-win-rate", type=float, default=0.70,
        help="Pool win rate required to exit recovery (default: 0.70)",
    )
    parser.add_argument(
        "--snapshot-win-rate", type=float, default=0.30,
        help="Quality gate for snapshots (default: 0.30, lower than pool-first training)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: models/size{N}/selfplay)",
    )
    parser.add_argument(
        "--discord-webhook", type=str, default=None,
        help="Discord webhook URL for training notifications",
    )
    parser.add_argument(
        "--discord-check-in", type=human_int, default=30,
        help="Minutes between periodic Discord check-in messages (default: 30)",
    )

    args = parser.parse_args()

    train(
        board_size=args.size,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        use_fog=not args.no_fog,
        snapshot_freq=args.snapshot_freq,
        pool_size=args.pool_size,
        advance_threshold=args.advance_threshold,
        backtrack_threshold=args.backtrack_threshold,
        min_steps_per_level=args.min_steps_per_level,
        recovery_win_rate=args.recovery_win_rate,
        snapshot_win_rate=args.snapshot_win_rate,
        output_dir=args.output_dir,
        discord_webhook=args.discord_webhook,
        discord_check_in=args.discord_check_in,
    )
