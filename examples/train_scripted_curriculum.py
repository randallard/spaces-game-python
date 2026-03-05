"""
Train an agent through scripted opponent levels, then transition to self-play.

Phase 1 — Scripted curriculum (levels 1-5):
  The agent plays 5-round fog games against scripted opponents of increasing
  difficulty. Each level gates at a configurable win rate (default 45%).

Phase 2 — Self-play:
  After clearing all 5 scripted levels, the agent's policy seeds the self-play
  pool. Progressive window self-play proceeds as in train_self_play.py.

This approach gives the agent structured exposure to opponent archetypes
(simple paths, reactive switching, traps, supermoves) before the open-ended
generalization pressure of self-play.

Usage:
    python examples/train_scripted_curriculum.py --size 2 --timesteps 5M
    python examples/train_scripted_curriculum.py --size 2 --timesteps 5M --discord-webhook URL

Monitor with:
    tensorboard --logdir logs/size{N}_scripted/
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
    ScriptedCurriculumCallback,
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
    """Find the simple pool file for a given board size."""
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
    board_size: int = 2,
    total_timesteps: int = 5_000_000,
    n_envs: int = 4,
    eval_freq: int = 2000,
    save_freq: int = 10_000,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.05,
    n_steps: int = 2048,
    batch_size: int = 64,
    use_fog: bool = True,
    # Scripted curriculum
    scripted_advance: float = 0.45,
    scripted_min_steps: int = 10_000,
    # Self-play (after scripted levels cleared)
    sp_advance: float = 0.55,
    sp_backtrack: float = 0.40,
    sp_min_steps: int = 50_000,
    snapshot_freq: int = 50_000,
    pool_size: int = 10,
    snapshot_win_rate: float = 0.30,
    recovery_win_rate: float = 0.55,
    output_dir: Optional[str] = None,
    discord_webhook: Optional[str] = None,
    discord_check_in: int = 30,
):
    """Train MaskablePPO: scripted curriculum then self-play."""
    simple_pool = find_simple_pool(board_size)
    opponent_pools = [simple_pool]
    phase_map = {0: [0]}

    if output_dir is None:
        output_dir = f"models/size{board_size}/scripted"

    max_construction_steps = board_size * 10

    print("=" * 70)
    print(f"SCRIPTED CURRICULUM -> SELF-PLAY -- Size {board_size}")
    print("=" * 70)
    print(f"Board size:        {board_size}x{board_size}")
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Fog of war:        {'ENABLED' if use_fog else 'DISABLED'}")
    print(f"Learning rate:     {learning_rate}")
    print(f"Entropy coeff:     {ent_coef}")
    print(f"N steps (total):   {n_steps} ({n_steps // n_envs} per env)")
    print(f"Batch size:        {batch_size}")
    print(f"Output directory:  {output_dir}")
    print(f"\nFallback pool:     {simple_pool}")
    print(f"\nPhase 1 — Scripted curriculum (levels 1-5):")
    print(f"  Advance gate:    {scripted_advance:.0%} game WR")
    print(f"  Min steps/level: {scripted_min_steps:,}")
    print(f"\nPhase 2 — Self-play (after scripted levels cleared):")
    print(f"  Advance thresh:  {sp_advance:.0%}")
    print(f"  Backtrack thresh:{sp_backtrack:.0%}")
    print(f"  Min steps/level: {sp_min_steps:,}")
    print(f"  Snapshot freq:   {snapshot_freq:,}")
    print(f"  Pool size:       {pool_size}")
    print(f"  Snapshot WR:     {snapshot_win_rate:.0%} (quality gate)")
    print(f"  Recovery WR:     {recovery_win_rate:.0%}")
    print("=" * 70)

    # Create directories
    log_dir = f"logs/size{board_size}_scripted"
    eval_dir = f"eval/size{board_size}_scripted"
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

    # Create evaluation environment (pool opponents for best_model tracking)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(rank=1000, seed=42, **env_kwargs)
    ])

    # Self-play callback (dormant until scripted levels cleared)
    # warmup_steps set extremely high — ScriptedCurriculumCallback will lower it to 0
    self_play_callback = SelfPlayCurriculumCallback(
        output_dir=output_dir,
        snapshot_freq=snapshot_freq,
        pool_size=pool_size,
        warmup_steps=total_timesteps * 10,  # effectively infinite until activated
        n_envs=n_envs,
        phase_callback=None,  # will be set below
        advance_threshold=sp_advance,
        backtrack_threshold=sp_backtrack,
        min_steps_per_level=sp_min_steps,
        recovery_win_rate=recovery_win_rate,
        snapshot_win_rate=snapshot_win_rate,
        board_size=board_size,
        opponent_pools=opponent_pools,
        phase_map=phase_map,
        use_fog=use_fog,
        sp_eval_freq=eval_freq,
        verbose=1,
    )

    # Scripted curriculum callback
    scripted_callback = ScriptedCurriculumCallback(
        eval_freq=eval_freq,
        eval_episodes=20,
        advance_threshold=scripted_advance,
        min_steps_per_level=scripted_min_steps,
        board_size=board_size,
        opponent_pools=opponent_pools,
        phase_map=phase_map,
        use_fog=use_fog,
        output_dir=output_dir,
        self_play_callback=self_play_callback,
        n_envs=n_envs,
        verbose=1,
    )

    # Wire self-play callback to read win rate from scripted callback
    self_play_callback.phase_callback = scripted_callback

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=output_dir,
        name_prefix="ppo_scripted",
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

    callbacks = [scripted_callback, self_play_callback, checkpoint_callback, eval_callback]

    if discord_webhook:
        discord_callback = DiscordNotifierCallback(
            webhook_url=discord_webhook,
            board_size=board_size,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            use_fog=use_fog,
            self_play=True,
            check_in_minutes=discord_check_in,
            phase_callback=scripted_callback,
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
    print("\nStarting scripted curriculum training...")
    print(f"Monitor progress with: tensorboard --logdir {log_dir}/")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = f"{output_dir}/ppo_scripted_final.zip"
    model.save(final_path)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {output_dir}/best/best_model.zip")
    if scripted_callback._all_levels_cleared:
        print(f"Scripted levels:   All 5 cleared")
        print(f"Max SP level:      {self_play_callback.max_level}")
        print(f"SP snapshots:      {len(self_play_callback.snapshot_paths)}")
    else:
        print(f"Scripted level:    {scripted_callback.current_scripted_level} "
              f"(did not clear all 5)")
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
        description="Train agent: scripted opponent curriculum then self-play",
    )
    parser.add_argument(
        "--size", type=int, default=2,
        help="Board size (default: 2)",
    )
    parser.add_argument(
        "--timesteps", type=human_int, default=5_000_000,
        help="Total training timesteps (default: 5M). Supports k/M/B suffixes.",
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

    # Scripted curriculum
    parser.add_argument(
        "--scripted-advance", type=float, default=0.45,
        help="Win rate to advance scripted level (default: 0.45)",
    )
    parser.add_argument(
        "--scripted-min-steps", type=human_int, default=10_000,
        help="Min steps before advancing scripted level (default: 10k)",
    )

    # Self-play
    parser.add_argument(
        "--sp-advance", type=float, default=0.55,
        help="Self-play: win rate to advance window level (default: 0.55)",
    )
    parser.add_argument(
        "--sp-backtrack", type=float, default=0.40,
        help="Self-play: win rate to backtrack a level (default: 0.40)",
    )
    parser.add_argument(
        "--sp-min-steps", type=human_int, default=50_000,
        help="Self-play: min steps before advancing level (default: 50k)",
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
        "--snapshot-win-rate", type=float, default=0.30,
        help="Quality gate for self-play snapshots (default: 0.30)",
    )
    parser.add_argument(
        "--recovery-win-rate", type=float, default=0.55,
        help="Win rate to exit self-play recovery (default: 0.55)",
    )

    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: models/size{N}/scripted)",
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
        scripted_advance=args.scripted_advance,
        scripted_min_steps=args.scripted_min_steps,
        sp_advance=args.sp_advance,
        sp_backtrack=args.sp_backtrack,
        sp_min_steps=args.sp_min_steps,
        snapshot_freq=args.snapshot_freq,
        pool_size=args.pool_size,
        snapshot_win_rate=args.snapshot_win_rate,
        recovery_win_rate=args.recovery_win_rate,
        output_dir=args.output_dir,
        discord_webhook=args.discord_webhook,
        discord_check_in=args.discord_check_in,
    )
