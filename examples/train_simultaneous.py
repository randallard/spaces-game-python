"""
Train an agent for simultaneous 5-round play (Stage 3).

The agent constructs a board each round without seeing the opponent's board.
After simulation, the opponent's full board is revealed. The agent must learn
to adapt across rounds based on opponent patterns.

Progressive opponent curriculum is built dynamically from the board pool files
in boards/sizeN/. Files are ordered by numeric prefix if present (e.g.
00_simple.json, 01_one_trap.json), otherwise by a known legacy order
(simple, one_trap, super_move, super_move_counter), with unknown files last.

Phases progress: each pool solo, then cumulative mixes, then all pools.

Usage:
    python examples/train_simultaneous.py --size 2
    python examples/train_simultaneous.py --size 2 --timesteps 500000
    python examples/train_simultaneous.py --size 2 --self-play --timesteps 2000000

Monitor with:
    tensorboard --logdir logs/size{N}_stage3/
"""

import os
import sys
import warnings
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
    discover_pools,
    build_phase_map,
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
    use_fog: bool = False,
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


def train(
    board_size: int = 2,
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    eval_freq: int = 2000,
    save_freq: int = 10_000,
    opponent_pools: Optional[List[str]] = None,
    resume_from: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_phase_steps: int = 10_000,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.05,
    n_steps: int = 2048,
    batch_size: int = 64,
    start_opponent_phase: Optional[int] = None,
    use_fog: bool = False,
    self_play: bool = False,
    snapshot_freq: int = 50_000,
    pool_size: int = 10,
    warmup_steps: int = 100_000,
    win_rate_threshold: float = 0.70,
    advance_threshold: float = 0.70,
    backtrack_threshold: float = 0.55,
    min_steps_per_level: int = 50_000,
    recovery_win_rate: float = 0.70,
    snapshot_win_rate: Optional[float] = None,
    discord_webhook: Optional[str] = None,
    discord_check_in: int = 30,
):
    """Train MaskablePPO agent for simultaneous 5-round play."""
    # Defaults — auto-discover pools from boards/sizeN/
    if opponent_pools is None:
        opponent_pools = discover_pools(board_size)
        if not opponent_pools:
            print(f"ERROR: No board pools found in boards/size{board_size}/")
            return

    stage = "stage4" if use_fog else "stage3"
    if output_dir is None:
        output_dir = f"models/size{board_size}/{stage}"

    max_construction_steps = board_size * 10
    phase_map = build_phase_map(len(opponent_pools))
    max_phase = max(phase_map.keys())

    print("=" * 70)
    stage_label = "Stage 4 - FOG OF WAR" if use_fog else "Stage 3"
    print(f"SIMULTANEOUS 5-ROUND PLAY ({stage_label}) - MaskablePPO")
    print("=" * 70)
    print(f"Board size:        {board_size}x{board_size}")
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Eval frequency:    {eval_freq:,} steps")
    print(f"Save frequency:    {save_freq:,} steps")
    print(f"Max steps/round:   {max_construction_steps}")
    print(f"Min phase steps:   {min_phase_steps:,}")
    print(f"Learning rate:     {learning_rate}")
    print(f"Entropy coeff:     {ent_coef}")
    print(f"N steps (total):   {n_steps} ({n_steps // n_envs} per env)")
    print(f"Batch size:        {batch_size}")
    print(f"Output directory:  {output_dir}")
    print(f"\nOpponent pools ({len(opponent_pools)}):")
    for i, p in enumerate(opponent_pools):
        print(f"  [{i}] {p}")
    print(f"\nProgressive opponent phases (max {max_phase}):")
    for phase in range(max_phase + 1):
        pool_indices = phase_map.get(phase, list(range(len(opponent_pools))))
        active = [opponent_pools[i] for i in pool_indices if i < len(opponent_pools)]
        names = [Path(p).stem for p in active]
        print(f"  Phase {phase}: {', '.join(names)}")
    if use_fog:
        print(f"\nFog of war:        ENABLED (partial opponent reveal)")
    if self_play:
        print(f"\nSelf-play enabled (progressive window curriculum):")
        print(f"  Snapshot freq:   {snapshot_freq:,} steps")
        print(f"  Pool size:       {pool_size}")
        print(f"  Warmup steps:    {warmup_steps:,}")
        print(f"  Advance thresh:  {advance_threshold:.0%} (win rate to advance level)")
        print(f"  Backtrack thresh:{backtrack_threshold:.0%} (win rate to back up)")
        print(f"  Min steps/level: {min_steps_per_level:,}")
        print(f"  Recovery WR:     {recovery_win_rate:.0%} (required to exit recovery)")
        effective_snap_wr = snapshot_win_rate if snapshot_win_rate is not None else (backtrack_threshold + advance_threshold) / 2.0
        print(f"  Snapshot WR:     {effective_snap_wr:.0%} (quality gate for snapshots)")
        if resume_from:
            print(f"  Seed model:      {resume_from} (permanent pool member)")
    if resume_from:
        print(f"\nResuming from:     {resume_from}")
    print("=" * 70)

    # Validate pools exist
    for path in opponent_pools:
        if not Path(path).exists():
            print(f"\nERROR: Board pool not found: {path}")
            return

    # Create directories
    log_dir = f"logs/size{board_size}_{stage}"
    eval_dir = f"eval/size{board_size}_{stage}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Common env kwargs
    initial_opponent_phase = start_opponent_phase if start_opponent_phase is not None else 0
    env_kwargs = dict(
        board_size=board_size,
        opponent_pools=opponent_pools,
        opponent_phase=initial_opponent_phase,
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
        win_rate_threshold=win_rate_threshold,
        valid_rate_threshold=0.90,
        min_steps_per_phase=min_phase_steps,
        max_phase=max_phase,
        board_size=board_size,
        opponent_pools=opponent_pools,
        eval_callback_env=eval_env,
        output_dir=output_dir,
        phase_map=phase_map,
        start_opponent_phase=start_opponent_phase,
        use_fog=use_fog,
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
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
    else:
        print("\nInitializing MaskablePPO agent...")
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
    print("\nStarting training...")
    print(f"Monitor progress with: tensorboard --logdir {log_dir}/")
    print("=" * 70)

    callbacks = [phase_callback, checkpoint_callback, eval_callback]

    self_play_callback = None
    if self_play:
        self_play_callback = SelfPlayCurriculumCallback(
            output_dir=output_dir,
            snapshot_freq=snapshot_freq,
            pool_size=pool_size,
            warmup_steps=warmup_steps,
            n_envs=n_envs,
            phase_callback=phase_callback,
            advance_threshold=advance_threshold,
            backtrack_threshold=backtrack_threshold,
            min_steps_per_level=min_steps_per_level,
            recovery_win_rate=recovery_win_rate,
            snapshot_win_rate=snapshot_win_rate,
            seed_model_path=resume_from,
            board_size=board_size,
            opponent_pools=opponent_pools,
            phase_map=phase_map,
            use_fog=use_fog,
            sp_eval_freq=eval_freq,
            verbose=1,
        )
        callbacks.append(self_play_callback)
        phase_callback.self_play_callback = self_play_callback

    if discord_webhook:
        discord_callback = DiscordNotifierCallback(
            webhook_url=discord_webhook,
            board_size=board_size,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            use_fog=use_fog,
            self_play=self_play,
            check_in_minutes=discord_check_in,
            phase_callback=phase_callback,
            self_play_callback=self_play_callback,
            verbose=1,
        )
        callbacks.append(discord_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
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


def _check_deprecated_flags(args):
    """Warn about deprecated CLI flags."""
    deprecated = {
        "self_play_block_steps": "--self-play-block-steps (replaced by --min-steps-per-level)",
        "pool_recovery_steps": "--pool-recovery-steps (recovery is now automatic)",
        "min_pool_win_rate": "--min-pool-win-rate (replaced by --backtrack-threshold)",
        "self_play_ratio": "--self-play-ratio (replaced by progressive window levels)",
    }
    for attr, msg in deprecated.items():
        if getattr(args, attr, None) is not None:
            warnings.warn(f"Deprecated flag {msg}. It will be ignored.", DeprecationWarning, stacklevel=2)


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
        description="Train simultaneous 5-round play agent (Stage 3)",
    )
    parser.add_argument(
        "--size", type=int, default=2,
        help="Board size (default: 2)",
    )
    parser.add_argument(
        "--timesteps", type=human_int, default=200_000,
        help="Total training timesteps (default: 200k). Supports k/M/B suffixes.",
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
    parser.add_argument(
        "--min-phase-steps", type=human_int, default=10_000,
        help="Minimum steps per curriculum phase before advancing (default: 10k)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4,
        help="Learning rate (default: 3e-4, try 1e-4 for larger boards)",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.05,
        help="Entropy coefficient for exploration (default: 0.05, try 0.1 for larger boards)",
    )
    parser.add_argument(
        "--n-steps", type=human_int, default=2048,
        help="Total rollout steps across all envs (default: 2048, try 4k-8k for larger boards)",
    )
    parser.add_argument(
        "--batch-size", type=human_int, default=64,
        help="Minibatch size (default: 64)",
    )
    parser.add_argument(
        "--start-opponent-phase", type=int, default=None,
        help="Skip construction and start at this opponent phase (use with --resume)",
    )
    parser.add_argument(
        "--fog", action="store_true",
        help="Enable fog of war (Stage 4): partial opponent board reveal after simulation",
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="Enable self-play with progressive window curriculum",
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
        "--warmup-steps", type=human_int, default=100_000,
        help="Steps before self-play activates (default: 100k)",
    )
    parser.add_argument(
        "--win-rate-threshold", type=float, default=0.70,
        help="Win rate required to advance curriculum phase (default: 0.70)",
    )
    parser.add_argument(
        "--advance-threshold", type=float, default=0.70,
        help="Self-play: win rate to advance window level (default: 0.70)",
    )
    parser.add_argument(
        "--backtrack-threshold", type=float, default=0.55,
        help="Self-play: win rate to back up a level (default: 0.55)",
    )
    parser.add_argument(
        "--min-steps-per-level", type=human_int, default=50_000,
        help="Self-play: minimum steps before advancing level (default: 50k)",
    )
    parser.add_argument(
        "--recovery-win-rate", type=float, default=0.70,
        help="Pool win rate required to exit recovery and resume self-play (default: 0.70)",
    )
    parser.add_argument(
        "--snapshot-win-rate", type=float, default=None,
        help="Pool win rate required to save a snapshot (default: midpoint of backtrack and advance thresholds)",
    )

    # Discord notifications
    parser.add_argument(
        "--discord-webhook", type=str, default=None,
        help="Discord webhook URL for training notifications (disabled if not set)",
    )
    parser.add_argument(
        "--discord-check-in", type=human_int, default=30,
        help="Minutes between periodic Discord check-in messages (default: 30)",
    )

    # Deprecated flags — kept for backwards compatibility, ignored with warning
    parser.add_argument("--self-play-block-steps", type=human_int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pool-recovery-steps", type=human_int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--min-pool-win-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--self-play-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--board-library", type=str, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    _check_deprecated_flags(args)

    if args.board_library is not None:
        warnings.warn("--board-library is deprecated (strict masking makes scaffolding unnecessary)", DeprecationWarning)

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
        min_phase_steps=args.min_phase_steps,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        start_opponent_phase=args.start_opponent_phase,
        use_fog=args.fog,
        self_play=args.self_play,
        snapshot_freq=args.snapshot_freq,
        pool_size=args.pool_size,
        warmup_steps=args.warmup_steps,
        win_rate_threshold=args.win_rate_threshold,
        advance_threshold=args.advance_threshold,
        backtrack_threshold=args.backtrack_threshold,
        min_steps_per_level=args.min_steps_per_level,
        recovery_win_rate=args.recovery_win_rate,
        snapshot_win_rate=args.snapshot_win_rate,
        discord_webhook=args.discord_webhook,
        discord_check_in=args.discord_check_in,
    )
