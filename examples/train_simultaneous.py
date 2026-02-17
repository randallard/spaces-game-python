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
    python examples/train_simultaneous.py --size 4 --board-library new_boards_4.json
    python examples/train_simultaneous.py --size 2 --board-pools boards/size2/simple.json,boards/size2/one_trap.json

Monitor with:
    tensorboard --logdir logs/size{N}_stage3/
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from spaces_game import SimultaneousPlayEnv


# Legacy ordering for pool files without numeric prefixes.
LEGACY_POOL_ORDER = ["simple", "one_trap", "super_move", "super_move_counter"]


def discover_pools(board_size: int) -> List[str]:
    """
    Discover all .json board pool files in boards/sizeN/.

    Sorting rules:
    - Files starting with a digit (e.g. 00_simple.json) sort by their numeric
      prefix.
    - Files without a numeric prefix sort by LEGACY_POOL_ORDER, with unknown
      names appended alphabetically after.

    Returns list of paths sorted in curriculum order.
    """
    pool_dir = Path(f"boards/size{board_size}")
    if not pool_dir.is_dir():
        return []

    numbered = []   # (number, path)
    legacy = []     # (order_index, path)
    unknown = []    # (stem, path)

    for p in pool_dir.glob("*.json"):
        match = re.match(r'^(\d+)', p.stem)
        if match:
            numbered.append((int(match.group(1)), str(p)))
        elif p.stem in LEGACY_POOL_ORDER:
            legacy.append((LEGACY_POOL_ORDER.index(p.stem), str(p)))
        else:
            unknown.append((p.stem, str(p)))

    # If any files have numeric prefixes, use that ordering for all numbered
    # files, then append any non-numbered legacy/unknown files after.
    numbered.sort(key=lambda x: x[0])
    legacy.sort(key=lambda x: x[0])
    unknown.sort(key=lambda x: x[0])

    pools = [path for _, path in numbered]
    pools += [path for _, path in legacy]
    pools += [path for _, path in unknown]
    return pools


def build_phase_map(num_pools: int) -> Dict[int, List[int]]:
    """
    Build a progressive opponent phase map for the given number of pools.

    Pattern:
    - Phase 0: pool 0 solo
    - Phase 1: pool 1 solo
    - Phase 2: pools 0+1 mixed
    - Phase 3: pool 2 solo
    - Phase 4: pools 0+1+2 mixed
    - ...
    - Final phase: all pools mixed

    For each pool after the first, we add two phases: solo then cumulative mix.
    Single pool gets just one phase.
    """
    if num_pools == 0:
        return {0: [0]}

    if num_pools == 1:
        return {0: [0]}

    phase_map: Dict[int, List[int]] = {}
    phase = 0

    # Phase 0: first pool solo
    phase_map[phase] = [0]
    phase += 1

    # For each subsequent pool: solo phase, then cumulative mix
    for i in range(1, num_pools):
        # Solo phase for this pool
        phase_map[phase] = [i]
        phase += 1
        # Cumulative mix of all pools seen so far
        phase_map[phase] = list(range(i + 1))
        phase += 1

    return phase_map


# Map opponent phase completions to difficulty checkpoint names.
# When the agent finishes training on phase N, save as the corresponding difficulty.
DIFFICULTY_CHECKPOINTS = {
    0: "beginner",       # Can build valid boards, beats simple opponents
    2: "intermediate",   # Handles traps and mixed opponents
}
# "expert" is saved at phase 5 completion OR at training end (whichever comes last)


def mask_fn(env: gym.Env) -> np.ndarray:
    """Get action masks from the environment."""
    return env.action_masks()


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


class SelfPlayCallback(BaseCallback):
    """Callback for self-play with block scheduling.

    Instead of per-round coin-flip mixing, uses dedicated blocks:
    - Self-play blocks: pure self-play (ratio=1.0) for block_steps
    - Pool recovery: pure pool opponents (ratio=0.0) for recovery_steps
      (triggered when pool win rate drops below min_pool_win_rate)

    After warmup_steps, every snapshot_freq steps:
    1. Save current model as a snapshot
    2. Prune oldest if pool exceeds pool_size
    3. Assign a random snapshot to each training env

    At block boundaries, evaluate against pool opponents. If win rate is
    above threshold, start another self-play block. If below, switch to
    pool recovery until the recovery period ends.

    Also saves skill-level milestone checkpoints based on eval win rate.
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
        self_play_ratio: float = 0.5,
        block_steps: int = 200_000,
        recovery_steps: int = 100_000,
        min_pool_win_rate: float = 0.60,
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
        self.self_play_ratio = self_play_ratio
        self.block_steps = block_steps
        self.recovery_steps = recovery_steps
        self.min_pool_win_rate = min_pool_win_rate
        self.recovery_win_rate = recovery_win_rate
        # Quality gate: default to midpoint of min_pool_win_rate and recovery_win_rate
        if snapshot_win_rate is not None:
            self.snapshot_win_rate = snapshot_win_rate
        else:
            self.snapshot_win_rate = (min_pool_win_rate + recovery_win_rate) / 2.0
        self.seed_model_path = seed_model_path
        self.snapshot_paths: List[str] = []
        self._warmup_complete = False
        self._last_snapshot_step = 0
        self._saved_tiers: set = set()
        self._best_win_rate = 0.0

        # Block scheduling state
        self._in_self_play_block = True  # Start with self-play after warmup
        self._block_start_step = 0  # n_calls when current block/recovery started
        self._block_count = 0
        self._recovery_count = 0

        # Create snapshot directory
        self.snapshot_dir = os.path.join(output_dir, "opponent_pool")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Seed pool with initial converged model (never pruned)
        self._seed_path: Optional[str] = None
        if seed_model_path and os.path.exists(seed_model_path):
            import shutil
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
            self._block_start_step = self.n_calls
            self._in_self_play_block = True
            if self.verbose >= 1:
                print(f"\n  SELF-PLAY: Warmup complete at {total_steps:,} steps")
                print(f"  SELF-PLAY: Starting self-play block 1 ({self.block_steps:,} steps)")
            self._take_snapshot()
            self._set_ratio(1.0)
            return True

        if self.n_calls - self._last_snapshot_step >= self.snapshot_freq:
            self._take_snapshot()

        # Check block boundaries
        self._check_block_transition()

        # Check for skill milestone checkpoints from phase callback's eval
        self._check_skill_milestones()

        return True

    def _check_block_transition(self):
        """Check if current block/recovery period has ended and transition."""
        steps_in_block = (self.n_calls - self._block_start_step) * self.n_envs

        if self._in_self_play_block:
            if steps_in_block >= self.block_steps:
                # Self-play block ended — check pool win rate
                pool_win_rate = self._get_latest_pool_win_rate()
                self._block_count += 1

                if pool_win_rate < self.min_pool_win_rate:
                    # Pool performance degraded — switch to recovery
                    self._in_self_play_block = False
                    self._block_start_step = self.n_calls
                    self._set_ratio(0.0)
                    self._recovery_count += 1
                    if self.verbose >= 1:
                        print(f"\n  SELF-PLAY: Block {self._block_count} done. "
                              f"Pool win rate {pool_win_rate:.1%} < {self.min_pool_win_rate:.1%}")
                        print(f"  SELF-PLAY: Starting pool recovery {self._recovery_count} "
                              f"({self.recovery_steps:,} steps)")
                else:
                    # Pool performance OK — start another self-play block
                    self._block_start_step = self.n_calls
                    if self.verbose >= 1:
                        print(f"\n  SELF-PLAY: Block {self._block_count} done. "
                              f"Pool win rate {pool_win_rate:.1%} >= {self.min_pool_win_rate:.1%}")
                        print(f"  SELF-PLAY: Starting self-play block {self._block_count + 1} "
                              f"({self.block_steps:,} steps)")
        else:
            # In pool recovery — require minimum steps AND win rate threshold
            if steps_in_block >= self.recovery_steps:
                pool_win_rate = self._get_latest_pool_win_rate()
                if pool_win_rate >= self.recovery_win_rate:
                    # Recovery done — back to self-play
                    self._in_self_play_block = True
                    self._block_start_step = self.n_calls
                    self._set_ratio(1.0)
                    if self.verbose >= 1:
                        print(f"\n  SELF-PLAY: Recovery {self._recovery_count} done. "
                              f"Pool win rate {pool_win_rate:.1%} >= {self.recovery_win_rate:.1%}")
                        print(f"  SELF-PLAY: Starting self-play block {self._block_count + 1} "
                              f"({self.block_steps:,} steps)")
                elif self.verbose >= 1 and steps_in_block % self.recovery_steps == 0:
                    print(f"\n  SELF-PLAY: Recovery {self._recovery_count} extending. "
                          f"Pool win rate {pool_win_rate:.1%} < {self.recovery_win_rate:.1%}")

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
                mode = "SELF-PLAY" if self._in_self_play_block else "RECOVERY"
                print(f"  {mode}: Snapshot skipped — pool win rate "
                      f"{pool_win_rate:.1%} < {self.snapshot_win_rate:.1%}")
            # Still assign opponents from existing pool
            if self._in_self_play_block:
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
            mode = "SELF-PLAY" if self._in_self_play_block else "RECOVERY"
            print(f"  {mode}: Snapshot saved ({len(self.snapshot_paths)} in pool, "
                  f"win rate {pool_win_rate:.1%})")

        # Assign random snapshots to training envs (only during self-play blocks)
        if self._in_self_play_block:
            self._assign_opponents()

    def _assign_opponents(self):
        """Assign a random snapshot to each training env."""
        candidates = list(self.snapshot_paths)
        if self._seed_path and self._seed_path not in candidates:
            candidates.append(self._seed_path)
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
                import shutil
                shutil.copy2(src, dst)
                if self.verbose >= 1:
                    print(f"  Skill tier '{tier}' -> {dst}")

        # Always save current model as expert
        expert_path = os.path.join(diff_dir, "expert.zip")
        self.model.save(expert_path)

        if self.verbose >= 1:
            print(f"\nSelf-play: {len(self.snapshot_paths)} snapshots in pool")
            print(f"Skill milestones saved: {sorted(self._saved_tiers)}")


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
    board_library_path: Optional[str] = None,
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
    self_play_ratio: float = 0.5,
    win_rate_threshold: float = 0.70,
    self_play_block_steps: int = 200_000,
    pool_recovery_steps: int = 100_000,
    min_pool_win_rate: float = 0.60,
    recovery_win_rate: float = 0.70,
    snapshot_win_rate: Optional[float] = None,
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

    if board_library_path is not None:
        print(f"WARNING: --board-library is deprecated (strict masking makes scaffolding unnecessary)")

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
        print(f"\nSelf-play enabled (block scheduling):")
        print(f"  Snapshot freq:   {snapshot_freq:,} steps")
        print(f"  Pool size:       {pool_size}")
        print(f"  Warmup steps:    {warmup_steps:,}")
        print(f"  Block steps:     {self_play_block_steps:,} (pure self-play per block)")
        print(f"  Recovery steps:  {pool_recovery_steps:,} (min pool-only before re-eval)")
        print(f"  Min pool WR:     {min_pool_win_rate:.0%} (triggers recovery)")
        print(f"  Recovery WR:     {recovery_win_rate:.0%} (required to exit recovery)")
        effective_snap_wr = snapshot_win_rate if snapshot_win_rate is not None else (min_pool_win_rate + recovery_win_rate) / 2.0
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

    if self_play:
        self_play_callback = SelfPlayCallback(
            output_dir=output_dir,
            snapshot_freq=snapshot_freq,
            pool_size=pool_size,
            warmup_steps=warmup_steps,
            n_envs=n_envs,
            phase_callback=phase_callback,
            self_play_ratio=self_play_ratio,
            block_steps=self_play_block_steps,
            recovery_steps=pool_recovery_steps,
            min_pool_win_rate=min_pool_win_rate,
            recovery_win_rate=recovery_win_rate,
            snapshot_win_rate=snapshot_win_rate,
            seed_model_path=resume_from,
            verbose=1,
        )
        callbacks.append(self_play_callback)

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
        "--board-library", type=str, default=None,
        help="Board library JSON for construction scaffolding (e.g. new_boards_3.json)",
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
        "--min-phase-steps", type=int, default=10_000,
        help="Minimum steps per curriculum phase before advancing (default: 10,000)",
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
        "--n-steps", type=int, default=2048,
        help="Total rollout steps across all envs (default: 2048, try 4096-8192 for larger boards)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
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
        help="Enable self-play with rolling opponent pool",
    )
    parser.add_argument(
        "--snapshot-freq", type=int, default=50_000,
        help="Steps between self-play snapshots (default: 50,000)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=10,
        help="Max self-play snapshots to keep (default: 10)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100_000,
        help="Steps before self-play activates (default: 100,000)",
    )
    parser.add_argument(
        "--self-play-ratio", type=float, default=0.5,
        help="Fraction of rounds using self-play opponent vs pool (default: 0.5)",
    )
    parser.add_argument(
        "--win-rate-threshold", type=float, default=0.70,
        help="Win rate required to advance curriculum phase (default: 0.70, try 0.55 for size 2)",
    )
    parser.add_argument(
        "--self-play-block-steps", type=int, default=200_000,
        help="Steps per pure self-play block (default: 200,000)",
    )
    parser.add_argument(
        "--pool-recovery-steps", type=int, default=100_000,
        help="Steps of pool-only training when pool win rate drops (default: 100,000)",
    )
    parser.add_argument(
        "--min-pool-win-rate", type=float, default=0.60,
        help="Pool win rate threshold to trigger recovery (default: 0.60)",
    )
    parser.add_argument(
        "--recovery-win-rate", type=float, default=0.70,
        help="Pool win rate required to exit recovery and resume self-play (default: 0.70)",
    )
    parser.add_argument(
        "--snapshot-win-rate", type=float, default=None,
        help="Pool win rate required to save a snapshot (default: midpoint of min-pool-win-rate and recovery-win-rate)",
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
        board_library_path=args.board_library,
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
        self_play_ratio=args.self_play_ratio,
        win_rate_threshold=args.win_rate_threshold,
        self_play_block_steps=args.self_play_block_steps,
        pool_recovery_steps=args.pool_recovery_steps,
        min_pool_win_rate=args.min_pool_win_rate,
        recovery_win_rate=args.recovery_win_rate,
        snapshot_win_rate=args.snapshot_win_rate,
    )
