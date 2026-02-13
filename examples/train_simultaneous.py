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
    Callback to advance training through two sequential curricula:

    1. Construction curriculum (if board_library_path provided):
       Pre-fills boards from library, gradually removing scaffolding.
       Advances on valid_rate >= threshold.

    2. Opponent curriculum (always):
       Advances opponent difficulty based on game win rate.
    """

    def __init__(
        self,
        eval_freq: int = 2000,
        eval_episodes: int = 20,
        win_rate_threshold: float = 0.70,
        valid_rate_threshold: float = 0.90,
        construction_valid_threshold: float = 0.95,
        min_steps_per_phase: int = 10000,
        max_phase: int = 4,
        board_size: int = 2,
        opponent_pools: Optional[List[str]] = None,
        board_library_path: Optional[str] = None,
        eval_callback_env: Optional[DummyVecEnv] = None,
        output_dir: str = "models/stage3",
        phase_map: Optional[Dict[int, List[int]]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.win_rate_threshold = win_rate_threshold
        self.valid_rate_threshold = valid_rate_threshold
        self.construction_valid_threshold = construction_valid_threshold
        self.min_steps_per_phase = min_steps_per_phase
        self.max_phase = max_phase
        self.current_phase = 0
        self.phase_history = []
        self.eval_callback_env = eval_callback_env
        self.output_dir = output_dir
        self._phase_start_step = 0

        # Construction curriculum state
        self.board_library_path = board_library_path
        self.construction_phase = 0  # current scaffolding phase
        self.max_construction_phase = 0  # computed from library
        self.in_construction_mode = False

        if board_library_path is not None:
            from spaces_game.board_loader import load_boards_from_json
            library = load_boards_from_json(board_library_path)
            if library:
                max_seq = max(len(b.sequence) for b in library)
                self.max_construction_phase = max_seq - 1  # -1: last phase = from scratch
                self.in_construction_mode = True

        # Dedicated single env for evaluation (must match training env's obs space)
        max_construction_steps = board_size * 10
        self._eval_env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            opponent_phase=0,
            board_library_path=board_library_path,
            max_construction_steps=max_construction_steps,
            phase_map=phase_map,
        )
        # Start eval env with scaffolding if in construction mode
        if self.in_construction_mode:
            self._eval_env.set_scaffolding(0)  # Phase 0: place goal only

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_maybe_advance()
        return True

    def _evaluate_and_maybe_advance(self):
        mode = "CONSTRUCTION" if self.in_construction_mode else "OPPONENT"
        phase = self.construction_phase if self.in_construction_mode else self.current_phase

        if self.verbose >= 1:
            print(f"\n{'='*70}")
            print(f"{mode} PHASE {phase} EVALUATION ({self.eval_episodes} games)")
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
        self.logger.record("curriculum/construction_phase", self.construction_phase)
        self.logger.record("curriculum/opponent_phase", self.current_phase)
        self.logger.record("curriculum/game_win_rate", game_win_rate)
        self.logger.record("curriculum/valid_rate", valid_rate)
        self.logger.record("curriculum/avg_reward", avg_reward)
        self.logger.record("curriculum/in_construction", int(self.in_construction_mode))

        if self.verbose >= 1:
            ties = self.eval_episodes - game_wins - game_losses
            print(f"  Game wins:  {game_win_rate:.1%} ({game_wins}W/{game_losses}L/{ties}T)")
            print(f"  Valid rate: {valid_rate:.1%} ({total_rounds_valid}/{total_rounds} rounds)")
            print(f"  Avg reward: {avg_reward:.2f}")

        # Advance phase (different logic for construction vs opponent mode)
        steps_at_phase = self.n_calls - self._phase_start_step

        if self.in_construction_mode:
            self._maybe_advance_construction(valid_rate, steps_at_phase)
        else:
            self._maybe_advance_opponent(game_win_rate, valid_rate, steps_at_phase)

        self.phase_history.append({
            "timestep": self.n_calls,
            "construction_phase": self.construction_phase,
            "opponent_phase": self.current_phase,
            "in_construction": self.in_construction_mode,
            "game_win_rate": game_win_rate,
            "valid_rate": valid_rate,
            "avg_reward": avg_reward,
        })

        if self.verbose >= 1:
            print(f"{'='*70}\n")

    def _maybe_advance_construction(self, valid_rate: float, steps_at_phase: int):
        """Advance construction scaffolding phase based on valid_rate."""
        if (valid_rate >= self.construction_valid_threshold and
                steps_at_phase >= self.min_steps_per_phase and
                self.construction_phase < self.max_construction_phase):
            self.construction_phase += 1
            self._phase_start_step = self.n_calls
            new_scaffolding = self.construction_phase

            # Update training envs
            try:
                self.training_env.env_method("set_scaffolding", new_scaffolding)
            except Exception as e:
                if self.verbose >= 1:
                    print(f"  Warning: Could not update training envs: {e}")

            # Update eval envs
            self._eval_env.set_scaffolding(new_scaffolding)
            if self.eval_callback_env is not None:
                try:
                    self.eval_callback_env.env_method(
                        "set_scaffolding", new_scaffolding,
                    )
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"  Warning: Could not update eval env: {e}")

            # Save checkpoint
            ckpt_path = f"{self.output_dir}/construction_phase_{self.construction_phase}_checkpoint.zip"
            self.model.save(ckpt_path)

            if self.verbose >= 1:
                print(f"\n  CONSTRUCTION PHASE ADVANCED: {self.construction_phase-1} -> {self.construction_phase}")
                print(f"  Scaffolding: remove {new_scaffolding} moves (+goal)")
                print(f"  Checkpoint saved: {ckpt_path}")

        # Check if construction curriculum is complete
        elif (valid_rate >= self.construction_valid_threshold and
                steps_at_phase >= self.min_steps_per_phase and
                self.construction_phase >= self.max_construction_phase):
            # Transition to opponent curriculum
            self.in_construction_mode = False
            self._phase_start_step = self.n_calls

            # Disable scaffolding
            try:
                self.training_env.env_method("set_scaffolding", -1)
            except Exception as e:
                if self.verbose >= 1:
                    print(f"  Warning: Could not update training envs: {e}")

            self._eval_env.set_scaffolding(-1)
            if self.eval_callback_env is not None:
                try:
                    self.eval_callback_env.env_method("set_scaffolding", -1)
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"  Warning: Could not update eval env: {e}")

            # Save checkpoint
            ckpt_path = f"{self.output_dir}/construction_complete_checkpoint.zip"
            self.model.save(ckpt_path)

            if self.verbose >= 1:
                print(f"\n  CONSTRUCTION CURRICULUM COMPLETE!")
                print(f"  Transitioning to opponent curriculum (phase 0)")
                print(f"  Checkpoint saved: {ckpt_path}")

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


def make_env(
    rank: int,
    seed: int = 0,
    board_size: int = 2,
    opponent_pools: Optional[List[str]] = None,
    opponent_phase: int = 0,
    max_construction_steps: int = 20,
    board_library_path: Optional[str] = None,
    scaffolding_moves_to_remove: int = -1,
    phase_map: Optional[Dict[int, List[int]]] = None,
):
    """Create a single environment instance with action masking."""
    def _init():
        env = SimultaneousPlayEnv(
            board_size=board_size,
            opponent_pools=opponent_pools,
            opponent_phase=opponent_phase,
            max_construction_steps=max_construction_steps,
            board_library_path=board_library_path,
            phase_map=phase_map,
        )
        if scaffolding_moves_to_remove >= 0:
            env.set_scaffolding(scaffolding_moves_to_remove)
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
):
    """Train MaskablePPO agent for simultaneous 5-round play."""
    # Defaults — auto-discover pools from boards/sizeN/
    if opponent_pools is None:
        opponent_pools = discover_pools(board_size)
        if not opponent_pools:
            print(f"ERROR: No board pools found in boards/size{board_size}/")
            return

    if output_dir is None:
        output_dir = f"models/size{board_size}/stage3"

    max_construction_steps = board_size * 10
    phase_map = build_phase_map(len(opponent_pools))
    max_phase = max(phase_map.keys())

    # Determine initial scaffolding
    initial_scaffolding = -1  # disabled by default
    if board_library_path is not None and Path(board_library_path).exists():
        initial_scaffolding = 0  # start at phase 0 (place goal only)

    print("=" * 70)
    print("SIMULTANEOUS 5-ROUND PLAY (Stage 3) - MaskablePPO")
    print("=" * 70)
    print(f"Board size:        {board_size}x{board_size}")
    print(f"Total timesteps:   {total_timesteps:,}")
    print(f"Parallel envs:     {n_envs}")
    print(f"Eval frequency:    {eval_freq:,} steps")
    print(f"Save frequency:    {save_freq:,} steps")
    print(f"Max steps/round:   {max_construction_steps}")
    print(f"Min phase steps:   {min_phase_steps:,}")
    print(f"Output directory:  {output_dir}")
    if board_library_path:
        print(f"Board library:     {board_library_path} (construction scaffolding)")
    print(f"\nOpponent pools ({len(opponent_pools)}):")
    for i, p in enumerate(opponent_pools):
        print(f"  [{i}] {p}")
    if board_library_path:
        print(f"\nConstruction curriculum: scaffold -> build from scratch -> opponent phases")
    print(f"\nProgressive opponent phases (max {max_phase}):")
    for phase in range(max_phase + 1):
        pool_indices = phase_map.get(phase, list(range(len(opponent_pools))))
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
        board_library_path=board_library_path,
        scaffolding_moves_to_remove=initial_scaffolding,
        phase_map=phase_map,
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
        construction_valid_threshold=0.95,
        min_steps_per_phase=min_phase_steps,
        max_phase=max_phase,
        board_size=board_size,
        opponent_pools=opponent_pools,
        board_library_path=board_library_path,
        eval_callback_env=eval_env,
        output_dir=output_dir,
        phase_map=phase_map,
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
    if board_library_path:
        print(f"Final construction phase: {phase_callback.construction_phase}/{phase_callback.max_construction_phase}")
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
    )
