# Training Journal: Fog of War Implementation (Stage 4)

**Date:** February 16, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Stage 4 — fog of war (partial observability)

---

## How We Got Here

The path to fog of war was a series of hard-won lessons, each one stripping away a layer of complexity until the core architecture was solid enough to build on.

### The Foundation: Strict Masking Eliminates Invalid Boards (Feb 14)

For months, training was bottlenecked by invalid boards. The agent would waste millions of steps learning what a valid board looks like, only to still produce ~2% invalid boards at convergence. The breakthrough was strict BFS masking: a reachability check in `_is_valid_placement()` that makes invalid boards structurally impossible at the action mask level. The agent can't build an invalid board because every unmasked action keeps the board on a completable path.

This one change eliminated construction scaffolding, simplified the training pipeline, and let us focus entirely on gameplay strategy.

### The Action Space: Flat Discrete Replaces MultiDiscrete (Feb 14)

The MultiDiscrete action space `[cell, type, done]` masked each dimension independently. The agent could pick a valid cell and a valid type, but the combination was invalid — wasting 30-50% of steps on a 16-cell board. The fix was a flat `Discrete(2 * n_cells + 1)` space where every unmasked action is guaranteed valid. Zero wasted steps.

### The Reward Signal: Shaping Is Load-Bearing for Size 4+ (Feb 14)

We tried simplifying rewards to pure game outcomes (0.0 construction shaping). It worked for size 2-3 but failed for size 4 — the sparse signal can't credit-assign through 10+ construction steps on a 16-cell board. The original shaping rewards (+0.1 piece, +0.3 row 0, +0.2 supermove) turned out to be load-bearing, not noise. They guide the agent through the construction sequence while strict masking prevents the exploit loops that made shaping dangerous before.

### Self-Play: One Source of Non-Stationarity at a Time (Feb 14-15)

Self-play collapsed twice before we got it right:

1. **Warmup too short**: 100k steps wasn't enough for size 4. The first snapshot had only 14% valid rate, creating a death spiral where weak opponents produce weaker training signal.

2. **Value network mismatch**: Resuming with changed rewards meant the value network was calibrated to old reward scales. Negative explained variance cascaded into policy collapse.

3. **Specialization collapse**: Pure self-play caused the agent to co-evolve into a narrow niche, winning 100% against itself but 5% against pool opponents.

The fix was pool opponent mixing at 0.5 ratio — half the rounds use self-play opponents, half use JSON pool opponents. This maintains breadth while still getting the counter-play benefits of self-play. Size 4 converged to ~78% win rate with all 5 difficulty tiers saved.

### Where Size 4 Landed (Feb 15)

- 100% valid board rate (strict masking makes this structural, not learned)
- ~78% win rate with self-play mixing, asymptotic by 2.05M steps
- All opponent phases cleared
- 5 difficulty tiers saved (beginner through advanced_plus)

The agent is strong enough that pool opponents are exhausted. Self-play opponents have actual patterns in `opponent_history` for the agent to exploit and counter-exploit across rounds. The meta-game works — the agent clearly adapts its construction strategy based on what the opponent played in previous rounds.

But it's adapting with full information. It sees the opponent's entire board after each round. That's not how the real game works.

---

## What Fog of War Changes

In the real game, after simulation the agent only sees opponent moves up to the step where its own round ended. If the agent hit a trap at step 3, it sees opponent moves 0-3 and nothing beyond. Traps are invisible unless the agent actually landed on one (the "sprung" trap).

This changes the meta-game fundamentally:
- **Full reveal**: "I saw your entire board. I know you like traps at (0,1). I'll avoid that next round."
- **Fog of war**: "I got stopped at step 3. I know you had pieces at (1,0) and (0,0), but I have no idea where your traps were — except the one I hit."

The agent must now infer and adapt from partial information. The `opponent_history` grid still exists, but it's fog-filtered — it only shows what the agent actually observed.

---

## Implementation: Option A (Train from Scratch)

We chose to start with fog active from step 1. The agent never sees full opponent boards during training. The rationale: the agent shouldn't develop a dependency on full information that it then has to unlearn.

### What Changed in the Env

**`use_fog` parameter** on `SimultaneousPlayEnv.__init__` (default `False`):
- When `False`: everything works exactly as before. Stage 3 models are unaffected.
- When `True`: opponent board encoding is fog-filtered, and `fog_outcomes` appears in the observation space.

**`_encode_opponent_board_fog(board, player_last_step, sprung_trap_pos)`**:
- Shows piece moves where `move.order - 1 <= player_last_step` (order is 1-based, step is 0-based)
- Hides ALL traps except the sprung trap (the one the player hit)
- Skips final moves as before

**`fog_outcomes` observation** — `Box(0, 1, shape=(5, 6))`:
Per-round signals providing structured metadata about what happened:
1. `opponent_steps_visible` — normalized by opponent's total moves
2. `opponent_hit_trap` — did the opponent hit one of the agent's traps?
3. `player_hit_trap` — did the agent hit an opponent trap?
4. `collision` — did both players collide?
5. `opponent_reached_goal` — proxy based on scoring and collision/trap status
6. `visible_opponent_traps` — count of sprung traps, normalized by max traps

These signals give the agent structured information about the round outcome beyond just the partial board grid. A human watching the game would notice "I hit a trap" and "the opponent scored high" — these channels encode that same information.

**`_finish_round()` modification**:
When `use_fog=True` and the board is valid, the simulation result's `SimulationDetails` provides `playerLastStep`, `playerHitTrap`, `playerTrapPosition`, and `collision`. These drive both the fog-filtered board encoding and the `fog_outcomes` signals.

### What Changed in Training

**`--fog` CLI flag** on `train_simultaneous.py`:
- Routes logs to `logs/sizeN_stage4/` and models to `models/sizeN/stage4/`
- Passes `use_fog=True` through to all envs (training, eval, and the progression callback's dedicated eval env)
- Everything else — opponent curriculum, self-play, checkpointing — works unchanged

### What Didn't Change

- `simulate_round()` — `SimulationDetails` already had all the fields we needed
- `RoundResult` — already has `collision` and `simulationDetails`
- Construction masking — the agent's own board construction is fully observable, fog only affects what it sees of the opponent
- Reward structure — same shaping rewards, same game outcome bonuses

---

## Training Plan

Start with size 3 (fast iteration, enough strategic depth for fog to matter):

```bash
python examples/train_simultaneous.py --size 3 --fog --timesteps 5000000
```

### What to Watch For

1. **Phase progression speed**: Expect slower than Stage 3 since the learning signal is noisier. The agent has to figure out opponent patterns from partial information.

2. **Valid rate**: Should stay at 100% — strict masking doesn't depend on fog. Construction is fully observable.

3. **Win rate ceiling**: Expect lower than Stage 3's ~78%. With partial information, perfect counter-play isn't possible. The question is how much the fog_outcomes metadata helps the agent compensate.

4. **fog_outcomes utilization**: After training, zero out the fog_outcomes and re-evaluate. If performance doesn't drop, the agent isn't using them and we can simplify.

5. **Round-over-round adaptation**: Does the agent change its construction strategy between rounds based on what it observed? This is the whole point of fog — if the agent builds the same board every round regardless of history, the fog_outcomes aren't helping.

### After Size 3

If results are promising, scale to size 4:

```bash
python examples/train_simultaneous.py --size 4 --fog --timesteps 5000000
```

Then add self-play:

```bash
# Pre-train against pool opponents first (lesson from Feb 14)
python examples/train_simultaneous.py --size 4 --fog --timesteps 3000000

# Then resume with self-play mixing
python examples/train_simultaneous.py --size 4 --fog --self-play --warmup-steps 0 \
    --resume models/size4/stage4/best/best_model.zip --timesteps 5000000
```

---

## Alternative: Fog Curriculum (Option B)

We chose Option A (fog from scratch), but there's an alternative approach documented in [EXPERIMENTS.md](../EXPERIMENTS.md) (Experiment 1B): train with full reveal first, then transition to fog. The idea is that the agent first learns what the fog signals *mean* (with full info to validate against), then adapts when the information becomes partial.

The tradeoff: faster initial convergence but potential over-reliance on signals that become noisy under fog. EXPERIMENTS.md also outlines signal ablation experiments (Experiment 3) and fog + self-play dynamics (Experiment 4) that may inform which approach works better for production training.

---

## Verification

12 unit tests in `tests/test_fog_of_war.py`:
- Obs space shape correct with `use_fog=True` vs `False`
- Fog-filtered encoding hides moves after `playerLastStep`
- Fog-filtered encoding hides traps except sprung trap
- `fog_outcomes` populated correctly after simulation
- Full round-trip through step/finish cycle
- All existing tests (26 strict masking, 10 parity, etc.) still pass
- Smoke test: 4096-step training run with `--fog` completes without errors

---

## Size 3 Fog Training Results (Pool Opponents)

Kicked off the first real fog training run:

```bash
python examples/train_simultaneous.py --size 3 --fog --timesteps 5000000
```

### Phase Progression

The agent cleared all 6 opponent phases by 256k steps — faster than Stage 3 full reveal (which took ~460k):

| Phase | Started at | Entry win rate |
|-------|-----------|----------------|
| 0 (simple solo) | 8k | 50% |
| 1 (one_trap solo) | 48k | 85% |
| 2 (simple + one_trap mix) | 88k | 65% |
| 3 (super_move solo) | 136k | 55% |
| 4 (cumulative mix) | 176k | 90% |
| 5 (super_move_counter) | 216k | 90% |
| 6 (all pools) | 256k | 75% |

### Convergence at Phase 6

Win rate stabilized at ~82% by 256k steps, flat through 1.14M+ steps:

| Step range | Avg win rate |
|------------|-------------|
| 256k-408k | 82.3% |
| 416k-568k | 80.5% |
| 576k-728k | 82.2% |
| 736k-888k | 81.3% |
| 896k-1.05M | 83.3% |
| 1.06M-1.17M | 83.0% |

Valid rate: 100% throughout (strict masking). Explained variance: ~0.18 (typical for partial observability).

This is surprisingly strong — higher than the Stage 3 full-reveal baseline (~75%). Against fixed pool opponents, partial information doesn't seem to hurt. The real test is self-play.

---

## Bug Fix: Self-Play Temp Env Missing use_fog

When starting fog + self-play:

```bash
python examples/train_simultaneous.py --size 3 --fog --self-play --warmup-steps 0 \
    --resume models/size3/stage4/best/best_model.zip --timesteps 5000000
```

Crashed immediately with `KeyError: 'fog_outcomes'`.

**Root cause**: `_build_opponent_board_from_model()` creates a temporary `SimultaneousPlayEnv` to drive the opponent model's construction. This temp env was created without `use_fog=True`, so its observations lacked `fog_outcomes`. When the fog-trained opponent model called `predict(obs)`, SB3's feature extractor couldn't find the expected key.

The crash manifested as an `EOFError` in the main process because the actual `KeyError` happened inside a SubprocVecEnv worker process — the worker died and the main process saw the pipe close.

**Fix**: One line — pass `use_fog=self.use_fog` when creating the temp env in `_build_opponent_board_from_model()`.

**Lesson**: Any code path that creates a temporary env for model inference must mirror all observation-space-affecting flags from the parent env. This is the same class of bug as the phase sync issue (Feb 4) — auxiliary envs drifting out of sync with the training env's configuration.

---

## Self-Play Block Scheduling (Feb 16)

The initial self-play attempt used per-round coin-flip mixing (`--self-play-ratio 0.5`): each round independently decides self-play vs pool opponent. After ~318k steps, win rate was volatile (76-87%) and explained variance spiked negative (-0.27). The mixing strategy has a fundamental problem: it interrupts learning in both directions. During a self-play stretch, a random pool board breaks the counter-play signal. During pool training, a self-play board injects non-stationarity.

### Block Scheduling Design

Replace the coin flip with dedicated blocks:

1. **Self-play block** (default 200k steps): Pure self-play (ratio=1.0). The agent plays exclusively against snapshots of itself. No interruptions from pool opponents.
2. **Block boundary check**: At the end of each block, check pool win rate from `OpponentProgressionCallback`'s evaluation.
3. **Pool recovery** (default 100k steps): If pool win rate drops below threshold (default 60%), switch to pure pool opponents (ratio=0.0). The agent re-grounds against known-good opponents.
4. **Resume**: After recovery, start another self-play block.

If pool win rate stays above threshold, skip recovery and start the next self-play block immediately.

### New CLI Flags

```bash
--self-play-block-steps 200000   # Steps per pure self-play block
--pool-recovery-steps 100000     # Steps of pool-only if degraded
--min-pool-win-rate 0.60         # Threshold to trigger recovery
```

### Training Command

```bash
python examples/train_simultaneous.py --size 3 --fog --self-play \
    --warmup-steps 0 \
    --resume models/size3/stage4/best/best_model.zip \
    --timesteps 5000000 \
    --self-play-block-steps 200000 \
    --pool-recovery-steps 100000 \
    --min-pool-win-rate 0.60
```

### Rationale

The key insight: **learning benefits from consistency**. When the agent is playing self-play, it needs sustained exposure to develop counter-strategies. When it needs to recover pool competence, it needs sustained pool exposure without self-play noise. The block boundaries create natural checkpoints where we can assess whether the agent is maintaining breadth (pool win rate) while developing depth (self-play adaptation).

This is the same principle as "one source of non-stationarity at a time" from the Feb 14 self-play collapse — but applied at a finer granularity within a single training run.
