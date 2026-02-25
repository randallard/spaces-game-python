# Training Journal: Self-Play from Scratch and History-Aware Opponents

**Date:** February 24, 2026
**Author:** Claude (with Ryan's direction)
**Stage:** Self-play architecture rethink

---

## The Problem: Agents That Ignore History

Ryan and I identified a structural issue with how we've been training self-play models. The pipeline has always been: train pool-only first, then resume with self-play. The pool phase gets the agent good at building valid boards and winning games. The self-play phase adds adaptive opponents.

The problem is subtle. During pool-only training, opponent boards are drawn randomly from JSON files. They have no relationship to what the agent built in previous rounds. The `opponent_history` observation — the encoded history of what the opponent played in rounds 1 through 4 — is pure noise during pool training. The model learns to ignore it. By the time self-play is added via `--resume`, that habit is already baked in. The agent plays the same board every round regardless of what the opponent did.

This matters because the whole point of a 5-round game is adaptation. A strong player should notice the opponent's patterns and adjust — maybe placing traps differently, changing piece paths, exploiting tendencies. Our current models don't do any of that.

---

## The Fix: Self-Play from Step 0

Ryan's solution was clean: skip pool pre-training entirely. Start self-play from the very first step. We created `examples/train_self_play.py` — a focused training script that:

- Uses only `00_simple.json` (or `simple.json`) as a minimal fallback pool
- Starts self-play immediately (warmup = 0)
- Has no opponent phase curriculum (single pool file, single phase)
- Enables fog by default (all production models are fog)
- Uses a lower snapshot quality gate (0.30 vs the usual 0.625 midpoint)

The key insight: with self-play from step 0, both the agent and its opponents are equally bad at the start. The relative signal is still meaningful — the agent just needs to be slightly better than its previous self. And critically, `opponent_history` is meaningful from round 1 because the opponent is another version of the agent, playing boards that reflect a learned strategy.

The biggest risk was early policy collapse. A random model produces weak snapshots, weak snapshots make weak opponents, and you get a death spiral. But strict BFS masking ensures boards are always structurally valid, and the quality gate prevents snapshotting garbage models. Recovery mode falls back to the simple pool if things go sideways.

---

## Validation Run: Size 3, 500k Steps

We ran a quick test to make sure the script worked before committing to a long run.

Results were encouraging:

| Metric | 8k steps | 24k steps | 32k steps | 200k steps |
|--------|----------|-----------|-----------|------------|
| Valid rate | 100% | 100% | 100% | 100% |
| Pool WR | 55% | 75% | 85% | 70% |
| SP eval WR | — | — | — | 100% |
| Window level | 0 | 0 | 0 | 1 |
| Reward | -48 | +104 | +131 | +93 |

No collapse. Valid rate held at 100% throughout (strict masking doing its job). The agent advanced to self-play level 1 by 200k steps — meaning it was beating its own snapshot consistently enough to level up. Pool win rate climbed from 55% to 85% in the first 30k steps, which is comparable to the pool-first pipeline.

---

## Production Run: Size 3, 10M Steps

Based on the validation, Ryan kicked off the full run:

```bash
nohup python examples/train_self_play.py \
    --size 3 \
    --timesteps 10M \
    --pool-size 20 \
    --discord-webhook URL > size-3-selfplay.txt 2>&1 &
```

We bumped `--pool-size` to 20 (up from the default 10) to push past the level 10 ceiling that capped previous self-play runs. Output goes to `models/size3/selfplay/`, logs to `logs/size3_selfplay/`.

Discord notifications will track progress. The real test will be whether the trained agent builds different boards across rounds when playing against it — that's the whole reason for this approach.

---

## What This Means for the Training Pipeline

If this works — if the size 3 agent demonstrates genuine round-over-round adaptation — `train_self_play.py` becomes the standard script for producing AI opponents at all sizes. The existing pool-first pipeline (`train_simultaneous.py`) still has its place for training difficulty tiers (beginner/easy/medium models captured mid-pool-training), but the top-tier "expert" opponents should come from self-play-from-scratch.

The training plan has been updated to document both approaches (Option A: self-play from scratch, Option B: pool-first legacy) with guidance on when to use each.

---

## Lessons

1. **Training order shapes what the model learns to attend to.** Pool-first training teaches the model that opponent history is noise. That's technically correct during pool training — and permanently damaging for self-play.
2. **Cold start isn't as scary as it sounds.** Strict masking + quality gates + recovery mode handle the "both sides are random" phase gracefully. The 500k validation showed zero collapse.
3. **Pool size matters for ceiling.** Previous runs all hit level 10 (the pool-size cap) and couldn't advance further. Bumping to 20 gives the curriculum room to grow.
