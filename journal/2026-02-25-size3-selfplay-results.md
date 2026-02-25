# Training Journal: Size 3 Self-Play from Scratch — Results

**Date:** February 25, 2026
**Author:** Claude (with Ryan's direction)
**Stage:** Self-play from scratch — first completed run

---

## Size 3 Finished: Level 20, 100% SP Win Rate

The size 3 self-play-from-scratch run completed overnight. 10M steps, ~11.7 hours wall time. The headline numbers:

| Metric | Final Value |
|--------|-------------|
| Max self-play level | 20 (pool-size cap) |
| SP eval win rate | 100% (sustained from ~6M steps onward) |
| Pool win rate | 35% (expected — model optimized for self-play, not pool) |
| Valid rate | 100% throughout |
| Explained variance | 0.688 |
| Rollout reward | +144 |
| Snapshots in pool | 20 |
| Level advancement saves | 83 |
| Recovery events | 0 (never needed) |

---

## Level Progression Timeline

The agent climbed steadily through all 20 levels with only two brief backtracks:

| Level | Steps | Notes |
|-------|-------|-------|
| 1 | 50k | First advance — within the first minute |
| 2 | 200k | |
| 5 | 800k | Roughly one level per 200k steps |
| 10 | 1.88M | Previous pool-size 10 ceiling |
| 15 | 2.81M | |
| 20 | 3.80M | Pool-size 20 ceiling reached |

After hitting level 20 at 3.8M steps, the remaining 6.2M steps were spent at the cap. SP eval win rate climbed from 80% to a sustained 100% during that consolidation period.

Two backtracks occurred — level 4→3 at 680k and level 9→8 at 1.68M — both recovered within ~100k steps. No recovery mode was ever triggered. The cold-start concern from the plan was unfounded for size 3.

---

## The Pool Win Rate Story

An interesting pattern: pool win rate peaked early (~85% at 32k steps during the validation run) and then steadily declined as the agent got deeper into self-play. By 10M steps it was 35%.

This isn't failure — it's expected. The simple pool boards are static opponents with predictable patterns. As the agent learns increasingly sophisticated self-play strategies, it's optimizing for a different game than "beat the pool." The pool WR oscillated between 20-55% from 3M steps onward, confirming the agent was fully focused on self-play.

This validates Ryan's reasoning: the pool is just a fallback floor, not the training signal. Self-play eval win rate is the metric that matters.

---

## What We Shipped

Ryan created a convenience script at `/usr/local/bin/train` that wraps the whole thing:

```bash
train --size 4                    # 10M steps, pool-size 20, discord, fog
train --size 5 --timesteps 20M   # override defaults
```

It runs detached via nohup, logs to `size-N-selfplay.txt`, and prints the PID and tensorboard command. No more remembering the full invocation.

Size 2 training is currently running. Size 4 will follow.

---

## Comparison: Self-Play from Scratch vs Pool-First + Resume

For size 3, we now have results from both approaches:

| Metric | Pool-first + resume (take 6) | Self-play from scratch |
|--------|------------------------------|----------------------|
| Max SP level | 10 (pool-size 10) | 20 (pool-size 20) |
| SP eval WR | ~90% | 100% |
| Total steps | 7.5M (pool + SP combined) | 10M (single run) |
| Wall time | ~16 hours (two runs) | ~11.7 hours |
| Recovery events | Multiple | 0 |
| Pool WR at end | ~40-50% | ~35% |

The self-play-from-scratch run reached a higher level with more stability and fewer total steps. The single-run simplicity is a real advantage — no manual intervention to switch from pool to self-play.

The open question remains: does this agent actually adapt its boards across rounds? The numbers look right, but we need to play against it to confirm. That's the next test.

---

## Lessons

1. **Pool-size 20 works.** The agent climbed all 20 levels without collapse. Previous runs were artificially capped at 10.
2. **Cold start was a non-issue for size 3.** Zero recovery events. Strict masking + quality gate handled the random-policy phase cleanly.
3. **Pool win rate is not the metric.** For self-play-from-scratch, pool WR declines as the agent specializes. SP eval WR is the only metric that matters.
4. **Convenience scripts compound.** Ryan's `train` wrapper saves maybe 30 seconds per invocation, but it removes the friction of looking up the command — which means training runs actually get started instead of deferred.

---

## Update: Size 2 Abandoned, Size 4 Started

We tried size 2 self-play-from-scratch and it confirmed what Ryan suspected: the board is too small for self-play to produce meaningful signal. SP eval win rate sat at exactly 50% from 200k through 484k steps — a pure coin flip. On a 2x2 board there are only 4 cells, so the space of valid boards is tiny. Both sides end up playing essentially the same board every time. Pool WR was volatile (100% → 15% → 95%) — noise, not learning. It entered recovery once at 450k steps.

The diagnosis is straightforward: size 2 doesn't have enough strategic depth for history-dependent adaptation to emerge. The existing pool-trained models are sufficient for that scale.

Ryan killed the size 2 run and started size 4 instead (`train --size 4`). With 16 cells and up to 3 traps, there's real room for board variety and round-over-round adaptation. This is the true test of the self-play-from-scratch approach at a scale where it should matter.
