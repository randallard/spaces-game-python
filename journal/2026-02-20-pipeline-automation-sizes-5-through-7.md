# Training Journal: Pipeline Automation and Sizes 5 Through 7

**Date:** February 20, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Stage 4 — fog of war, self-play, automation

---

## The Big Picture

Ryan wanted to push through sizes 5, 6, and 7 in a single session. Not one-at-a-time babysitting — he wanted to kick off training, get Discord notifications on his phone, and check in when something interesting happened. We got there, but not without a detour.

---

## Fog-First Training: Stage 3 is Dead

The headline result: **Stage 3 (full reveal) is unnecessary.**

Size 5 was our test case. Instead of training with full opponent board reveal first, we went straight to fog of war. The reasoning: size 3 fog from scratch converged at 82% in just 256K steps without any Stage 3 pretraining. The agent never develops full-information dependencies, so there's nothing to unlearn.

Results validated the approach:

| Size | Fog Pool Convergence | SP Level 10 | SP WR | Total Steps |
|------|---------------------|-------------|-------|-------------|
| 5 | 400K steps (all 7 phases) | 2M steps | ~90% | ~2.5M |
| 6 | 1.5M steps (all 7 phases) | 1.9M steps | 100% | ~6M |
| 7 | 1.6M steps (all 7 phases) | In progress | TBD | TBD |

Every size follows the same pattern: pool training converges in 1-4M steps, self-play reaches level 10 within another 2M. The fog-first pipeline is now the standard recipe for new sizes.

---

## The Pipeline Script (And Why It Stalled)

Ryan asked me to automate the full training flow: pool-only until converged, then self-play, with Discord updates. Straightforward enough — except my first implementation used `subprocess.PIPE` for the training process's stdout.

Python's pipe buffer is about 64KB. Training produces a *lot* of stdout. The buffer filled up, the training process blocked on write, and the whole thing silently stalled at 0.12M steps. The monitoring script kept reading stale TensorBoard data every 5 minutes, dutifully reporting the same numbers for 6 hours. The training process was alive but frozen.

The fix was embarrassingly simple: redirect stdout to a file instead of a pipe.

```python
# Broken: pipe buffer fills → training blocks
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

# Fixed: file never blocks
proc = subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT)
```

Classic case of a monitoring system that monitors itself instead of the thing it's supposed to monitor. The pipeline script was happy, the TB files were stale, and the actual training was wedged. Ryan caught it the next morning when the Discord updates showed no progress.

---

## Size 6: The Largest Board Yet

36 cells, max 5 traps. Ryan hand-created the board pools following the same 4-category structure as sizes 3-5 (simple, mixed_traps, super_move, super_move_counter). I generated the size 6 pools programmatically based on size 5 patterns — same board styles scaled to 6x6.

Pool training cleared all phases by 1.5M steps but then oscillated at 55-80% WR at phase 6. After 4.1M steps with no upward trend, Ryan called it and we moved to self-play. The self-play run was textbook: level 10 by 1.9M steps, 100% SP eval WR.

Model sizes scale linearly and stay tiny — size 6 production models are about 1MB each. All 15 production models across sizes 2-6 total roughly 10MB. Railway hosting costs won't budge.

---

## Discord + VNC: Development on the Go

The real workflow improvement this session wasn't the code — it was the feedback loop. Ryan set up:

1. **Discord webhook notifications** from the training callbacks (see [DISCORD_SETUP.md](../DISCORD_SETUP.md)) — milestone alerts on phase advances, self-play level changes, recovery events, plus periodic check-ins with win rate trends and commentary
2. **Standalone monitor scripts** for custom monitoring (convergence detection, self-play WR gates)
3. **VNC from his phone** to the training machine for quick TensorBoard checks

The combination means Ryan can kick off a multi-hour training run, go about his day, and get a Discord ping when something needs attention. The pipeline script automates the pool→self-play transition, so even the "converged, time for self-play" decision happens automatically.

For setup details on Discord notifications, see [DISCORD_SETUP.md](../DISCORD_SETUP.md). The built-in `--discord-webhook` and `--discord-check-in` flags handle the standard notifications. For custom monitoring (like the convergence-detection scripts we used this session), see `scripts/discord_monitor.py` for the pattern.

---

## Size 7: Automated Pipeline

With the fixed pipeline script, size 7 was hands-off. Ryan added the board pools, I started the pipeline, and the pool training converged at 1.6M steps. Self-play is running now with hourly Discord updates.

The training command for the full pipeline:
```bash
python -u /tmp/size7_train_pipeline_v2.py "$DISCORD_WEBHOOK"
```

Which internally runs:
```bash
# Phase 1: Pool-only (auto-stops when phase 6 reached + 1M steps)
python examples/train_simultaneous.py \
    --size 7 --fog \
    --timesteps 10000000 \
    --learning-rate 1e-4 --ent-coef 0.1 --n-steps 4096 \
    --discord-webhook "$DISCORD_WEBHOOK" --discord-check-in 60

# Phase 2: Self-play (auto-stops at level 10 + 100% SP WR)
python examples/train_simultaneous.py \
    --size 7 --fog --self-play \
    --resume models/size7/stage4/best/best_model.zip \
    --timesteps 10000000 \
    --learning-rate 1e-4 --ent-coef 0.1 --n-steps 4096 \
    --discord-webhook "$DISCORD_WEBHOOK" --discord-check-in 60
```

---

## What's Deployed

All sizes 2-6 are fully trained and deployed:

| Size | Model Path | SP Level | SP WR |
|------|-----------|----------|-------|
| 2 | `models/size2/stage3/` | N/A (pool sufficient) | N/A |
| 3 | `models/size3/stage4/` | 10 | ~90% |
| 4 | `models/size4/stage4/` | 10 | 80% |
| 5 | `models/size5/stage4/` | 10 | ~90% |
| 6 | `models/size6/stage4/` | 10 | 100% |

Size 7 will join this table once self-play converges. Each size has beginner, intermediate, and expert models plus level advancement snapshots accessible via the inference server's indexed model selection.

---

## Lessons Learned

1. **Stage 3 is dead** — fog-first training works for every size tested. Skip full-reveal pretraining.
2. **Never use `subprocess.PIPE` for long-running processes** unless you actively drain it. File redirection is simpler and never blocks.
3. **The pool→self-play pattern is mechanical** — every size follows the same curve. This is ready to be a one-command pipeline.
4. **Discord + VNC = async development** — training doesn't need babysitting when the feedback loop is tight.
