# Training Architecture: Hardware Journey

This project trains reinforcement learning agents locally on consumer hardware. The goal from the start has been to avoid cloud training dependencies — no AWS SageMaker, no GCP TPU pods, no per-hour GPU rental. Everything runs on hardware we own, which means lower ongoing costs and no surprises when a training run takes longer than expected. The trade-off is patience: sequential training, one run at a time, and careful memory management.

The machine started as a Dell Optiplex desktop pulled from a pile of decommissioned office PCs. Over time we've upgraded it piece by piece as training demands grew. This document tracks what we've built, what worked, and where the limits are.

**Current state**: Same Optiplex board with the i5-3470 and 8GB DDR3 RAM, now running an RTX 3060 12GB GPU. Very effective for one training run and one Claude Code session simultaneously. It froze up when we tried adding a second Claude Code session to update the frontend — the Node install process pushed memory over the edge. One training + one dev session is the practical ceiling on this hardware.

---

## Current Specs

| Component | Spec | Notes |
|-----------|------|-------|
| **Motherboard** | Dell Optiplex (LGA 1155) | Office surplus, rock solid |
| **CPU** | Intel Core i5-3470 @ 3.20GHz | 4 cores, no hyperthreading |
| **RAM** | 8GB DDR3 | ~3GB available during training |
| **GPU** | NVIDIA GeForce RTX 3060 12GB | The big upgrade — handles RL fine |
| **OS** | Omarchy (Arch Linux) | Lightweight, minimal overhead |
| **PyTorch** | 2.1.0 + CUDA | GPU-accelerated policy network updates |

---

## Hardware Progression

### Phase 1: Bare Bones Optiplex

The starting point. i5-3470, 8GB DDR3, no discrete GPU. Good enough for Stage 0 (deck selection) and initial Stage 1 (board construction) experiments with size 2 boards. Training was CPU-bound and slow, but the game environment is simple enough that it worked.

- 4 parallel envs (`--envs 4`) was the sweet spot
- ~50K steps/minute on size 2
- Size 3 boards fit in memory with careful management

### Phase 2: RTX 3060 12GB

Added the GPU. This was the single biggest improvement — not because RL training is GPU-heavy (it's mostly CPU-bound for environment stepping), but because PyTorch policy network updates got dramatically faster. The 12GB VRAM is massive overkill for our model sizes but means we never worry about GPU memory.

- Same CPU/RAM constraints apply
- Training throughput improved for the gradient update portion
- Size 4 training became practical (16-cell observation space)
- One training run saturates a CPU core at 99%, uses ~1.8GB resident memory

---

## Practical Limits

### What Works

- **One training run at a time**: Size 2-4, all stages, pool or self-play
- **One Claude Code session alongside training**: For code changes, debugging, reviewing TensorBoard
- **4 parallel envs**: Best balance of throughput vs memory on 8GB

### What Doesn't Work

- **Two training runs**: Second run pushes into swap, both slow to 30-40% speed
- **Training + heavy dev work**: Node installs, large builds, or a second Claude Code session can trigger OOM or freeze
- **`--envs 8` or higher**: Memory pressure causes swap thrashing
- **Size 5+ without tuning**: Larger observation spaces (25 cells) will need `--envs 2` to fit in memory

### Memory Budget

```
Total RAM:        ~7.7GB
OS + services:    ~2.0GB
Training run:     ~1.8GB (4 envs, size 4)
Claude Code:      ~1.5-2.0GB
Available:        ~2-3GB headroom

Adding Node install: ~1-2GB spike -> OOM/freeze
```

---

## Upgrade Path

If we upgrade RAM from 8GB to 16-32GB DDR3 (the Optiplex board maxes at 32GB):

- **16GB**: Comfortable dual training or training + heavy dev. `--envs 8` becomes viable.
- **32GB**: `--envs 16`, size 5 training with margin, parallel experiments.

CPU is the harder bottleneck — the i5-3470 only has 4 physical cores. Environment stepping is single-threaded per env, so more envs doesn't help past the core count. A CPU upgrade would mean a new motherboard.

---

## Training Performance Reference

| Board Size | Cells | Typical Timesteps | Approx Time (4 envs) | Memory |
|-----------|-------|-------------------|----------------------|--------|
| 2 | 4 | 200K | ~10 min | ~1.2GB |
| 3 | 9 | 2M | ~1-2 hours | ~1.5GB |
| 4 | 16 | 5-10M | ~5-10 hours | ~1.8GB |
| 5 | 25 | TBD | TBD (likely `--envs 2`) | ~2.5GB est. |

Self-play adds ~20-30% overhead (snapshot loading, opponent model inference).

---

## Tips for This Hardware

1. **Check free memory before training**: `free -h` — if available < 3GB, close browsers/editors first
2. **Use `--envs 4`** for sizes 2-4. Drop to `--envs 2` for size 5+.
3. **Don't run Node/npm alongside training** — install dependencies before starting a training run
4. **TensorBoard is lightweight** — safe to run alongside training
5. **Sequential, not parallel** — finish one size before starting the next
6. **Monitor with `htop`** — if swap usage climbs past 500MB, something needs to stop
