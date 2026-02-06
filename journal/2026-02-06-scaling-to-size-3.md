# Training Journal: Scaling to Size 3

**Date:** February 6, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Size 3 - Full Pipeline (Stage 1 + Stage 2)

---

## The Problem

Ryan kicked off a size 3 training run and it wasn't going well. The agent was stuck at phase 1 of the reverse curriculum, win rates oscillating wildly between 10-65%, never stabilizing enough to advance. After 50k steps it had barely moved.

When Ryan brought me in to diagnose, the answer was hiding in plain sight: **we'd skipped Stage 1 entirely.**

---

## What We Missed

The size 2 success story had two stages that built on each other:

1. **Stage 1** (Board Selection): Agent learns which of the 8 curated boards beats which opponent. 100k timesteps to 100% optimal play. This produced a frozen policy at `models/construction/best/best_model.zip`.

2. **Stage 2** (Reverse Curriculum): Uses the frozen Stage 1 model to pick *strategically good* base boards, then teaches the agent to build those boards from scratch.

For size 3, `models/size3/stage1/` was **empty**. Stage 2 was falling back to random board selection -- the agent was trying to complete arbitrary boards against random opponents with no strategic foundation. No wonder it was floundering.

There was also a practical issue: `train_construction.py` was hardcoded to `new_boards_2.json` with no `--size` parameter. It literally couldn't train Stage 1 for anything other than size 2.

---

## The Fix

Parameterized `train_construction.py` to accept `--size`, `--board-library`, and `--output-dir`. Now the full pipeline works for any board size:

```bash
# Stage 1: Learn board selection (new!)
python examples/train_construction.py --size 3 --timesteps 200000

# Stage 2: Reverse curriculum with frozen Stage 1
python examples/train_reverse_curriculum.py --size 3 --timesteps 500000
```

Also fixed a default path mismatch in `train_reverse_curriculum.py` -- it was looking for `my_boards_{N}.json` while the actual files are `new_boards_{N}.json`. And the Stage 1 model path now correctly points to the `best/best_model.zip` subdirectory that EvalCallback produces.

---

## Size 3: The Numbers

### The Board Library

14 curated size-3 boards (vs 8 for size 2), ranging from 4 to 8 moves per board. The action space is 9 cells (vs 4 for size 2), so the combinatorial complexity is significantly higher.

### Stage 1 Results

200k timesteps, ~10 minutes.

| Metric | Value |
|--------|-------|
| Mean reward | 112-113 |
| Eval reward | 112 +/- 3.32 |
| Explained variance | 0.98 |
| Entropy | -0.52 (converged) |

Mean reward of 112 = +100 win bonus + ~12 points of margin across 5 rounds. The agent is winning essentially every game with consistent margins. Tight standard deviation confirms it's not getting lucky -- it knows which board beats which opponent across all 14 matchups.

### Stage 2 Results

Ran to 300k of a planned 500k before Ryan killed it (curves had plateaued).

| Metric | Value |
|--------|-------|
| Final phase | 10 (full from-scratch) |
| Valid rate | 100% |
| Win rate | 65-85% (oscillating) |
| Avg reward | ~28-33 |
| Eval mean reward | ~32 |

Night and day compared to the previous attempt. The agent advanced through all 10 curriculum phases by ~75k steps, then spent the remaining 225k at phase 10 building valid boards from scratch and winning the majority of games.

The agent converged on building minimal valid boards (4 moves, the shortest in the library). That's a rational optimization -- simple boards that win -- but it means the agent isn't exploring the more complex 7-8 move board structures.

---

## The Difference Stage 1 Makes

This is the headline takeaway. Same environment, same reward structure, same hyperparameters. The only variable was whether Stage 1 existed:

| | Without Stage 1 | With Stage 1 |
|--|-----------------|--------------|
| Phases reached | 1 | 10 |
| Valid rate | 100% (trivial phases) | 100% (all phases) |
| Win rate | 10-50% (unstable) | 65-85% |
| Steps to advance past phase 1 | Never | ~12k |

Stage 1 gives Stage 2 a massive leg up by ensuring the base boards being completed are already known to be good counters to the specific opponent. Without it, the agent is completing random boards and hoping for the best.

---

## Files Changed

- `examples/train_construction.py` - Added `--size`, `--board-library`, `--output-dir` parameters; size-based output paths (`models/size{N}/stage1/`, `logs/size{N}_stage1/`)
- `examples/train_reverse_curriculum.py` - Fixed default board library path (`new_boards_` not `my_boards_`), fixed Stage 1 model path to include `best/` subdirectory

---

## What's Next

The size 3 agent is functional but has room to grow. Win rate plateaued at 65-85% rather than climbing. Some options:

1. **Longer training** -- resume from the 300k checkpoint with more timesteps
2. **Reward tuning** -- the agent settles for minimal boards; could incentivize complexity
3. **Board library expansion** -- 14 boards may not provide enough strategic diversity
4. **Scale further** -- sizes 4 and 5 are now straightforward with the parameterized pipeline

The important thing is that the pipeline works. Stage 1 then Stage 2, for any board size. That was the missing piece.

---

*"The foundation matters more than the curriculum."*
