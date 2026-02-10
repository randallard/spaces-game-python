# Training Journal: Scoring Fix, No-Revisit Masking, and Fog of War Research

**Date:** February 9, 2026
**Author:** Claude (with Ryan's guidance)
**Stage:** Size 3 Stage 3 refinement, fog of war investigation

---

## Starting Point

Size 3 Stage 3 had completed training (construction C0-C6 + opponent O0-O5) with 95% win rate and 100% valid boards. Ryan was playing against the agent and noticed something odd: the agent was building boards with 24-30 move sequences on a 3x3 grid, oscillating back and forth between cells to farm forward-movement points.

---

## Bug 1: Scoring — Oscillation Farming

The simulation engine (`simulation.py`) awarded +1 point for every forward move, even revisiting the same row. An agent could bounce between row 0 and row 1 repeatedly, collecting a point each time it moved "forward." The sequences got absurd — 24+ moves for a board that only needs 3-4.

**Fix:** Track `player_best_row` and `opponent_best_row`. Only award points the first time a piece reaches a new forward row. One variable each, straightforward logic.

The TypeScript parity tests (all 52) confirmed this matches the reference implementation — the TypeScript version already used first-visit scoring. Our Python code was the one that was wrong.

**Lesson:** Test with actual gameplay. The training metrics looked great (95% wins!) but playing a single game revealed the exploit immediately.

---

## Bug 2: Explanation Text — Same Scoring Bug

The technical explanation in `cli.py` had its own copy of the scoring logic for step-by-step display. Same bug — showed "+1 point (forward movement)" for every forward move instead of just first-visit. The actual simulation scores were correct (the display was just misleading), but it made the oscillation look worse than it was score-wise.

**Fix:** Added `player_best_row` / `opponent_best_row` tracking to the explanation renderer, matching the simulation logic.

---

## Fix 3: No-Revisit Action Masking

Even with the scoring fix, the agent was still building oscillating boards — it just didn't earn points from the oscillation anymore. The sequences were still wastefully long.

Ryan asked whether we should add negative rewards or masking. Masking is clearly better:
- Completely prevents the behavior (no oscillation boards can be built)
- No reward complexity (negative rewards interfere with learning)
- Reduces wasted exploration
- Consistent with our existing action masking pattern

**Implementation:** Added `piece_visited_positions` set to track cells the piece has been on. `_is_valid_placement()` rejects piece moves to already-visited cells. Action masks flow through automatically.

This needed to be applied in multiple places:
- `simultaneous_play_env.py` — training env (step + scaffolding prefill)
- `reverse_builder_env.py` — Stage 2 env (reconstruct + step)
- `play_against_agent.py` — inference during play

---

## The Play Script Bug

After deploying the no-revisit masking, Ryan reported the agent was STILL oscillating during play. Turns out `play_against_agent.py` has its own copy of the construction loop (`_agent_build_board_blind()`) that manually drives construction outside of `env.step()`. It wasn't updating `piece_visited_positions`.

This led to an important insight about MaskablePPO: **masking is enforcement, not learning.** The model doesn't learn to avoid masked actions — they just get blocked during training. The underlying policy has arbitrary probabilities for those actions. Remove the mask (even accidentally, as in our play script bug) and the behavior comes right back.

Ryan pointed out this will matter for the app integration too — any code that drives agent construction needs to track visited positions. We added documentation in the env class docstring and TRAINING_PLAN.md to make sure this is visible when we build the app.

---

## Design Decision: Agent Rule vs Game Rule

Ryan caught an important distinction when I started adding no-revisit checks to `validation.py` (the game rules validator). Revisiting cells isn't *illegal* — it's just *bad strategy*. The scoring fix already ensures zero benefit from it. Humans should be free to discover this on their own.

So we reverted the validation changes. The no-revisit rule lives only in the agent's action masking, not in the game rules. `is_board_playable()` remains the authoritative game rules checker for all consumers (app, CLI, training).

---

## Fog of War Research

Investigated the TypeScript reference implementation to understand fog of war for Stage 4. Key findings:

**Three tiers of information:**
1. **Fully visible:** Opponent piece moves up to their last executed step, sprung trap positions, round outcome
2. **Partially visible (existence only):** When the opponent places a trap during an executed step, the explanation reveals "a trap was set" — but not where
3. **Completely hidden:** Unexecuted moves, unsprung trap locations

The "trap exists but no location" signal is an interesting design challenge for the observation space. We documented all three tiers in TRAINING_PLAN.md.

---

## Other Improvements

- **Model selection timestamps:** Added modification dates to the model selection menu in `play_against_agent.py`, matching the training run display format
- **TRAINING_PLAN.md:** Major update — Stage 3 marked complete for both sizes, Stage 4 expanded with fog visibility rules, scoring rule documented, no-revisit app integration note added

---

## Files Changed

- `spaces_game/simulation.py` — First-visit forward movement scoring
- `spaces_game/cli.py` — Explanation text scoring fix (matching simulation)
- `spaces_game/simultaneous_play_env.py` — No-revisit action masking, app integration docs
- `spaces_game/reverse_builder_env.py` — No-revisit action masking
- `examples/play_against_agent.py` — No-revisit in inference loop, model date display
- `TRAINING_PLAN.md` — Status updates, scoring rule, no-revisit docs, fog of war details

---

## What's Next

- Retrain size 3 with both fixes (scoring + no-revisit masking)
- Design difficulty-level training (beginner/intermediate/expert checkpoints)
- Implement Stage 4 (fog of war) in SimultaneousPlayEnv

---

*Masking is enforcement, not learning. If you lift the mask, the behavior comes right back.*
