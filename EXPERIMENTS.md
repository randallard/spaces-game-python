# Experiments: Fog of War Training and Beyond

**Author:** Claude (with Ryan's guidance)
**Created:** February 16, 2026

## Summary

With Stage 3 complete for sizes 2, 3, and 4 (flat Discrete action space, strict BFS masking, self-play with pool mixing), the next frontier is **fog of war** — partial observability where the agent can't see the opponent's full board after simulation.

This document outlines experiments to explore different approaches to fog of war training, as well as broader experiments comparing training methodologies. The goal is to build intuition about what works before committing to a single approach for production training.

All experiments should be run at **size 3** first (fast iteration, enough strategic depth for fog to matter), then validated at size 4 if results are promising.

---

## Experiment 1: Fog from Scratch vs Fog Curriculum

**Question:** Does the agent learn fog reasoning better when it never sees full boards, or when it starts with full visibility and transitions to fog?

### 1A: Fog from Scratch

Train from random weights with fog active from step 1. The agent never sees full opponent boards — only partial reveals filtered by `opponentLastStep`, sprung traps, and `fog_outcomes` metadata.

```bash
python examples/train_simultaneous.py --size 3 --fog --timesteps 5000000
```

**Hypothesis:** The agent learns to reason under uncertainty from the start. It never develops a dependency on full information, so its strategies are natively fog-compatible. May take longer to converge since the learning signal is noisier.

**What to track:**
- Phase progression speed vs Stage 3 (full reveal) baseline
- Win rate at each phase — expect lower ceiling than Stage 3
- Does the agent learn to use `fog_outcomes` signals? (Check if zeroing them out degrades performance)
- Construction quality — does fog hurt board-building even though construction is fully observable?

### 1B: Fog Curriculum

Train from random weights with `fog_outcomes` in the observation space but filled with **ground truth** initially (full opponent info encoded in the fog signals). After the agent clears all opponent phases with full reveal, activate fog filtering.

```bash
# Phase 1: full reveal with fog_outcomes present (ground truth)
python examples/train_simultaneous.py --size 3 --fog-curriculum --timesteps 3000000

# Phase 2: activate fog (resume, obs space unchanged)
python examples/train_simultaneous.py --size 3 --fog --timesteps 5000000 \
    --resume models/size3/stage4/best/best_model.zip
```

**Hypothesis:** The agent first learns what the fog signals *mean* (with full info to validate against), then adapts to partial info. Faster initial convergence but may over-rely on signals that become noisy under fog.

**What to track:**
- Does full-reveal pre-training help or hurt fog performance?
- How long does the transition period take? (Win rate drop when fog activates, recovery time)
- Compare final win rate to Experiment 1A at the same total timestep budget

### Comparison Metrics

| Metric | 1A (Scratch) | 1B (Curriculum) |
|---|---|---|
| Steps to clear all phases | ? | ? |
| Win rate at final phase | ? | ? |
| Win rate delta when fog_outcomes zeroed | ? | ? |
| Total timesteps to convergence | ? | ? |

**Infrastructure status (Feb 16, 2026):** Experiment 1A is ready to run. The `--fog` flag on `train_simultaneous.py` enables fog-filtered opponent encoding + `fog_outcomes` observation. See `journal/2026-02-16-fog-of-war-implementation.md` for implementation details. Experiment 1B (`--fog-curriculum`) is not yet implemented — would require a ground-truth mode that fills `fog_outcomes` with full information before transitioning to actual fog.

**What carries over from existing work:**
- Opponent board pools (`boards/size{N}/`) — unchanged
- Strict BFS masking — unchanged, construction is fully observable under fog
- Self-play infrastructure — unchanged, but self-play opponent also builds under fog
- `_encode_opponent_board_fog()` — implemented, filters by `playerLastStep` and sprung trap
- `simulate_round()` return value — already contains `SimulationDetails` with `playerLastStep`, trap info, etc.

---

## Experiment 2: LLM Opponent vs RL Opponent

**Question:** How does an LLM-based opponent (via MCP) compare to our RL-trained agents in actual gameplay?

This isn't about using MCP for training — it's about demonstrating the difference in opponent quality between an LLM that reasons about the game from its general knowledge vs an RL agent that learned strategy through millions of self-play games.

### Setup

**RL Agent:** Our trained Stage 3 expert model. Builds boards using the policy network with action masking. Has played millions of games and discovered strategies empirically.

**LLM Agent:** Claude (or another LLM) connected via MCP server. Given the game rules, board state, opponent history, and asked to construct a board. Uses reasoning and general intelligence but has never actually played the game.

### MCP Server Design

Build a lightweight MCP server that exposes:
- `construct_board(board_size, round_num, scores, opponent_history)` — LLM builds a board given game context
- `get_valid_moves(board_state)` — returns valid placements (uses our existing masking logic)
- `evaluate_board(board)` — validates a proposed board

The LLM would call `get_valid_moves` iteratively, reasoning about each placement, and build a board step by step — similar to how a human uses the interactive builder.

### Experiment Protocol

Run a tournament:
1. **RL Expert vs Pool Opponents** — baseline (we already know this: ~65-78% win rate)
2. **LLM vs Pool Opponents** — how does the LLM do against the same fixed opponents?
3. **RL Expert vs LLM** — head-to-head across 100 games
4. **LLM vs RL Beginner** — can the LLM at least beat a weak RL agent?

### What to Track

| Metric | RL Agent | LLM Agent |
|---|---|---|
| Valid board rate | ~100% | ? (with masking tools, should be high) |
| Win rate vs pools | 65-78% | ? |
| Win rate head-to-head | ? | ? |
| Avg time per board | ~1ms | ~2-5s (API latency) |
| Cost per game | ~$0 | ~$0.05-0.50 (API calls) |
| Strategy adaptation across rounds | Learned from self-play | Reasoned from context |

**Hypothesis:** The RL agent wins decisively. It has internalized millions of games of strategic experience into its policy weights. The LLM has general reasoning ability but no game-specific optimization. The LLM might produce creative or surprising boards, but they won't be systematically optimized for winning.

**The interesting question:** Does the LLM show any round-over-round adaptation? It can see opponent history — does it reason about patterns, or does it treat each round independently?

**What carries over from existing work:**
- Existing RL models as one side of the matchup
- Board validation (`is_board_playable()`) for validating LLM outputs
- Simulation engine for running the games
- Action masking logic exposed via MCP tools

---

## Experiment 3: Fog Signal Ablation

**Question:** Which fog signals actually help the agent? Is the `fog_outcomes` metadata necessary, or can the agent learn from partial boards alone?

### Variants

**3A: Partial boards only** — `opponent_history` fog-filtered, no `fog_outcomes` field at all. The agent only sees the truncated grid.

**3B: Partial boards + outcome booleans** — Add `opponent_hit_trap`, `player_hit_trap`, `collision`, `opponent_reached_goal` per round. The agent knows *what happened* but not how many traps or how far the opponent got.

**3C: Full fog_outcomes (current implementation)** — All 6 channels: opponent_steps_visible, opponent_hit_trap, player_hit_trap, collision, opponent_reached_goal, visible_opponent_traps. Maximum structured information available to a human watching the play-by-play.

### What to Track

Compare all three at the same timestep budget:
- Win rate at final phase
- Does the agent adapt strategy across rounds? (Measure board diversity round-over-round)
- Performance delta between 3A/3B/3C tells us which signals the agent actually uses

**Hypothesis:** 3C > 3B > 3A, but the gap between 3B and 3C may be small. The outcome booleans are the most actionable signals. Knowing the exact step count and trap count is more subtle — the agent may not learn to exploit them within a reasonable training budget.

**What carries over from existing work:**
- Everything from Experiment 1 — this runs on top of whichever fog approach wins
- The current `fog_outcomes` implementation (6 channels) corresponds to variant 3C. Variants 3A and 3B would require code changes to reduce the signal set

---

## Experiment 4: Fog + Self-Play Dynamics

**Question:** Does self-play under fog produce different emergent strategies than self-play with full reveal?

Under full reveal, the opponent's entire board is visible after each round. The meta-game is about predicting and countering the opponent's *next* board based on their *previous* boards. Under fog, the agent only gets partial information — so the meta-game shifts toward inference and deception.

### What to Look For

- **Trap placement patterns:** Under fog, traps placed after `opponentLastStep` are invisible. Does the agent learn to place traps late in the sequence to hide them?
- **Deceptive construction:** Does the agent learn to build boards that look one way from the partial view but are actually different? (e.g., piece path suggests left column, but a hidden trap covers the right)
- **Information gathering:** Does the agent occasionally build boards designed to *reveal* the opponent's strategy (e.g., a board that forces the opponent deep into their sequence, exposing more of their layout)?

### Protocol

Train two agents to convergence:
1. **Full-reveal self-play** (our existing Stage 3 models)
2. **Fog self-play** (new Stage 4)

Then cross-evaluate:
- Fog agent vs full-reveal agent (under fog rules)
- Fog agent vs full-reveal agent (under full-reveal rules)

**Hypothesis:** The fog-trained agent develops more robust, less exploitable strategies because it can't rely on seeing the opponent's full board. The full-reveal agent may have higher win rates in its native environment but be brittle when information is limited.

**What carries over from existing work:**
- Stage 3 self-play models as the full-reveal baseline
- Self-play infrastructure (`SelfPlayCallback`, snapshot pool, `--self-play-ratio`)

---

## Future Experiment Ideas

- **Transfer learning across sizes:** Can a size-3 fog agent's weights initialize a size-4 fog agent? (Obs space dimensions differ, but could we transfer the policy trunk?)
- **Asymmetric fog:** One player has fog, the other doesn't. Does the fog player learn defensive strategies?
- **Human-in-the-loop evaluation:** Have human players rate board quality from RL vs LLM opponents. Which feels more "intelligent"?
- **Reward shaping for fog:** Are the existing construction shaping rewards (+0.1 piece, +0.3 row 0) still optimal under fog, or does partial observability need different incentives?

---

## Running Experiments

All experiments should follow the same protocol:
1. Document hypothesis before training
2. Use TensorBoard for monitoring (`tensorboard --logdir logs/`)
3. Save phase_history.json for post-hoc analysis
4. Run at size 3 first (fast iteration), size 4 to validate
5. Record results in this document or in journal entries

Size 3 baseline for comparison:
- **Full reveal, pool only:** All phases cleared by 256k steps, 70-95% win rate at phase 6
- **Full reveal + self-play:** Asymptotic at ~65% win rate by 1.2M steps
