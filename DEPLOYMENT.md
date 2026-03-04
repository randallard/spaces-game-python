# Deployment Guide: Railway

This document covers how the Spaces Game Python inference server is deployed to Railway, how the Node frontend connects to it, and what needs to change to support fog of war agent selection.

## Architecture Overview

```
┌─────────────────────┐        POST /construct-board        ┌──────────────────────────┐
│                     │ ──────────────────────────────────▸ │                          │
│  spaces-game-node   │                                     │  spaces-game-python      │
│  (Node/TypeScript)  │ ◂────────────────────────────────── │  (FastAPI on Railway)    │
│                     │        { board, valid, ... }         │                          │
│  - Game UI          │                                     │  - Model registry        │
│  - Simulation       │        GET /info                    │  - Board construction    │
│  - Player input     │ ──────────────────────────────────▸ │  - Skill levels          │
│                     │        { sizes, models }            │  - Opponent pools        │
└─────────────────────┘                                     └──────────────────────────┘
      Railway                                                     Railway
```

Both services run on Railway. The Node app handles the game UI, player interaction, and simulation. When the AI agent needs to build a board, the Node app sends a POST request to the Python inference server with the current game state. The server loads the appropriate trained model, constructs a board, and returns it.

## Current Deployment (Stage 3)

### What Gets Deployed

The Python inference server runs as a single Railway web service. It serves trained RL models that construct boards with full opponent board visibility (Stage 3).

**Deployed files:**
- `inference_server/` — FastAPI application
- `spaces_game/` — core game engine (types, simulation, validation, envs)
- `boards/` — opponent board pools (JSON)
- `models/size{N}/stage3/` — trained model checkpoints (3 per board size)
- `Procfile` — `web: python -m inference_server.main`
- `requirements.txt` — Python dependencies (CPU-only PyTorch)

### Model Files in Git

Only production models are committed. The `.gitignore` allowlists:

**Skill-level checkpoints** (3 per board size per stage — used by the difficulty picker):
```
models/size*/stage{3,4}/beginner.zip
models/size*/stage{3,4}/intermediate.zip
models/size*/stage{3,4}/expert.zip
```

**Browse-menu models** (any `.zip` in these subdirectories — shown in "Browse all available models"):
```
models/size*/stage{3,4}/level_advancement/*.zip
models/size*/stage{3,4}/more_available_models/*.zip
```

To add a model to the browse menu, copy it into the appropriate `more_available_models/` directory:
```bash
mkdir -p models/size3/stage4/more_available_models/
cp my_checkpoint.zip models/size3/stage4/more_available_models/
git add models/size3/stage4/more_available_models/my_checkpoint.zip
```

The model's filename (minus `.zip`) becomes its display label in the UI.

All other training artifacts (logs, intermediate checkpoints, eval data) are gitignored.

### Environment Variables

Railway sets these automatically or via the dashboard:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | (Railway-provided) | Server listen port |
| `INFERENCE_MODELS_DIR` | `models/` | Path to model checkpoints |
| `INFERENCE_BOARDS_DIR` | `boards/` | Path to opponent board pools |
| `INFERENCE_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |

For production, set `INFERENCE_CORS_ORIGINS` to the Node app's Railway URL.

### API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check (Railway uses this) |
| GET | `/info` | Loaded models, supported board sizes |
| GET | `/models` | All available models (powers "Browse all available models" UI) |
| POST | `/construct-board` | Build a board using the AI agent |

**`POST /construct-board` request:**
```json
{
    "board_size": 3,
    "round_num": 0,
    "agent_score": 0,
    "opponent_score": 0,
    "opponent_history": [],
    "skill_level": "intermediate"
}
```

**Response:**
```json
{
    "board": {
        "sequence": [...],
        "boardSize": 3,
        "grid": [...]
    },
    "valid": true,
    "attempts_used": 1,
    "model_info": {
        "skill_level": "intermediate",
        "deterministic": false,
        "uses_masks": true,
        "model_board_size": 3
    }
}
```

**Skill levels:** `beginner`, `beginner_plus`, `intermediate`, `intermediate_plus`, `advanced`, `advanced_plus`, `test_fail`, `scripted_1`, `scripted_2`, `scripted_3`, `scripted_4`

Each RL skill level maps to a checkpoint (early/mid/advanced) and a sampling mode (stochastic/deterministic). Lower skill = earlier checkpoint + stochastic sampling. Higher skill = best checkpoint + deterministic.

**Scripted agents** (`scripted_1` through `scripted_4`) bypass the RL model pipeline entirely. They use deterministic board-building strategies and require no model files or opponent pools. They accept an optional `round_scores` field in the request for reactive behavior.

| Level | Name | Traps | Column Strategy | Reactive Behavior |
|-------|------|-------|-----------------|-------------------|
| `scripted_1` | Simple | None | One switch at a random round | None |
| `scripted_2` | Reactive | None | Switch column if lost last round | Score-aware |
| `scripted_3` | Trapper | Adjacent to start | Switch column if lost last round | Score-aware + traps |
| `scripted_4` | Adaptive | Adjacent to start | Always switch columns between rounds | Drops trap after losing 2 in a row |

**`round_scores` format** (optional, used by scripted agents):
```json
{
    "round_scores": [
        {"agent": 1.0, "opponent": 0.0},
        {"agent": 0.0, "opponent": 2.0}
    ]
}
```
Scores are from the AI agent's perspective (agent = AI's points, opponent = player's points for that round).

### Deployment Workflow

```bash
# 1. Train models (local machine with GPU)
python examples/train_simultaneous.py --size 3 --timesteps 2000000

# 2. Copy production checkpoints
cp models/size3/stage3/difficulty/beginner.zip models/size3/stage3/beginner.zip
cp models/size3/stage3/difficulty/intermediate.zip models/size3/stage3/intermediate.zip
cp models/size3/stage3/difficulty/expert.zip models/size3/stage3/expert.zip

# 3. Commit and push (triggers Railway redeployment)
git add models/size3/stage3/*.zip
git commit -m "Update size 3 models"
git push
```

Railway automatically:
1. Detects the push to main
2. Installs dependencies from `requirements.txt`
3. Runs `web: python -m inference_server.main` (from `Procfile`)
4. Model registry discovers and loads all `.zip` files on startup
5. Server starts accepting requests

### How the Node App Calls the Inference Server

The Node app (`spaces-game-node`) has an AI agent integration module that:

1. Calls `POST /construct-board` with the current game state (board size, round, scores, opponent history, skill level)
2. If the response has `valid: true` — uses the board, simulates the round
3. If `valid: false` — retries up to 5 times, then offers the player a forfeit option
4. The `test_fail` skill level always returns an invalid board (for testing the retry/forfeit UI flow)

---

## Adding Fog of War (Stage 4)

With fog of war implemented in the training environment (`SimultaneousPlayEnv(use_fog=True)`), the next step is letting players choose whether to play against a Stage 3 agent (full reveal) or a Stage 4 agent (fog of war). This requires changes to three components:

### 1. Inference Server Changes

**Model registry — discover Stage 4 models:**

The `ModelRegistry` currently only looks in `models/size{N}/stage3/`. It needs to also scan `models/size{N}/stage4/` and track which stage each model belongs to.

```
models/
├── size3/
│   ├── stage3/          # Full reveal models (existing)
│   │   ├── beginner.zip
│   │   ├── intermediate.zip
│   │   └── expert.zip
│   └── stage4/          # Fog of war models (new)
│       ├── beginner.zip
│       ├── intermediate.zip
│       └── expert.zip
```

The registry cache key changes from `(board_size, checkpoint_type)` to `(board_size, stage, checkpoint_type)`.

**API — add `agent_type` to request:**

The `/construct-board` endpoint needs a new field to select which agent to use:

```json
{
    "board_size": 3,
    "round_num": 0,
    "agent_score": 0,
    "opponent_score": 0,
    "opponent_history": [],
    "skill_level": "intermediate",
    "agent_type": "fog"
}
```

`agent_type` values: `"standard"` (Stage 3, default) or `"fog"` (Stage 4).

The server uses `agent_type` to select the right model directory. If a fog model isn't available for the requested size, fall back to standard or return a clear error.

**Inference — fog-aware board construction:**

`build_board_for_round()` currently creates a `SimultaneousPlayEnv` with default settings. For fog models, it needs to pass `use_fog=True` so the env's observation space matches what the model was trained on. The fog model expects `fog_outcomes` in its obs space — if the env doesn't provide it, SB3 will crash on shape mismatch.

The opponent history encoding also changes: the server needs to decide whether to send full or fog-filtered opponent boards. Since the inference server doesn't have simulation details (the Node app runs simulation), the opponent history sent from the frontend is already what the human player saw — which under fog rules is already partial. So the server can pass it through as-is. The `fog_outcomes` for past rounds would need to be sent from the frontend or filled with zeros (the model can handle zeros — it sees them for future rounds during training).

**`/info` endpoint — expose available agent types:**

The `/info` response should indicate which agent types are available per board size:

```json
{
    "supported_board_sizes": [2, 3, 4],
    "loaded_models": {
        "size3": {
            "standard": [{"checkpoint": "early", "path": "..."}],
            "fog": [{"checkpoint": "early", "path": "..."}]
        }
    }
}
```

### 2. `.gitignore` — Allow Stage 4 Models

Add Stage 4 model paths to the allowlist:

```gitignore
!models/size*/stage4/
models/size*/stage4/*
!models/size*/stage4/beginner.zip
!models/size*/stage4/intermediate.zip
!models/size*/stage4/expert.zip
```

### 3. Node App Changes

The Node app needs updates to support agent type selection:

**Game setup UI — agent type picker:**

When starting a game against the AI, the player chooses:
- Board size (2, 3, or 4)
- Difficulty (beginner through advanced_plus)
- **Agent type: "Standard" or "Fog of War"** (new)

The UI should indicate what each means:
- **Standard**: The AI saw your full board after each round during training. It learned to counter specific board patterns.
- **Fog of War**: The AI only saw partial boards during training. It learned to infer your strategy from limited information — just like you do.

**Request construction — pass `agent_type`:**

The `requestAiAgentBoard()` function adds `agent_type` to the POST body. Default to `"standard"` for backward compatibility.

**Fog of war display:**

The existing `--fog` display mode in `play_against_agent.py` shows the human player a partial view of the opponent's board. The Node app should do the same when `agent_type` is `"fog"`:
- After simulation, show the human player only opponent moves up to `playerLastStep`
- Show only the sprung trap (if any)
- Display fog outcome indicators (hit trap, collision, etc.)

This is a display-side change — the simulation still runs with full information. The fog filtering happens when rendering the round results.

**`/info` integration — disable fog option when unavailable:**

On page load, the Node app calls `GET /info`. If fog models aren't available for the selected board size, grey out the "Fog of War" agent type option.

### 4. Fog of War Deployment Workflow

```bash
# 1. Train fog models (local machine with GPU)
python examples/train_simultaneous.py --size 3 --fog --timesteps 5000000

# 2. Copy production checkpoints
cp models/size3/stage4/difficulty/beginner.zip models/size3/stage4/beginner.zip
cp models/size3/stage4/difficulty/intermediate.zip models/size3/stage4/intermediate.zip
cp models/size3/stage4/difficulty/expert.zip models/size3/stage4/expert.zip

# 3. Commit and push
git add models/size3/stage4/*.zip
git commit -m "Add size 3 fog of war models"
git push

# 4. Railway auto-deploys, registry discovers stage4/ models on startup
# 5. Node app sees fog models in /info, enables fog agent type picker
```

### 5. Backward Compatibility

All changes are additive:
- `agent_type` defaults to `"standard"` — existing Node app versions work without changes
- Stage 3 models and paths are unchanged
- The `/construct-board` endpoint accepts requests with or without `agent_type`
- The `/info` endpoint adds fields but doesn't remove any

---

## Deployment Checklist

### Before First Deploy (one-time setup)
- [ ] Create Railway project for the Python inference server
- [ ] Set environment variables: `INFERENCE_CORS_ORIGINS` (Node app URL)
- [ ] Verify `Procfile` and `requirements.txt` are committed
- [ ] Push to main — Railway auto-deploys

### Per-Size Model Deploy
- [ ] Train models locally (Stage 3 and/or Stage 4)
- [ ] Copy beginner/intermediate/expert checkpoints to `models/size{N}/stage{3,4}/`
- [ ] Test locally: `python -m inference_server.main` then `curl localhost:8100/info`
- [ ] Commit model `.zip` files and push
- [ ] Verify Railway deployment: `curl <railway-url>/health`
- [ ] Verify from Node app: start a game, check AI constructs valid boards

### Adding Fog of War Support
- [x] Train Stage 4 fog models for target sizes
- [x] Update `.gitignore` for `stage4/` paths
- [x] Update `model_registry.py` to discover `stage4/` directories
- [x] Add `agent_type` field to `ConstructBoardRequest`
- [x] Update `build_board_for_round()` to pass `use_fog=True` for fog models
- [x] Update `/info` to expose available agent types
- [ ] Deploy Python server
- [ ] Update Node app: agent type picker UI, pass `agent_type` in requests, fog display mode
- [ ] Deploy Node app

---

## Troubleshooting

**"No models are loaded" (503):** Model `.zip` files aren't in the expected directory. Check `INFERENCE_MODELS_DIR` and verify the directory structure matches `models/size{N}/stage3/*.zip`.

**Board size not supported (400):** No models exist for that board size. Train and deploy models first.

**CORS errors in browser:** `INFERENCE_CORS_ORIGINS` doesn't include the Node app's URL. Set it in Railway's environment variables (comma-separated for multiple origins).

**Model shape mismatch (500):** The model was trained on a different board size or with a different observation space (e.g., Stage 3 model loaded for a fog request). Check that the right model is in the right `stage{N}/` directory.

**Slow cold start:** PyTorch + model loading takes 10-20 seconds on Railway's starter tier. The health check may fail during this window. Railway's restart policy handles this automatically.

---

## Related Documentation

- [TRAINING_PLAN.md](TRAINING_PLAN.md) — Full training curriculum, Stage 3 and Stage 4 details
- [EXPERIMENTS.md](EXPERIMENTS.md) — Fog of war experiments (signal ablation, fog curriculum, self-play dynamics)
- [journal/2026-02-16-fog-of-war-implementation.md](journal/2026-02-16-fog-of-war-implementation.md) — Implementation details and training lessons
- [journal/2026-02-11-inference-server-and-validation.md](journal/2026-02-11-inference-server-and-validation.md) — Inference server design history
