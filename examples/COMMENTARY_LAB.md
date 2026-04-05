# Commentary Lab Reference

Run 5-round games between any combination of agents, observe round-by-round, and collect notes.

```
python examples/commentary_lab.py --size <N> --agent-a <AGENT> --agent-b <AGENT> [OPTIONS]
```

## Agents

| Agent | Description |
|-------|-------------|
| `human` | You play — build boards interactively or pick from a library |
| `scripted_1` | Straight paths, no traps. Cycles columns predictably |
| `scripted_2` | Straight paths, no traps. Reacts to score — switches column after losing |
| `scripted_3` | Trap boards. Same column logic as level 2 |
| `scripted_4` | Trap boards. Rotates column each round |
| `scripted_5` | Adaptive — reads opponent history, uses supermoves, counters patterns |
| `beginner` | RL model (stage 4 training) |
| `intermediate` | RL model (stage 4 training) |
| `expert` | RL model (stage 4 training) |
| `path/to/model.zip` | Any .zip model file directly |

## Board Sizes

Models exist for sizes **2 through 9**. Board pools exist for sizes **2 through 10**.

Scripted agents work at any size.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--size` | 3 | Board dimension (NxN) |
| `--agent-a` | required | Agent A specification |
| `--agent-b` | required | Agent B specification |
| `--output-dir` | `lab_sessions/` | Where session JSON + markdown get saved |
| `--board-library` | `my_boards.json` | Board library file for human mode |

## Human Mode

When playing as `human`, each round you choose:

1. **Build interactively** — uses the color-coded direction builder (`move up`, `trap left`, `supermove right`, `finish`). After building, you can save the board to your library.
2. **Pick from library** — loads boards from your `--board-library` file, or falls back to `boards/sizeN/` pool files if no library exists.

## Examples

```bash
# Watch two scripted agents
python examples/commentary_lab.py --size 3 --agent-a scripted_1 --agent-b scripted_3

# Scripted vs RL
python examples/commentary_lab.py --size 4 --agent-a scripted_5 --agent-b expert

# RL vs RL
python examples/commentary_lab.py --size 3 --agent-a beginner --agent-b expert

# Play against a scripted opponent
python examples/commentary_lab.py --size 3 --agent-a human --agent-b scripted_3

# Play against an RL model
python examples/commentary_lab.py --size 3 --agent-a human --agent-b expert

# Play against expert on a bigger board with a custom board library
python examples/commentary_lab.py --size 6 --agent-a human --agent-b expert --board-library boards/my_favorites.json

# Two RL models on a large board
python examples/commentary_lab.py --size 7 --agent-a intermediate --agent-b expert

# Direct model path
python examples/commentary_lab.py --size 3 --agent-a models/size3/stage4/expert.zip --agent-b scripted_5
```

## Session Output

After each game, a JSON data file and a markdown summary are saved to `--output-dir`. The JSON includes full board paths, trap positions, scores, collisions, and your notes. The markdown is a readable recap.

## Fog of War

Boards are displayed with fog of war by default. This mirrors real gameplay — you don't see the opponent's full strategy.

- **Human player's own board**: always fully visible (no fog)
- **All other boards**: fogged — only shows piece moves up to the step where the opposing piece stopped, and only the trap that was actually sprung (if any). Hidden cells show `·`, hidden trace steps show `???`.
- **AI vs AI spectating**: both boards fogged — you see what each side would have known

This applies to both the ASCII board display and the step-by-step simulation trace.

## During a Game

After each round you'll see:
- ASCII boards (your board in full, opponent boards under fog with `[FOG]` label)
- Step-by-step simulation trace (fogged — hidden steps show as `???`)
- Score box with running totals
- Note prompt — type observations, hit Enter twice to finish (or once to skip)

At game end: full recap and optional overall game notes.
