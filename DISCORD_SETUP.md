# Discord Training Monitor Setup

Get Discord notifications during RL training runs — milestone alerts when phases advance and periodic check-ins with trend analysis.

For CLI flag reference, see [TRAINING_PLAN.md](TRAINING_PLAN.md#discord-notifications).

---

## 1. Create a Discord Webhook

1. Open your Discord server
2. Go to **Server Settings** > **Integrations** > **Webhooks**
3. Click **New Webhook**
4. Choose a channel (e.g., `#training-alerts`)
5. Optionally rename it (e.g., "Spaces Training Bot")
6. Click **Copy Webhook URL**

The URL looks like: `https://discord.com/api/webhooks/123456789/abcdefg...`

## 2. Use It

Pass the webhook URL when launching training:

```bash
python examples/train_simultaneous.py \
    --size 3 --fog --self-play \
    --timesteps 7.5M \
    --discord-webhook "https://discord.com/api/webhooks/YOUR/WEBHOOK"
```

Optionally set the check-in interval (default: 30 minutes):

```bash
    --discord-check-in 15   # check-in every 15 minutes
```

No webhook URL = no Discord code runs, no errors.

## 3. What You'll See

### Milestone Alerts (immediate, event-driven)

These fire once per state transition — not on every step.

| Event | When | Color |
|-------|------|-------|
| Training started | `model.learn()` begins | Blue |
| Phase advanced | Opponent phase increases | Blue |
| All phases cleared | Final phase reached | Green |
| Self-play level advanced | Window level increases | Blue |
| Self-play level backtracked | Window level decreases | Orange |
| Entered pool recovery | Win rate collapsed | Red |
| Exited pool recovery | Win rate recovered | Green |
| Training complete | `model.learn()` returns | Green |

### Periodic Check-Ins (every N minutes)

Summary embed with:
- Progress (steps / total, percentage)
- Current phase and win rate
- Self-play level and snapshot count
- Trend commentary

Example check-in:

> **Check-In — Size 3 Fog+Self-Play**
>
> Win rate climbing steadily — training progressing well.
>
> Progress: 2.1M / 7.5M (28%)
> Phase: 6/6 | Win Rate: 74% | Valid Rate: 100%
> Self-Play: Level 2, 5 snapshots

### Commentary Heuristics

The check-in includes automated analysis of recent evaluation history:

| Pattern | Commentary |
|---------|-----------|
| Win rate improving (>5% over last 5 evals) | "Win rate climbing steadily — training progressing well." |
| Win rate flat over 10+ evals | "Win rate plateaued at ~75%. May be near convergence." |
| Win rate declining (>5% drop) | "Win rate declining — watch for policy collapse." |
| Valid rate below 95% | "Valid rate dropped — unusual with strict masking, check logs." |
| Win rate range >15% during self-play | "Win rate volatile during self-play — normal during adaptation." |
| In pool recovery | "In pool recovery — waiting for win rate to stabilize." |

## 4. Environment Variable (Optional)

To avoid pasting the webhook URL every time, export it:

```bash
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR/WEBHOOK"

python examples/train_simultaneous.py \
    --size 3 --timesteps 2M \
    --discord-webhook "$DISCORD_WEBHOOK"
```

Add to `~/.bashrc` or `~/.zshrc` to persist across sessions.

## 5. Troubleshooting

**No messages appearing**: Check the webhook URL is correct. Run a quick test:
```bash
curl -H "Content-Type: application/json" \
     -d '{"content": "Test message"}' \
     "https://discord.com/api/webhooks/YOUR/WEBHOOK"
```

**Webhook failures don't crash training**: All Discord sends are wrapped in try/except. If the webhook is unreachable (network issue, rate limit, invalid URL), training continues normally and a warning is printed to the console.

**Rate limits**: Discord webhooks allow ~30 messages per minute. Phase advances are infrequent, and check-ins default to every 30 minutes, so rate limits shouldn't be an issue. If you set `--discord-check-in 1` on a fast-advancing size-2 run, you might hit limits — increase the interval.

**Testing locally**: Use a short run to verify messages arrive:
```bash
python examples/train_simultaneous.py \
    --size 2 --timesteps 50k \
    --discord-webhook "$DISCORD_WEBHOOK" \
    --discord-check-in 1
```
Size 2 advances phases quickly, so you'll see milestone alerts within a minute.
