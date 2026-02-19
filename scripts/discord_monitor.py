"""Standalone Discord monitor for training runs. Posts check-ins every N minutes."""

import sys
import time
import json
import urllib.request
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WEBHOOK_URL = sys.argv[1]
LOGDIR = sys.argv[2]
INTERVAL_MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 30
LABEL = sys.argv[4] if len(sys.argv) > 4 else "Training"
TOTAL_STEPS = int(float(sys.argv[5])) if len(sys.argv) > 5 else 10_000_000


def latest(ea, tag):
    events = ea.Scalars(tag)
    return events[-1].value if events else None


def latest_step(ea):
    for tag in ["self_play/window_level", "train/explained_variance", "rollout/ep_rew_mean"]:
        events = ea.Scalars(tag)
        if events:
            return events[-1].step
    return 0


def send_check_in(ea):
    step = latest_step(ea)
    sp_wr = latest(ea, "self_play/sp_eval_win_rate")
    level = latest(ea, "self_play/window_level")
    pool_wr = latest(ea, "self_play/pool_win_rate")
    ev = latest(ea, "train/explained_variance")
    valid = latest(ea, "curriculum/valid_rate")
    recovery = latest(ea, "self_play/in_recovery")
    game_wr = latest(ea, "curriculum/game_win_rate")

    pct = step / TOTAL_STEPS if TOTAL_STEPS > 0 else 0

    fields = [
        {"name": "Steps", "value": f"{step/1e6:.2f}M / {TOTAL_STEPS/1e6:.0f}M ({pct:.0%})", "inline": False},
    ]

    if level is not None:
        fields.append({"name": "SP Window Level", "value": f"{int(level)}", "inline": True})
    if sp_wr is not None:
        fields.append({"name": "SP Eval WR", "value": f"{sp_wr:.0%}", "inline": True})
    if pool_wr is not None:
        fields.append({"name": "Pool WR", "value": f"{pool_wr:.0%}", "inline": True})
    if game_wr is not None and level is None:
        fields.append({"name": "Game WR", "value": f"{game_wr:.0%}", "inline": True})
    if valid is not None:
        fields.append({"name": "Valid Rate", "value": f"{valid:.0%}", "inline": True})
    if ev is not None:
        fields.append({"name": "Explained Var", "value": f"{ev:.3f}", "inline": True})
    if recovery is not None:
        fields.append({"name": "Recovery", "value": "YES" if recovery > 0 else "No", "inline": True})

    desc_parts = [f"Step {step/1e6:.2f}M ({pct:.0%})"]
    if level is not None:
        desc_parts.append(f"level {int(level)}")
    if sp_wr is not None:
        desc_parts.append(f"SP WR {sp_wr:.0%}")
    elif game_wr is not None:
        desc_parts.append(f"WR {game_wr:.0%}")

    embed = {
        "title": f"{LABEL} — Check-In",
        "description": ", ".join(desc_parts),
        "color": 0xE74C3C if (recovery and recovery > 0) else 0x2ECC71 if (sp_wr and sp_wr >= 0.8) else 0x3498DB,
        "fields": fields,
    }

    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "SpacesGameMonitor/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[{time.strftime('%H:%M:%S')}] Sent check-in: {', '.join(desc_parts)} (status {resp.status})")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Failed to send: {e}")


def main():
    print(f"Discord monitor: {LABEL}")
    print(f"  Logdir: {LOGDIR}")
    print(f"  Interval: {INTERVAL_MIN} min")
    print(f"  Total steps: {TOTAL_STEPS:,}")

    while True:
        try:
            ea = EventAccumulator(LOGDIR)
            ea.Reload()
            step = latest_step(ea)
            if step > 0:
                send_check_in(ea)
                if step >= TOTAL_STEPS:
                    print("Training complete, exiting monitor.")
                    break
            else:
                print(f"[{time.strftime('%H:%M:%S')}] No data yet, waiting...")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")

        time.sleep(INTERVAL_MIN * 60)


if __name__ == "__main__":
    main()
