"""Pool-only training to capture beginner/easy/medium difficulty snapshots.

Runs fog pool-only training for each specified size, saves snapshots at:
  - Beginner:  phase 1 (mixed_traps solo) at 65% WR
  - Easy:      phase 2 (simple+mixed combined) at 65% WR
  - Medium:    entering phase 3 (super_move solo)

Stops training once medium is captured (no need to run to convergence).
Does NOT run self-play — this is only for lower difficulty tiers.

If the monitor misses a phase (fast sizes), falls back to phase checkpoints
saved by the training callback (phase_N_checkpoint.zip).

Usage:
  python scripts/train_pool_difficulty.py SIZES [WEBHOOK_URL]

  SIZES is a comma-separated list or range, e.g.:
    python scripts/train_pool_difficulty.py 2,3,4,5,6,7
    python scripts/train_pool_difficulty.py 2-7
    python scripts/train_pool_difficulty.py 3 https://discord.com/api/webhooks/...

  WEBHOOK_URL is optional — if omitted, no Discord notifications are sent.
"""
import sys
import os
import time
import json
import subprocess
import glob
import urllib.request
import shutil

sys.stdout.reconfigure(line_buffering=True)


def parse_sizes(arg):
    """Parse '2-7' or '2,3,4,5,6,7' into a list of ints."""
    sizes = []
    for part in arg.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            sizes.extend(range(int(lo), int(hi) + 1))
        else:
            sizes.append(int(part))
    return sizes


if len(sys.argv) < 2:
    print("Usage: python scripts/train_pool_difficulty.py SIZES [WEBHOOK_URL]")
    print("  SIZES: comma-separated or range (e.g. 2-7 or 2,3,5)")
    sys.exit(1)

SIZES = parse_sizes(sys.argv[1])
WEBHOOK = sys.argv[2] if len(sys.argv) >= 3 else None

# Find venv python
VENV_PYTHON = os.path.join(os.getcwd(), "venv", "bin", "python")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = os.path.join(os.getcwd(), ".venv", "bin", "python")
if not os.path.exists(VENV_PYTHON):
    print("ERROR: No venv found at venv/ or .venv/")
    sys.exit(1)


def send_discord(msg, color=0x3498DB, title="Pool Difficulty Pipeline"):
    if not WEBHOOK:
        return
    embed = {"title": title, "description": msg, "color": color}
    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "SpacesGameMonitor/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"  [{time.strftime('%H:%M:%S')}] Discord sent (status {resp.status})")
    except Exception as e:
        print(f"  [{time.strftime('%H:%M:%S')}] Discord failed: {e}")


def get_metrics(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    runs = sorted(glob.glob(os.path.join(logdir, "*")))
    if not runs:
        return None
    ea = EventAccumulator(runs[-1])
    ea.Reload()

    def latest(tag):
        try:
            events = ea.Scalars(tag)
            return events[-1].value if events else None
        except Exception:
            return None

    def latest_step():
        for tag in ["train/explained_variance", "rollout/ep_rew_mean"]:
            try:
                events = ea.Scalars(tag)
                if events:
                    return events[-1].step
            except Exception:
                pass
        return 0

    return {
        "step": latest_step(),
        "phase": latest("curriculum/opponent_phase"),
        "wr": latest("curriculum/game_win_rate"),
        "ev": latest("train/explained_variance"),
    }


def save_snapshot(stage_dir, name, label):
    """Copy the current best model to a named snapshot."""
    best = os.path.join(stage_dir, "best", "best_model.zip")
    if not os.path.exists(best):
        print(f"  WARNING: No best model to save as {name}")
        return False
    diff_dir = os.path.join(stage_dir, "difficulty")
    os.makedirs(diff_dir, exist_ok=True)
    dest = os.path.join(diff_dir, name)
    shutil.copy2(best, dest)
    print(f"  Saved {label}: {dest}")
    return True


def recover_from_phase_checkpoints(diff_stage_dir, dst_diff):
    """Fall back to phase checkpoints saved by the training callback."""
    recovered = {"beginner": False, "easy": False, "medium": False}

    mapping = {
        "beginner.zip": ["phase_0_checkpoint.zip", "phase_1_checkpoint.zip"],
        "easy.zip": ["phase_1_checkpoint.zip", "phase_2_checkpoint.zip"],
        "medium.zip": ["phase_3_checkpoint.zip", "phase_2_checkpoint.zip"],
    }

    os.makedirs(dst_diff, exist_ok=True)
    for name, candidates in mapping.items():
        dest = os.path.join(dst_diff, name)
        if os.path.exists(dest):
            recovered[name.replace(".zip", "")] = True
            continue
        for candidate in candidates:
            src = os.path.join(diff_stage_dir, candidate)
            if os.path.exists(src):
                shutil.copy2(src, dest)
                print(f"  Recovered {name} from {candidate}")
                recovered[name.replace(".zip", "")] = True
                break

    return recovered


def train_size(size):
    """Run pool-only training for one size, capturing difficulty snapshots."""
    label = f"Size {size}"
    logdir = f"logs/size{size}_stage4/"
    stage_dir = f"models/size{size}/stage4"
    diff_stage_dir = f"models/size{size}/stage4_difficulty"

    # Check if boards exist
    board_dir = f"boards/size{size}"
    if not os.path.isdir(board_dir):
        print(f"  SKIP: No board directory at {board_dir}")
        return False

    # Check if snapshots already exist in main stage dir
    dst_diff = os.path.join(stage_dir, "difficulty")
    existing = []
    for name in ["beginner.zip", "easy.zip", "medium.zip"]:
        if os.path.exists(os.path.join(dst_diff, name)):
            existing.append(name)
    if len(existing) == 3:
        print(f"  SKIP: All 3 snapshots already exist in {dst_diff}")
        return True

    cmd = [
        VENV_PYTHON, "examples/train_simultaneous.py",
        "--size", str(size), "--fog",
        "--timesteps", "10000000",
        "--learning-rate", "1e-4",
        "--ent-coef", "0.1",
        "--n-steps", "4096",
        "--output-dir", diff_stage_dir,
    ]
    if WEBHOOK:
        cmd.extend(["--discord-webhook", WEBHOOK, "--discord-check-in", "60"])

    logfile = f"/tmp/size{size}_pool_difficulty.log"
    with open(logfile, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    saved_beginner = False
    saved_easy = False
    saved_medium = False
    MONITOR_INTERVAL = 60  # check every 1 min
    DISCORD_INTERVAL = 600  # Discord update every 10 min
    last_discord = time.time()
    prev_step = 0
    stall_count = 0

    while proc.poll() is None:
        time.sleep(MONITOR_INTERVAL)

        m = get_metrics(logdir)
        if m is None or m["step"] == 0:
            continue

        phase = int(m["phase"]) if m["phase"] is not None else 0
        wr = m["wr"] or 0
        ev = m["ev"] or 0
        step = m["step"]

        status = f"Step {step/1e6:.2f}M | Phase {phase} | WR {wr:.0%} | EV {ev:.3f}"
        saved_str = f" | Saved: {'+'.join(n for n, s in [('B', saved_beginner), ('E', saved_easy), ('M', saved_medium)] if s) or 'none'}"
        print(f"  [{time.strftime('%H:%M:%S')}] {status}{saved_str}")

        # Periodic Discord updates
        now = time.time()
        if now - last_discord >= DISCORD_INTERVAL:
            send_discord(
                f"{label}: {status}{saved_str}",
                color=0x3498DB,
            )
            last_discord = now

        # Stall detection
        if step == prev_step:
            stall_count += 1
            if stall_count >= 30:  # 30 min at 1-min interval
                print(f"  WARNING: No progress for 30 min!")
                send_discord(f"{label}: WARNING — no progress for 30 min!", color=0xE74C3C)
                stall_count = 0  # reset so we don't spam
        else:
            stall_count = 0
            prev_step = step

        # Beginner: phase 1 at 65% WR, or past phase 1 (grab what we can)
        if not saved_beginner:
            if phase == 1 and wr >= 0.65:
                saved_beginner = save_snapshot(diff_stage_dir, "beginner.zip", f"{label} Beginner (phase 1, {wr:.0%} WR)")
                send_discord(f"{label}: Saved beginner (phase 1, {wr:.0%} WR)", color=0xF39C12)
            elif phase >= 2:
                # Missed phase 1 — use phase_0 or phase_1 checkpoint from training callback
                for ckpt in ["phase_1_checkpoint.zip", "phase_0_checkpoint.zip"]:
                    src = os.path.join(diff_stage_dir, ckpt)
                    if os.path.exists(src):
                        os.makedirs(os.path.join(diff_stage_dir, "difficulty"), exist_ok=True)
                        shutil.copy2(src, os.path.join(diff_stage_dir, "difficulty", "beginner.zip"))
                        saved_beginner = True
                        print(f"  Recovered beginner from {ckpt}")
                        send_discord(f"{label}: Recovered beginner from {ckpt}", color=0xF39C12)
                        break

        # Easy: phase 2 at 65% WR, or past phase 2
        if not saved_easy:
            if phase == 2 and wr >= 0.65:
                saved_easy = save_snapshot(diff_stage_dir, "easy.zip", f"{label} Easy (phase 2, {wr:.0%} WR)")
                send_discord(f"{label}: Saved easy (phase 2, {wr:.0%} WR)", color=0xF39C12)
            elif phase >= 3:
                for ckpt in ["phase_2_checkpoint.zip", "phase_1_checkpoint.zip"]:
                    src = os.path.join(diff_stage_dir, ckpt)
                    if os.path.exists(src):
                        os.makedirs(os.path.join(diff_stage_dir, "difficulty"), exist_ok=True)
                        shutil.copy2(src, os.path.join(diff_stage_dir, "difficulty", "easy.zip"))
                        saved_easy = True
                        print(f"  Recovered easy from {ckpt}")
                        send_discord(f"{label}: Recovered easy from {ckpt}", color=0xF39C12)
                        break

        # Medium: entering phase 3, or past phase 3
        if not saved_medium:
            if phase == 3:
                saved_medium = save_snapshot(diff_stage_dir, "medium.zip", f"{label} Medium (entering phase 3)")
                send_discord(f"{label}: Saved medium (entering phase 3)", color=0xF39C12)
            elif phase >= 4:
                for ckpt in ["phase_3_checkpoint.zip", "phase_2_checkpoint.zip"]:
                    src = os.path.join(diff_stage_dir, ckpt)
                    if os.path.exists(src):
                        os.makedirs(os.path.join(diff_stage_dir, "difficulty"), exist_ok=True)
                        shutil.copy2(src, os.path.join(diff_stage_dir, "difficulty", "medium.zip"))
                        saved_medium = True
                        print(f"  Recovered medium from {ckpt}")
                        send_discord(f"{label}: Recovered medium from {ckpt}", color=0xF39C12)
                        break

        # All captured — stop training
        if saved_beginner and saved_easy and saved_medium:
            print(f"  All 3 snapshots captured! Stopping training.")
            send_discord(f"{label}: All 3 snapshots captured! Stopping training.", color=0x2ECC71)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break

    # Process exited — try to recover any missing snapshots from phase checkpoints
    if not (saved_beginner and saved_easy and saved_medium):
        print(f"  Training ended, checking for phase checkpoints to recover missing snapshots...")
        recovered = recover_from_phase_checkpoints(diff_stage_dir, os.path.join(diff_stage_dir, "difficulty"))
        if not saved_beginner and recovered["beginner"]:
            saved_beginner = True
        if not saved_easy and recovered["easy"]:
            saved_easy = True
        if not saved_medium and recovered["medium"]:
            saved_medium = True

    if not (saved_beginner and saved_easy and saved_medium):
        missing = []
        if not saved_beginner:
            missing.append("beginner")
        if not saved_easy:
            missing.append("easy")
        if not saved_medium:
            missing.append("medium")
        print(f"  WARNING: Could not capture: {', '.join(missing)}")
        send_discord(f"{label}: WARNING — missing: {', '.join(missing)}", color=0xE74C3C)
        return False

    # Copy snapshots to the main stage4 difficulty dir
    src_diff = os.path.join(diff_stage_dir, "difficulty")
    os.makedirs(dst_diff, exist_ok=True)
    for name in ["beginner.zip", "easy.zip", "medium.zip"]:
        src = os.path.join(src_diff, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_diff, name))
            print(f"  Deployed {name} to {dst_diff}")

    return True


# ============ Main ============
print("=" * 60)
print("Pool-Only Difficulty Snapshot Pipeline")
print(f"Sizes: {SIZES}")
print(f"Discord: {'enabled' if WEBHOOK else 'disabled'}")
print("=" * 60)

send_discord(
    f"Starting pool difficulty pipeline for sizes {SIZES}\n\n"
    f"Capturing: beginner (phase 1 @ 65% WR), easy (phase 2 @ 65% WR), medium (entering phase 3)",
    color=0x3498DB,
)

results = {}
for size in SIZES:
    print(f"\n{'='*40}")
    print(f"Size {size}")
    print(f"{'='*40}")
    send_discord(f"Starting size {size} pool training for difficulty snapshots", color=0x3498DB)
    ok = train_size(size)
    results[size] = ok
    status = "OK" if ok else "INCOMPLETE"
    print(f"  Result: {status}")
    send_discord(
        f"Size {size}: {status}",
        color=0x2ECC71 if ok else 0xE74C3C,
    )

# Summary
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
summary_lines = []
for size in SIZES:
    status = "OK" if results[size] else "INCOMPLETE"
    print(f"  Size {size}: {status}")
    summary_lines.append(f"Size {size}: {status}")

send_discord(
    "Pool difficulty pipeline complete!\n\n" + "\n".join(summary_lines),
    color=0x2ECC71 if all(results.values()) else 0xF39C12,
)
print("\nDone!")
