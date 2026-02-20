"""Fog training pipeline: pool-only until converged, then self-play.

Saves difficulty-tier snapshots throughout:
  - Beginner:  saved mid-phase 1 (mixed_traps) at 65% WR — barely competent
  - Easy:      saved mid-phase 2 (simple+mixed combined) at 65% WR
  - Medium:    saved at the start of phase 3 (super_move) — weakest at new challenge
  - Hard:      self-play level 3-5 snapshots (from level_advancement/)
  - Expert:    self-play best model (level 10 / final)

Usage: python scripts/train_pipeline.py SIZE WEBHOOK_URL
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

if len(sys.argv) < 3:
    print("Usage: python scripts/train_pipeline.py SIZE WEBHOOK_URL")
    sys.exit(1)

SIZE = sys.argv[1]
WEBHOOK = sys.argv[2]

# Find venv python
VENV_PYTHON = os.path.join(os.getcwd(), "venv", "bin", "python")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = os.path.join(os.getcwd(), ".venv", "bin", "python")
if not os.path.exists(VENV_PYTHON):
    print("ERROR: No venv found at venv/ or .venv/")
    sys.exit(1)

BASE_ARGS = [
    VENV_PYTHON, "examples/train_simultaneous.py",
    "--size", SIZE, "--fog",
    "--timesteps", "10000000",
    "--learning-rate", "1e-4",
    "--ent-coef", "0.1",
    "--n-steps", "4096",
    "--discord-webhook", WEBHOOK,
    "--discord-check-in", "60",
]
LOGDIR = f"logs/size{SIZE}_stage4/"
STAGE_DIR = f"models/size{SIZE}/stage4"
DIFF_DIR = os.path.join(STAGE_DIR, "difficulty")
LABEL = f"Size {SIZE} Fog"

# ============ Helpers ============

def send_discord(msg, color=0x3498DB):
    embed = {"title": f"{LABEL} Pipeline", "description": msg, "color": color}
    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK, data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "SpacesGameMonitor/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[{time.strftime('%H:%M:%S')}] Discord sent (status {resp.status})")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Discord failed: {e}")


def get_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    runs = sorted(glob.glob(os.path.join(LOGDIR, "*")))
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
        for tag in ["self_play/window_level", "train/explained_variance", "rollout/ep_rew_mean"]:
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
        "valid": latest("curriculum/valid_rate"),
        "wr": latest("curriculum/game_win_rate"),
        "ev": latest("train/explained_variance"),
        "sp_level": latest("self_play/window_level"),
        "sp_wr": latest("self_play/sp_eval_win_rate"),
        "pool_wr": latest("self_play/pool_win_rate"),
    }


def save_snapshot(name, label):
    """Copy the current best model to a named snapshot."""
    best = os.path.join(STAGE_DIR, "best", "best_model.zip")
    if not os.path.exists(best):
        print(f"[{time.strftime('%H:%M:%S')}] WARNING: No best model to save as {name}")
        return
    os.makedirs(DIFF_DIR, exist_ok=True)
    dest = os.path.join(DIFF_DIR, name)
    shutil.copy2(best, dest)
    print(f"[{time.strftime('%H:%M:%S')}] Saved {label} snapshot: {dest}")
    send_discord(f"Saved {label} snapshot: `{name}`", color=0xF39C12)


def run_training(cmd, phase_name, check_fn, logfile):
    with open(logfile, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    MONITOR_INTERVAL = 300
    last_discord = 0
    DISCORD_INTERVAL = 3600
    prev_step = 0
    stall_count = 0

    while proc.poll() is None:
        time.sleep(MONITOR_INTERVAL)
        m = get_metrics()
        if m is None or m["step"] == 0:
            continue

        now = time.time()
        if now - last_discord >= DISCORD_INTERVAL:
            check_fn(m, send_update=True)
            last_discord = now

        if m["step"] == prev_step:
            stall_count += 1
            if stall_count >= 6:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: No progress for 30 min!")
                send_discord("WARNING: No training progress for 30+ minutes!", color=0xE74C3C)
        else:
            stall_count = 0
            prev_step = m["step"]

        result = check_fn(m, send_update=False)
        if result:
            print(f"[{time.strftime('%H:%M:%S')}] {phase_name} converged!")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return True

    print(f"[{time.strftime('%H:%M:%S')}] {phase_name} process exited with code {proc.returncode}")
    return proc.returncode == 0


# ============ Phase 1: Pool-only training ============
print("=" * 60)
print(f"PHASE 1: {LABEL} Pool-Only Training")
print("=" * 60)
send_discord(f"Starting {LABEL.lower()} pool-only training", color=0x3498DB)

# Track difficulty snapshot saves
saved_beginner = False
saved_easy = False
saved_medium = False


def check_pool(m, send_update=False):
    global saved_beginner, saved_easy, saved_medium
    phase = int(m["phase"]) if m["phase"] is not None else 0
    wr = m["wr"] or 0
    ev = m["ev"] or 0
    step = m["step"]
    status = f"Step {step/1e6:.2f}M | Phase {phase} | WR {wr:.0%} | EV {ev:.3f}"
    print(f"[{time.strftime('%H:%M:%S')}] {status}")

    if send_update:
        send_discord(f"Pool training: {status}", color=0x3498DB)

    # Save difficulty snapshots at specific moments during pool training.
    # These capture weak models for lower difficulty tiers.

    # Beginner: still learning mixed_traps (phase 1), WR just hitting 65%
    if phase == 1 and wr >= 0.65 and not saved_beginner:
        save_snapshot("beginner.zip", "Beginner (phase 1, 65% WR)")
        saved_beginner = True

    # Easy: still learning simple+mixed combined (phase 2), WR just hitting 65%
    if phase == 2 and wr >= 0.65 and not saved_easy:
        save_snapshot("easy.zip", "Easy (phase 2, 65% WR)")
        saved_easy = True

    # Medium: just entered super_move solo (phase 3) — weakest at this new challenge
    if phase == 3 and not saved_medium:
        save_snapshot("medium.zip", "Medium (entering phase 3)")
        saved_medium = True

    # Convergence check: phase 6 reached + 1M steps
    if phase >= 6 and step >= 1_000_000:
        save_snapshot("pool_best.zip", "Pool-only best (backup)")
        send_discord(
            f"Pool training converged!\n\nStep: {step/1e6:.2f}M\nPhase: {phase}\nWR: {wr:.0%}\nEV: {ev:.3f}\n\nStarting self-play...",
            color=0x2ECC71,
        )
        return True
    return False


pool_ok = run_training(
    BASE_ARGS,
    "Pool training",
    check_pool,
    f"/tmp/size{SIZE}_pool_training.log",
)
time.sleep(5)

# ============ Phase 2: Self-play training ============
print("\n" + "=" * 60)
print(f"PHASE 2: {LABEL} Self-Play Training")
print("=" * 60)

best_model = os.path.join(STAGE_DIR, "best", "best_model.zip")
if not os.path.exists(best_model):
    send_discord("ERROR: No best model found for self-play!", color=0xE74C3C)
    sys.exit(1)

send_discord(f"Starting {LABEL.lower()} self-play from best pool model", color=0x9B59B6)


def check_sp(m, send_update=False):
    sp_level = int(m["sp_level"]) if m["sp_level"] is not None else 0
    sp_wr = m["sp_wr"] if m["sp_wr"] is not None else 0
    ev = m["ev"] or 0
    step = m["step"]
    status = f"Step {step/1e6:.2f}M | Level {sp_level} | SP WR {sp_wr:.0%} | EV {ev:.3f}"
    print(f"[{time.strftime('%H:%M:%S')}] {status}")
    if send_update:
        send_discord(f"Self-play: {status}", color=0x9B59B6)
    if sp_level >= 10 and sp_wr >= 1.0:
        pool_wr = m["pool_wr"] or 0
        send_discord(
            f"Self-play COMPLETE!\n\nLevel: {sp_level}\nSP WR: {sp_wr:.0%}\nPool WR: {pool_wr:.0%}\nStep: {step/1e6:.2f}M",
            color=0x2ECC71,
        )
        return True
    return False


sp_ok = run_training(
    BASE_ARGS + ["--self-play", "--resume", best_model],
    "Self-play",
    check_sp,
    f"/tmp/size{SIZE}_sp_training.log",
)

# ============ Deploy production models ============
print("\n" + "=" * 60)
print(f"DEPLOYING: {LABEL} Production Models")
print("=" * 60)

# Expert = self-play best
best = os.path.join(STAGE_DIR, "best", "best_model.zip")
if os.path.exists(best):
    shutil.copy2(best, os.path.join(STAGE_DIR, "expert.zip"))
    print("Deployed expert.zip (self-play best)")

# Hard = self-play level 4 snapshot (mid-range self-play)
level_dir = os.path.join(STAGE_DIR, "level_advancement")
if os.path.isdir(level_dir):
    # Pick a mid-range level for "hard" — prefer level 4-5
    for lvl in [5, 4, 3, 6, 7]:
        hard_src = os.path.join(level_dir, f"level_{lvl}.zip")
        if os.path.exists(hard_src):
            shutil.copy2(hard_src, os.path.join(STAGE_DIR, "hard.zip"))
            print(f"Deployed hard.zip (self-play level {lvl})")
            break

# Medium, Easy, Beginner come from pool phase snapshots (saved during Phase 1)
for name in ["medium.zip", "easy.zip", "beginner.zip"]:
    src = os.path.join(DIFF_DIR, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(STAGE_DIR, name))
        print(f"Deployed {name}")
    else:
        print(f"WARNING: {name} not found in {DIFF_DIR}")

# Summary
deployed = []
for name in ["beginner.zip", "easy.zip", "medium.zip", "hard.zip", "expert.zip"]:
    if os.path.exists(os.path.join(STAGE_DIR, name)):
        deployed.append(name.replace(".zip", ""))

send_discord(
    f"{LABEL} pipeline COMPLETE!\n\nDeployed: {', '.join(deployed)}\n\nDifficulty tiers:\n"
    f"- Beginner: pool phase 0-1\n- Easy: pool phase 2-3\n- Medium: full pool\n"
    f"- Hard: self-play mid-level\n- Expert: self-play best",
    color=0x2ECC71,
)
print(f"\nDone! Deployed: {', '.join(deployed)}")
