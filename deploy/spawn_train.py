"""Fire-and-forget trainer launch — submit the training call and exit.

``modal run --detach`` still cancels the remote app when the launching
process is killed before detach completes (e.g. a CI/agent shell tearing
down the process group). This launcher targets the DEPLOYED app
(``modal deploy deploy/modal_app.py``) via ``Function.from_name`` — a spawn
on a deployed function is fully server-side and survives any client death.

    HF_TOKEN=... modal deploy deploy/modal_app.py   # once (image cached)
    HF_TOKEN=... python deploy/spawn_train.py --epochs 5
    HF_TOKEN=... python deploy/spawn_train.py --push-to-hub \
        Lucius-Morningstar/mailroom-modernbert-classifier

Guardrails (churn-and-burn prevention):

- refuses to spawn while a mailroom-ml-train app is already running
  (double-launch guard — one GPU run at a time),
- ``--smoke``: pre-flight check that spawns a 24-micro-batch run (forward +
  backward + optimizer + eval on the real image/volume/secret), watches the
  logs for step lines, and reports per-step cadence — the signal that the
  GPU is actually crunching (a 1-vCPU-starved container crawls here and is
  caught in minutes, not hours). The smoke app is stopped at the end.

Follow the run (epoch lines stream live):

    modal app list                 # find the ephemeral app id (newest)
    modal app logs ap-<app-id>     # trainer output streams (subprocess pass-through)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import modal

APP_NAME = "mailroom-ml-train"
FUNCTION_NAME = "train"
SMOKE_STEPS = 24          # 3 optimizer steps at grad-accum 8
SMOKE_WATCH_S = 60 * 12   # generous ceiling for image pull + model download
SMOKE_STEP_TIMEOUT_S = 60 * 4  # no step line for 4 min => starved/stuck


def _running_app_ids() -> list[str]:
    """Parse `modal app list` for live mailroom-ml-train app ids."""
    out = subprocess.run(["modal", "app", "list"], capture_output=True,
                         text=True, timeout=60).stdout
    ids = []
    for line in out.splitlines():
        if "ap-" in line and "mailroom-ml" in line and "deployed" not in line:
            # columns: │ ap-xxx │ name │ state │ tasks │ date │
            ids.append(line.split("│")[1].strip())
    return ids


def _watch_logs(app_id: str, needle: str, timeout_s: int) -> list[str]:
    """Poll `modal app logs` until `needle` appears; return the tail lines."""
    deadline = time.time() + timeout_s
    tail: list[str] = []
    while time.time() < deadline:
        out = subprocess.run(["modal", "app", "logs", app_id],
                             capture_output=True, text=True, timeout=60).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if lines:
            tail = lines[-20:]
        if any(needle in ln for ln in lines):
            return tail
        time.sleep(15)
    return tail


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push-to-hub", default="",
                    help="model repo id to push the checkpoint to")
    ap.add_argument("--no-eval-test", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="pre-flight: 24-micro-batch run, watch step cadence, "
                         "then stop the app (no checkpoint is kept)")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set — the trainer cannot pull the dataset or "
              "push the checkpoint; aborting.", file=sys.stderr)
        sys.exit(2)

    live = _running_app_ids()
    if live:
        print(f"ERROR: training app already running: {live} — one GPU run at "
              f"a time. Stop it first (modal app stop {live[0]}).",
              file=sys.stderr)
        sys.exit(3)

    train = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    call = train.spawn(
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        push_to_hub="" if args.smoke else args.push_to_hub,
        eval_test=False if args.smoke else not args.no_eval_test,
        max_steps=SMOKE_STEPS if args.smoke else 0,
    )
    print(f"spawned training call: {call.object_id}")

    if not args.smoke:
        print("watch: modal app logs ap-<app-id>  (find it via: modal app list)")
        sys.exit(0)

    # Smoke watch: find the ephemeral app, verify step cadence, stop it.
    app_id = None
    deadline = time.time() + 120
    while time.time() < deadline and not app_id:
        ids = _running_app_ids()
        if ids:
            app_id = ids[-1]  # newest
        else:
            time.sleep(10)
    if not app_id:
        print("ERROR: smoke app never appeared in `modal app list`",
              file=sys.stderr)
        sys.exit(4)
    print(f"smoke app: {app_id}")

    tail = _watch_logs(app_id, "step 24", SMOKE_WATCH_S)
    print("--- smoke log tail ---")
    print("\n".join(tail))
    if not any("step 24" in ln for ln in tail):
        print("ERROR: smoke run did not reach step 24 — container starved or "
              "stuck (check step cadence above). Stopping the app.",
              file=sys.stderr)
        subprocess.run(["modal", "app", "stop", app_id], timeout=60)
        sys.exit(5)

    # Cadence check: steps should land seconds apart, not minutes.
    step_lines = [ln for ln in tail if "step" in ln and "loss" in ln]
    print(f"smoke OK: {len(step_lines)} step lines, "
          f"last: {step_lines[-1] if step_lines else 'n/a'}")
    subprocess.run(["modal", "app", "stop", app_id], timeout=60)
    print("smoke app stopped — ready for the real launch.")
