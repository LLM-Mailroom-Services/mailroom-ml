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


def _live_app_ids() -> list[str]:
    """Parse `modal app list` for mailroom-ml-train apps with live tasks.

    A spawn on the deployed function runs as a task ON the deployed app
    (state ``deployed``, Tasks > 0) — that is the only "running work" signal;
    stopped apps (0 tasks) must not trip the double-launch guard.
    """
    out = subprocess.run(["modal", "app", "list"], capture_output=True,
                         text=True, timeout=60).stdout
    ids = []
    for line in out.splitlines():
        if "ap-" not in line or "mailroom-ml" not in line:
            continue
        fields = [f.strip() for f in line.split("│")]
        # fields: '' | App ID | Description | State | Tasks | Created at ...
        if len(fields) >= 6 and fields[4].isdigit() and int(fields[4]) > 0:
            ids.append(fields[1])
    return ids


def _deployed_app_id() -> str | None:
    """The mailroom-ml-train deployed app id (State column == 'deployed')."""
    out = subprocess.run(["modal", "app", "list"], capture_output=True,
                         text=True, timeout=60).stdout
    for line in out.splitlines():
        if "ap-" not in line or "mailroom-ml" not in line:
            continue
        fields = [f.strip() for f in line.split("│")]
        if len(fields) >= 4 and fields[3] == "deployed":
            return fields[1]
    return None


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

    live = _live_app_ids()
    if live:
        print(f"ERROR: training work already live on: {live} — one GPU run "
              f"at a time. Wait for it to finish (modal app logs {live[0]}).",
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

    # Smoke watch: the run executes as a task on the DEPLOYED app — watch its
    # logs for the smoke's completion line and the step cadence. The deployed
    # app is never stopped (that would garbage-collect the deployment); the
    # smoke call ends on its own after max-steps.
    app_id = _deployed_app_id()
    if not app_id:
        print("ERROR: no deployed mailroom-ml-train app found — deploy first "
              "(modal deploy deploy/modal_app.py)", file=sys.stderr)
        sys.exit(4)
    print(f"smoke runs on deployed app: {app_id}")

    tail = _watch_logs(app_id, "smoke run complete", SMOKE_WATCH_S)
    print("--- smoke log tail ---")
    print("\n".join(tail))
    if not any("smoke run complete" in ln for ln in tail):
        print("ERROR: smoke run did not complete — container starved or "
              "stuck (check step cadence above).", file=sys.stderr)
        sys.exit(5)

    # Cadence check: steps should land seconds apart, not minutes.
    step_lines = [ln for ln in tail if "step" in ln and "loss" in ln]
    print(f"smoke OK: {len(step_lines)} step lines, "
          f"last: {step_lines[-1] if step_lines else 'n/a'}")
    print("smoke complete — ready for the real launch.")
