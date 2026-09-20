"""Fire-and-forget trainer launch — submit the training call and exit.

Budget guard (2026-09-19): before a real (non-smoke) spawn, the estimated
run cost is compared against BUDGET_CEILING_USD (default $4.32, the Option A
envelope). The estimate uses the smoke run's measured training cadence and
dataset size (``~/.cache/mailroom-ml/smoke_metrics.json``), and a real launch
is REFUSED when no smoke metrics are on file — the smoke must run first so
the guard has real numbers, not guesses. A projection over the ceiling
aborts with exit code 6 unless ``--force`` is passed. The ceiling bounds the
*estimate* only — the app timeout is the hard wall on a pathological run, so
watch the first epoch's wall time.

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
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

APP_NAME = "mailroom-ml-train"
FUNCTION_NAME = "train"
SMOKE_STEPS = 24          # 3 optimizer steps at grad-accum 8
SMOKE_LOG_EVERY = 4       # step lines every 4 micro-batches during the smoke
SMOKE_WATCH_S = 60 * 12   # generous ceiling for image pull + model download
SMOKE_STEP_TIMEOUT_S = 60 * 4  # no step line for 4 min => starved/stuck

# ---------------------------------------------------------------------------
# Budget guard (2026-09-19) — explicit overspend protection for the real run.
#
# Rates from modal.com/pricing (fetched 2026-09-19): L4 $0.000222/s, CPU
# $0.0000131/core/s, memory $0.00000222/GiB/s. Option A (L4 + cpu=2 + 10 GiB)
# is therefore $0.9734/h all-in — the estimate multiplies that by the smoke's
# measured (epochs x steps/epoch x seconds/step) plus a fixed startup
# overhead. The guard refuses to launch a REAL run when the estimate exceeds
# BUDGET_CEILING_USD ($4.32, the plan envelope) unless --force is passed.
ALL_IN_USD_PER_HOUR = round(
    (0.000222 + 2 * 0.0000131 + 10 * 0.00000222) * 3600, 6)
BUDGET_CEILING_USD = 4.32
STARTUP_OVERHEAD_S = 60 * 30      # cold image pull + weight download allowance
SMOKE_METRICS_ENV = "MAILROOM_ML_SMOKE_METRICS"
SMOKE_METRICS_DEFAULT = Path.home() / ".cache" / "mailroom-ml" / "smoke_metrics.json"


def _smoke_metrics_path() -> Path:
    return Path(os.environ.get(SMOKE_METRICS_ENV, str(SMOKE_METRICS_DEFAULT)))


def _load_smoke_metrics() -> dict:
    p = _smoke_metrics_path()
    if not p.is_file():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_smoke_metrics(metrics: dict) -> Path:
    p = _smoke_metrics_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return p


def _sec_per_step_from_lines(lines: list[str]) -> float | None:
    """Median wall-clock seconds per micro-batch from timestamped step lines.

    Parses `modal app logs --timestamps` output: each trainer step line is
    ``<ts> ... step N loss ...`` (ts = ``YYYY-MM-DD HH:MM:SS``). Median across
    consecutive step deltas is robust to a single slow poll interval.
    """
    steps: list[tuple[datetime, int]] = []
    for ln in lines:
        m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\bstep (\d+) loss", ln)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        steps.append((ts, int(m.group(2))))
    if len(steps) < 2:
        return None
    deltas = []
    for (t1, n1), (t2, n2) in zip(steps, steps[1:], strict=False):
        dt = (t2 - t1).total_seconds()
        dn = n2 - n1
        if dt > 0 and dn > 0:
            deltas.append(dt / dn)
    if not deltas:
        return None
    return float(sorted(deltas)[len(deltas) // 2])


def _train_windows_from_lines(lines: list[str]) -> int | None:
    """The trainer's ``windows: train N / validation M`` line (smoke prints it
    once after dataset pull — the dataset size is unknown locally otherwise)."""
    for ln in lines:
        m = re.search(r"windows: train (\d+) / validation", ln)
        if m:
            return int(m.group(1))
    return None


def _estimate_run_cost_usd(epochs: int, steps_per_epoch: int,
                           sec_per_step: float,
                           usd_per_hour: float = ALL_IN_USD_PER_HOUR,
                           startup_overhead_s: float = STARTUP_OVERHEAD_S) -> float:
    """Estimated all-in Modal cost for epochs at the measured cadence."""
    hours = (epochs * steps_per_epoch * sec_per_step + startup_overhead_s) / 3600
    return hours * usd_per_hour


def _steps_per_epoch(train_windows: int, batch_size: int) -> int:
    """Micro-batches per epoch for the smoke-measured dataset size."""
    return math.ceil(train_windows / batch_size)


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
    """Poll `modal app logs --timestamps` until `needle` appears; return all
    observed lines (deduplicated) so downstream cadence parsing sees the full
    timeline, not just the tail."""
    deadline = time.time() + timeout_s
    seen: set[str] = set()
    all_lines: list[str] = []
    while time.time() < deadline:
        out = subprocess.run(
            ["modal", "app", "logs", "--timestamps", app_id],
            capture_output=True, text=True, timeout=60).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                all_lines.append(ln)
        if any(needle in ln for ln in lines):
            break
        time.sleep(15)
    return all_lines


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
    ap.add_argument("--budget", type=float, default=BUDGET_CEILING_USD,
                    help=f"cost ceiling for the budget guard (default "
                         f"${BUDGET_CEILING_USD:.2f}); applies to real launches")
    ap.add_argument("--force", action="store_true",
                    help="launch even when the estimated cost exceeds the "
                         "budget ceiling (or no smoke metrics are on file)")
    ap.add_argument("--resume", default="",
                    help="checkpoint bundle dir on the volume (e.g. "
                         "/checkpoints/latest) to continue a cut run from — "
                         "the trainer resumes at the next epoch; pass the "
                         "ORIGINAL --epochs (the trainer continues to it)")
    ap.add_argument("--trainer-extra", action="append", default=None,
                    help="extra trainer flags passed through verbatim "
                         "(repeatable; the '=' form is REQUIRED — argparse "
                         "treats a space-separated value as a new option: "
                         "--trainer-extra=--label-smoothing=0.05 "
                         "--trainer-extra=--mlp-heads). Audit levers: "
                         "--loss-lambda-dt, --label-smoothing, --weight-mode, "
                         "--weight-cap, --mlp-heads, --head-dropout, "
                         "--freeze-backbone-epochs, --early-stop-patience, "
                         "--weight-decay, --betas, --eps, "
                         "--subclass-min-train-rows")
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

    if not args.smoke:
        # Budget guard — estimate the real run from the smoke's measured
        # cadence (or refuse when no prior smoke measured it; --force skips).
        metrics = _load_smoke_metrics()
        if metrics:
            steps_per_epoch = _steps_per_epoch(
                metrics["train_windows"], args.batch_size)
            est = _estimate_run_cost_usd(
                args.epochs, steps_per_epoch, metrics["sec_per_step"])
            print(f"[budget] {args.epochs} epochs x {steps_per_epoch} steps/epoch "
                  f"x {metrics['sec_per_step']:.1f}s/step (smoke-measured) "
                  f"=> est. ${est:.2f} [{ALL_IN_USD_PER_HOUR:.4f}/h all-in "
                  f"(L4+cpu2+10GiB) + {STARTUP_OVERHEAD_S}s startup]")
            if est > args.budget and not args.force:
                print(f"ERROR: estimated run cost ${est:.2f} exceeds the "
                      f"--budget ceiling ${args.budget:.2f} — refusing to "
                      f"spawn. Adjust --epochs/--batch-size, re-run --smoke, "
                      f"or pass --force to override.", file=sys.stderr)
                sys.exit(6)
        elif not args.force:
            print("ERROR: no smoke metrics on file (run "
                  "`python deploy/spawn_train.py --smoke` first) — cannot "
                  "estimate cost; pass --force to launch blind.",
                  file=sys.stderr)
            sys.exit(6)

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
        log_every=SMOKE_LOG_EVERY if args.smoke else 0,
        resume="" if args.smoke else args.resume,
        trainer_extra=None if args.smoke else args.trainer_extra,
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

    all_lines = _watch_logs(app_id, "smoke run complete", SMOKE_WATCH_S)
    print("--- smoke log (observed) ---")
    print("\n".join(all_lines[-25:]))
    if not any("smoke run complete" in ln for ln in all_lines):
        print("ERROR: smoke run did not complete — container starved or "
              "stuck (check step cadence above).", file=sys.stderr)
        sys.exit(5)

    # Cadence check: steps should land seconds apart, not minutes.
    step_lines = [ln for ln in all_lines if "step" in ln and "loss" in ln]
    sec_per_step = _sec_per_step_from_lines(all_lines)
    train_windows = _train_windows_from_lines(all_lines)
    if sec_per_step is None or train_windows is None:
        print(f"ERROR: smoke completed but cadence/dataset size unparsable "
              f"({len(step_lines)} step lines, "
              f"{'windows line' if train_windows else 'no windows line'}) — "
              f"cannot budget the real run; investigate before launching.",
              file=sys.stderr)
        sys.exit(5)

    # Persist measured metrics so the real launch can budget itself.
    metrics = {"sec_per_step": sec_per_step, "train_windows": train_windows,
               "batch_size": args.batch_size,
               "smoke_steps": SMOKE_STEPS, "measured_at": time.strftime(
                   "%Y-%m-%dT%H:%M:%S%z")}
    metrics_path = _save_smoke_metrics(metrics)
    print(f"smoke OK: {len(step_lines)} step lines, "
          f"cadence {sec_per_step:.1f}s/step, "
          f"train windows {train_windows}, saved -> {metrics_path}")

    steps_per_epoch = _steps_per_epoch(train_windows, args.batch_size)
    est = _estimate_run_cost_usd(args.epochs, steps_per_epoch, sec_per_step)
    print(f"projection: {args.epochs} epochs x {steps_per_epoch} steps/epoch "
          f"x {sec_per_step:.1f}s/step => est. ${est:.2f} all-in "
          f"(L4+cpu2+10GiB @ ${ALL_IN_USD_PER_HOUR:.4f}/h + startup)")
    print("smoke complete — ready for the real launch.")
