"""Fire-and-forget trainer launch — submit the training call and exit.

``modal run --detach`` still cancels the remote app when the launching
process is killed before detach completes (e.g. a CI/agent shell tearing
down the process group). This launcher targets the DEPLOYED app
(``modal deploy deploy/modal_app.py``) via ``Function.lookup`` — a spawn on
a deployed function is fully server-side and survives any client death.

    HF_TOKEN=... modal deploy deploy/modal_app.py   # once (image cached)
    HF_TOKEN=... python deploy/spawn_train.py --epochs 5
    HF_TOKEN=... python deploy/spawn_train.py --push-to-hub \
        Lucius-Morningstar/mailroom-modernbert-classifier

Follow the run (epoch lines stream live):

    modal app list                 # find the ephemeral app id (newest)
    modal app logs ap-<app-id>     # trainer output streams (subprocess pass-through)
"""
from __future__ import annotations

import argparse
import os
import sys

import modal

APP_NAME = "mailroom-ml-train"
FUNCTION_NAME = "train"

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
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set — the trainer cannot pull the dataset or "
              "push the checkpoint; aborting.", file=sys.stderr)
        sys.exit(2)

    train = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    call = train.spawn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        eval_test=not args.no_eval_test,
    )
    print(f"spawned training call: {call.object_id}")
