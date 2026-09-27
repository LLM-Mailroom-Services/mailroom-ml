#!/usr/bin/env python3
"""Fetch Enron test jsonl inputs for ``build_heldout_plus.py`` (CPU, pinned)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import ENRON_DEDUP_REPO, ENRON_DEDUP_REVISION  # noqa: E402

OUT = ROOT / "data" / "enrichment" / "enron_heldout"
PATTERNS = (
    "ground_truth/test.jsonl",
    "blind/test.jsonl",
)


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN required to download Enron heldout inputs")
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    marker_gt = OUT / "ground_truth" / "test.jsonl"
    marker_bl = OUT / "blind" / "test.jsonl"
    if marker_gt.is_file() and marker_bl.is_file():
        print(f"[fetch_enron_heldout] cache hit under {OUT}", flush=True)
        return 0
    print(f"[fetch_enron_heldout] {ENRON_DEDUP_REPO} @ {ENRON_DEDUP_REVISION[:8]}…",
          flush=True)
    snapshot_download(
        repo_id=ENRON_DEDUP_REPO,
        repo_type="dataset",
        revision=ENRON_DEDUP_REVISION,
        local_dir=str(OUT),
        allow_patterns=list(PATTERNS),
    )
    if not marker_gt.is_file() or not marker_bl.is_file():
        raise SystemExit(f"expected jsonl under {OUT}/{{ground_truth,blind}}/test.jsonl")
    print(f"[fetch_enron_heldout] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
