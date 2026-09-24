#!/usr/bin/env python3
"""Download the pinned mailroom-finetune parquet snapshot into ``data/``.

Populates ``data/parquet/{ground_truth,default}/`` so ``load_corpus_rows()`` and
``@pytest.mark.fullcorpus`` tests can run locally.

Usage:
    uv run python training/fetch_corpus.py
    uv run python training/fetch_corpus.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_b = Path(__file__).resolve()
while not (_b / ".git").is_dir():
    _parent = _b.parent
    if _parent == _b:
        raise SystemExit(f"repo root not found above {Path(__file__).resolve()}")
    _b = _parent
ROOT = _b
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import (  # noqa: E402
    CORPUS_DOCUMENT_COUNT,
    DATA_DIR,
    FINETUNE_REPO,
    FINETUNE_REVISION,
)
from mailroom_ml.dataset import load_corpus_rows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=DATA_DIR,
        help=f"download root (default: {DATA_DIR})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned download only",
    )
    args = ap.parse_args()
    marker = args.local_dir / "parquet" / "ground_truth" / "train"
    if args.dry_run:
        print(f"repo_id     : {FINETUNE_REPO}")
        print(f"revision    : {FINETUNE_REVISION}")
        print(f"local_dir   : {args.local_dir}")
        print(f"verify via  : load_corpus_rows() -> {CORPUS_DOCUMENT_COUNT} rows")
        print(f"marker path : {marker}")
        return 0

    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snapshot_download(
        repo_id=FINETUNE_REPO,
        repo_type="dataset",
        revision=FINETUNE_REVISION,
        local_dir=str(args.local_dir),
    )
    rows = load_corpus_rows()
    if len(rows) != CORPUS_DOCUMENT_COUNT:
        print(
            f"ERROR: expected {CORPUS_DOCUMENT_COUNT} rows, got {len(rows)}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {len(rows)} corpus rows at {args.local_dir / 'parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
