#!/usr/bin/env python3
"""CPU preflight: held-out-plus pool labels + window_document (no model, no GPU)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # training/<area>/<script>.py -> repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training" / "eval"))

# eval CLI lives in training/eval/
from eval_modernbert import _load_eval_docs  # noqa: E402

from mailroom_ml.config import MAX_TOKENS, STAGE_DIR  # noqa: E402
from mailroom_ml.labels import normalize_subclass  # noqa: E402
from mailroom_ml.windows import window_document  # noqa: E402

OVERLAP = 512


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plus-dir", type=Path, default=Path("data/heldout_plus_v1"))
    ap.add_argument("--stage", type=Path, default=STAGE_DIR)
    args = ap.parse_args(argv)

    audit_path = args.plus_dir / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("gates_ok"):
            print("FAIL: audit gates_ok is false", flush=True)
            return 1

    docs = _load_eval_docs(args.stage, "heldout-plus")
    n = len(docs)
    errors = 0
    label_bad = 0
    empty = 0
    title_fn = 0
    n_windows = 0
    multi = 0
    max_w = 0
    for _, row in docs.iterrows():
        fn = str(row["filename"])
        title = str(row.get("title") or "")
        text = str(row.get("doc_text") or "")
        dt = str(row["doc_type"])
        sc = str(row["subclass"])
        if not text.strip():
            empty += 1
            errors += 1
            continue
        if title.strip() == fn:
            title_fn += 1
            errors += 1
        try:
            if normalize_subclass(dt, sc) != sc:
                label_bad += 1
                errors += 1
        except Exception:
            label_bad += 1
            errors += 1
        try:
            wins = window_document(title, text, max_tokens=MAX_TOKENS, overlap=OVERLAP)
        except Exception as exc:
            print(f"window error {fn}: {exc}", flush=True)
            errors += 1
            continue
        nw = len(wins)
        n_windows += nw
        max_w = max(max_w, nw)
        if nw > 1:
            multi += 1

    summary = {
        "n_docs": n,
        "window_errors": errors,
        "label_mismatches": label_bad,
        "empty_bodies": empty,
        "title_eq_filename": title_fn,
        "single_window_docs": n - multi,
        "multi_window_docs": multi,
        "max_windows_per_doc": max_w,
        "total_windows": n_windows,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if n != 1323 or errors:
        print("FAIL: preflight", flush=True)
        return 1
    print("preflight heldout-plus OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
