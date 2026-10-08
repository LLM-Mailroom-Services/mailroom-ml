#!/usr/bin/env python3
"""Score a Tier-2 blind candidate pool with a loaded classifier bundle (#26).

Writes ``doc_type_conf``, ``subclass_conf``, and ``agreement`` so
``assemble_enrichment.py --tiers 2`` can run without hand-picked columns.

    uv run python training/eval/score_blind_pool.py \\
        --pool data/enrichment/enron_blind.parquet \\
        --checkpoint artifacts/pytorch/model \\
        --out data/enrichment/blind_scored.parquet

Does not invent confidences: rows the classifier could not score (oversize,
no windows, a classifier error) are kept with zero scores, ``scored=false``
and a ``score_reason``.  The unscored count is printed per reason, and the
run fails (exit 3) when NO row could be scored — a broken bundle must not
silently write an all-zero pool.  No Hub writes.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # training/<area>/<script>.py -> repo root
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mailroom_ml.config import (  # noqa: E402
    BLIND_REQUIRED_COLUMNS,
    MAX_TOKENS,
)
from mailroom_ml.inference import (  # noqa: E402
    BundleLoadError,
    BundleUnavailable,
    classify_document,
    load_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text-col", default="doc_text")
    ap.add_argument("--title-col", default="title")
    ap.add_argument("--limit", type=int, default=0)
    return ap


def _read_pool(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"unsupported pool suffix {path.suffix}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bundle = load_bundle(args.checkpoint)
    except (BundleUnavailable, BundleLoadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    df = _read_pool(args.pool)
    if args.limit:
        df = df.head(args.limit).copy()
    rows = []
    unscored: Counter = Counter()
    for r in df.to_dict("records"):
        title = str(r.get(args.title_col) or "")
        text = str(r.get(args.text_col) or r.get("body") or "")
        rec = dict(r)
        try:
            out = classify_document(
                bundle, title, text, max_tokens=MAX_TOKENS,
                filename=str(r.get("filename") or r.get("id") or ""))
            # scored only when the classifier actually produced confidences:
            # oversize / no-window early returns carry status "ok" but no
            # calibrated_confidence
            scored = (out.get("status") == "ok"
                      and out.get("calibrated_confidence") is not None)
            rec["doc_type_conf"] = float(out.get("calibrated_confidence") or 0.0)
            rec["subclass_conf"] = float(out.get("subclass_confidence") or 0.0)
            rec["agreement"] = float(out.get("agreement") or 0.0)
            rec["scored"] = scored
            rec["score_reason"] = "" if scored else str(
                out.get("reason") or out.get("status") or "unscored")
        except Exception as exc:  # noqa: BLE001 — keep the row, zero scores
            rec["doc_type_conf"] = 0.0
            rec["subclass_conf"] = 0.0
            rec["agreement"] = 0.0
            rec["scored"] = False
            rec["score_reason"] = f"error:{type(exc).__name__}"
        if not rec["scored"]:
            unscored[rec["score_reason"]] += 1
        rows.append(rec)
    out = pd.DataFrame(rows)
    for col in BLIND_REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() in {".parquet", ".pq"}:
        out.to_parquet(args.out, index=False)
    else:
        out.to_json(args.out, orient="records", lines=True)
    n_unscored = sum(unscored.values())
    print(f"scored {len(out) - n_unscored}/{len(out)} rows -> {args.out}")
    if unscored:
        print(f"unscored by reason: {dict(sorted(unscored.items()))}",
              file=sys.stderr)
    if len(out) and n_unscored == len(out):
        print("ERROR: no row could be scored — check the bundle / pool "
              "columns before assembling Tier 2", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
