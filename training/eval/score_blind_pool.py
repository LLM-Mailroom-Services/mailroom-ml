#!/usr/bin/env python3
"""Score a Tier-2 blind candidate pool with a loaded classifier bundle (#26).

Writes ``doc_type_conf``, ``subclass_conf``, and ``agreement`` so
``assemble_enrichment.py --tiers 2`` can run without hand-picked columns.

    uv run python training/score_blind_pool.py \\
        --pool data/enrichment/enron_blind.parquet \\
        --checkpoint artifacts/pytorch/model \\
        --out data/enrichment/blind_scored.parquet

Does not invent confidences: rows that fail the classifier are kept with
zero scores and ``scored=false``.  No Hub writes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    for r in df.to_dict("records"):
        title = str(r.get(args.title_col) or "")
        text = str(r.get(args.text_col) or r.get("body") or "")
        rec = dict(r)
        try:
            out = classify_document(
                bundle, title, text, max_tokens=MAX_TOKENS,
                filename=str(r.get("filename") or r.get("id") or ""))
            rec["doc_type_conf"] = float(out.get("calibrated_confidence") or 0.0)
            rec["subclass_conf"] = float(out.get("subclass_confidence") or 0.0)
            rec["agreement"] = float(out.get("agreement") or 0.0)
            rec["scored"] = out.get("status") == "ok"
        except Exception:  # noqa: BLE001 — keep the row, zero scores
            rec["doc_type_conf"] = 0.0
            rec["subclass_conf"] = 0.0
            rec["agreement"] = 0.0
            rec["scored"] = False
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
    print(f"scored {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
