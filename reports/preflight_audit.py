#!/usr/bin/env python3
"""Pre-flight data audit for the ModernBERT training set (read-only).

Measures, on the ACTUAL staged parquet:
  A. schema + row counts + split integrity
  B. doc_type + per-head subclass support (train/val), val adequacy
  C. TRUE token lengths with the real ModernBERT tokenizer + truncation rate
  D. input-format check: is every window `title + "\n\n" + body`?
  E. text hygiene: empty/near-empty docs, dup windows, title quality
  F. windowing: docs per n_windows, overlap sanity, coverage
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "modernbert_training" / "stage"
sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import MODEL_ID, MAX_TOKENS  # noqa: E402


def load(cfg: str, split: str) -> pd.DataFrame:
    p = STAGE / "data" / cfg / split
    return pd.concat([pd.read_parquet(f) for f in sorted(p.glob("*.parquet"))],
                     ignore_index=True)


def main() -> int:
    out: dict = {}
    docs = pd.concat([load("documents", s) for s in ("train", "validation", "test")],
                     ignore_index=True)
    wins = pd.concat([load("windows", s) for s in ("train", "validation")],
                     ignore_index=True)

    # ---- A. schema / counts -------------------------------------------------
    out["A_schema"] = {
        "documents_columns": list(docs.columns),
        "windows_columns": list(wins.columns),
        "documents_rows": int(len(docs)),
        "windows_rows": int(len(wins)),
        "documents_by_split": docs["split"].value_counts().to_dict(),
        "windows_by_split": wins["split"].value_counts().to_dict(),
        "dup_filenames_across_splits": int(
            docs["filename"].duplicated().sum()),
    }

    # ---- B. label support ---------------------------------------------------
    dt = docs.groupby(["split", "doc_type"]).size().unstack(fill_value=0)
    out["B_doc_type_support"] = dt.to_dict()
    sub = (docs.groupby(["doc_type", "subclass", "split"]).size()
           .unstack(fill_value=0))
    sub = sub.reindex(columns=["train", "validation", "test"], fill_value=0)
    out["B_subclass_support"] = {
        f"{c}/{s}": {k: int(v) for k, v in row.items()}
        for (c, s), row in sub.iterrows()
    }
    # val adequacy: how many subclass cells have < 5 val rows
    val = sub["validation"]
    out["B_val_adequacy"] = {
        "subclass_cells_total": int(len(val)),
        "val_lt_1": int((val < 1).sum()),
        "val_lt_3": int((val < 3).sum()),
        "val_lt_5": int((val < 5).sum()),
        "val_median": float(val.median()),
        "val_min": int(val.min()),
        "val_max": int(val.max()),
    }
    # train support for subclass heads
    tr = sub["train"]
    out["B_train_support"] = {
        "subclass_cells_total": int(len(tr)),
        "train_lt_5": int((tr < 5).sum()),
        "train_lt_10": int((tr < 10).sum()),
        "train_lt_20": int((tr < 20).sum()),
        "train_median": float(tr.median()),
        "train_min": int(tr.min()),
    }

    # ---- C. TRUE token lengths (real tokenizer) -----------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.model_max_length = 1 << 30

    lens = []
    trunc = 0
    for t in wins["text"].tolist():
        n = len(tok(t, add_special_tokens=True)["input_ids"])
        lens.append(n)
        if n > MAX_TOKENS:
            trunc += 1
    s = pd.Series(lens)
    out["C_true_tokens"] = {
        "n": int(len(s)),
        "min": int(s.min()), "p50": float(s.median()),
        "p90": float(s.quantile(0.90)), "p99": float(s.quantile(0.99)),
        "max": int(s.max()), "mean": float(s.mean()),
        "over_8192": int(trunc),
        "over_8192_pct": round(100 * trunc / len(s), 2),
        "at_or_over_8000": int((s >= 8000).sum()),
        "at_or_over_8000_pct": round(100 * (s >= 8000).mean(), 2),
        "under_512": int((s < 512).sum()),
        "under_512_pct": round(100 * (s < 512).mean(), 2),
    }
    # per doc_type token profile
    wins2 = wins.copy()
    wins2["true_tokens"] = lens
    out["C_tokens_by_doctype"] = {
        str(k): {"n": int(len(g)), "p50": float(g["true_tokens"].median()),
                 "p90": float(g["true_tokens"].quantile(0.90)),
                 "max": int(g["true_tokens"].max()),
                 "over_8192": int((g["true_tokens"] > MAX_TOKENS).sum())}
        for k, g in wins2.groupby("doc_type")
    }

    # ---- D. input format: title + "\n\n" + body -----------------------------
    # reconstruct expected prefix from the documents table
    title_by_fn = dict(zip(docs["filename"], docs["title"]))
    fmt_ok = 0
    fmt_bad = []
    for r in wins.itertuples():
        title = str(title_by_fn.get(r.filename, ""))
        if title and r.text.startswith(title + "\n\n"):
            fmt_ok += 1
        elif not title:
            fmt_ok += 1  # no title -> body only is the documented fallback
        else:
            fmt_bad.append((r.filename, r.window_index, title[:40],
                            r.text[:40]))
    out["D_format"] = {
        "windows_with_title_prefix": fmt_ok,
        "windows_missing_title_prefix": len(fmt_bad),
        "examples": fmt_bad[:5],
    }

    # ---- E. text hygiene ----------------------------------------------------
    empty_docs = docs[docs["doc_text"].astype(str).str.strip().str.len() == 0]
    short_docs = docs[docs["doc_text"].astype(str).str.len() < 200]
    out["E_hygiene"] = {
        "empty_doc_text": int(len(empty_docs)),
        "docs_under_200_chars": int(len(short_docs)),
        "empty_titles": int((docs["title"].astype(str).str.strip() == "").sum()),
        "title_eq_filename": int((docs["title"] == docs["filename"]).sum()),
        "dup_window_text": int(wins["text"].duplicated().sum()),
        "dup_window_text_pct": round(100 * wins["text"].duplicated().mean(), 2),
        "doc_text_len_p50": int(docs["doc_text"].astype(str).str.len().median()),
        "doc_text_len_max": int(docs["doc_text"].astype(str).str.len().max()),
    }

    # ---- F. windowing -------------------------------------------------------
    nw = docs[docs["split"] != "test"].groupby("filename").size()
    wins_per_doc = wins.groupby("filename").size()
    out["F_windowing"] = {
        "docs_with_windows": int(len(wins_per_doc)),
        "docs_single_window": int((wins_per_doc == 1).sum()),
        "docs_single_window_pct": round(100 * (wins_per_doc == 1).mean(), 2),
        "docs_multi_window": int((wins_per_doc > 1).sum()),
        "max_windows_per_doc": int(wins_per_doc.max()),
        "windows_per_doc_p50": float(wins_per_doc.median()),
        "windows_per_doc_p90": float(wins_per_doc.quantile(0.90)),
        "n_windows_col_matches": bool(
            (wins.groupby("filename")["n_windows"].nunique() == 1).all()),
    }

    print(json.dumps(out, indent=2, default=str))
    (ROOT / "reports" / "preflight_audit.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())