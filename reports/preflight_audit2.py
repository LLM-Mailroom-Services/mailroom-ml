#!/usr/bin/env python3
"""Pre-flight audit part 2: label leakage, title quality, text provenance."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "modernbert_training" / "stage"
sys.path.insert(0, str(ROOT / "src"))


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

    # ---- G. filename -> label leakage --------------------------------------
    # does the filename literally contain the doc_type or subclass token?
    def fn_has(fn: str, label: str) -> bool:
        f = re.sub(r"[^a-z0-9]", "", str(fn).lower())
        lab = re.sub(r"[^a-z0-9]", "", str(label).lower())
        return bool(lab) and lab in f

    dt_leak = docs.apply(lambda r: fn_has(r["filename"], r["doc_type"]), axis=1)
    sub_leak = docs.apply(lambda r: fn_has(r["filename"], r["subclass"]), axis=1)
    out["G_filename_leak"] = {
        "doc_type_token_in_filename": int(dt_leak.sum()),
        "doc_type_token_in_filename_pct": round(100 * dt_leak.mean(), 2),
        "subclass_token_in_filename": int(sub_leak.sum()),
        "subclass_token_in_filename_pct": round(100 * sub_leak.mean(), 2),
        "by_doc_type": {
            str(k): {
                "n": int(len(g)),
                "dt_in_fn": int(g.apply(
                    lambda r: fn_has(r["filename"], r["doc_type"]), axis=1).sum()),
                "sub_in_fn": int(g.apply(
                    lambda r: fn_has(r["filename"], r["subclass"]), axis=1).sum()),
            }
            for k, g in docs.groupby("doc_type")
        },
    }

    # ---- H. title quality ---------------------------------------------------
    title_is_fn = (docs["title"] == docs["filename"])
    out["H_title"] = {
        "title_eq_filename": int(title_is_fn.sum()),
        "title_eq_filename_pct": round(100 * title_is_fn.mean(), 2),
        "title_ne_filename": int((~title_is_fn).sum()),
        "by_doc_type_title_eq_fn": {
            str(k): round(100 * (g["title"] == g["filename"]).mean(), 2)
            for k, g in docs.groupby("doc_type")
        },
        "sample_real_titles": docs[~title_is_fn]["title"].head(8).tolist(),
        "sample_fn_titles": docs[title_is_fn]["filename"].head(8).tolist(),
    }

    # ---- I. text provenance / normalization ---------------------------------
    # heuristics: does doc_text look normalized (single-spaced, no form feeds,
    # no OCR artifacts) or raw?
    dt = docs["doc_text"].astype(str)
    out["I_text_provenance"] = {
        "has_form_feed": int(dt.str.contains("\f").sum()),
        "has_multiple_blank_lines": int(dt.str.contains(r"\n\s*\n\s*\n").sum()),
        "has_trailing_ws_lines": int(dt.str.contains(r"[ \t]+\n").sum()),
        "has_control_chars": int(dt.str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f]").sum()),
        "starts_with_whitespace": int(dt.str.match(r"^\s").sum()),
        "sample_head": dt.iloc[0][:300],
        "sample_correspondence": docs[docs["doc_type"] == "correspondence"]["doc_text"].iloc[0][:300],
        "sample_contract": docs[docs["doc_type"] == "contract"]["doc_text"].iloc[0][:300],
    }

    # ---- J. duplicate windows ----------------------------------------------
    dup_mask = wins["text"].duplicated(keep=False)
    dups = wins[dup_mask].sort_values("text")
    out["J_dup_windows"] = {
        "n_dup_rows": int(dup_mask.sum()),
        "groups": int(dups["text"].nunique()),
        "examples": [
            {"filenames": sorted(g["filename"].unique().tolist()),
             "doc_types": sorted(g["doc_type"].unique().tolist()),
             "text_head": str(g["text"].iloc[0])[:80]}
            for _, g in list(dups.groupby("text"))[:5]
        ],
    }

    # ---- K. short docs ------------------------------------------------------
    short = docs[docs["doc_text"].astype(str).str.len() < 200]
    out["K_short_docs"] = {
        "n": int(len(short)),
        "by_doc_type": short["doc_type"].value_counts().to_dict(),
        "len_min": int(short["doc_text"].astype(str).str.len().min()),
        "len_p50": float(short["doc_text"].astype(str).str.len().median()),
        "examples": [
            {"fn": r.filename, "dt": r.doc_type, "len": len(str(r.doc_text)),
             "head": str(r.doc_text)[:120]}
            for r in short.head(5).itertuples()
        ],
    }

    # ---- L. window overlap sanity ------------------------------------------
    # for multi-window docs, check consecutive windows share content (512 tok)
    multi = wins[wins["n_windows"] > 1]
    out["L_overlap"] = {
        "multi_window_rows": int(len(multi)),
        "docs_multi": int(multi["filename"].nunique()),
    }

    print(json.dumps(out, indent=2, default=str))
    (ROOT / "reports" / "preflight_audit2.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())