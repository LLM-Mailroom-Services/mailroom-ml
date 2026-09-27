#!/usr/bin/env python3
"""Build the held-out-plus v1 extension for ModernBERT eval (CPU-only).

Appends 1,000 fresh correspondence/email documents from the Enron
deduplicated corpus (``Lucius-Morningstar/enron-correspondence-dedup``,
test split) to the canonical 323-doc held-out test — cleaned through the
EXACT training path (``deterministic_normalize`` + semantic-only titles,
never the filename fallback) and gated by the same leak audits that guard
the published training set:

- filename overlap vs train+val+canonical-test == 0,
- content-sha overlap under a different filename == 0 (``dedup_by_sha``
  semantics),
- ``filename_leak_audit(... )["clean"]`` is True (``title_eq_filename`` and
  ``title_looks_like_filename`` both zero),
- every row normalizes to a sanctioned (doc_type, subclass) pair.

Insurance (CMS) and v8-corpus pools were surveyed 2026-09-27 and are fully
absorbed into the training corpus (CMS train 332/375 filenames already in
train+val; v8 0 filename-fresh rows) — they contribute nothing fresh, so v1
is correspondence-only.  Corporate-record top-up and the CUAD clause ->
subclass mapping are tracked as phase 2 (no fresh pool exists on the Hub).

Usage (from the repo root, ``data/parquet`` snapshot + Enron files present):

    uv run python training/build_heldout_plus.py \\
        --enron-gt /tmp/heldout-probe/enron/ground_truth/test.jsonl \\
        --enron-blind /tmp/heldout-probe/enron/blind/test.jsonl \\
        --n 1000 --seed 42 --out data/heldout_plus_v1

Writes ``documents.parquet``, ``manifest.json`` and ``audit.json`` under
``--out``.  ``data/`` never commits (Hub re-fetch); the audit JSON is the
durable record.  No GPU, no network when inputs are local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mailroom_ml.dataset import (
    dedup_by_sha,
    filename_leak_audit,
)
from mailroom_ml.labels import normalize_subclass
from mailroom_ml.normalize import deterministic_normalize
from mailroom_ml.preprocessing import build_title

ENRON_REPO = "Lucius-Morningstar/enron-correspondence-dedup"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enron-gt", required=True, type=Path)
    ap.add_argument("--enron-blind", required=True, type=Path)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/heldout_plus_v1"))
    args = ap.parse_args(argv)

    from mailroom_ml.dataset import build_documents, load_corpus_rows

    corpus = build_documents(load_corpus_rows())
    pool = corpus[corpus["split"].isin(["train", "validation"])]
    canon_test = corpus[corpus["split"] == "test"]
    pool_fn = set(pool["filename"].astype(str))
    test_fn = set(canon_test["filename"].astype(str))
    pool_sha = {
        str(s): set(fns)
        for s, fns in pool.groupby(pool["content_sha256"].astype(str))[
            "filename"
        ].apply(set).items()
        if str(s).strip()
    }

    gt = {}
    for line in args.enron_gt.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gt[str(r["filename"])] = r
    blind = {}
    for line in args.enron_blind.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            blind[str(r["filename"])] = r

    stats: dict[str, int] = {
        "enron_gt_rows": len(gt),
        "enron_blind_rows": len(blind),
    }
    joined = 0
    skipped_no_blind = 0
    skipped_filename_overlap = 0
    skipped_sha_overlap = 0
    skipped_empty = 0
    skipped_label = 0
    recs: list[dict] = []
    for fn in sorted(gt):
        grow = gt[fn]
        brow = blind.get(fn)
        if brow is None:
            skipped_no_blind += 1
            continue
        if fn in pool_fn or fn in test_fn:
            skipped_filename_overlap += 1
            continue
        dt = str(grow.get("expected") or "")
        sc_raw = grow.get("expected_subclass") or ""
        if dt != "correspondence":
            skipped_label += 1
            continue
        try:
            sc = normalize_subclass(dt, sc_raw)
        except Exception:
            skipped_label += 1
            continue
        row = {
            "filename": fn,
            "doc_text": str(brow.get("text") or ""),
            "expected": dt,
            "expected_subclass": sc_raw,
            "metadata": {"subject": str(brow.get("subject") or "")},
            "split": "test",
        }
        title = deterministic_normalize(build_title(row))[0]
        text = deterministic_normalize(row["doc_text"])[0]
        if not text.strip():
            skipped_empty += 1
            continue
        sha = _sha(text)
        pool_fns = pool_sha.get(sha)
        if pool_fns and fn not in pool_fns:
            skipped_sha_overlap += 1
            continue
        joined += 1
        recs.append(
            {
                "filename": fn,
                "document_id": "",
                "content_sha256": sha,
                "source_revision": f"{ENRON_REPO}@test",
                "title": title,
                "doc_text": text,
                "doc_type": dt,
                "subclass": sc,
                "corpus_split": "test",
                "split": "test",
            }
        )
    stats.update(
        {
            "skipped_no_blind": skipped_no_blind,
            "skipped_filename_overlap": skipped_filename_overlap,
            "skipped_sha_overlap": skipped_sha_overlap,
            "skipped_empty": skipped_empty,
            "skipped_label": skipped_label,
            "fresh_joined": joined,
        }
    )

    # Deterministic draw: seeded shuffle of sorted filenames, take --n.
    rng = np.random.RandomState(args.seed)
    order = sorted(r["filename"] for r in recs)
    rng.shuffle(order)
    keep = set(order[: max(0, args.n)])
    docs = pd.DataFrame([r for r in recs if r["filename"] in keep])
    docs = docs.sort_values("filename").reset_index(drop=True)
    stats["selected"] = int(len(docs))

    # Gates (same audits as the training build).
    leak = filename_leak_audit(docs)
    # Cross-split filename check: selection vs train+val+canonical-test.
    corpus_fn = pd.Series(
        list(pool_fn | test_fn), name="filename"
    ).to_frame()
    corpus_fn["split"] = "corpus"
    sel_fn = pd.DataFrame({"filename": sorted(keep), "split": "heldout-plus"})
    by_fn = (
        pd.concat([corpus_fn, sel_fn], ignore_index=True)
        .groupby("filename")["split"]
        .apply(lambda s: sorted(set(s)))
    )
    dup_splits = {fn: splits for fn, splits in by_fn.items() if len(splits) > 1}
    audit = {
        "sources": {
            "corpus_test": int(len(canon_test)),
            "enron_repo": ENRON_REPO,
            "enron_split": "test",
        },
        "filter_stats": stats,
        "filename_leak_audit": leak,
        "cross_split_duplicates_in_selection": dup_splits,
        "per_doc_type": docs["doc_type"].value_counts().to_dict(),
        "per_subclass": docs["subclass"].value_counts().to_dict(),
        "title_nonempty": int((docs["title"].astype(str).str.strip() != "").sum()),
        "seed": args.seed,
    }
    gates_ok = (
        leak["clean"]
        and not dup_splits
        and int(len(docs)) == args.n
    )
    audit["gates_ok"] = bool(gates_ok)

    args.out.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(args.out / "documents.parquet", index=False)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "name": "heldout-plus-v1",
                "n_docs": int(len(docs)),
                "seed": args.seed,
                "enron_repo": ENRON_REPO,
                "cleaning": "deterministic_normalize + semantic-only titles "
                "(subject -> exhibit_description -> empty; never filename)",
                "dedup": "filename exclusion + content-sha (different "
                "filename) vs train+val and canonical test",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not gates_ok:
        print("GATES FAILED", flush=True)
        return 1
    print(f"wrote {args.out / 'documents.parquet'} ({len(docs)} docs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
