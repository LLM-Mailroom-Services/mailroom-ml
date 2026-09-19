"""Corpus loading, splits, windowing and stage/verify (port of ``prep.py``).

Port of ``modernbert/prep.py`` (Mailroom-Corpus-EDA commit cf096fa),
re-sourced to the WORKING corpus copy ``Lucius-Morningstar/mailroom-finetune``
@ ``FINETUNE_REVISION`` (a byte-schema-identical duplicate of the canonical
eval corpus): ``ground_truth`` config carries the labels, ``default`` carries
``doc_text`` + ``metadata``; the two are joined **on filename, never
positionally**, and GT columns are deduped keep-first.

Contract with consumers (lucius's inference/training code):

- ``build_documents`` columns — ``filename, document_id, content_sha256,
  source_revision, title, doc_text, doc_type, subclass, corpus_split,
  token_estimate`` + ``split`` (train/validation/test);
- corpus ``train`` (2,979) -> 90/10 stratified train/validation (seeded,
  deterministic ``RandomState`` — byte-stable rebuilds); corpus ``test``
  (323) is held out ENTIRELY — it never touches training;
- ``build_windows`` covers train+validation only — the fine-tune surface.

New leak-control primitives for the enrichment tier: ``grouped_split``
(families/groups never straddle splits), ``dedup_by_sha`` (exact
``content_sha256`` + filename-set guard) and ``leakage_audit`` (cross-split
filename dups, near-dup titles, per-class validation share).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mailroom_ml.config import (
    DATA_DIR,
    DOC_TYPES,
    FINETUNE_REPO,
    FINETUNE_REVISION,
    MAX_TOKENS,
    RANDOM_STATE,
    VAL_FRACTION,
    WINDOW_OVERLAP_TOKENS,
)
from mailroom_ml.labels import SUBCLASS_BY_CLASS, label_maps, normalize_subclass
from mailroom_ml.preprocessing import build_title
from mailroom_ml.provenance import build_manifest, record_manifest_sha
from mailroom_ml.windows import estimate_tokens, window_document

__all__ = [
    "load_corpus_rows",
    "build_documents",
    "stratified_split",
    "grouped_split",
    "dedup_by_sha",
    "leakage_audit",
    "build_windows",
    "class_weights",
    "stage",
    "verify_stage",
]

_PARQUET_DIR = DATA_DIR / "parquet"
_ALIAS_KEY_RE = re.compile(r"[^a-z0-9]")


# ---------------------------------------------------------------------------
# Corpus loading (local snapshot of FINETUNE_REPO @ FINETUNE_REVISION)
# ---------------------------------------------------------------------------

def _expand_gt_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Promote the v9 ``gt_fields`` JSON column to top-level GT columns.

    Values are kept as native JSON (strings stay strings; nested label maps
    stay JSON-encoded strings, which the consumers' ``_parse_labels()``
    handles).
    """
    parsed = df["gt_fields"].apply(
        lambda v: json.loads(v)
        if isinstance(v, str) and v.strip()
        else (v if isinstance(v, dict) else {})
    )
    expanded = pd.DataFrame(parsed.tolist(), index=df.index)
    return pd.concat([df, expanded], axis=1)


def _load_config(cfg: str) -> pd.DataFrame:
    """Read one snapshot config (``parquet/{cfg}/{train,test}``) deterministically.

    Split directories are scanned in sorted order; the ``split`` column is
    assigned from the directory when the file does not carry one (the
    ``default`` config), and ``ground_truth`` gets its ``gt_fields`` JSON
    promoted to top-level columns.
    """
    frames = []
    for split in ("train", "test"):
        p = _PARQUET_DIR / cfg / split
        for f in sorted(p.glob("*.parquet")):
            df = pd.read_parquet(f)
            if "split" not in df.columns:
                df = df.assign(split=split)  # default config: split implicit in directory
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if cfg == "ground_truth" and "gt_fields" in df.columns:
        df = _expand_gt_fields(df)
    return df


def load_corpus_rows() -> list[dict]:
    """GT rows + doc_text + title metadata, joined by filename (sorted).

    Ground-truth labels from the ``ground_truth`` config are joined with
    ``doc_text``/``metadata`` from the ``default`` config **by filename**
    (never positionally — silent row-order drift is a data defect); GT
    columns are deduped keep-first before the join.

    Verifies the pinned-corpus invariants AT LOAD (corrupt data is loud):
    3,302 total rows, 2,979 train / 323 test, and exactly the 5 committed
    ``DOC_TYPES`` — a mismatch raises ``ValueError`` with the observed
    counts rather than silently training on a drifted snapshot.  A missing
    local snapshot raises ``FileNotFoundError`` naming the fetch path.
    """
    marker = _PARQUET_DIR / "ground_truth" / "train"
    if not (marker.exists() and list(marker.glob("*.parquet"))):
        raise FileNotFoundError(
            f"local corpus snapshot absent: expected parquet configs under "
            f"{_PARQUET_DIR} (ground_truth + default). Fetch it first, e.g. "
            f"huggingface_hub.snapshot_download('{FINETUNE_REPO}', "
            f"repo_type='dataset', revision='{FINETUNE_REVISION}', "
            f"local_dir='{DATA_DIR}')"
        )
    gt = _load_config("ground_truth")
    # gt_fields expansion can mirror top-level matter columns (relationships,
    # related_document_ids) — keep the canonical top-level values.
    gt = gt.loc[:, ~gt.columns.duplicated(keep="first")]
    blind = _load_config("default")
    text_by_fn = dict(zip(blind["filename"], blind["doc_text"], strict=False))
    md_by_fn = dict(zip(blind["filename"], blind["metadata"], strict=False))
    rows = []
    for r in gt.sort_values("filename").to_dict("records"):
        fn = str(r["filename"])
        r["doc_text"] = str(text_by_fn.get(fn, ""))
        r["metadata"] = md_by_fn.get(fn) or {}
        rows.append(r)

    # verify at load: pinned row counts + doc_type vocab — corrupt data is loud.
    gt_counts = Counter(gt["split"]) if "split" in gt.columns else Counter()
    observed_types = set(gt["expected"]) if "expected" in gt.columns else set()
    if (len(rows) != 3302 or gt_counts.get("train") != 2979
            or gt_counts.get("test") != 323):
        raise ValueError(
            f"corpus pin mismatch for {FINETUNE_REPO}@{FINETUNE_REVISION}: "
            f"expected 3,302 rows (2,979 train / 323 test), got {len(rows)} "
            f"rows {dict(gt_counts)} from data/parquet"
        )
    if observed_types != set(DOC_TYPES):
        raise ValueError(
            f"corpus pin mismatch: expected doc_types {set(DOC_TYPES)}, got "
            f"{sorted(observed_types)} from data/parquet"
        )
    return rows


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame, stratify_col: str = "doc_type",
                     val_fraction: float = VAL_FRACTION,
                     seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Deterministic 90/10 stratified split (by class, seeded RandomState).

    Per class: rows sorted by filename, shuffled with ``np.random.RandomState
    (seed)`` (the legacy RandomState algorithm is frozen across numpy
    versions — no sklearn dependency, byte-stable rebuilds), first
    ``round(n * val_fraction)`` rows -> validation.  Returns a copy with the
    ``split`` column set to train/validation.
    """
    rng = np.random.RandomState(seed)
    out = []
    for _, grp in df.sort_values("filename").groupby(stratify_col, sort=True):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val = set(idx[:n_val])
        grp = grp.copy()
        grp["split"] = np.where(grp.index.isin(val), "validation", "train")
        out.append(grp)
    return pd.concat(out).sort_values("filename").reset_index(drop=True)


def grouped_split(df: pd.DataFrame, family_col: str, val_fraction: float,
                  seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Split so that GROUPS (families) never straddle train/validation.

    The unit of randomization is the group, not the row: unique family
    values are sorted, shuffled with ``np.random.RandomState(seed)`` (same
    frozen-algorithm discipline as ``stratified_split``), and the first
    ``round(n_groups * val_fraction)`` groups go to validation WHOLE.  Any
    future enrichment row keyed by a seen ``family_col`` value lands in the
    same split — no cross-split leakage by construction.

    Returns a copy sorted by filename with the ``split`` column set.
    """
    rng = np.random.RandomState(seed)
    families = sorted({str(f) for f in df[family_col].dropna().unique()})
    rng.shuffle(families)
    n_val = int(round(len(families) * val_fraction))
    val = set(families[:n_val])
    out = df.copy()
    out["split"] = np.where(
        out[family_col].astype(str).isin(val), "validation", "train")
    return out.sort_values("filename").reset_index(drop=True)


def dedup_by_sha(df: pd.DataFrame, pool_df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``df`` rows whose exact ``content_sha256`` already exists in ``pool_df``.

    The filename-set guard: a ``df`` row is a leak duplicate only when its
    sha matches a ``pool_df`` row under a DIFFERENT filename.  A match with
    the same filename is the same document identity (e.g. a re-export of the
    pool row) — kept, not deduped.  Rows with an empty ``content_sha256``
    are opaque and kept (no silent drops on missing hashes).

    Deterministic: the pool is grouped by sha in sorted order; ``df`` row
    order is preserved, index reset.
    """
    df = df.reset_index(drop=True)
    if df.empty or pool_df.empty:
        return df
    pool = pool_df[
        pool_df["content_sha256"].notna()
        & pool_df["content_sha256"].astype(str).str.strip().ne("")
    ]
    if pool.empty:
        return df
    sha_to_filenames = {
        sha: set(fns)
        for sha, fns in pool.groupby(pool["content_sha256"].astype(str))["filename"].apply(set).items()
    }

    def _is_pool_duplicate(row: pd.Series) -> bool:
        sha = str(row["content_sha256"] or "").strip()
        if not sha:
            return False
        pool_fns = sha_to_filenames.get(sha)
        if not pool_fns:
            return False
        return str(row["filename"]) not in pool_fns

    return df[~df.apply(_is_pool_duplicate, axis=1)].reset_index(drop=True)


def leakage_audit(df: pd.DataFrame) -> dict[str, Any]:
    """Cross-split leak audit: duplicate filenames, near-dup titles, val share.

    Returns a deterministic dict (sorted keys / values):

    - ``duplicate_filenames_across_splits`` — filename -> sorted split list
      for every filename appearing in more than one split;
    - ``title_duplicates`` — ``exact_groups`` (identical title strings),
      ``folded_groups`` (case/separator-folded near-dups), and up to 3
      exemplar groups (title, row count, filenames);
    - ``per_class_val_share`` — validation share among non-test rows per
      doc_type (0.0 when a class has no train+val rows).
    """
    result: dict[str, Any] = {}

    # duplicate filenames across splits
    by_fn = df.groupby("filename")["split"].apply(
        lambda s: sorted(set(s))).sort_index()
    dup = {fn: splits for fn, splits in by_fn.items() if len(splits) > 1}
    result["duplicate_filenames_across_splits"] = dup

    # near-dup titles: exact + case/separator-folded counts, with exemplars
    titles = df["title"].astype(str)
    exact_dup = titles[titles.duplicated(keep=False)].value_counts()
    exact_dup = exact_dup[exact_dup > 1]
    folded_series = titles.map(lambda t: _ALIAS_KEY_RE.sub("", t.lower()))
    folded_dup = folded_series.value_counts()
    folded_dup = folded_dup[folded_dup > 1]
    exemplars = []
    for fold, n in sorted(folded_dup.items(), key=lambda kv: (-kv[1], kv[0]))[:3]:
        rows = df[folded_series == fold][["title", "filename", "split"]]
        exemplars.append({
            "folded_title": fold,
            "n_rows": int(n),
            "example_title": str(rows["title"].iloc[0]),
            "filenames": sorted(rows["filename"].astype(str).tolist()),
            "splits": sorted(rows["split"].astype(str).unique().tolist()),
        })
    result["title_duplicates"] = {
        "exact_groups": int(len(exact_dup)),
        "folded_groups": int(len(folded_dup)),
        "examples": exemplars,
    }

    # per-class validation share over non-test rows
    active = df[df["split"].isin(["train", "validation"])]
    share: dict[str, float] = {}
    for cls, grp in active.groupby("doc_type", sort=True):
        n_val = int((grp["split"] == "validation").sum())
        share[str(cls)] = n_val / len(grp) if len(grp) else 0.0
    result["per_class_val_share"] = share
    return result


# ---------------------------------------------------------------------------
# Documents / windows
# ---------------------------------------------------------------------------

def build_documents(rows: list[dict]) -> pd.DataFrame:
    """One row per document: labels (canonical), title, text, split, stats.

    Columns (contract): ``filename, document_id, content_sha256,
    source_revision, title, doc_text, doc_type, subclass, corpus_split,
    token_estimate`` + ``split``.  Corpus test rows are held out entirely;
    corpus train rows get the seeded 90/10 stratified split.
    """
    recs = []
    for r in rows:
        doc_type = str(r["expected"])
        recs.append({
            "filename": str(r["filename"]),
            "document_id": str(r.get("document_id") or ""),
            "content_sha256": str(r.get("content_sha256") or ""),
            "source_revision": str(r.get("source_revision") or FINETUNE_REVISION),
            "title": build_title(r),
            "doc_text": str(r["doc_text"]),
            "doc_type": doc_type,
            "subclass": normalize_subclass(doc_type, r.get("expected_subclass")),
            "corpus_split": str(r.get("split") or ""),
            "token_estimate": estimate_tokens(str(r["doc_text"])),
        })
    df = pd.DataFrame(recs).sort_values("filename").reset_index(drop=True)
    # corpus test rows are held out entirely; train rows get the 90/10 split.
    train = stratified_split(df[df["corpus_split"] == "train"].copy())
    test = df[df["corpus_split"] == "test"].copy()
    test["split"] = "test"
    return pd.concat([train, test]).sort_values("filename").reset_index(drop=True)


def build_windows(docs: pd.DataFrame, max_tokens: int = MAX_TOKENS,
                  overlap: int = WINDOW_OVERLAP_TOKENS) -> pd.DataFrame:
    """One row per 8,192-token window (train+validation only — the fine-tune
    surface; the held-out test split stays document-level)."""
    recs = []
    for r in docs[docs["split"] != "test"].to_dict("records"):
        windows = window_document(r["title"], r["doc_text"], max_tokens, overlap)
        for i, text in enumerate(windows):
            recs.append({
                "filename": r["filename"],
                "window_index": i,
                "n_windows": len(windows),
                "text": text,
                "doc_type": r["doc_type"],
                "subclass": r["subclass"],
                "split": r["split"],
                "window_tokens": estimate_tokens(text),
            })
    return pd.DataFrame(recs).sort_values(["filename", "window_index"]).reset_index(drop=True)


def class_weights(df: pd.DataFrame, label_col: str) -> dict[str, float]:
    """Inverse-frequency class weights over the TRAIN split (val/test excluded)."""
    counts = Counter(df[df["split"] == "train"][label_col])
    total = sum(counts.values())
    weights = {k: total / (len(counts) * v) for k, v in counts.items()}
    return dict(sorted(weights.items()))


# ---------------------------------------------------------------------------
# Stage + verify
# ---------------------------------------------------------------------------

def stage(stage_dir: Path, rows: list[dict] | None = None,
          with_windows: bool = True) -> dict:
    """Build + write the staged training set (parquet + sidecars + manifest).

    Layout (byte-deterministic — sorted rows, seeded split, no timestamps):

        data/documents/{train,validation,test}/*.parquet  one row per document
        data/windows/{train,validation}/*.parquet         one row per window
        dataset_info.json   root card declaring the documents/windows configs
        labels.json         id2label per head + class weights (train split)
        vocabularies.json   canonical subclass vocab per doc_type
        manifest.txt        build facts + sha256s

    The ``data/<config>/<split>/`` layout + root ``dataset_info.json`` is the
    canonical raw-parquet repo shape: the datasets-server indexes
    ``documents``/``windows`` as real configs, so
    ``load_dataset(repo, "documents")`` works natively. (The previous
    ``parquet/`` namespace is the server's own export format — uploading
    there made the server flatten every file into one broken ``default``
    config.)

    Returns a stats dict (``counts`` per config/split + ``manifest_sha256``)
    for the publish CLI.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if rows is None:
        rows = load_corpus_rows()
    docs = build_documents(rows)
    maps = label_maps(docs)

    def _write(df: pd.DataFrame, cfg: str, split: str) -> int:
        d = stage_dir / "data" / cfg / split
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       d / f"{split}-00000-of-00001.parquet")
        return len(df)

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        counts.setdefault("documents", {})[split] = _write(
            docs[docs["split"] == split], "documents", split)
    if with_windows:
        wins = build_windows(docs)
        for split in ("train", "validation"):
            counts.setdefault("windows", {})[split] = _write(
                wins[wins["split"] == split], "windows", split)

    _write_dataset_info(stage_dir, counts)

    sidecars = {
        "labels.json": json.dumps(maps, sort_keys=True, indent=2),
        "vocabularies.json": json.dumps(
            {k: list(v) for k, v in SUBCLASS_BY_CLASS.items()},
            sort_keys=True, indent=2),
    }
    for name, content in sidecars.items():
        (stage_dir / name).write_text(content + "\n", encoding="utf-8")

    manifest = build_manifest(docs, counts, maps)
    (stage_dir / "manifest.txt").write_text(manifest, encoding="utf-8")
    return {"counts": counts, "manifest_sha256": record_manifest_sha(manifest)}


def _write_dataset_info(stage_dir: Path, counts: dict[str, dict[str, int]]) -> None:
    """Root ``dataset_info.json`` declaring the documents/windows configs.

    The datasets-server indexes a raw-parquet repo from this file: without
    it, every parquet file lands in one broken ``default`` config
    (heterogeneous schemas — ``doc_text`` vs ``text`` — under one roof) and
    ``load_dataset(repo, "documents")`` fails with "BuilderConfig not found.
    Available: ['default']".
    """
    import pyarrow.parquet as pq

    # arrow type name -> datasets-server dtype name (large_string is what
    # pa.Table.from_pandas emits for str columns)
    _DTYPE = {"large_string": "string", "string": "string",
              "int64": "int64", "double": "float64", "bool": "bool"}

    def _config(cfg: str, splits: dict[str, int]) -> dict:
        d = stage_dir / "data" / cfg
        first = sorted((d / next(iter(splits))).glob("*.parquet"))[0]
        schema = pq.read_schema(first)
        features = {
            name: {"dtype": _DTYPE.get(str(schema.field(name).type),
                                       str(schema.field(name).type)),
                   "_type": "Value"}
            for name in schema.names
        }
        return {
            "description": f"{cfg} config of the mailroom-modernbert-training set",
            "features": features,
            "splits": {
                split: {
                    "name": split,
                    "num_bytes": sum(f.stat().st_size
                                     for f in (d / split).glob("*.parquet")),
                    "num_examples": n,
                    "dataset_name": "Lucius-Morningstar/mailroom-modernbert-training",
                }
                for split, n in splits.items()
            },
        }

    info = {cfg: _config(cfg, splits) for cfg, splits in counts.items()}
    (stage_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8")


def verify_stage(stage_dir: Path) -> dict:
    """Integrity checks over a staged tree: counts, splits, vocab, leakage.

    Checks: every expected config/split file exists (the ``windows`` config
    is OPTIONAL — a documents-only stage built with ``--no-windows`` skips
    windows checks, but a present windows dir must carry train+validation,
    never ``test``); the ``split`` column matches its directory; doc_type is
    within the 5 committed classes and every subclass within its class
    vocabulary; NO filename appears in more than one split.  Returns ``ok``,
    ``problems``, ``rows``, ``splits`` and per-config ``counts``.
    """

    problems: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    configs = ("documents",) if not (stage_dir / "data" / "windows").exists() \
        else ("documents", "windows")
    for cfg in configs:
        for split in ("train", "validation", "test"):
            files = sorted((stage_dir / "data" / cfg / split).glob("*.parquet"))
            if cfg == "windows" and split == "test":
                if files:
                    problems.append("windows/test must not exist (test held out)")
                continue
            if not files:
                problems.append(f"missing {cfg}/{split}")
                continue
            df = pd.read_parquet(files[0])
            counts.setdefault(cfg, {})[split] = int(len(df))
            if (df["split"] != split).any():
                problems.append(f"{cfg}/{split}: split column mismatch")
            if cfg == "documents":
                bad = df[~df["doc_type"].isin(DOC_TYPES)]
                if len(bad):
                    problems.append(f"documents/{split}: {len(bad)} bad doc_type")
                for cls, grp in df.groupby("doc_type"):
                    allowed = set(SUBCLASS_BY_CLASS[cls])
                    bad_sub = grp[~grp["subclass"].isin(allowed)]
                    if len(bad_sub):
                        problems.append(f"documents/{split}: {cls} bad subclass: "
                                        f"{sorted(bad_sub['subclass'].unique())}")
    # leakage: no filename appears in more than one split
    docs = pd.concat([
        pd.read_parquet(f)
        for split in ("train", "validation", "test")
        for f in sorted((stage_dir / "data" / "documents" / split).glob("*.parquet"))
    ])
    dup = docs["filename"].duplicated()
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate filenames across splits")
    return {"ok": not problems, "problems": problems,
            "rows": int(len(docs)),
            "splits": docs["split"].value_counts().to_dict(),
            "counts": counts}
