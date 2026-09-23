"""Enrichment/augmentation core — the Tier-1/2/3 assemblers (plan §6).

Implements the augmentation pillar of the intake-classifier plan
(docs/intake-classifier-combined-plan.md):

- **Tier 1 — source-matched enrichment** (§6.2/§6.3/§6.4): Enron GT-subset
  expansion (aeslc_join/llm_zero_shot lineage rows, deduped against the
  canonical corpus by ``content_sha256``, capped at ``cap_mult`` x the
  correspondence train rows) plus the insurance pools — CMS rendered
  (carrier/inpatient/outpatient/pde), GNOTHEIA polycontexts (property),
  BDR tabular (auto; rows without a rendered ``doc_text`` are flagged
  ``needs_render`` and skipped — the render step is not reproducible here),
  INSURBIAS narratives (adopted only when a ``narrative`` subclass exists on
  the insurance head — it does not today, so rows come back in the reject
  register, never force-fitted).
- **Tier 2 — pseudo-label distillation scaffold** (§6.2): confidence gates
  (doc_type >= 0.95 AND subclass >= 0.9 AND agreement >= 0.9), cap <= 30% of
  correspondence train rows, balanced per-subclass stratification, and a
  grouped-split enforcement seam (thread families never straddle train/val).
  No live labeling — candidates carry fake-able confidence columns.
- **Tier 3 — constrained-synthesis machinery** (§6.5): eligibility table,
  tier caps ``((5,3),(15,2),(30,1),(75,0))``, mixture-cap math (<=40% global,
  <=50% per-subclass batch, synthetic weight 0.6), the ``LabelCard`` spec,
  and the seven gates.  Real logic: schema gate, lexical n-gram
  contamination, entity overlap; stubs (``not_run``): rule-cue checks,
  independent adjudication, embedding diversity, model disagreement.  A
  ``not_run`` decision blocks adoption — nothing is "adopted" until the
  ladder A/B (plan §6.1) measures it.

Disciplines carried in (issues #52/#57/#75): every source is revision-pinned
(config pins carried as defaults; local overrides record a content-sha
revision), every rejected/cut row lands in the audit register with a reason
(never silently dropped), subclass surfaces are checked against the
OBSERVED head (never a fabricated label — unresolvable rows are rejects),
and every function is pure: pools arrive as DataFrames, out come DataFrames
+ reject registers.  No network, no Hub, no LLM, no GPU anywhere in this
module.

Assembler row contract (superset of the trainer documents schema):
``filename, document_id, content_sha256, source_revision, title, doc_text,
doc_type, subclass, corpus_split, token_estimate, split`` (split is always
"train" — the ladder rule: nothing new enters validation/test, ever) plus
the provenance columns ``source_corpus, purpose, label_source,
label_confidence, example_weight, lineage, tier``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mailroom_ml.config import (
    BDR_REPO,
    BDR_REVISION,
    CMS_POOL_REPO,
    CMS_POOL_REVISION,
    CUAD_FULL_REPO,
    CUAD_FULL_REVISION,
    ENRON_DEDUP_REPO,
    ENRON_DEDUP_REVISION,
    GNOTHEIA_REPO,
    GNOTHEIA_REVISION,
    INSURBIAS_REPO,
    INSURBIAS_REVISION,
    RANDOM_STATE,
    SYNTHETIC_ELIGIBILITY_MAX_AUTHENTIC,
    SYNTHETIC_MAX_GLOBAL_SHARE,
    SYNTHETIC_MAX_PER_SUBCLASS_SHARE,
    SYNTHETIC_TIER_CAPS,
    VAL_FRACTION,
)
from mailroom_ml.dataset import dedup_by_sha, grouped_split
from mailroom_ml.labels import normalize_subclass, observed_label_surfaces
from mailroom_ml.windows import estimate_tokens

__all__ = [
    "PURPOSE_CANONICAL",
    "PURPOSE_TRAIN_ONLY",
    "DOCS_SCHEMA_COLUMNS",
    "PROVENANCE_COLUMNS",
    "WINDOW_SCHEMA_COLUMNS",
    "PoolResult",
    "PseudoResult",
    "Eligibility",
    "LabelCard",
    "GateResult",
    "GateReport",
    "AuditStore",
    "content_sha256",
    "assemble_enron_gt",
    "assemble_cuad_pool",
    "assemble_cms_pool",
    "assemble_gnotheia_pool",
    "assemble_bdr_pool",
    "assemble_insurbias_pool",
    "combine_tier1",
    "apply_insurance_cap",
    "assemble_pseudo_labels",
    "grouped_split_seam",
    "assign_grouped_split",
    "build_enrichment_windows",
    "evaluate_eligibility",
    "synthetic_cap",
    "mixture_caps",
    "apply_mixture_caps",
    "rows_frame",
    "gate_schema",
    "gate_lexical_contamination",
    "gate_rule_cues",
    "gate_entity_overlap",
    "gate_independent_adjudication",
    "gate_embedding_diversity",
    "gate_model_disagreement",
    "run_seven_gates",
    "human_spot_audit",
]

# ---------------------------------------------------------------------------
# Constants / contracts
# ---------------------------------------------------------------------------

PURPOSE_CANONICAL = "canonical"
PURPOSE_TRAIN_ONLY = "train_only"

# The trainer's documents schema (build_documents contract) — enrichment
# rows must drop EXACTLY to these columns when written into the stage layout.
DOCS_SCHEMA_COLUMNS: tuple[str, ...] = (
    "filename", "document_id", "content_sha256", "source_revision", "title",
    "doc_text", "doc_type", "subclass", "corpus_split", "token_estimate",
    "split",
)
# §5 provenance columns carried per-row by the assemblers (sidecar on write).
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_corpus", "purpose", "label_source", "label_confidence",
    "example_weight", "lineage", "tier",
)
WINDOW_SCHEMA_COLUMNS: tuple[str, ...] = (
    "filename", "window_index", "n_windows", "text", "doc_type", "subclass",
    "split", "window_tokens",
)

# §6.3 tail-prioritization order: enrichment targets subclass tails
# (property/auto/narrative), not raw class volume.
TAIL_PRIORITY: tuple[str, ...] = (
    "property", "auto", "narrative", "pde", "carrier", "inpatient",
    "outpatient",
)

ENRON_LINEAGE_COLS: tuple[str, ...] = ("aeslc_join", "llm_zero_shot")
ENRON_LABEL_SOURCE = "enron_gt"
CUAD_LABEL_SOURCE = "cuad_full"
PSEUDO_LABEL_SOURCE = "pseudo_enron"
SYNTHETIC_LABEL_SOURCE = "synthetic_card"

# Cap policy for the CUAD contract pool (issue #113): contract had no tier
# entry, so the §6.2 source-matched rule is reused — the pool may add at most
# ``cap_mult`` x the current contract TRAIN rows *in total* (i.e. a new-row
# allowance of ``cap_mult - 1`` x today's mass).  CLI knob: ``cuad_cap_mult``.
CUAD_CAP_MULT = 2.0

# Tier-1 Enron rows are source-matched authentic GT (``label_source=
# enron_gt``), never synthetic, so neither the Tier-3 synthesis tier table
# nor the §6.5 synthetic global share applies.  Neutralising the tier table
# to a flat x1 leaves only the per-subclass share bound (added <= authentic)
# in force — the "no subclass more than doubled" guard used to balance the
# Enron pool; the §6.2 global cap + balanced stratification already bound
# the batch total.
_TIER1_SHARE_ONLY_TIERS: tuple[tuple[int, int], ...] = ((0, 1),)

_FALSEY_TOKENS = {"", "none", "null", "nan", "false", "0"}


def _nested_field(value: Any, key: str) -> Any:
    """Read ``key`` from a nested JSON object that may be a dict or a JSON
    string (the CUAD pool carries ``input``/``metadata`` as objects; a
    CSV/parquet re-export may carry them as encoded strings).  Missing or
    unparseable values return ``None`` — the caller decides the loud reject.
    """
    if isinstance(value, dict):
        return value.get(key)
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed.get(key) if isinstance(parsed, dict) else None
    return None


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------

def content_sha256(text: str) -> str:
    """Hex sha256 of the utf-8 document text — the canonical content hash.

    Computed over exactly the ``doc_text`` bytes this package stores, so a
    rebuild on identical pool content reproduces identical hashes (and the
    same dedup decisions).
    """
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _truthy(value: Any) -> bool:
    """Truthiness for lineage/flag columns ('' / 'none' / 'nan' -> False)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not math.isnan(float(value))
    return str(value).strip().lower() not in _FALSEY_TOKENS


def _require_columns(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{where}: pool is missing required columns {missing} — corrupt "
            f"input is loud, nothing is silently co-opted"
        )


def _doc_record(
    *,
    filename: str,
    doc_text: str,
    doc_type: str,
    subclass: str,
    title: str,
    source_corpus: str,
    source_revision: str,
    label_source: str,
    label_confidence: float,
    example_weight: float = 1.0,
    lineage: str = "",
    tier: int = 1,
) -> dict[str, Any]:
    """One enrichment document row (trainer schema + §5 provenance columns)."""
    return {
        "filename": str(filename),
        "document_id": "",
        "content_sha256": content_sha256(doc_text),
        "source_revision": str(source_revision),
        "title": str(title),
        "doc_text": str(doc_text),
        "doc_type": doc_type,
        "subclass": subclass,
        "corpus_split": "",
        "token_estimate": estimate_tokens(str(doc_text)),
        # Ladder rule (plan §6.1): enrichment rows are train-only.
        "split": "train",
        "source_corpus": str(source_corpus),
        "purpose": PURPOSE_TRAIN_ONLY,
        "label_source": label_source,
        "label_confidence": float(label_confidence),
        "example_weight": float(example_weight),
        "lineage": lineage,
        "tier": int(tier),
    }


def _rows_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    cols = list(DOCS_SCHEMA_COLUMNS) + list(PROVENANCE_COLUMNS)
    return pd.DataFrame(records, columns=cols).sort_values("filename") \
        .reset_index(drop=True)


def rows_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Public constructor for enrichment document rows (documented contract:
    trainer schema + §5 provenance columns, sorted by filename)."""
    return _rows_frame(records)


def _reject_frame(rejects: list[dict[str, str]]) -> pd.DataFrame:
    if rejects:
        df = pd.DataFrame(rejects, columns=["filename", "reason", "detail"])
    else:
        df = pd.DataFrame(columns=["filename", "reason", "detail"], dtype=str)
    return df.sort_values(["filename", "reason"]).reset_index(drop=True)


def _head_surface(canonical_docs: pd.DataFrame, doc_type: str) -> tuple[str, ...]:
    """Observed head vocabulary for ``doc_type`` over the canonical docs."""
    surfaces = observed_label_surfaces(canonical_docs)
    return surfaces.get(doc_type, ())


def _normalized_subclass(doc_type: str, value: Any) -> str:
    """Canonical subclass key via the labels.py canon (#57 — single canon)."""
    return normalize_subclass(doc_type, value)


def _is_bad_subclass(subclass: str, head: tuple[str, ...]) -> bool:
    """True when the normalized subclass is off the observed head (#75:
    never a fabricated label — the row is a reject instead)."""
    return subclass not in head


def _dedup_within_pool(records: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[dict[str, str]]]:
    """Within-pool dedup by content_sha256 — first-occurrence by filename.

    Rows are ordered by (filename, sha) so the first occurrence wins;
    duplicates are rejected with a reason, never silently dropped.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for r in sorted(records, key=lambda r: (str(r["filename"]), str(r["content_sha256"]))):
        sha = str(r["content_sha256"])
        if sha in seen:
            rejects.append({
                "filename": str(r["filename"]), "reason": "duplicate_sha",
                "detail": "same content_sha256 earlier in pool (first-occurrence wins)",
            })
            continue
        seen.add(sha)
        kept.append(r)
    return kept, rejects


def _reject_filename_collisions(
        records: list[dict[str, Any]],
        canonical_docs: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Reject rows whose filename collides with a canonical row of DIFFERENT
    content (loud — never a silent rename).  Same-name same-sha rows are the
    same document identity and are kept (the dedup_by_sha discipline).
    Canonical rows with an EMPTY hash are opaque (same rule as
    ``dedup_by_sha`` — no collision can be proven against them)."""
    by_fn: dict[str, set[str]] = {}
    canon = canonical_docs[["filename", "content_sha256"]]
    for fn, sha in zip(canon["filename"].astype(str),
                       canon["content_sha256"].astype(str), strict=False):
        if not sha.strip():
            continue  # opaque hash — nothing can be proven, never dropped
        by_fn.setdefault(fn, set()).add(str(sha))
    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for r in sorted(records, key=lambda r: str(r["filename"])):
        fn = str(r["filename"])
        shas = by_fn.get(fn)
        if shas is not None and str(r["content_sha256"]) not in shas:
            rejects.append({
                "filename": fn, "reason": "filename_collision",
                "detail": "filename collides with a canonical row of different content",
            })
            continue
        kept.append(r)
    return kept, rejects


def _dedup_vs_canonical(
    rows: pd.DataFrame,
    canonical_docs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Sha-dedup enrichment rows against the canonical corpus.

    Enrichment is purely ADDITIVE, so any row whose ``content_sha256``
    already exists in canonical is a duplicate and is dropped — **including a
    same-filename match**.  The library's ``dedup_by_sha`` keeps
    same-filename matches as "same document identity", which is correct when
    deduping a corpus against itself but wrong here: keeping it double-counts
    the document in train and, when the canonical row sits in
    validation/test, leaks it across the split (#112 finding — CMS would have
    leaked 74 val/test rows).
    """
    canon_shas = set(canonical_docs["content_sha256"].astype(str)) - {"", "nan"}
    dup_mask = rows["content_sha256"].astype(str).isin(canon_shas)
    rejects: list[dict[str, str]] = [
        {"filename": fn, "reason": "duplicate_sha_canonical",
         "detail": "content_sha256 already in corpus (same or different filename)"}
        for fn in sorted(rows.loc[dup_mask, "filename"])
    ]
    return rows.loc[~dup_mask].reset_index(drop=True), rejects


def _finalize_pool(
    records: list[dict[str, Any]],
    canonical_docs: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Common Tier-1 hygiene: within-pool dedup, canonical sha dedup, and the
    filename-collision check (in a fixed order, all loud)."""
    kept, rejects = _dedup_within_pool(records)
    kept, collisions = _reject_filename_collisions(kept, canonical_docs)
    rejects.extend(collisions)
    rows = _rows_frame(kept)
    deduped, canonical_rejects = _dedup_vs_canonical(rows, canonical_docs)
    rejects.extend(canonical_rejects)
    return deduped, rejects


# ---------------------------------------------------------------------------
# Tier 1 — Enron GT-subset expansion
# ---------------------------------------------------------------------------

@dataclass
class PoolResult:
    """One pool's assembly: adopted rows + the reject register.

    ``rows`` carries the trainer schema + provenance columns, split="train".
    ``rejected`` carries every row that did NOT get adopted, with a reason
    (the audit store, never a silent drop).  ``needs_render`` is the BDR
    tabular subset that requires the decision-letter render step.
    """
    rows: pd.DataFrame
    rejected: pd.DataFrame
    needs_render: pd.DataFrame = field(default_factory=pd.DataFrame)


def _enron_cap(correspondence_train_rows: int, cap_mult: float) -> int:
    """New Enron GT rows allowed: ``cap_mult`` x current correspondence
    train rows total, i.e. at most ``cap_mult - 1`` x today's count new."""
    return max(0, int(round(cap_mult * correspondence_train_rows))
               - correspondence_train_rows)


def assemble_enron_gt(
    gt_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    cap_mult: float = 2.0,
    source_corpus: str = ENRON_DEDUP_REPO,
    source_revision: str = ENRON_DEDUP_REVISION,
    text_col: str = "doc_text",
    doc_type_col: str = "expected",
    subclass_col: str = "expected_subclass",
    lineage_cols: tuple[str, ...] | None = ENRON_LINEAGE_COLS,
    head_subclasses: tuple[str, ...] | None = None,
    label_confidence: float = 1.0,
) -> PoolResult:
    """Tier-1 Enron ground-truth expansion (plan §6.2).

    Adopts ``ground_truth``-config rows that are not already in the canonical
    corpus (filename/sha-deduped).  Labels are exact GT
    (``label_source="enron_gt"``, confidence 1.0); provenance is revision-
    pinned.  Cap: ``cap_mult`` (default 2.0) x the current correspondence
    train rows, cut loud when hit.  Subclass values are normalized through
    the canonical correspondence surface; rows resolving outside the
    OBSERVED head are rejects (issue #75 posture — never a fabricated
    label).

    ``lineage_cols``: ``None`` opts into the *weak-lineage* accept-all mode
    (every source-matched GT row, lineage marker ``enron_dedup_gt``).
    Necessary because the pinned enron pool exposes **no** ``aeslc_join``/
    ``llm_zero_shot`` column (#112 finding) — the plan §6.2 lineage filter
    cannot be evaluated at the pin.  A non-empty tuple that is entirely
    absent still raises (corrupt/renamed input stays loud).
    """
    if gt_df.empty:
        return PoolResult(_rows_frame([]), _reject_frame([]), _reject_frame([]))
    _require_columns(gt_df, [text_col, doc_type_col, subclass_col],
                     "assemble_enron_gt")
    requested = tuple(lineage_cols or ())
    present_lineage = [c for c in requested if c in gt_df.columns]
    if requested and not present_lineage:
        raise ValueError(
            "assemble_enron_gt: ground_truth config carries none of the "
            f"lineage columns {lineage_cols} — pass lineage_cols=None to "
            "accept every source-matched GT row (weak-lineage opt-in, not "
            "the default)")
    head = head_subclasses if head_subclasses is not None \
        else _head_surface(canonical_docs, "correspondence")
    n_corr_train = int(
        ((canonical_docs["doc_type"] == "correspondence")
         & (canonical_docs["split"] == "train")).sum())
    cap = _enron_cap(n_corr_train, cap_mult)

    records: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for r in gt_df.sort_values("filename").to_dict("records"):
        fn = str(r["filename"])
        if requested:
            lineage = "+".join(c for c in present_lineage if _truthy(r.get(c)))
            if not lineage:
                rejects.append({"filename": fn, "reason": "no_lineage",
                                "detail": "row shares neither aeslc_join nor llm_zero_shot lineage"})
                continue
        else:
            # weak-lineage opt-in: the pin exposes no lineage column, so the
            # marker records that the row was taken source-matched-but-unfiltered.
            lineage = "enron_dedup_gt"
        doc_type = str(r[doc_type_col] or "").strip()
        if doc_type and doc_type != "correspondence":
            rejects.append({"filename": fn, "reason": "not_correspondence",
                            "detail": f"doc_type {doc_type!r} is not correspondence"})
            continue
        text = str(r[text_col] or "")
        if not text.strip():
            rejects.append({"filename": fn, "reason": "missing_doc_text",
                            "detail": "GT row carries no document text"})
            continue
        subclass = _normalized_subclass("correspondence", r.get(subclass_col))
        if _is_bad_subclass(subclass, head):
            rejects.append({"filename": fn, "reason": "unresolvable_subclass",
                            "detail": f"{r.get(subclass_col)!r} resolves to {subclass!r}, "
                                      f"not on the observed correspondence head {tuple(head)}"})
            continue
        records.append(_doc_record(
            filename=fn, doc_text=text, doc_type="correspondence",
            subclass=subclass, title=str(r.get("subject") or ""),
            source_corpus=source_corpus, source_revision=source_revision,
            label_source=ENRON_LABEL_SOURCE, label_confidence=label_confidence,
            lineage=lineage, tier=1))

    rows, post = _finalize_pool(records, canonical_docs)
    rejects.extend(post)
    if rows.empty:
        return PoolResult(rows, _reject_frame(rejects), _reject_frame([]))
    # ---- subclass balance (issue #114) -----------------------------------
    # The §6.2 global cap decides HOW MANY new rows enter; the balanced
    # stratification seam decides WHICH.  A ~98%-email pool must not be
    # allowed to donate email rows at the expense of the tail subclasses —
    # that is exactly the majority collapse the head is suffering from.
    # ``cap_2x`` stays the reason when the pool overflows the global cap;
    # otherwise the per-subclass slot cut is the reason.
    over_global = len(rows) > cap
    balanced, balance_rejects = _balance_by_subclass(
        rows.to_dict("records"), budget=cap,
        reason="cap_2x" if over_global else "enron_balance_cut",
        subclass_universe=head)
    rejects.extend(balance_rejects)
    balanced_rows = _rows_frame(balanced)
    # ---- §6.5 mixture share bound ----------------------------------------
    # Tier-1 Enron rows are authentic GT, not synthetic: the Tier-3 tier
    # table is neutralised (flat x1) and the synthetic global share is
    # disabled (the §6.2 cap + balance already bound the batch total), so
    # only the per-subclass share bound applies — added <= authentic; no
    # subclass is ever more than doubled.  A subclass whose bound is 0
    # (unknown/zero support) is a loud ``mixture_cap`` reject, never a
    # silent drop.
    auth_counts: dict[str, int] = {}
    corr = canonical_docs[(canonical_docs["doc_type"] == "correspondence")
                          & (canonical_docs["split"] != "test")]
    for sc, n in corr["subclass"].value_counts().items():
        auth_counts[str(sc)] = int(n)
    if not balanced_rows.empty:
        for sc in balanced_rows["subclass"].unique():
            auth_counts.setdefault(str(sc), 0)
    capped, mixture_rejects = apply_mixture_caps(
        balanced_rows, auth_counts, global_share=0.99,
        tier_caps=_TIER1_SHARE_ONLY_TIERS)
    rejects.extend(mixture_rejects.to_dict("records"))
    return PoolResult(capped, _reject_frame(rejects), _reject_frame([]))


def assemble_cuad_pool(
    pool_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    cap_mult: float = CUAD_CAP_MULT,
    source_corpus: str = CUAD_FULL_REPO,
    source_revision: str = CUAD_FULL_REVISION,
    head_subclasses: tuple[str, ...] | None = None,
    label_confidence: float = 1.0,
) -> PoolResult:
    """Tier-1 CUAD contract pool (issue #113) — the contract head's
    source-matched augmentation.

    Each row is ``{id, input:{doc_text,...}, expected, metadata, tags,...}``;
    the subclass is ``metadata.category`` (a CUAD folder family such as
    ``"Consulting Agreements"``) resolved through
    ``SUBTYPE_ALIASES`` + ``normalize_subclass("contract", ...)`` (the #57
    single canon) and head-checked against the OBSERVED contract surface —
    never force-fitted.  A category that resolves to the ``other`` fallback
    is a loud reject (the contract head carries ``other`` as an
    inference-only token with zero authentic support — #116 — so enrichment
    must not manufacture rows for it).  Rows are sha-deduped against the
    canonical corpus by the common Tier-1 hygiene.

    **Cap policy:** contract had no tier entry, so the §6.2 source-matched
    rule is reused — the pool may add at most ``cap_mult`` x the current
    contract TRAIN rows in total (default 2.0 -> at most double the current
    contract train mass).  Exceeding rows are cut loud with reason
    ``cap_contract``.

    Titles are deliberately empty: CUAD's ``metadata.document_id`` is a
    filename-derived identifier that routinely names the family
    ("…Marketing Agreement"), which would leak the label into the model
    input (the 2026-09-20 leak-law).  Body-only windows only.
    """
    if pool_df.empty:
        return PoolResult(_rows_frame([]), _reject_frame([]), _reject_frame([]))
    _require_columns(pool_df, ["id", "input", "metadata"], "assemble_cuad_pool")
    head = head_subclasses if head_subclasses is not None \
        else _head_surface(canonical_docs, "contract")
    n_contract_train = int(
        ((canonical_docs["doc_type"] == "contract")
         & (canonical_docs["split"] == "train")).sum())
    cap = _enron_cap(n_contract_train, cap_mult)

    records: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for r in pool_df.sort_values("id").to_dict("records"):
        fn = str(r["id"])
        text = str(_nested_field(r.get("input"), "doc_text") or "")
        if not text.strip():
            rejects.append({"filename": fn, "reason": "missing_doc_text",
                            "detail": "CUAD row carries no input.doc_text"})
            continue
        category = _nested_field(r.get("metadata"), "category")
        subclass = _normalized_subclass("contract", category)
        if subclass == "other":
            rejects.append({
                "filename": fn, "reason": "unresolvable_subclass",
                "detail": f"metadata.category {category!r} does not resolve to "
                          "a canonical CUAD contract family (the `other` "
                          "fallback is inference-only — never force-fit)"})
            continue
        if _is_bad_subclass(subclass, head):
            rejects.append({
                "filename": fn, "reason": "subclass_not_on_head",
                "detail": f"category {category!r} resolves to {subclass!r}, "
                          f"not on the observed contract head {tuple(head)}"})
            continue
        records.append(_doc_record(
            filename=fn, doc_text=text, doc_type="contract", subclass=subclass,
            title="", source_corpus=source_corpus,
            source_revision=source_revision, label_source=CUAD_LABEL_SOURCE,
            label_confidence=label_confidence, lineage="cuad_full", tier=1))

    rows, post = _finalize_pool(records, canonical_docs)
    rejects.extend(post)
    if len(rows) > cap:
        cut = rows.iloc[cap:]
        rejects.extend({
            "filename": fn, "reason": "cap_contract",
            "detail": f"CUAD contract cap {cap} (cap_mult={cap_mult} x "
                      f"{n_contract_train} contract train rows) exceeded",
        } for fn in cut["filename"])
        rows = rows.iloc[:cap].reset_index(drop=True)
    return PoolResult(rows, _reject_frame(rejects), _reject_frame([]))


# ---------------------------------------------------------------------------
# Tier 1 — insurance pools (plan §6.3)
# ---------------------------------------------------------------------------

def _assemble_insurance_pool(
    pool_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    adopted_subclass: str | None,
    source_corpus: str,
    source_revision: str,
    text_col: str,
    subclass_col: str | None,
    title_col: str | None,
    head_subclasses: tuple[str, ...] | None,
    label_source: str,
    render_required: bool = False,
    exact_subclass: bool = False,
) -> PoolResult:
    """Shared Tier-1 insurance pipeline (per-pool wrappers set the policy)."""
    if pool_df.empty:
        return PoolResult(_rows_frame([]), _reject_frame([]), _reject_frame([]))
    if text_col not in pool_df.columns:
        if render_required:
            # tabular pool carries no rendered text at all: every row needs
            # the decision-letter render step (plan §6.3) — register + skip.
            pending = [{"filename": str(fn), "reason": "needs_render",
                        "detail": "tabular record requires the decision-letter render step"}
                       for fn in pool_df["filename"]]
            return PoolResult(_rows_frame([]), _reject_frame([]),
                              _reject_frame(pending))
        raise ValueError(
            f"insurance pool: missing required columns ['{text_col}'] — "
            f"corrupt input is loud, nothing is silently co-opted")
    head = head_subclasses if head_subclasses is not None \
        else _head_surface(canonical_docs, "insurance_claim")

    records: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    pending_render: list[dict[str, str]] = []
    for r in pool_df.sort_values("filename").to_dict("records"):
        fn = str(r["filename"])
        text = str(r[text_col] or "")
        if render_required and not text.strip():
            # tabular rows need the decision-letter render step; not
            # reproducible here -> register + skip (plan §6.3).
            pending_render.append({"filename": fn, "reason": "needs_render",
                                   "detail": "tabular record requires the decision-letter render step"})
            continue
        if not text.strip():
            rejects.append({"filename": fn, "reason": "missing_doc_text",
                            "detail": "pool row carries no document text"})
            continue
        raw_subclass = adopted_subclass if adopted_subclass is not None else \
            str(r.get(subclass_col) or "") if subclass_col else ""
        # exact_subclass: the pool's label IS the canonical token verbatim
        # (e.g. INSURBIAS narratives) — the labels canon (#57) may not carry
        # the token YET, so normalization would silently fold it to "other";
        # assign verbatim and let the observed-head check decide adoption.
        subclass = raw_subclass if exact_subclass and raw_subclass else \
            _normalized_subclass("insurance_claim", raw_subclass) \
            if raw_subclass else ""
        if not subclass:
            rejects.append({"filename": fn, "reason": "missing_subclass",
                            "detail": "pool row resolves to no insurance subclass"})
            continue
        if _is_bad_subclass(subclass, head):
            rejects.append({"filename": fn, "reason": "subclass_not_on_head",
                            "detail": f"{subclass!r} not on the insurance head "
                                      f"{tuple(head)} (never force-fit)"})
            continue
        title = str(r[title_col] or "") if title_col else ""
        records.append(_doc_record(
            filename=fn, doc_text=text, doc_type="insurance_claim",
            subclass=subclass, title=title, source_corpus=source_corpus,
            source_revision=source_revision, label_source=label_source,
            label_confidence=1.0, lineage="", tier=1))

    rows, post = _finalize_pool(records, canonical_docs)
    rejects.extend(post)
    return PoolResult(rows, _reject_frame(rejects), _reject_frame(pending_render))


def assemble_cms_pool(
    pool_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    source_corpus: str = CMS_POOL_REPO,
    source_revision: str = CMS_POOL_REVISION,
    text_col: str = "doc_text",
    subclass_col: str = "expected_subclass",
    title_col: str | None = None,
    head_subclasses: tuple[str, ...] | None = None,
) -> PoolResult:
    """Tier-1 CMS DE-SynPUF rendered pool -> insurance subclasses from the
    pool's own labels (carrier/inpatient/outpatient/pde), head-checked and
    sha-deduped against the canonical corpus."""
    return _assemble_insurance_pool(
        pool_df, canonical_docs, adopted_subclass=None,
        source_corpus=source_corpus, source_revision=source_revision,
        text_col=text_col, subclass_col=subclass_col, title_col=title_col,
        head_subclasses=head_subclasses,
        label_source="cms_desynpuf_rendered", render_required=False)


def assemble_gnotheia_pool(
    pool_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    source_corpus: str = GNOTHEIA_REPO,
    source_revision: str = GNOTHEIA_REVISION,
    text_col: str = "doc_text",
    title_col: str | None = None,
    head_subclasses: tuple[str, ...] | None = None,
) -> PoolResult:
    """Tier-1 GNOTHEIA polycontexts -> property (direct: the docs are
    rendered property claims), head-checked + sha-deduped."""
    return _assemble_insurance_pool(
        pool_df, canonical_docs, adopted_subclass="property",
        source_corpus=source_corpus, source_revision=source_revision,
        text_col=text_col, subclass_col=None, title_col=title_col,
        head_subclasses=head_subclasses,
        label_source="gnotheia_polycontext", render_required=False)


def assemble_bdr_pool(
    pool_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    source_corpus: str = BDR_REPO,
    source_revision: str = BDR_REVISION,
    text_col: str = "doc_text",
    title_col: str | None = None,
    head_subclasses: tuple[str, ...] | None = None,
) -> PoolResult:
    """Tier-1 BDR tabular records -> auto.  Tabular rows without a rendered
    decision letter are flagged ``needs_render`` and skipped by default —
    the render step is only adoptable once it is reproducible."""
    return _assemble_insurance_pool(
        pool_df, canonical_docs, adopted_subclass="auto",
        source_corpus=source_corpus, source_revision=source_revision,
        text_col=text_col, subclass_col=None, title_col=title_col,
        head_subclasses=head_subclasses,
        label_source="bdr_decision_letter", render_required=True)


def assemble_insurbias_pool(
    pool_df: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    source_corpus: str = INSURBIAS_REPO,
    source_revision: str = INSURBIAS_REVISION,
    text_col: str = "claim_narrative",
    title_col: str | None = None,
    narrative_subclass: str = "narrative",
    head_subclasses: tuple[str, ...] | None = None,
) -> PoolResult:
    """Tier-1 INSURBIAS claim narratives -> the ``narrative`` subclass, but
    ONLY when that subclass exists on the insurance head.  The cmitted head
    (``INSURANCE_SUBCLASSES``) has no narrative token, so with the default
    head these rows are rejects — recorded loud, ready for a future head
    extension, never force-fitted into a neighbor class."""
    return _assemble_insurance_pool(
        pool_df, canonical_docs, adopted_subclass=narrative_subclass,
        source_corpus=source_corpus, source_revision=source_revision,
        text_col=text_col, subclass_col=None, title_col=title_col,
        head_subclasses=head_subclasses,
        label_source="insurbias_narrative", render_required=False,
        exact_subclass=True)


def combine_tier1(
    results: list[tuple[str, PoolResult]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate Tier-1 pool results with CROSS-POOL sha dedup.

    First pool in ``results`` wins on a sha collision (earlier pools are the
    closer source at the ladder step); later duplicates are rejected with a
    reason.  Deterministic: pools are processed in the given order, rows
    sorted by filename.
    """
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    rejects: list[dict[str, str]] = []
    for pool_name, res in results:
        rows = res.rows
        if rows.empty:
            continue
        dupe_mask = rows["content_sha256"].isin(seen)
        if dupe_mask.any():
            rejects.extend({
                "filename": fn, "reason": "duplicate_sha_cross_pool",
                "detail": f"content_sha256 already adopted by an earlier pool ({pool_name})",
            } for fn in rows.loc[dupe_mask, "filename"])
            rows = rows[~dupe_mask]
        seen.update(rows["content_sha256"])
        frames.append(rows)
    combined = pd.concat(frames, ignore_index=True) if frames \
        else _rows_frame([])
    return combined.sort_values("filename").reset_index(drop=True), \
        _reject_frame(rejects)


def apply_insurance_cap(
    rows: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    class_cap_mult: float = 2.0,
    tail_priority: tuple[str, ...] = TAIL_PRIORITY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """§6.3 imbalance guard: total insurance train rows <= ``class_cap_mult``
    x the class size.  When the budget binds, tail subclasses (property/auto/
    narrative first) are prioritized — enrichment targets subclass tails,
    not raw class volume.  Cut rows are rejected with the reason
    ``cap_insurance_class``."""
    current = int(((canonical_docs["doc_type"] == "insurance_claim")
                   & (canonical_docs["split"] == "train")).sum())
    budget = max(0, int(round(class_cap_mult * current)) - current)
    if len(rows) <= budget:
        return rows, _reject_frame([])
    priority = {sc: i for i, sc in enumerate(tail_priority)}
    order = sorted(
        rows.to_dict("records"),
        key=lambda r: (priority.get(str(r["subclass"]), len(priority)),
                       str(r["filename"])))
    kept = order[:budget]
    cut = order[budget:]
    rejects = [{
        "filename": str(r["filename"]), "reason": "cap_insurance_class",
        "detail": f"insurance class cap {budget} new rows (cap_mult="
                  f"{class_cap_mult} x {current} train rows) exceeded; "
                  f"tail priority kept {str(r['subclass'])}",
    } for r in cut]
    return _rows_frame(kept), _reject_frame(rejects)


# ---------------------------------------------------------------------------
# Tier 2 — Enron pseudo-label distillation scaffold (plan §6.2)
# ---------------------------------------------------------------------------

@dataclass
class PseudoResult:
    """Tier-2 output: adopted pseudo-labeled rows + reject register."""
    rows: pd.DataFrame
    rejected: pd.DataFrame


def grouped_split_seam(
    rows: pd.DataFrame,
    existing_docs: pd.DataFrame,
    family_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped-split enforcement seam: rows whose family appears among the
    canonical VALIDATION or TEST families are rejected — thread families
    never straddle train vs val (plan §6.2, decision D6).  When
    ``existing_docs`` lacks the family column nothing can be blocked and the
    seam is a no-op (the caller reports the inactivity loudly); a missing
    family column on the ROWS side is a data defect and raises."""
    if not rows.empty and family_col not in rows.columns:
        raise ValueError(
            f"grouped_split_seam: family_col {family_col!r} missing from "
            f"the enrichment rows")
    if family_col not in existing_docs.columns:
        return rows, _reject_frame([])
    blocked = set()
    for split in ("validation", "test"):
        present = existing_docs[existing_docs["split"] == split]
        if not present.empty:
            blocked.update(
                str(f) for f in present[family_col].dropna().unique())
    if rows.empty or not blocked:
        return rows.copy(), _reject_frame([])
    fam = rows[family_col].astype(str)
    mask = fam.isin(blocked)
    kept = rows[~mask].reset_index(drop=True)
    rejects = [{
        "filename": str(r["filename"]), "reason": "family_straddles_val_test",
        "detail": f"family {str(r[family_col])!r} already sits in a canonical "
                  "validation/test family — never straddle splits",
    } for r in rows.loc[mask].to_dict("records")]
    return kept, _reject_frame(rejects)


def assign_grouped_split(
    rows: pd.DataFrame,
    family_col: str,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Reuse ``dataset.grouped_split`` for enrichment families (the upgrade
    path for any future val-including ladder step — decision D6).

    Groups (families) are the randomization unit: they never straddle
    train/validation.  Every row must carry a non-null family value — a
    missing family is a data defect, not a row-level fallback.
    """
    if rows.empty:
        return rows.copy()
    if family_col not in rows.columns:
        raise ValueError(
            f"assign_grouped_split: family_col {family_col!r} missing from rows")
    if rows[family_col].isna().any() or (rows[family_col].astype(str) == "").any():
        raise ValueError(
            "assign_grouped_split: rows with empty family values cannot be "
            "split by family — corrupt data is loud")
    return grouped_split(rows, family_col, val_fraction, seed)


_PSEUDO_DOC_TYPE_CONF = 0.95
_PSEUDO_SUBCLASS_CONF = 0.90
_PSEUDO_AGREEMENT = 0.90
_PSEUDO_MAX_FRACTION = 0.30
_PSEUDO_EXAMPLE_WEIGHT = 0.5


def _conf(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _balance_by_subclass(
    records: list[dict[str, Any]],
    budget: int,
    *,
    reason: str = "pseudo_balance_cut",
    subclass_universe: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Balanced per-subclass stratification inside a global ``budget``.

    Deterministic: each subclass (sorted) contributes its top rows by
    (confidence desc, filename); every subclass gets ``budget // n`` slots,
    the remainder goes to the subclasses with the most remaining candidates
    (ties by subclass name).  Unpicked candidates are rejected with
    ``reason`` (default ``pseudo_balance_cut`` for Tier-2; Tier-1 Enron
    passes its own labels via the caller).

    ``subclass_universe`` optionally widens the slot denominator to the full
    observed head (e.g. all correspondence subclasses) even when the pool
    carries no candidates for some — the balance target is the head, not
    merely the subclasses that happened to appear.
    """
    if budget <= 0 or not records:
        return [], [{
            "filename": str(r["filename"]), "reason": reason,
            "detail": f"balance budget {budget} exhausted",
        } for r in records]
    by_sub: dict[str, list[dict[str, Any]]] = {
        str(sc): [] for sc in subclass_universe}
    for r in records:
        by_sub.setdefault(str(r["subclass"]), []).append(r)
    for lst in by_sub.values():
        lst.sort(key=lambda r: (-float(r["label_confidence"]), str(r["filename"])))
    subs = sorted(by_sub)
    base = budget // len(subs)
    remainder = budget - base * len(subs)
    # remainder slots -> subclasses with the most candidates, subclass name tiebreak
    leftover_order = sorted(subs, key=lambda s: (-len(by_sub[s]), s))
    extra = {s: 0 for s in subs}
    for s in leftover_order[:remainder]:
        extra[s] += 1
    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for s in subs:
        slot = base + extra[s]
        taken = by_sub[s][:slot]
        kept.extend(taken)
        rejects.extend({
            "filename": str(r["filename"]), "reason": reason,
            "detail": f"balanced stratification slot {slot} for subclass "
                      f"{s!r} filled by higher-confidence rows",
        } for r in by_sub[s][slot:])
    kept.sort(key=lambda r: str(r["filename"]))
    return kept, rejects


def assemble_pseudo_labels(
    candidates: pd.DataFrame,
    canonical_docs: pd.DataFrame,
    *,
    doc_type_conf_col: str = "doc_type_conf",
    subclass_conf_col: str = "subclass_conf",
    agreement_col: str = "agreement",
    pred_doc_type_col: str = "pred_doc_type",
    pred_subclass_col: str = "pred_subclass",
    doc_type_conf: float = _PSEUDO_DOC_TYPE_CONF,
    subclass_conf: float = _PSEUDO_SUBCLASS_CONF,
    agreement: float = _PSEUDO_AGREEMENT,
    max_fraction: float = _PSEUDO_MAX_FRACTION,
    family_col: str | None = None,
    source_corpus: str = ENRON_DEDUP_REPO,
    source_revision: str = ENRON_DEDUP_REVISION,
    head_subclasses: tuple[str, ...] | None = None,
) -> PseudoResult:
    """Tier-2 pseudo-label scaffold (plan §6.2): gate -> seam -> cap ->
    balance.  Candidates carry model confidences (doc_type/subclass/agreement
    columns); nothing is labeled here.  Kept rows get
    ``label_source="pseudo_enron"``, ``label_confidence`` = the joint gate
    minimum, ``example_weight=0.5``, and land in train only."""
    if candidates.empty:
        return PseudoResult(_rows_frame([]), _reject_frame([]))
    required = [doc_type_conf_col, subclass_conf_col, agreement_col,
                pred_subclass_col]
    _require_columns(candidates, required, "assemble_pseudo_labels")
    if pred_doc_type_col and pred_doc_type_col not in candidates.columns:
        # optional: the blind Enron pool is correspondence by construction
        pred_doc_type_col = None
    if family_col and family_col not in candidates.columns:
        family_col = None
    head = head_subclasses if head_subclasses is not None \
        else _head_surface(canonical_docs, "correspondence")
    n_corr_train = int(
        ((canonical_docs["doc_type"] == "correspondence")
         & (canonical_docs["split"] == "train")).sum())
    budget = int(round(max_fraction * n_corr_train))

    records: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for r in candidates.sort_values("filename").to_dict("records"):
        fn = str(r["filename"])
        dt_conf = _conf(r.get(doc_type_conf_col))
        sc_conf = _conf(r.get(subclass_conf_col))
        ag_conf = _conf(r.get(agreement_col))
        if dt_conf is None or sc_conf is None or ag_conf is None:
            rejects.append({"filename": fn, "reason": "missing_confidence",
                            "detail": "confidence columns must be finite floats"})
            continue
        if pred_doc_type_col and str(r.get(pred_doc_type_col) or "") not in (
                "", "correspondence"):
            rejects.append({"filename": fn, "reason": "not_correspondence",
                            "detail": f"pred_doc_type {r.get(pred_doc_type_col)!r}"})
            continue
        if dt_conf < doc_type_conf:
            rejects.append({"filename": fn, "reason": "below_doc_type_conf",
                            "detail": f"doc_type_conf {dt_conf} < {doc_type_conf}"})
            continue
        if sc_conf < subclass_conf:
            rejects.append({"filename": fn, "reason": "below_subclass_conf",
                            "detail": f"subclass_conf {sc_conf} < {subclass_conf}"})
            continue
        if ag_conf < agreement:
            rejects.append({"filename": fn, "reason": "below_agreement",
                            "detail": f"agreement {ag_conf} < {agreement}"})
            continue
        text = str(r.get("doc_text") or "")
        if not text.strip():
            rejects.append({"filename": fn, "reason": "missing_doc_text",
                            "detail": "candidate carries no text"})
            continue
        subclass = _normalized_subclass(
            "correspondence", r.get(pred_subclass_col))
        if _is_bad_subclass(subclass, head):
            rejects.append({"filename": fn, "reason": "unresolvable_subclass",
                            "detail": f"{r.get(pred_subclass_col)!r} -> {subclass!r} "
                                      f"not on the observed head {tuple(head)}"})
            continue
        label_conf = round(min(dt_conf, sc_conf, ag_conf), 4)
        records.append(_doc_record(
            filename=fn, doc_text=text, doc_type="correspondence",
            subclass=subclass, title=str(r.get("title") or ""),
            source_corpus=source_corpus, source_revision=source_revision,
            label_source=PSEUDO_LABEL_SOURCE, label_confidence=label_conf,
            example_weight=_PSEUDO_EXAMPLE_WEIGHT, lineage="pseudo_enron",
            tier=2))

    rows = _rows_frame(records)
    rows, dedup_rejects = _dedup_vs_canonical(rows, canonical_docs)
    rejects.extend(dedup_rejects)
    # grouped-split seam (families never straddle train/val): the candidate
    # family rides along as a temporary column under the caller's family name
    # and is stripped after the seam check.
    if family_col is not None:
        fam = candidates.set_index("filename")[family_col].astype(str)
        rows = rows.copy()
        rows[family_col] = rows["filename"].map(fam).fillna("")
        rows, seam_rejects = grouped_split_seam(rows, canonical_docs, family_col)
        rows = rows.drop(columns=[family_col]).reset_index(drop=True)
    else:
        seam_rejects = _reject_frame([])
    rejects.extend(seam_rejects.to_dict("records"))
    # global cap: <= max_fraction of correspondence train rows
    kept = rows
    if len(kept) > budget:
        order = kept.sort_values(
            ["label_confidence", "filename"],
            ascending=[False, True]).reset_index(drop=True)
        cut = order.iloc[budget:]
        rejects.extend({
            "filename": fn, "reason": "pseudo_cap",
            "detail": f"pseudo cap {budget} ({max_fraction} x {n_corr_train} "
                      "correspondence train rows) exceeded",
        } for fn in cut["filename"])
        kept = order.iloc[:budget].reset_index(drop=True)
    # balanced stratification by subclass within the cap
    kept_records = kept.to_dict("records")
    balanced, balance_rejects = _balance_by_subclass(kept_records, len(kept))
    rejects.extend(balance_rejects)
    return PseudoResult(_rows_frame(balanced), _reject_frame(rejects))


# ---------------------------------------------------------------------------
# Tier 3 — constrained LLM synthesis machinery (plan §6.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelCard:
    """Structured label card: the generator prompt contract (plan §6.5).

    Fictional entities are mandatory (``fictional_entities=True``), the
    card's ``prohibited_facts`` are the hard constraints, and ``subclass`` is
    the exact canonical head token the candidate must carry.
    """
    parent_class: str
    subclass: str
    positive_cues: tuple[str, ...] = ()
    negative_cues: tuple[str, ...] = ()
    title_patterns: tuple[str, ...] = ()
    required_structure: tuple[str, ...] = ()
    target_length: int = 300
    variation_dims: tuple[str, ...] = ()
    prohibited_facts: tuple[str, ...] = ()
    fictional_entities: bool = True


@dataclass(frozen=True)
class Eligibility:
    """One subclass's synthesis eligibility verdict (plan §6.5)."""
    subclass: str
    authentic_count: int
    eligible: bool
    cap: int
    reasons: tuple[str, ...]


def synthetic_cap(
    authentic_count: int,
    tier_caps: tuple[tuple[int, int], ...] = SYNTHETIC_TIER_CAPS,
) -> int:
    """Tier-cap multiplier from the (5,3),(15,2),(30,1),(75,0) table.

    Each tuple is ``(authentic_threshold, cap_multiplier)``: below the first
    threshold the multiplier is 0 (0-4 authentic -> no synthetic-only labels
    — LLM-routed / other), 5-14 -> x3, 15-29 -> x2, 30-74 -> x1, 75+ -> none.
    Returns the max synthetic rows (multiplier x count).
    """
    mult = 0
    for threshold, m in sorted(tier_caps):
        if authentic_count >= threshold:
            mult = m
        else:
            break
    return mult * authentic_count


def evaluate_eligibility(
    authentic_counts: dict[str, int],
    *,
    max_authentic: int = SYNTHETIC_ELIGIBILITY_MAX_AUTHENTIC,
    tier_caps: tuple[tuple[int, int], ...] = SYNTHETIC_TIER_CAPS,
    f1_by_subclass: dict[str, float] | None = None,
    f1_floor: float = 0.65,
    confusion: dict[str, Iterable[str]] | None = None,
    subclass_class: dict[str, str] | None = None,
    min_sibling_share: float = 0.10,
) -> dict[str, Eligibility]:
    """§6.5 eligibility per subclass: < ``max_authentic`` authentic rows,
    macro-F1 below floor, high confusion with a neighbor, or sibling
    underrepresentation.  Never for labels with adequate authentic
    diversity; a zero tier cap (0-4 or 75+ authentic) blocks adoption
    regardless of the reasons."""
    class_totals: dict[str, int] = {}
    if subclass_class:
        for sc, cls in subclass_class.items():
            class_totals[cls] = class_totals.get(cls, 0) + authentic_counts.get(sc, 0)
    out: dict[str, Eligibility] = {}
    for subclass in sorted(authentic_counts):
        n = int(authentic_counts[subclass])
        reasons: list[str] = []
        if n < max_authentic:
            reasons.append("below_authentic_floor")
        if f1_by_subclass and f1_by_subclass.get(subclass, 1.0) < f1_floor:
            reasons.append("macro_f1_below_floor")
        if confusion and any(True for _ in confusion.get(subclass, ())):
            reasons.append("high_confusion_with_neighbor")
        if subclass_class and subclass in subclass_class:
            total = class_totals[subclass_class[subclass]]
            share = n / total if total else 0.0
            if 0 < share < min_sibling_share:
                reasons.append("sibling_underrepresented")
        cap = synthetic_cap(n, tier_caps)
        eligible = bool(reasons) and cap > 0
        if not eligible and cap == 0 and n >= max_authentic:
            reasons.append("tier_cap_zero")
        out[subclass] = Eligibility(
            subclass=subclass, authentic_count=n, eligible=eligible,
            cap=cap, reasons=tuple(reasons))
    return out


def mixture_caps(
    authentic_counts: dict[str, int],
    *,
    global_share: float = SYNTHETIC_MAX_GLOBAL_SHARE,
    per_subclass_share: float = SYNTHETIC_MAX_PER_SUBCLASS_SHARE,
    tier_caps: tuple[tuple[int, int], ...] = SYNTHETIC_TIER_CAPS,
) -> dict[str, int]:
    """Mixture-cap math: per-subclass allowance and the global synthetic cap.

    Per subclass: ``min(tier cap, the per-subclass batch share bound
    syn / (auth + syn) <= per_subclass_share)``.  Global: total synthetic
    <= ``global_share / (1 - global_share)`` x the authentic total.

    The global bound is allocated by a **need-weighted water-fill** (issue
    #115): every subclass's allowance is raised together until the smaller
    allowances cap out, then the survivors share the remaining budget.  A
    residual smaller than the number of survivors goes to the largest unmet
    allowances (exact ties resolve by subclass name), so no subclass with a
    positive allowance is starved merely for sorting late and the result is
    invariant to the input's key order.

    Returns per-subclass allowances plus ``__global_cap__`` and
    ``__authentic_total__`` keys.
    """
    total = sum(int(v) for v in authentic_counts.values())
    per: dict[str, int] = {}
    for sc, n in authentic_counts.items():
        tier = synthetic_cap(int(n), tier_caps)
        share_bound = int(math.floor(int(n) * per_subclass_share
                                     / (1 - per_subclass_share)))
        per[sc] = max(0, min(tier, share_bound))
    global_cap = int(math.floor(total * global_share / (1 - global_share)))

    # Water-fill the global cap across the per-subclass allowances; `uncapped`
    # is only ever raised in lockstep, so the result is value-symmetric and
    # invariant to the input's key order.
    allocated: dict[str, int] = {sc: 0 for sc in per}
    remaining = global_cap
    uncapped = list(per)
    level = 0
    for bound in sorted(set(per.values())):
        if remaining <= 0:
            break
        group = [sc for sc in uncapped if per[sc] == bound]
        if not group:
            continue
        cost = (bound - level) * len(uncapped)
        if cost <= remaining:
            for sc in uncapped:
                allocated[sc] = bound
            remaining -= cost
            level = bound
            uncapped = [sc for sc in uncapped if sc not in set(group)]
        else:
            each = remaining // len(uncapped)
            for sc in uncapped:
                allocated[sc] = level + each
            remaining -= each * len(uncapped)
            # Residual (< number of survivors) goes to the largest unmet
            # allowances; exact ties resolve by canonical name so the result
            # is invariant to input order.  A rename can only ever move a
            # unit between subclasses of IDENTICAL support.
            if remaining > 0:
                order = sorted(uncapped,
                               key=lambda sc: (allocated[sc] - per[sc], sc))
                for sc in order[:remaining]:
                    if allocated[sc] < per[sc]:
                        allocated[sc] += 1
                        remaining -= 1
            break
    allocated["__global_cap__"] = global_cap
    allocated["__authentic_total__"] = total
    return allocated


def apply_mixture_caps(
    rows: pd.DataFrame,
    authentic_counts: dict[str, int],
    *,
    global_share: float = SYNTHETIC_MAX_GLOBAL_SHARE,
    per_subclass_share: float = SYNTHETIC_MAX_PER_SUBCLASS_SHARE,
    tier_caps: tuple[tuple[int, int], ...] = SYNTHETIC_TIER_CAPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut adopted synthetic rows to the §6.5 mixture caps (kept, rejected)."""
    caps = mixture_caps(authentic_counts, global_share=global_share,
                        per_subclass_share=per_subclass_share,
                        tier_caps=tier_caps)
    kept: list[dict[str, Any]] = []
    rejects: list[dict[str, str]] = []
    for subclass, allowed in sorted(caps.items()):
        if subclass.startswith("__"):
            continue
        sub_rows = [r for r in rows.to_dict("records")
                    if str(r["subclass"]) == subclass]
        take = sub_rows[:allowed]
        kept.extend(take)
        rejects.extend({
            "filename": str(r["filename"]), "reason": "mixture_cap",
            "detail": f"subclass {subclass!r} allowance {allowed} "
                      f"(global {global_share}, per-subclass {per_subclass_share}) exceeded",
        } for r in sub_rows[allowed:])
    kept_df = _rows_frame(kept) if kept else _rows_frame([])
    return kept_df.sort_values("filename").reset_index(drop=True), \
        _reject_frame(rejects)


# ---------------------------------------------------------------------------
# Tier 3 — the seven gates + audit store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    """One gate's verdict.  ``decision`` is pass | reject | not_run; a
    ``not_run`` (stub) gate blocks adoption — the report is never a pass
    while a gate was not executed."""
    gate: str
    decision: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    """Outcome of the ordered seven-gate run over one candidate."""
    candidate_id: str
    gates: list[GateResult]
    passed: bool
    first_failure: GateResult | None
    audit: dict[str, Any]


class AuditStore:
    """Reject register — records never influence fitting (never fitted).

    Holds every rejected/cut row from the assemblers and gate pipeline so a
    rebuild re-adjudicates honestly instead of re-using a rejection.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self._records.append(dict(record))

    def extend(self, records: Iterable[dict[str, Any]]) -> None:
        for r in records:
            self.add(r)

    def add_rejected_frame(self, pool: str, df: pd.DataFrame) -> None:
        for r in df.to_dict("records"):
            self._records.append({
                "pool": pool, "filename": str(r["filename"]),
                "reason": str(r["reason"]), "detail": str(r.get("detail") or ""),
            })

    @property
    def records(self) -> list[dict[str, Any]]:
        return sorted(
            self._records,
            key=lambda r: (str(r.get("pool", "")), str(r.get("filename", "")),
                           str(r.get("reason", ""))))

    def __len__(self) -> int:
        return len(self._records)


# word n-grams for the contamination gate
_WORD_RE = re.compile(r"[a-z0-9]+")


def _word_ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(str(text).lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def gate_schema(candidate: dict[str, Any], card: LabelCard,
                *args: Any, **kwargs: Any) -> GateResult:
    """Gate 1 (real): the candidate satisfies the label-card schema —
    non-empty text, exact parent class + subclass, title present."""
    del args, kwargs
    text = str(candidate.get("doc_text") or "").strip()
    if not text:
        return GateResult("schema", "reject", "empty doc_text")
    if str(candidate.get("parent_class") or "") != card.parent_class:
        return GateResult(
            "schema", "reject",
            f"parent_class {candidate.get('parent_class')!r} != "
            f"{card.parent_class!r}")
    if str(candidate.get("subclass") or "") != card.subclass:
        return GateResult(
            "schema", "reject",
            f"subclass {candidate.get('subclass')!r} != {card.subclass!r}")
    if not str(candidate.get("title") or "").strip():
        return GateResult("schema", "reject", "empty title")
    return GateResult("schema", "pass", "schema satisfied")


def gate_lexical_contamination(
    candidate_text: str,
    corpus_texts: Iterable[str],
    *,
    n: int = 5,
    max_overlaps: int = 0,
    **kwargs: Any,
) -> GateResult:
    """Gate 2 (real): word n-gram overlap of the candidate with the training
    corpus.  A candidate sharing more than ``max_overlaps`` of its n-grams
    with ANY corpus row is a contamination reject — never paraphrase of a
    training document (plan §6.5)."""
    del kwargs
    cand_grams = _word_ngrams(candidate_text, n)
    if not cand_grams:
        return GateResult("lexical_contamination", "pass",
                          "candidate has no word n-grams")
    corpus_grams: set[tuple[str, ...]] = set()
    for text in corpus_texts:
        corpus_grams |= _word_ngrams(text, n)
    hits = cand_grams & corpus_grams
    if len(hits) > max_overlaps:
        worst = max(hits, key=lambda g: (len(g), g))
        return GateResult(
            "lexical_contamination", "reject",
            f"{len(hits)} of the candidate's {n}-grams appear in the corpus",
            {"overlap_count": len(hits), "max_overlaps": max_overlaps,
             "exemplar_ngram": " ".join(worst)})
    return GateResult("lexical_contamination", "pass",
                      f"no {n}-gram overlaps above the threshold",
                      {"overlap_count": len(hits)})


def gate_rule_cues(candidate: dict[str, Any], card: LabelCard,
                   *args: Any, **kwargs: Any) -> GateResult:
    """Gate 3 (STUB): rule-based positive/negative cue checks per label card.

    Scaffold only — the cue checker is not implemented; ``not_run`` blocks
    adoption until it lands (no gate, no adoption).
    """
    del candidate, card, args, kwargs
    return GateResult("rule_cues", "not_run",
                      "stub: rule-based cue checker not implemented (Tier-3 scaffold)")


def gate_entity_overlap(
    candidate_text: str,
    corpus_texts: Iterable[str] = (),
    *,
    known_entities: Iterable[str] = (),
    **kwargs: Any,
) -> GateResult:
    """Gate 4a (real, basic): prohibited entity overlap.  Rejects when a
    known real-world entity appears in the candidate (fictional entities are
    mandatory), or when the candidate shares a capitalized proper-noun span
    with a corpus row (basic extractive-near-dup probe)."""
    del kwargs
    text_l = str(candidate_text).lower()
    for ent in known_entities:
        if str(ent).strip().lower() and str(ent).strip().lower() in text_l:
            return GateResult(
                "entity_overlap", "reject",
                f"known entity {str(ent)!r} present — cards must use fictional entities",
                {"entity": str(ent)})
    corpus_spans: set[str] = set()
    for t in corpus_texts:
        corpus_spans |= set(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", str(t)))
    cand_spans = set(re.findall(
        r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", str(candidate_text)))
    hits = {s.lower() for s in cand_spans & corpus_spans}
    hits = {h for h in hits if h not in {"ltd", "inc", "corp", "co", "llc"}}
    if hits:
        return GateResult(
            "entity_overlap", "reject",
            "candidate shares proper-noun spans with corpus rows",
            {"spans": sorted(hits)})
    return GateResult("entity_overlap", "pass",
                      "no known entities, no corpus proper-noun overlap")


def gate_independent_adjudication(
    candidate: dict[str, Any], card: LabelCard,
    *args: Any, **kwargs: Any) -> GateResult:
    """Gate 5 (STUB): a second model — never the generator — adjudicates."""
    del candidate, card, args, kwargs
    return GateResult("independent_adjudication", "not_run",
                      "stub: independent adjudicator not wired (second model required)")


def gate_embedding_diversity(
    candidate: dict[str, Any], *args: Any, **kwargs: Any) -> GateResult:
    """Gate 6 (STUB): candidate embedding must clear the pool's diversity
    floor (chromatic-diversity baseline from the blind Enron pool)."""
    del candidate, args, kwargs
    return GateResult("embedding_diversity", "not_run",
                      "stub: embedding diversity check not implemented")


def gate_model_disagreement(
    candidate: dict[str, Any], *args: Any, **kwargs: Any) -> GateResult:
    """Gate 7 (STUB): the fine-tune model must not disagree with the card."""
    del candidate, args, kwargs
    return GateResult("model_disagreement", "not_run",
                      "stub: model-disagreement filter not implemented")


def run_seven_gates(
    candidate: dict[str, Any],
    card: LabelCard,
    corpus_texts: Iterable[str],
    *,
    known_entities: Iterable[str] = (),
    n_gram: int = 5,
    audit: AuditStore | None = None,
) -> GateReport:
    """Run the ordered seven gates; short-circuit at the first non-pass.

    A ``not_run`` stub blocks the candidate (reason names the missing gate)
    — the pipeline cannot pass until every gate executed.  On failure the
    candidate is written to ``audit`` (never fitted).
    """
    gates = [
        gate_schema(candidate, card),
        gate_lexical_contamination(str(candidate.get("doc_text") or ""),
                                   corpus_texts, n=n_gram),
        gate_rule_cues(candidate, card),
        gate_entity_overlap(str(candidate.get("doc_text") or ""),
                            corpus_texts, known_entities=known_entities),
        gate_independent_adjudication(candidate, card),
        gate_embedding_diversity(candidate),
        gate_model_disagreement(candidate),
    ]
    results: list[GateResult] = []
    failed: GateResult | None = None
    for g in gates:
        results.append(g)
        if g.decision != "pass":
            failed = g
            break
    passed = failed is None
    cid = str(candidate.get("id") or candidate.get("filename") or "?")
    audit_record = {
        "candidate_id": cid, "card": f"{card.parent_class}/{card.subclass}",
        "passed": passed,
        "first_failure": failed.gate if failed else None,
        "reason": failed.reason if failed else "",
    }
    if audit is not None:
        audit.add(audit_record)
    return GateReport(candidate_id=cid, gates=tuple(results), passed=passed,
                      first_failure=failed, audit=audit_record)


def human_spot_audit(
    rows: pd.DataFrame,
    authentic_counts: dict[str, int],
    *,
    min_authentic: int = 5,
) -> dict[str, int]:
    """Human spot-audit seam (plan §6.5): adopted synthetic rows whose
    subclass has fewer than ``min_authentic`` authentic rows must be
    spot-audited by a human before adoption.  Returns {subclass: n_rows
    pending}; the CLI prints the list — the hook is the audit, not
    automation."""
    counts = rows["subclass"].value_counts().to_dict() if not rows.empty else {}
    return {
        str(sc): int(n) for sc, n in sorted(counts.items())
        if authentic_counts.get(str(sc), 0) < min_authentic
    }


# ---------------------------------------------------------------------------
# Windows for enrichment rows (committed window schema)
# ---------------------------------------------------------------------------

def build_enrichment_windows(
    doc_records: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """Window rows for enrichment documents (the committed window schema:
    filename, window_index, n_windows, text, doc_type, subclass, split,
    window_tokens).  ``doc_records`` must carry a ``windows`` list (already
    produced by the pinned windower — the caller's seam; this function is
    pure).  Enrichment windows are TRAIN-only like their documents.
    """
    recs: list[dict[str, Any]] = []
    for r in sorted(doc_records, key=lambda r: str(r["filename"])):
        windows = list(r.get("windows") or [str(r.get("doc_text") or "")])
        for i, text in enumerate(windows):
            recs.append({
                "filename": str(r["filename"]),
                "window_index": i,
                "n_windows": len(windows),
                "text": text,
                "doc_type": str(r["doc_type"]),
                "subclass": str(r["subclass"]),
                "split": "train",
                "window_tokens": estimate_tokens(text),
            })
    return pd.DataFrame(recs, columns=list(WINDOW_SCHEMA_COLUMNS)) \
        .sort_values(["filename", "window_index"]).reset_index(drop=True)
