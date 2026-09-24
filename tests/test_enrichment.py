"""Hermetic tests for the enrichment/augmentation layer (mailroom-ml U4).

Covers ``mailroom_ml.enrichment`` (Tier-1 assemblers per pool, Tier-2
pseudo-label scaffold, Tier-3 eligibility/mixture/seven gates) and the
``training/assemble_enrichment.py`` CLI.  Fully hermetic: tiny injected
DataFrames/CSVs, no network, no Hub, no LLM, no GPU.  Byte-determinism of
every written artifact is asserted (no timestamps, re-runs reproduce
identical bytes).

Plan contract under test: docs/intake-classifier-combined-plan.md §6.1-6.5,
§14 (U4 evidence gate: gates unit-tested; provenance), §15 issues #52
(revision pins), #57 (single labels canon), #75 (observed head, never
fabricated labels).
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from conftest import (  # type: ignore[import-not-found]
    ROOT,
    fixture_rows,
    requires_transformers,
)
from mailroom_ml import config as cfg
from mailroom_ml.dataset import build_documents
from mailroom_ml.enrichment import (
    DOCS_SCHEMA_COLUMNS,
    ENRON_DEDUP_REPO,
    ENRON_DEDUP_REVISION,
    PROVENANCE_COLUMNS,
    PURPOSE_TRAIN_ONLY,
    WINDOW_SCHEMA_COLUMNS,
    AuditStore,
    LabelCard,
    apply_insurance_cap,
    apply_mixture_caps,
    assemble_bdr_pool,
    assemble_cms_pool,
    assemble_cuad_pool,
    assemble_enron_gt,
    assemble_gnotheia_pool,
    assemble_insurbias_pool,
    assemble_pseudo_labels,
    assign_grouped_split,
    build_enrichment_windows,
    combine_tier1,
    content_sha256,
    evaluate_eligibility,
    gate_embedding_diversity,
    gate_entity_overlap,
    gate_independent_adjudication,
    gate_lexical_contamination,
    gate_model_disagreement,
    gate_rule_cues,
    gate_schema,
    grouped_split_seam,
    human_spot_audit,
    mixture_caps,
    run_seven_gates,
    synthetic_cap,
)
from mailroom_ml.labels import INSURANCE_SUBCLASSES, label_maps

CLI_PATH = ROOT / "training" / "assemble_enrichment.py"
_spec = importlib.util.spec_from_file_location("assemble_enrichment_cli", CLI_PATH)
assemble_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assemble_cli)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Helpers (tiny injected frames — hermetic by construction)
# ---------------------------------------------------------------------------

def _enron_gt_rows() -> pd.DataFrame:
    """Five ground_truth-config rows: 2 aeslc, 1 llm_zero_shot, 1 no lineage,
    1 non-correspondence."""
    return pd.DataFrame([
        {"filename": "enron_a1.txt", "expected": "correspondence",
         "expected_subclass": "letter", "doc_text": "Enron letter body one.",
         "aeslc_join": "1", "llm_zero_shot": "", "subject": "Re: one"},
        {"filename": "enron_a2.txt", "expected": "correspondence",
         "expected_subclass": "notice", "doc_text": "Enron notice body two.",
         "aeslc_join": "aeslc_001", "llm_zero_shot": ""},
        {"filename": "enron_z3.txt", "expected": "correspondence",
         "expected_subclass": "attorney_demand", "doc_text": "Demand body three.",
         "aeslc_join": "", "llm_zero_shot": "zs_007"},
        {"filename": "enron_nl.txt", "expected": "correspondence",
         "expected_subclass": "letter", "doc_text": "No lineage here.",
         "aeslc_join": "", "llm_zero_shot": ""},
        {"filename": "enron_contract.txt", "expected": "contract",
         "expected_subclass": "consulting", "doc_text": "A contract body.",
         "aeslc_join": "1", "llm_zero_shot": ""},
    ])


def _canonical() -> pd.DataFrame:
    """Canonical docs from the committed fixture rows (real build path).

    Backfills empty ``content_sha256`` with the true content hash — the real
    corpus carries per-row hashes; the dedup discipline is only meaningful
    against hashed rows.
    """
    docs = build_documents(fixture_rows())
    docs = docs.copy()
    docs["content_sha256"] = [
        c if str(c).strip() else content_sha256(str(t))
        for c, t in zip(docs["content_sha256"], docs["doc_text"], strict=False)]
    return docs


def _mini_canonical(n_ins: int = 60, n_corr: int = 100, *,
                    ins_subclasses: tuple[str, ...] = ("carrier",),
                    corr_subclasses: tuple[str, ...] = ("email", "letter"),
                    family_col: str | None = None) -> pd.DataFrame:
    """A small canonical docs frame with configurable insurance/correspondence
    volume for cap + seam tests (schema = build_documents contract)."""
    rows = []
    for i in range(n_ins):
        sc = ins_subclasses[i % len(ins_subclasses)]
        rows.append({"filename": f"ins_gt_{i:03d}.txt", "document_id": "",
                     "content_sha256": content_sha256(f"ins {i}"),
                     "source_revision": cfg.FINETUNE_REVISION,
                     "title": f"ins {i}", "doc_text": f"claim {i}",
                     "doc_type": "insurance_claim", "subclass": sc,
                     "corpus_split": "train", "token_estimate": 1,
                     "split": "train"})
    for i in range(n_corr):
        sc = corr_subclasses[i % len(corr_subclasses)]
        rows.append({"filename": f"corr_gt_{i:03d}.txt", "document_id": "",
                     "content_sha256": content_sha256(f"corr {i}"),
                     "source_revision": cfg.FINETUNE_REVISION,
                     "title": f"corr {i}", "doc_text": f"email {i}",
                     "doc_type": "correspondence", "subclass": sc,
                     "corpus_split": "train", "token_estimate": 1,
                     "split": "train"})
    df = pd.DataFrame(rows)
    if family_col:
        df[family_col] = [f"fam_{i % 7:02d}" for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# content hash + provenance plumbing
# ---------------------------------------------------------------------------

def test_content_sha256_deterministic_and_distinct():
    a, b = content_sha256("same text"), content_sha256("same text")
    assert a == b == content_sha256("same text")
    assert len(a) == 64
    assert a != content_sha256("same text!")
    import hashlib
    assert a == hashlib.sha256(b"same text").hexdigest()


def test_tier1_rows_carry_provenance_columns():
    res = assemble_enron_gt(_enron_gt_rows(), _canonical())
    assert set(PROVENANCE_COLUMNS) <= set(res.rows.columns)
    assert set(DOCS_SCHEMA_COLUMNS) <= set(res.rows.columns)
    assert (res.rows["purpose"] == PURPOSE_TRAIN_ONLY).all()
    assert (res.rows["split"] == "train").all()  # ladder rule: train-only
    assert (res.rows["tier"] == 1).all()


# ---------------------------------------------------------------------------
# Tier 1 — Enron GT expansion
# ---------------------------------------------------------------------------

def test_enron_gt_lineage_filter_and_mapping():
    canonical = _canonical()
    res = assemble_enron_gt(_enron_gt_rows(), canonical)
    kept = set(res.rows["filename"])
    # lineage rows kept; no-lineage + non-correspondence rejected
    assert kept == {"enron_a1.txt", "enron_a2.txt", "enron_z3.txt"}
    reasons = dict(zip(res.rejected["filename"], res.rejected["reason"], strict=False))
    assert reasons["enron_nl.txt"] == "no_lineage"
    assert reasons["enron_contract.txt"] == "not_correspondence"
    # labels surface through the canon; source pinned; exact GT confidence
    row = res.rows[res.rows["filename"] == "enron_a1.txt"].iloc[0]
    assert row["doc_type"] == "correspondence" and row["subclass"] == "letter"
    assert row["label_source"] == "enron_gt"
    assert row["label_confidence"] == 1.0
    assert row["source_corpus"] == ENRON_DEDUP_REPO
    assert row["source_revision"] == ENRON_DEDUP_REVISION
    assert row["lineage"] == "aeslc_join"
    assert row["content_sha256"] == content_sha256("Enron letter body one.")


def test_enron_gt_dedup_drops_any_canonical_sha_match():
    canonical = _canonical()
    canonical_text = canonical[
        canonical["filename"] == "enron_letter_002.txt"]["doc_text"].iloc[0]
    # same filename + same sha: the document is ALREADY in the corpus, so it
    # is dropped — enrichment is additive; keeping it would double-count the
    # doc in train and, when the canonical row sits in val/test, leak it
    # across the split (#112 finding).
    identity = pd.DataFrame([{
        "filename": "enron_letter_002.txt", "expected": "correspondence",
        "expected_subclass": "letter", "doc_text": canonical_text,
        "aeslc_join": "1"}])
    res = assemble_enron_gt(identity, canonical)
    assert res.rows.empty
    reasons = dict(zip(res.rejected["filename"], res.rejected["reason"], strict=False))
    assert reasons["enron_letter_002.txt"] == "duplicate_sha_canonical"
    # different filename + same sha -> also dropped against the corpus
    copy = pd.DataFrame([{
        "filename": "enron_letter_copy.txt", "expected": "correspondence",
        "expected_subclass": "letter", "doc_text": canonical_text,
        "aeslc_join": "1"}])
    res2 = assemble_enron_gt(copy, canonical)
    assert res2.rows.empty
    reasons2 = dict(zip(res2.rejected["filename"], res2.rejected["reason"], strict=False))
    assert reasons2["enron_letter_copy.txt"] == "duplicate_sha_canonical"


def test_enron_gt_weak_lineage_opt_in_accepts_without_lineage_columns():
    """lineage_cols=None: the pin exposes no aeslc_join/llm_zero_shot column
    (#112), so every source-matched GT row is accepted, head-checked, and
    marked weak-lineage."""
    canonical = _canonical()
    gt = _enron_gt_rows().drop(columns=["aeslc_join", "llm_zero_shot"])
    res = assemble_enron_gt(gt, canonical, lineage_cols=None)
    assert not res.rows.empty
    assert (res.rows["lineage"] == "enron_dedup_gt").all()
    # a non-empty requested tuple that is absent stays loud
    with pytest.raises(ValueError, match="lineage columns"):
        assemble_enron_gt(gt, canonical)


def test_enron_gt_cap_two_x_correspondence_train():
    canonical = _canonical()
    n_corr_train = int(((canonical["doc_type"] == "correspondence")
                        & (canonical["split"] == "train")).sum())
    assert n_corr_train == 3
    cap = int(round(2.0 * n_corr_train)) - n_corr_train  # the plan cap
    groups = [_enron_gt_rows()] * 4
    gt = pd.concat([g for g in groups], ignore_index=True)
    gt["filename"] = [f"e{i:03d}.txt" for i in range(len(gt))]
    # distinct text per row so within-pool dedup does not pre-empt the cap
    gt["doc_text"] = [f"body {i}" for i in range(len(gt))]
    res = assemble_enron_gt(gt, canonical)
    assert len(res.rows) == cap
    cut = res.rejected[res.rejected["reason"] == "cap_2x"]
    assert len(cut) == 12 - cap
    # configurable cap multiplier
    res3 = assemble_enron_gt(gt, canonical, cap_mult=3.0)
    # the §6.5 share bound (added <= authentic = 1 per subclass here) binds
    # before the 3x global cap: at most one new row per subclass survives
    assert len(res3.rows) == 3
    assert (res3.rejected["reason"] == "mixture_cap").sum() == 3


def test_enron_gt_balanced_does_not_deepen_email_majority():
    """#114 regression: a ~98%-email Enron pool must not raise any subclass's
    share above the head's uniform target balance."""
    subs = ("email", "memo", "letter", "notice", "press_release", "demand",
            "meeting_request", "attorney_demand")
    canonical = _mini_canonical(n_corr=80, corr_subclasses=subs)  # 10 each
    rows = [{"filename": f"en_email_{i:04d}.txt", "expected": "correspondence",
             "expected_subclass": "email", "doc_text": f"email body {i}",
             "aeslc_join": "1"} for i in range(196)]
    for sc in subs[1:]:
        for i in range(12):
            rows.append({"filename": f"en_{sc}_{i:02d}.txt",
                         "expected": "correspondence",
                         "expected_subclass": sc,
                         "doc_text": f"{sc} body {i}", "aeslc_join": "1"})
    gt = pd.DataFrame(rows)
    n_corr_train = int(((canonical["doc_type"] == "correspondence")
                        & (canonical["split"] == "train")).sum())
    cap = int(round(2.0 * n_corr_train)) - n_corr_train
    res = assemble_enron_gt(gt, canonical)
    assert not res.rows.empty
    assert len(res.rows) <= cap                       # §6.2 global cap holds
    share = res.rows["subclass"].value_counts(normalize=True)
    target = 1.0 / len(subs)                          # head's uniform balance
    assert share.max() <= target + 1e-9, share.to_dict()
    assert share.get("email", 0.0) <= target + 1e-9
    # rejections stay loud: cap_2x (global overflow) / balance / mixture
    assert {"cap_2x", "enron_balance_cut", "mixture_cap"} & set(res.rejected["reason"])


def test_enron_gt_observed_head_guard():
    canonical = _canonical()
    # 'email' is on the CANONICAL correspondence enum but NOT observed in the
    # fixture GT -> a reject (issue #75 posture), never a fabricated label
    gt = pd.DataFrame([
        {"filename": "enron_email.txt", "expected": "correspondence",
         "expected_subclass": "email", "doc_text": "body",
         "aeslc_join": "1"},
        {"filename": "enron_junk.txt", "expected": "correspondence",
         "expected_subclass": "voicemail", "doc_text": "body",
         "aeslc_join": "1"},
    ])
    res = assemble_enron_gt(gt, canonical)
    assert res.rows.empty
    reasons = dict(zip(res.rejected["filename"], res.rejected["reason"], strict=False))
    assert reasons == {"enron_email.txt": "unresolvable_subclass",
                       "enron_junk.txt": "unresolvable_subclass"}


def test_enron_gt_missing_lineage_cols_raises():
    gt = pd.DataFrame([{"filename": "x.txt", "expected": "correspondence",
                        "expected_subclass": "letter", "doc_text": "body"}])
    with pytest.raises(ValueError, match="lineage"):
        assemble_enron_gt(gt, _canonical())


# ---------------------------------------------------------------------------
# Tier 1 — CUAD contract pool (#113)
# ---------------------------------------------------------------------------

def test_cuad_pool_maps_category_and_head_checks():
    canonical = _canonical()
    pool = pd.DataFrame([
        {"id": "cuad-c1", "input": {"doc_text": "Consulting agreement body."},
         "metadata": {"category": "Consulting Agreements"}},
        {"id": "cuad-m1", "input": {"doc_text": "Marketing agreement body."},
         "metadata": {"category": "Marketing"}},
        {"id": "cuad-x1", "input": {"doc_text": "Mystery body."},
         "metadata": {"category": "Bogus Family"}},
        {"id": "cuad-no-text", "input": {"doc_text": "  "},
         "metadata": {"category": "Marketing"}},
        # nested objects may arrive JSON-encoded (CSV/parquet re-export)
        {"id": "cuad-str", "input": '{"doc_text": "Hosting agreement body."}',
         "metadata": '{"category": "Hosting"}'},
    ])
    res = assemble_cuad_pool(pool, canonical)
    kept = dict(zip(res.rows["filename"], res.rows["subclass"],
                    strict=False))
    assert kept == {"cuad-c1": "consulting", "cuad-m1": "marketing",
                    "cuad-str": "hosting"}
    assert (res.rows["doc_type"] == "contract").all()
    assert (res.rows["label_source"] == "cuad_full").all()
    assert (res.rows["split"] == "train").all()          # ladder rule
    assert (res.rows["title"] == "").all()  # no title/filename label leak
    reasons = dict(zip(res.rejected["filename"], res.rejected["reason"],
                       strict=False))
    # an unknown family must NOT be force-fit into the `other` fallback
    assert reasons["cuad-x1"] == "unresolvable_subclass"
    assert reasons["cuad-no-text"] == "missing_doc_text"


def test_cuad_pool_caps_and_dedups_vs_canonical():
    canonical = _canonical()
    pool = pd.DataFrame([
        {"id": f"cuad-{i:02d}", "input": {"doc_text": f"contract body {i}"},
         "metadata": {"category": "Supply"}} for i in range(6)])
    res = assemble_cuad_pool(pool, canonical)
    # contract train rows = 3 -> cap_mult 2.0 -> cap = 3 new rows
    assert len(res.rows) == 3
    assert len(res.rejected[res.rejected["reason"] == "cap_contract"]) == 3
    # sha-dedup vs the canonical corpus (different filename, same content)
    canonical_text = canonical[
        canonical["filename"] == "cuad_consulting_001.txt"]["doc_text"].iloc[0]
    dup = pd.DataFrame([{
        "id": "cuad-copy",
        "input": {"doc_text": canonical_text},
        "metadata": {"category": "Consulting Agreements"}}])
    res2 = assemble_cuad_pool(dup, canonical)
    assert res2.rows.empty
    assert res2.rejected["reason"].iloc[0] == "duplicate_sha_canonical"


def test_read_pool_dir_supports_jsonl(tmp_path):
    """The CUAD pool is JSONL; `_read_pool_dir` must read it (parquet/csv
    precedence elsewhere is untouched)."""
    d = tmp_path / "cuad_repo"
    d.mkdir()
    (d / "manifest.json").write_text("{}", encoding="utf-8")  # not a pool
    (d / "cuad.jsonl").write_text(
        json.dumps({"id": "x", "input": {"doc_text": "t"},
                    "metadata": {"category": "Marketing"}}) + "\n",
        encoding="utf-8")
    df = assemble_cli._read_pool_dir(d, "cuad")
    assert list(df["id"]) == ["x"]
    assert df["metadata"].iloc[0]["category"] == "Marketing"


# ---------------------------------------------------------------------------
# Tier 1 — insurance pools
# ---------------------------------------------------------------------------

def test_cms_pool_maps_subclasses_and_guards_head():
    pool = pd.DataFrame([
        {"filename": "cms_c.txt", "expected_subclass": "carrier", "doc_text": "EOB carrier"},
        {"filename": "cms_i.txt", "expected_subclass": "inpatient", "doc_text": "EOB inpatient"},
        {"filename": "cms_o.txt", "expected_subclass": "outpatient", "doc_text": "EOB outpatient"},
        {"filename": "cms_p.txt", "expected_subclass": "pde", "doc_text": "EOB pde"},
        {"filename": "cms_j.txt", "expected_subclass": "junk", "doc_text": "EOB junk"},
    ])
    res = assemble_cms_pool(pool, _canonical(),
                            head_subclasses=INSURANCE_SUBCLASSES)
    assert set(res.rows["subclass"]) == {"carrier", "inpatient", "outpatient", "pde"}
    assert (res.rows["doc_type"] == "insurance_claim").all()
    assert res.rows["label_source"].eq("cms_desynpuf_rendered").all()
    rej = res.rejected[res.rejected["filename"] == "cms_j.txt"]
    assert rej["reason"].iloc[0] == "subclass_not_on_head"


def test_gnotheia_pool_all_property():
    pool = pd.DataFrame([
        {"filename": "g_1.txt", "doc_text": "polycontext one"},
        {"filename": "g_2.txt", "doc_text": "polycontext two"},
    ])
    res = assemble_gnotheia_pool(pool, _canonical(),
                                 head_subclasses=INSURANCE_SUBCLASSES)
    assert (res.rows["subclass"] == "property").all()
    assert len(res.rows) == 2
    # default head = OBSERVED fixture surface ({carrier, outpatient}) -> property
    # is off-head and must NOT be force-fit
    res_default = assemble_gnotheia_pool(pool, _canonical())
    assert res_default.rows.empty
    assert (res_default.rejected["reason"] == "subclass_not_on_head").all()


def test_bdr_pool_needs_render_skipped_by_default():
    tabular = pd.DataFrame([
        {"filename": "bdr_1.txt", "claim_amount": "1000"},   # no doc_text
        {"filename": "bdr_2.txt", "claim_amount": "2000"},
    ])
    res = assemble_bdr_pool(tabular, _canonical(),
                            head_subclasses=INSURANCE_SUBCLASSES)
    assert res.rows.empty                      # render step not reproducible
    assert set(res.needs_render["filename"]) == {"bdr_1.txt", "bdr_2.txt"}
    rendered = pd.DataFrame([
        {"filename": "bdr_3.txt", "doc_text": "Decision letter three"},
    ])
    res2 = assemble_bdr_pool(rendered, _canonical(),
                             head_subclasses=INSURANCE_SUBCLASSES)
    assert set(res2.rows["subclass"]) == {"auto"}
    assert res2.needs_render.empty


def test_insurbias_narrative_requires_head_token():
    pool = pd.DataFrame([
        {"filename": "ib_1.txt", "claim_narrative": "Claim narrative one"},
        {"filename": "ib_2.txt", "claim_narrative": "Claim narrative two"},
    ])
    # default head has no narrative token -> loud rejects, never force-fit
    res = assemble_insurbias_pool(pool, _canonical())
    assert res.rows.empty
    assert (res.rejected["reason"] == "subclass_not_on_head").all()
    # a future head extension adopts them as narrative
    head = INSURANCE_SUBCLASSES + ("narrative",)
    res2 = assemble_insurbias_pool(pool, _canonical(), head_subclasses=head)
    assert (res2.rows["subclass"] == "narrative").all()
    assert len(res2.rows) == 2


def test_insurance_within_pool_dedup_first_occurrence_wins():
    text = "Shared EOB body."
    pool = pd.DataFrame([
        {"filename": "cms_dup_a.txt", "expected_subclass": "carrier", "doc_text": text},
        {"filename": "cms_dup_b.txt", "expected_subclass": "carrier", "doc_text": text},
    ])
    res = assemble_cms_pool(pool, _canonical(),
                            head_subclasses=INSURANCE_SUBCLASSES)
    assert list(res.rows["filename"]) == ["cms_dup_a.txt"]
    rej = res.rejected[res.rejected["reason"] == "duplicate_sha"]
    assert set(rej["filename"]) == {"cms_dup_b.txt"}


def test_insurance_cap_prioritizes_subclass_tails():
    canonical = _mini_canonical(n_ins=50, ins_subclasses=("carrier",))
    pool = pd.DataFrame([
        {"filename": f"g_p_{i:03d}.txt", "doc_text": f"prop {i}"}
        for i in range(30)] +
        [{"filename": f"b_a_{i:03d}.txt", "doc_text": f"auto {i}"}
         for i in range(30)] +
        [{"filename": f"cms_c{i:03d}.txt", "expected_subclass": "carrier",
          "doc_text": f"carrier {i}"} for i in range(30)])
    gno = assemble_gnotheia_pool(pool.iloc[:30], canonical,
                                 head_subclasses=INSURANCE_SUBCLASSES)
    bdr = assemble_bdr_pool(pool.iloc[30:60], canonical,
                            head_subclasses=INSURANCE_SUBCLASSES)
    cms = assemble_cms_pool(pool.iloc[60:], canonical,
                            head_subclasses=INSURANCE_SUBCLASSES)
    combined, _ = combine_tier1([("gnotheia", gno), ("bdr", bdr), ("cms", cms)])
    ins = combined[combined["doc_type"] == "insurance_claim"]
    # the cms carrier rows are genuinely assembled pre-cap (not rejected
    # off-head): 30 property + 30 auto + 30 carrier
    assert (ins["subclass"] == "carrier").sum() == 30
    kept, cut = apply_insurance_cap(ins, canonical, class_cap_mult=2.0)
    # budget = 2*50 - 50 = 50 new rows; tails (property, auto) fill it first
    assert len(kept) == 50
    counts = kept["subclass"].value_counts().to_dict()
    assert counts["property"] == 30
    assert counts["auto"] == 20
    assert "carrier" not in counts
    # all 30 assembled carrier rows are genuinely cut by tail priority
    assert sum(fn.startswith("cms_c") for fn in cut["filename"]) == 30
    assert (cut["reason"] == "cap_insurance_class").all()
    # under budget -> no cut at all
    kept2, cut2 = apply_insurance_cap(ins.iloc[:10], canonical,
                                      class_cap_mult=2.0)
    assert len(kept2) == 10 and cut2.empty


def test_combine_tier1_cross_pool_dedup():
    text = "Same doc in two pools."
    a = pd.DataFrame([{"filename": "a_1.txt", "expected_subclass": "carrier",
                       "doc_text": text}])  # pool 1 (earlier wins)
    b = pd.DataFrame([{"filename": "b_1.txt", "doc_text": text},  # cross-pool dup
                      {"filename": "b_2.txt", "doc_text": text + " unique"}])
    ra = assemble_cms_pool(a, _canonical(), head_subclasses=INSURANCE_SUBCLASSES)
    rb = assemble_gnotheia_pool(b, _canonical(),
                                head_subclasses=INSURANCE_SUBCLASSES)
    rows, rejects = combine_tier1([("cms", ra), ("gnotheia", rb)])
    # b_1 duplicates a_1's sha (same text) -> cross-pool drop; b_2 unique
    assert len(rows[rows["content_sha256"] == content_sha256(text)]) == 1
    assert set(rows[rows["content_sha256"] == content_sha256(text)]["filename"]) == {"a_1.txt"}
    # the earlier pool (cms) genuinely wins the cross-pool sha collision
    assert rows[rows["filename"] == "a_1.txt"]["subclass"].iloc[0] == "carrier"
    assert "b_2.txt" in set(rows["filename"])
    rej = rejects[rejects["reason"] == "duplicate_sha_cross_pool"]
    assert set(rej["filename"]) == {"b_1.txt"}


# ---------------------------------------------------------------------------
# Tier 2 — pseudo-label scaffold
# ---------------------------------------------------------------------------

def _cands(n: int, subclass: str, start_conf: float = 0.99,
           prefix: str = "cand") -> pd.DataFrame:
    return pd.DataFrame([{
        "filename": f"{prefix}_{i:03d}.txt", "doc_text": f"blind body {i}",
        "pred_doc_type": "correspondence", "pred_subclass": subclass,
        "doc_type_conf": start_conf - i * 0.001,
        "subclass_conf": start_conf - i * 0.001,
        "agreement": start_conf - i * 0.001,
    } for i in range(n)])


def test_pseudo_confidence_gates():
    cands = pd.DataFrame([
        {"filename": "ok.txt", "doc_text": "b", "pred_doc_type": "correspondence",
         "pred_subclass": "letter", "doc_type_conf": 0.96, "subclass_conf": 0.95,
         "agreement": 0.95},
        {"filename": "dt.txt", "doc_text": "b", "pred_doc_type": "correspondence",
         "pred_subclass": "letter", "doc_type_conf": 0.94, "subclass_conf": 0.95,
         "agreement": 0.95},
        {"filename": "sc.txt", "doc_text": "b", "pred_doc_type": "correspondence",
         "pred_subclass": "letter", "doc_type_conf": 0.96, "subclass_conf": 0.89,
         "agreement": 0.95},
        {"filename": "ag.txt", "doc_text": "b", "pred_doc_type": "correspondence",
         "pred_subclass": "letter", "doc_type_conf": 0.96, "subclass_conf": 0.95,
         "agreement": 0.89},
        {"filename": "na.txt", "doc_text": "b", "pred_doc_type": "correspondence",
         "pred_subclass": "letter", "doc_type_conf": 0.96, "subclass_conf": "x",
         "agreement": 0.95},
        {"filename": "wt.txt", "doc_text": "b", "pred_doc_type": "contract",
         "pred_subclass": "letter", "doc_type_conf": 0.96, "subclass_conf": 0.95,
         "agreement": 0.95},
    ])
    res = assemble_pseudo_labels(
        cands, _canonical(), head_subclasses=("letter",))
    assert list(res.rows["filename"]) == ["ok.txt"]
    reasons = dict(zip(res.rejected["filename"], res.rejected["reason"], strict=False))
    assert reasons == {"dt.txt": "below_doc_type_conf", "sc.txt": "below_subclass_conf",
                       "ag.txt": "below_agreement", "na.txt": "missing_confidence",
                       "wt.txt": "not_correspondence"}
    row = res.rows.iloc[0]
    assert row["label_source"] == "pseudo_enron"
    assert row["label_confidence"] == 0.95
    assert row["example_weight"] == 0.5
    assert row["purpose"] == PURPOSE_TRAIN_ONLY and row["split"] == "train"


def test_pseudo_cap_and_balanced_stratification():
    canonical = _mini_canonical(n_corr=100, corr_subclasses=("email", "letter", "memo"))
    cands = pd.concat([_cands(40, "email", prefix="em"),
                       _cands(40, "letter", prefix="lt"),
                       _cands(40, "memo", prefix="mm")], ignore_index=True)
    res = assemble_pseudo_labels(
        cands, canonical, head_subclasses=("email", "letter", "memo"))
    cap = int(round(0.30 * 100))
    assert len(res.rows) == cap
    assert (res.rows["subclass"].value_counts() == cap // 3).all()
    assert res.rejected["reason"].isin(["pseudo_cap", "pseudo_balance_cut"]).all()


def test_pseudo_grouped_split_seam_blocks_family_straddle():
    canonical = _mini_canonical(n_corr=12, family_col="thread_id")
    # families 0..6 split 90/10 by family value: some land in validation
    canonical["split"] = "train"
    canonical.loc[canonical["thread_id"] == "fam_00", "split"] = "validation"
    cands = pd.DataFrame([
        {"filename": "c_train.txt", "doc_text": "a", "pred_doc_type": "correspondence",
         "pred_subclass": "email", "doc_type_conf": 0.99, "subclass_conf": 0.99,
         "agreement": 0.99, "thread_id": "fam_03"},   # canonical train family
        {"filename": "c_val.txt", "doc_text": "b", "pred_doc_type": "correspondence",
         "pred_subclass": "email", "doc_type_conf": 0.99, "subclass_conf": 0.99,
         "agreement": 0.99, "thread_id": "fam_00"},   # canonical VALIDATION family
    ])
    res = assemble_pseudo_labels(
        cands, canonical, head_subclasses=("email",), family_col="thread_id")
    assert list(res.rows["filename"]) == ["c_train.txt"]
    rej = res.rejected[res.rejected["filename"] == "c_val.txt"]
    assert rej["reason"].iloc[0] == "family_straddles_val_test"


def test_grouped_split_seam_noop_without_family_column():
    rows = _cands(2, "letter")
    rows["thread_id"] = ["t_a", "t_b"]
    kept, rejected = grouped_split_seam(rows, _canonical(), "thread_id")
    assert len(kept) == 2 and rejected.empty  # canonical has no families -> no-op
    # ...but a family column missing on the ROWS side is a data defect
    with pytest.raises(ValueError, match="family_col"):
        grouped_split_seam(_cands(1, "letter"), _canonical(), "thread_id")


def test_assign_grouped_split_families_never_straddle():
    df = pd.DataFrame({
        "filename": [f"r{i:04d}" for i in range(20)],
        "doc_type": ["correspondence"] * 20,
        "family": [f"fam_{i // 2:02d}" for i in range(20)],
    })
    out = assign_grouped_split(df, "family", val_fraction=0.5)
    assert set(out["split"]) == {"train", "validation"}
    fam_to_split = out.groupby("family")["split"].unique().apply(tuple).to_dict()
    assert all(len(s) == 1 for s in fam_to_split.values()), fam_to_split
    # rows without a family are a data defect, not a fallback
    df.loc[0, "family"] = ""
    with pytest.raises(ValueError, match="family"):
        assign_grouped_split(df, "family")


# ---------------------------------------------------------------------------
# Tier 3 — eligibility, mixture math, seven gates
# ---------------------------------------------------------------------------

def test_synthetic_cap_table():
    # (5,3),(15,2),(30,1),(75,0): table from config, plan §6.5
    assert [synthetic_cap(n) for n in (3, 4, 5, 14, 15, 29, 30, 74, 75, 100)] \
        == [0, 0, 15, 42, 30, 58, 30, 74, 0, 0]


def test_eligibility_table():
    counts = {"hosting": 10, "consulting": 2, "transportation": 80, "license": 16}
    scc = {"hosting": "contract", "consulting": "contract",
           "transportation": "contract", "license": "contract"}
    verdicts = evaluate_eligibility(counts, subclass_class=scc)
    assert verdicts["hosting"].eligible is True
    assert verdicts["hosting"].cap == 30
    assert verdicts["consulting"].eligible is False   # 0-4 -> no synthetic-only
    assert verdicts["consulting"].cap == 0
    assert verdicts["transportation"].eligible is False  # 75+ -> none
    assert "tier_cap_zero" in verdicts["transportation"].reasons
    # f1 below floor is an additional eligibility reason
    v = evaluate_eligibility(counts, f1_by_subclass={"hosting": 0.4},
                             f1_floor=0.65)
    assert "macro_f1_below_floor" in v["hosting"].reasons
    # high confusion with a neighbor also qualifies
    v2 = evaluate_eligibility(counts, confusion={"hosting": ["license"]})
    assert "high_confusion_with_neighbor" in v2["hosting"].reasons


def test_mixture_math():
    caps = mixture_caps({"a": 60, "b": 40})
    # global: <= 0.4/(1-0.4) x 100 = 66; per-subclass 50%: s <= auth
    assert caps["__global_cap__"] == 66
    assert caps["__authentic_total__"] == 100
    # need-weighted water-fill levels a and b equally until the budget binds
    assert caps["a"] == 33 and caps["b"] == 33
    assert caps["a"] + caps["b"] <= caps["__global_cap__"]
    # per-subclass share bound alone binds when the global cap does not
    caps2 = mixture_caps({"a": 10, "b": 10}, global_share=0.99)
    assert caps2["a"] == 10 and caps2["b"] == 10


def test_mixture_caps_order_invariant_and_no_alphabet_starvation():
    """#115: renaming/permuting subclasses must not change the allocation,
    and an alphabetically-late tail must not be starved to 0."""
    counts = {"affiliate": 9, "agency": 13, "license": 41, "maintenance": 30,
              "outsourcing": 18, "transportation": 11}
    base = mixture_caps(counts)
    # permutation of input key order -> identical allocation
    import random
    keys = list(counts)
    random.Random(0).shuffle(keys)
    shuffled = {k: counts[k] for k in keys}
    assert mixture_caps(shuffled) == base
    assert mixture_caps(dict(reversed(list(counts.items())))) == base
    # the old greedy sorted-name pass starved the late-alphabet tails; the
    # water-fill fills the smallest allowances first, so none is zero
    assert base["transportation"] > 0
    assert base["outsourcing"] > 0
    assert (base["affiliate"] + base["agency"] + base["license"]
            + base["maintenance"] + base["outsourcing"]
            + base["transportation"]) <= base["__global_cap__"]


def test_apply_mixture_caps_cuts_rows():
    rows = pd.concat([_rows_t3(f"a{i:03d}", "a") for i in range(70)] +
                     [_rows_t3(f"b{i:03d}", "b") for i in range(10)],
                     ignore_index=True)
    kept, cut = apply_mixture_caps(rows, {"a": 60, "b": 40})
    # water-fill allowances: a=33, b=33; b only has 10 candidates
    assert len(kept) == 43 and len(cut) == 37
    assert kept["subclass"].value_counts().to_dict() == {"a": 33, "b": 10}
    assert (cut["reason"] == "mixture_cap").all()


def _rows_t3(fn: str, subclass: str) -> pd.DataFrame:
    from mailroom_ml.enrichment import rows_frame
    return rows_frame([{
        "filename": fn, "doc_text": f"candidate {subclass} {fn}",
        "doc_type": "contract" if subclass == "a" else "contract",
        "subclass": subclass, "title": fn,
        "source_corpus": "synthetic:contract/a", "source_revision": "card",
        "label_source": "synthetic_card", "label_confidence": 1.0,
        "example_weight": 0.6, "lineage": "label_card+7gates", "tier": 3,
    }])


def test_gate_schema_validation():
    card = LabelCard(parent_class="contract", subclass="hosting")
    ok = {"doc_text": "x", "parent_class": "contract", "subclass": "hosting",
          "title": "t"}
    assert gate_schema(ok, card).decision == "pass"
    assert gate_schema({**ok, "doc_text": " "}, card).decision == "reject"
    assert gate_schema({**ok, "parent_class": "merger_agreement"}, card).decision == "reject"
    assert gate_schema({**ok, "subclass": "license"}, card).decision == "reject"
    assert gate_schema({**ok, "title": ""}, card).decision == "reject"


def test_gate_lexical_contamination_detects_overlap():
    corpus = ["This consulting agreement sets out the parties obligations.",
              "Other unrelated text about parking permits."]
    verbatim = corpus[0]
    assert gate_lexical_contamination(verbatim, corpus).decision == "reject"
    fresh = ("The sky is blue and the weather is pleasant for walking "
             "in the park today with friends.")
    assert gate_lexical_contamination(fresh, corpus).decision == "pass"
    # shorter n-grams are more sensitive; n is a knob
    assert gate_lexical_contamination(fresh, corpus, n=3).decision == "pass"


def test_gate_entity_overlap():
    corpus = ["Acme Corp delivered the inventory on time."]
    text_with_known = "Vertex Dynamics signed with Acme Corp yesterday."
    res = gate_entity_overlap(text_with_known, (), known_entities=("acme corp",))
    assert res.decision == "reject" and "fictional" in res.reason
    # proper-noun span shared with a corpus row -> reject (basic extractive probe)
    res2 = gate_entity_overlap("Welcome to Acme Corp quarterly review.", corpus)
    assert res2.decision == "reject"
    res3 = gate_entity_overlap("Vertex Dynamics held its board meeting.", corpus)
    assert res3.decision == "pass"


def test_stub_gates_are_not_run_not_pass():
    card = LabelCard(parent_class="contract", subclass="hosting")
    cand = {"doc_text": "x", "parent_class": "contract", "subclass": "hosting"}
    assert gate_rule_cues(cand, card).decision == "not_run"
    assert gate_independent_adjudication(cand, card).decision == "not_run"
    assert gate_embedding_diversity(cand).decision == "not_run"
    assert gate_model_disagreement(cand).decision == "not_run"


def test_run_seven_gates_short_circuits_and_audits():
    audit = AuditStore()
    corpus = ["The hosting agreement includes service level targets."]
    card = LabelCard(parent_class="contract", subclass="hosting")
    contaminated = {"id": "c1", "parent_class": "contract",
                    "subclass": "hosting", "title": "t",
                    "doc_text": corpus[0]}
    report = run_seven_gates(contaminated, card, corpus, audit=audit)
    assert report.passed is False
    assert report.first_failure is not None
    assert report.first_failure.gate == "lexical_contamination"
    assert len(report.gates) == 2  # schema passed, lexical rejected -> stop
    assert len(audit) == 1
    # a clean candidate still cannot pass while stub gates are not_run:
    # nothing is adopted without the full gate pipeline
    clean = {"id": "c2", "parent_class": "contract", "subclass": "hosting",
             "title": "t", "doc_text": "Vertex Dynamics hosted a migration."}
    report2 = run_seven_gates(clean, card, corpus, known_entities=("acme corp",),
                              audit=audit)
    assert report2.passed is False
    assert report2.first_failure.gate == "rule_cues"
    assert len(audit) == 2
    assert all(r["passed"] is False for r in audit.records)


def test_run_seven_gates_does_not_run_later_gates_after_failure(monkeypatch):
    audit = AuditStore()
    corpus = ["The hosting agreement includes service level targets."]
    card = LabelCard(parent_class="contract", subclass="hosting")
    contaminated = {"id": "c1", "parent_class": "contract",
                    "subclass": "hosting", "title": "t",
                    "doc_text": corpus[0]}
    calls = {"model_disagreement": 0}

    def _spy(*args, **kwargs):
        calls["model_disagreement"] += 1
        return gate_model_disagreement(*args, **kwargs)

    monkeypatch.setattr(
        "mailroom_ml.enrichment.gate_model_disagreement", _spy)
    report = run_seven_gates(contaminated, card, corpus, audit=audit)
    assert report.passed is False
    assert calls["model_disagreement"] == 0


def test_human_spot_audit_seam():
    rows = pd.concat([_rows_t3(f"h{i:03d}", "hosting") for i in range(3)] +
                     [_rows_t3(f"l{i:03d}", "license") for i in range(2)],
                     ignore_index=True)
    pending = human_spot_audit(rows, {"hosting": 2, "license": 60},
                               min_authentic=5)
    assert pending == {"hosting": 3}  # license has an authentic floor


# ---------------------------------------------------------------------------
# Windows + deterministic rebuilds
# ---------------------------------------------------------------------------

def test_build_enrichment_windows_committed_schema():
    recs = [{"filename": "w2.txt", "title": "T2", "doc_text": "b",
             "doc_type": "correspondence", "subclass": "letter",
             "windows": ["T2\n\nw2a", "T2\n\nw2b"]},
            {"filename": "w1.txt", "title": "T1", "doc_text": "a",
             "doc_type": "insurance_claim", "subclass": "property",
             "windows": ["T1\n\nw1a"]}]
    wins = build_enrichment_windows(recs)
    assert list(wins.columns) == list(WINDOW_SCHEMA_COLUMNS)
    assert (wins["split"] == "train").all()          # windows train-only too
    assert wins["filename"].tolist() == ["w1.txt", "w2.txt", "w2.txt"]
    assert wins[wins["filename"] == "w2.txt"]["window_index"].tolist() == [0, 1]
    assert (wins["window_tokens"] >= 1).all()


def test_assembler_determinism_two_runs_identical():
    gt = _enron_gt_rows()
    canonical = _canonical()
    a = assemble_enron_gt(gt, canonical)
    b = assemble_enron_gt(gt, canonical)
    assert a.rows.equals(b.rows)
    assert a.rejected.equals(b.rejected)


def test_label_maps_regenerate_over_merged_docs():
    canonical = _canonical()
    res = assemble_gnotheia_pool(
        pd.DataFrame([{"filename": "g_1.txt", "doc_text": "polycontext one"}]),
        canonical, head_subclasses=INSURANCE_SUBCLASSES)
    merged = pd.concat([canonical, res.rows[list(DOCS_SCHEMA_COLUMNS)]],
                       ignore_index=True)
    maps = label_maps(merged)
    # enrichment rows widened the insurance head to the observed property key
    assert "property" in maps["insurance_claim"]["labels"]
    # weights recomputed over train only, enrichment included
    assert "property" in maps["insurance_claim"]["weights"]


# ---------------------------------------------------------------------------
# CLI — assembly into the committed stage layout
# ---------------------------------------------------------------------------

def write_stage(stage_dir: Path, docs: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for split in ("train", "validation", "test"):
        d = stage_dir / "data" / "documents" / split
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(docs[docs["split"] == split],
                                            preserve_index=False),
                       d / f"{split}-00000-of-00001.parquet")
    # a real staged tree carries the windows config for train+validation;
    # empty canonical windows are enough for verify to accept train-only
    # enrichment windows
    empty_wins = pd.DataFrame(columns=("filename", "window_index", "n_windows",
                                       "text", "doc_type", "subclass", "split",
                                       "window_tokens"), dtype=str)
    for split in ("train", "validation"):
        d = stage_dir / "data" / "windows" / split
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(empty_wins, preserve_index=False),
                       d / f"{split}-00000-of-00001.parquet")
    (stage_dir / "manifest.txt").write_text(
        "mailroom-modernbert-training manifest\n"
        "source_repo      : fake @ 0000\n"
        "rows_total       : 18\n", encoding="utf-8")


def _cli_args(stage_dir: Path, pools: dict[str, Path], *extra: str) -> list[str]:
    args = ["--stage", str(stage_dir), "--tiers", "1", "--no-windows"]
    for name, path in pools.items():
        args += [f"--{name}-pool", str(path)]
    return args + list(extra)


def _tiny_enron_pool(tmp_path: Path) -> Path:
    p = tmp_path / "enron_gt.csv"
    pd.DataFrame([
        {"filename": "enron_x1.txt", "expected": "correspondence",
         "expected_subclass": "letter", "doc_text": "Enron letter body one.",
         "aeslc_join": "1", "llm_zero_shot": "", "subject": "Re: one"},
        {"filename": "enron_x2.txt", "expected": "correspondence",
         "expected_subclass": "notice", "doc_text": "Enron notice body two.",
         "aeslc_join": "", "llm_zero_shot": "zs_1"},
    ]).to_csv(p, index=False)
    return p


def _tiny_gnotheia_pool(tmp_path: Path) -> Path:
    p = tmp_path / "gnotheia.csv"
    pd.DataFrame([
        {"filename": "g_1.txt", "doc_text": "polycontext property one"},
        {"filename": "g_2.txt", "doc_text": "polycontext property two"},
    ]).to_csv(p, index=False)
    return p


def _tiny_cuad_pool(tmp_path: Path) -> Path:
    """A tiny synthetic CUAD JSONL — never the 38 MB Hub file (hermetic)."""
    p = tmp_path / "cuad.jsonl"
    lines = [
        {"id": "cuad_cl_001", "input": {"doc_text": "Consulting agreement one."},
         "expected": {"clause_count": 1},
         "metadata": {"category": "Consulting Agreements"}, "tags": []},
        {"id": "cuad_tr_001", "input": {"doc_text": "Transportation agreement one."},
         "expected": {"clause_count": 1},
         "metadata": {"category": "Transportation"}, "tags": []},
    ]
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n",
                 encoding="utf-8")
    return p


def _empty_pools(tmp_path: Path, names: tuple[str, ...]) -> dict[str, Path]:
    out = {}
    for name in names:
        p = tmp_path / f"{name}.csv"
        pd.DataFrame(columns=["filename", "doc_text"]).to_csv(p, index=False)
        out[name] = p
    return out


def test_cli_dry_run_deterministic_and_writes_nothing(tmp_path, capsys):
    stage_dir = tmp_path / "stage"
    write_stage(stage_dir, _canonical())
    pools = {"enron": _tiny_enron_pool(tmp_path), "cuad": _tiny_cuad_pool(tmp_path),
             "cms": _empty_pools(tmp_path, ("cms",))["cms"],
             "gnotheia": _tiny_gnotheia_pool(tmp_path),
             "bdr": _empty_pools(tmp_path, ("bdr",))["bdr"],
             "insurbias": _empty_pools(tmp_path, ("insurbias",))["insurbias"]}
    out1 = assemble_cli.main(_cli_args(stage_dir, pools, "--dry-run"))
    first = capsys.readouterr().out
    out2 = assemble_cli.main(_cli_args(stage_dir, pools, "--dry-run"))
    second = capsys.readouterr().out
    assert out1 == 0 and out2 == 0
    assert first == second, "dry-run must be byte-deterministic"
    assert "dry-run: no files written" in first
    assert "enrichment-note" in first and "train ONLY" in first
    assert "tier1 enron" in first and '"kept": 2' in first
    # NOTHING was written: no parquet, no sidecars, manifest untouched
    assert not (stage_dir / "data" / "documents" / "train" / "enrichment-00000-of-00001.parquet").exists()
    assert not (stage_dir / "enrichment_provenance.parquet").exists()
    assert not (stage_dir / "enrichment_audit.jsonl").exists()
    assert "enrichment" not in (stage_dir / "manifest.txt").read_text(encoding="utf-8")



def test_cli_tier1_writes_marry_the_stage_layout(tmp_path, capsys):
    stage_dir = tmp_path / "stage"
    canonical = _canonical()
    write_stage(stage_dir, canonical)
    pools = {"enron": _tiny_enron_pool(tmp_path),
             "cuad": _tiny_cuad_pool(tmp_path),
             "cms": _empty_pools(tmp_path, ("cms",))["cms"],
             "gnotheia": _tiny_gnotheia_pool(tmp_path),
             "bdr": _empty_pools(tmp_path, ("bdr",))["bdr"],
             "insurbias": _empty_pools(tmp_path, ("insurbias",))["insurbias"]}
    rc = assemble_cli.main(_cli_args(stage_dir, pools))
    assert rc == 0
    out = capsys.readouterr().out
    assert "verify ok" in out
    # enrichment rows land in documents/train ONLY — never val/test
    doc_train = stage_dir / "data" / "documents" / "train"
    enrichment = pd.read_parquet(doc_train / "enrichment-00000-of-00001.parquet")
    assert list(enrichment.columns) == list(DOCS_SCHEMA_COLUMNS)
    assert (enrichment["split"] == "train").all()
    # gnotheia is off the OBSERVED fixture head -> only enron + CUAD rows adopt
    assert len(enrichment) == 4
    assert set(enrichment["filename"]) == {
        "enron_x1.txt", "enron_x2.txt", "cuad_cl_001", "cuad_tr_001"}
    assert set(enrichment[enrichment["filename"].str.startswith("cuad")]["subclass"]) == {
        "consulting", "transportation"}
    for split in ("validation", "test"):
        files = list((stage_dir / "data" / "documents" / split).glob("enrichment-*"))
        assert not files, f"enrichment must never land in {split}"
    # provenance sidecar carries the §5 columns, filename-joined
    prov = pd.read_parquet(stage_dir / "enrichment_provenance.parquet")
    assert set(prov.columns) >= {"filename", "source_corpus", "source_revision",
                                 "purpose", "label_source", "label_confidence",
                                 "example_weight", "lineage", "tier"}
    assert (prov["purpose"] == PURPOSE_TRAIN_ONLY).all()
    # only the two wired Tier-1 sources adopt under the fixture head
    assert set(prov["label_source"]) == {"enron_gt", "cuad_full"}
    # audit store: enron cap cuts? no — 2 enron rows under cap; gnotheia property
    # is off the OBSERVED fixture head -> rejected loud in enrichment_audit.jsonl
    audit_lines = (stage_dir / "enrichment_audit.jsonl").read_text(encoding="utf-8").splitlines()
    audit_recs = [json.loads(line) for line in audit_lines if line.strip()]
    reasons = {r["reason"] for r in audit_recs}
    assert "subclass_not_on_head" in reasons  # gnotheia rejected (fixture head)
    # labels.json regenerated over canonical + enrichment docs
    labels = json.loads((stage_dir / "labels.json").read_text(encoding="utf-8"))
    assert "letter" in labels["correspondence"]["labels"]
    # manifest carries the deterministic enrichment block — no timestamps
    manifest = (stage_dir / "manifest.txt").read_text(encoding="utf-8")
    assert assemble_cli.MANIFEST_START in manifest
    assert "tier1" in manifest
    assert not re.search(r"\d{4}-\d{2}-\d{2}", manifest)
    # held-out test split is byte-identical to the pre-enrichment state
    before_test = canonical[canonical["split"] == "test"].sort_values("filename")
    after_test = pd.read_parquet(stage_dir / "data" / "documents" / "test" / "test-00000-of-00001.parquet")
    assert before_test.reset_index(drop=True).equals(after_test)


def test_cli_rerun_is_byte_identical(tmp_path):
    stage_dir = tmp_path / "stage"
    write_stage(stage_dir, _canonical())
    pools = {"enron": _tiny_enron_pool(tmp_path),
             "cuad": _tiny_cuad_pool(tmp_path),
             "cms": _empty_pools(tmp_path, ("cms",))["cms"],
             "gnotheia": _tiny_gnotheia_pool(tmp_path),
             "bdr": _empty_pools(tmp_path, ("bdr",))["bdr"],
             "insurbias": _empty_pools(tmp_path, ("insurbias",))["insurbias"]}
    assert assemble_cli.main(_cli_args(stage_dir, pools)) == 0
    snapshot = {
        rel: (stage_dir / rel).read_bytes()
        for rel in ("manifest.txt", "labels.json", "enrichment_audit.jsonl",
                    "enrichment_provenance.parquet")
    }
    snap_doc = (stage_dir / "data" / "documents" / "train"
                / "enrichment-00000-of-00001.parquet").read_bytes()
    assert assemble_cli.main(_cli_args(stage_dir, pools)) == 0
    for rel, blob in snapshot.items():
        assert (stage_dir / rel).read_bytes() == blob, f"{rel} drifted across reruns"
    assert (stage_dir / "data" / "documents" / "train"
            / "enrichment-00000-of-00001.parquet").read_bytes() == snap_doc
    assert not re.search(rb"\d{4}-\d{2}-\d{2}",
                         (stage_dir / "enrichment_audit.jsonl").read_bytes())


def test_cli_tier2_requires_blind_pool(tmp_path, capsys):
    stage_dir = tmp_path / "stage"
    write_stage(stage_dir, _canonical())
    args = ["--stage", str(stage_dir), "--tiers", "2", "--no-windows"]
    assert assemble_cli.main(args) == 2
    assert "blind-pool" in capsys.readouterr().err


def test_cli_tier2_end_to_end(tmp_path):
    stage_dir = tmp_path / "stage"
    canonical = _mini_canonical(n_corr=40, corr_subclasses=("email", "letter"))
    write_stage(stage_dir, canonical)
    blind = tmp_path / "blind.csv"
    pd.concat([_cands(10, "email", prefix="em"), _cands(10, "letter", prefix="lt")],
              ignore_index=True).to_csv(blind, index=False)
    args = ["--stage", str(stage_dir), "--tiers", "2", "--no-windows",
            "--blind-pool", str(blind)]
    rc = assemble_cli.main(args)
    assert rc == 0
    enrichment = pd.read_parquet(stage_dir / "data" / "documents" / "train"
                                 / "enrichment-00000-of-00001.parquet")
    assert len(enrichment) == int(round(0.30 * 40))  # the 30% cap
    assert (enrichment["subclass"].value_counts() == 6).all()  # balanced
    prov = pd.read_parquet(stage_dir / "enrichment_provenance.parquet")
    assert (prov["label_source"] == "pseudo_enron").all()


@requires_transformers
@pytest.mark.train
def test_cli_windows_marry_the_windows_layout(tmp_path):
    stage_dir = tmp_path / "stage"
    write_stage(stage_dir, _canonical())
    pools = {"enron": _tiny_enron_pool(tmp_path),
             "cuad": _tiny_cuad_pool(tmp_path),
             "cms": _empty_pools(tmp_path, ("cms",))["cms"],
             "gnotheia": _empty_pools(tmp_path, ("gnotheia",))["gnotheia"],
             "bdr": _empty_pools(tmp_path, ("bdr",))["bdr"],
             "insurbias": _empty_pools(tmp_path, ("insurbias",))["insurbias"]}
    args = ["--stage", str(stage_dir), "--tiers", "1"]
    for name, path in pools.items():
        args += [f"--{name}-pool", str(path)]
    assert assemble_cli.main(args) == 0
    wins = pd.read_parquet(stage_dir / "data" / "windows" / "train"
                           / "enrichment-00000-of-00001.parquet")
    assert list(wins.columns) == list(WINDOW_SCHEMA_COLUMNS)
    assert (wins["split"] == "train").all()
    assert wins["window_tokens"].ge(1).all()
    assert (wins["filename"].isin(
        pd.read_parquet(stage_dir / "data" / "documents" / "train"
                        / "enrichment-00000-of-00001.parquet")["filename"])).all()


def test_cli_tier3_eligibility_and_gates(tmp_path):
    stage_dir = tmp_path / "stage"
    canonical = _mini_canonical(n_corr=40, corr_subclasses=("email", "letter"))
    write_stage(stage_dir, canonical)
    cards = tmp_path / "cards.json"
    cards.write_text(json.dumps([
        {"parent_class": "contract", "subclass": "hosting",
         "positive_cues": ["hosting"], "negative_cues": [],
         "title_patterns": [], "required_structure": [], "target_length": 200},
    ]), encoding="utf-8")
    candidates = tmp_path / "cands.jsonl"
    candidates.write_text(
        json.dumps({"id": "s1", "parent_class": "contract", "subclass": "hosting",
                    "title": "Hosting Agreement", "doc_text": "Vertex Dynamics "
                    "agreed to host the production systems for three years."})
        + "\n", encoding="utf-8")
    args = ["--stage", str(stage_dir), "--tiers", "3", "--no-windows",
            "--tier3-cards", str(cards),
            "--tier3-candidates", str(candidates)]
    rc = assemble_cli.main(args)
    assert rc == 0
    audit = [json.loads(line) for line in
             (stage_dir / "enrichment_audit.jsonl").read_text(encoding="utf-8")
             .splitlines() if line.strip()]
    # stub gates (rule_cues + 3 more) block adoption in the scaffold
    assert all(not r["passed"] for r in audit if "candidate_id" in r)
    admission = [r for r in audit if r.get("candidate_id") == "s1"]
    assert admission and admission[0]["first_failure"] == "rule_cues"
    # eligibility/mixture facts land in the manifest block
    manifest = (stage_dir / "manifest.txt").read_text(encoding="utf-8")
    assert "tier3" in manifest


def test_cli_missing_stage_is_loud(tmp_path, capsys):
    args = ["--stage", str(tmp_path / "nope"), "--tiers", "1", "--no-windows",
            "--enron-pool", str(_tiny_enron_pool(tmp_path))]
    assert assemble_cli.main(args) == 2
    assert "build_dataset.py" in capsys.readouterr().err


def test_cli_unknown_cap_is_loud(tmp_path, capsys):
    stage_dir = tmp_path / "stage"
    write_stage(stage_dir, _canonical())
    args = ["--stage", str(stage_dir), "--tiers", "1", "--no-windows",
            "--caps", "bogus=1.0"]
    with pytest.raises(SystemExit):
        assemble_cli.main(args)
