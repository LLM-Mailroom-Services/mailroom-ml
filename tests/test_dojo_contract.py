"""mailroom-ml stays aligned with the Dojo (llm-dojo-scoring), the scoring core.

Training and serving read the hand-kept tables in ``labels.py``, ``normalize.py``
and ``config.py``; these tests hold them to the vendored Dojo contract. Eval and
gate reports call the Dojo itself; these tests pin that path too.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from mailroom_ml import config, labels, normalize
from mailroom_ml.dojo_contract import DOJO_COMMIT, DOJO_REQUIREMENT, load_contract
from mailroom_ml.m9a_gates import gates_all_met, m9a_gate_rows

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_contract()


def test_pin_is_one_commit_everywhere():
    """The dojo extra, the lockfile and the eval image requirement name the same commit."""
    assert len(DOJO_COMMIT) == 40 and all(c in "0123456789abcdef" for c in DOJO_COMMIT)
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["dojo"] == [DOJO_REQUIREMENT]
    assert DOJO_COMMIT in (ROOT / "uv.lock").read_text()


def test_vendored_contract_is_the_pinned_dojo_contract():
    """The vendored JSON is byte-for-byte what the pinned Dojo ships."""
    dojo_ml = pytest.importorskip("llm_dojo_scoring.ml")
    assert CONTRACT == dojo_ml.load_contract()
    assert CONTRACT["schema"] == "dojo.ml_contract/1"


def test_class_and_label_tables_match_the_contract():
    """Doc types, CUAD keys and aliases, MAUD types and subclass catalogs agree."""
    assert list(config.DOC_TYPES) == CONTRACT["doc_types"]
    assert set(labels.CONTRACT_SUBTYPE_KEYS) == {s["key"] for s in CONTRACT["contract_subtypes"]}
    assert labels.CONTRACT_SUBTYPE_LABELS == {s["key"]: s["label"] for s in CONTRACT["contract_subtypes"]}
    assert labels.SUBTYPE_ALIASES == CONTRACT["subtype_aliases"]
    assert list(labels.MAUD_CONSIDERATION_TYPES) == CONTRACT["maud_consideration_types"]
    unknown = CONTRACT["subclass_unknown"]
    for doc_type in CONTRACT["doc_types"]:
        dojo = set(CONTRACT["subclasses"][doc_type])
        ours = set(labels.SUBCLASS_BY_CLASS[doc_type])
        assert dojo <= ours <= dojo | {unknown}, doc_type


def test_every_corpus_surface_normalizes_like_the_dojo():
    """normalize_subclass gives the Dojo's canonical key for every surface it lists."""
    for doc_type, table in CONTRACT["subclass_normalization"].items():
        for surface, key in table.items():
            assert labels.normalize_subclass(doc_type, surface) == key, (doc_type, surface)


def test_intake_clerk_reproduces_the_dojo_cases():
    """Same prep steps, and the same output and stats on every golden case."""
    intake = CONTRACT["intake"]
    assert list(normalize.INTAKE_PREP_STEPS) == intake["prep_steps"]
    for case in intake["cases"]:
        output, stats = normalize.deterministic_normalize(case["input"])
        assert (output, stats) == (case["output"], case["stats"]), case["name"]


PER_DOC = [
    # contract head: one right, one wrong subclass, one abstain (None) -> misses
    {"gt_doc_type": "contract", "pred_doc_type": "contract", "gt_subclass": "license",
     "pred_subclass": "license", "fast_path": True, "doc_type_confidence": 0.97,
     "subclass_confidence": 0.9},
    {"gt_doc_type": "contract", "pred_doc_type": "contract", "gt_subclass": "supply",
     "pred_subclass": "license", "fast_path": False, "doc_type_confidence": 0.8,
     "subclass_confidence": 0.5},
    {"gt_doc_type": "contract", "pred_doc_type": "contract", "gt_subclass": "service",
     "pred_subclass": None, "fast_path": False, "doc_type_confidence": 0.7},
    # a llm_overflow doc stays out of the contract head (class not predicted right)
    {"gt_doc_type": "contract", "pred_doc_type": "llm_overflow", "gt_subclass": "agency",
     "pred_subclass": None, "fast_path": False},
    {"gt_doc_type": "correspondence", "pred_doc_type": "correspondence",
     "gt_subclass": "email", "pred_subclass": "email", "fast_path": True,
     "doc_type_confidence": 0.99, "subclass_confidence": 0.95},
]


def _harness_macro_f1(pairs):
    """The eval harness's own per-head definition, restated for the parity check."""
    classes = sorted({g for g, _ in pairs})
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def test_dojo_scorecard_matches_the_harness_head_definition():
    """The Dojo's per-class subclass F1 equals the harness's conditional per-head F1."""
    pytest.importorskip("llm_dojo_scoring.ml")
    from mailroom_ml.dojo_metrics import dojo_scorecard

    card = dojo_scorecard(PER_DOC, [0.9, 0.6], [True, False])
    pairs = [(d["gt_subclass"], d["pred_subclass"]) for d in PER_DOC
             if d["gt_doc_type"] == "contract" and d["pred_doc_type"] == "contract"]
    contract = card["by_class"]["contract"]
    assert contract["n"] == 4 and contract["n_class_right"] == 3
    assert contract["subclass_macro_f1"] == round(_harness_macro_f1(pairs), 6)
    assert card["by_class"]["correspondence"]["subclass_macro_f1"] == 1.0
    assert card["metrics"]["doc_type_accuracy"] == 0.8
    assert card["metrics"]["fast_path_coverage"] == 0.4
    assert card["window_ece"] is not None
    assert card["dojo"]["commit"] == DOJO_COMMIT


def test_gates_read_the_dojo_scorecard():
    """With a dojo block the gates use the Dojo's numbers, not the harness fields."""
    report = {
        "per_head": {"contract": {"macro_f1": 0.0}, "correspondence": {"macro_f1": 0.0}},
        "doc_type_accuracy": 0.0,
        "window_calibration": {"ece": 1.0},
        "dojo": {
            "by_class": {"contract": {"subclass_macro_f1": 0.31},
                         "correspondence": {"subclass_macro_f1": 0.4}},
            "metrics": {"doc_type_accuracy": 0.93},
            "window_ece": 0.03,
        },
    }
    assert [actual for _n, actual, _t, _h in m9a_gate_rows(report)] == [0.31, 0.4, 0.93, 0.03]
    assert gates_all_met(report)
    report["dojo"]["by_class"]["contract"]["subclass_macro_f1"] = None
    assert not gates_all_met(report)  # a missing Dojo figure fails, it never falls back


def test_reports_without_a_dojo_block_still_gate_on_their_own_fields():
    """Committed reports from before the Dojo pin keep rendering their recorded gates."""
    report = {"per_head": {"contract": {"macro_f1": 0.25}, "correspondence": {"macro_f1": 0.3}},
              "doc_type_accuracy": 0.9, "window_calibration": {"ece": 0.04}}
    assert [actual for _n, actual, _t, _h in m9a_gate_rows(report)] == [0.25, 0.3, 0.9, 0.04]
