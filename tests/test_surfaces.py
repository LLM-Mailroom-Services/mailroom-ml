"""Surface-derivation tests (mailroom-issues #66/#67/#68/#75).

Head vocabs must be derived from the pinned GT — train+validation rows only;
the held-out test split must never shape a head and zero-row enum labels are
not trainable classes: correspondence is 8-key (zero ``voicemail``, zero
``other``), corporate_record is the observed 10 keys (no
``certificate_of_formation``), insurance is 6-key.  The full-corpus test is
the #75 surface-drift CI guardrail.
"""
from __future__ import annotations

import pandas as pd
import pytest

from conftest import fixture_rows
from mailroom_ml.config import (
    CANONICAL_CORRESPONDENCE_SUBCLASSES,
    DATA_DIR,
    FINETUNE_REVISION,
    INSURANCE_SUBCLASSES,
    OBSERVED_CORPORATE_RECORD_SUBCLASSES,
)
from mailroom_ml.dataset import build_documents, load_corpus_rows
from mailroom_ml.labels import (
    CONTRACT_SUBTYPE_KEYS,
    MAUD_CONSIDERATION_TYPES,
    SUBCLASS_BY_CLASS,
    label_maps,
    observed_label_surfaces,
)


def test_observed_surfaces_deterministic_subset_of_canon():
    docs = build_documents(fixture_rows())
    a = observed_label_surfaces(docs)
    b = observed_label_surfaces(docs)
    assert a == b, "surface derivation must be deterministic"
    assert list(a) == list(b)  # stable head order
    for cls, surf in a.items():
        assert set(surf) <= set(SUBCLASS_BY_CLASS[cls]), (
            f"{cls} surface must be a subset of the canonical enum")
        assert len(set(surf)) == len(surf), f"{cls} surface has duplicate keys"


def test_surface_order_sanction_then_sorted_drift():
    # observed keys are ordered by the sanctioned config tuple; keys NOT in
    # the sanction (drift, #75) are appended in sorted order
    docs = pd.DataFrame({
        "filename": ["a", "b", "c", "d", "e"],
        "doc_type": ["correspondence"] * 5,
        "subclass": ["meeting_request", "email", "voicemail", "memo", "voicemail"],
        "split": ["train"] * 5,
    })
    assert observed_label_surfaces(docs)["correspondence"] == (
        "email", "memo", "meeting_request", "voicemail")


def test_observed_surfaces_never_contain_zero_row_labels():
    docs = build_documents(fixture_rows())
    surfaces = observed_label_surfaces(docs)
    for cls, surf in surfaces.items():
        if cls in ("contract", "merger_agreement"):
            continue  # sanctioned canon heads may include unobserved keys
        observed_rows = set(
            docs[(docs["split"] != "test") & (docs["doc_type"] == cls)]["subclass"])
        assert set(surf) == observed_rows, cls
        assert surf, f"{cls} head must not be empty on the fixture"


def test_observed_surfaces_exclude_test_rows():
    # a subclass present ONLY on held-out test rows must never enter a head
    docs = pd.DataFrame({
        "filename": ["a", "b", "c", "d"],
        "doc_type": ["correspondence"] * 4,
        "subclass": ["email", "memo", "voicemail", "voicemail"],
        "split": ["train", "validation", "test", "test"],
    })
    assert observed_label_surfaces(docs)["correspondence"] == ("email", "memo")
    # fixture-level: cms_outpatient_001 (outpatient) is the held-out test row
    fixture_docs = build_documents(fixture_rows())
    assert observed_label_surfaces(fixture_docs)["insurance_claim"] == ("carrier",)


def test_fixture_parity_correspondence_head():
    docs = build_documents(fixture_rows())
    surfaces = observed_label_surfaces(docs)
    # the fixture observes 3 correspondence keys — the head IS the observed
    # set, in sanctioned config order (letter < notice < attorney_demand)
    assert surfaces["correspondence"] == ("letter", "notice", "attorney_demand")
    # when the data covers all 8 canonical keys, the head equals the canon (#66)
    full = pd.DataFrame({
        "filename": [f"f{i}" for i in range(8)],
        "doc_type": ["correspondence"] * 8,
        "subclass": list(CANONICAL_CORRESPONDENCE_SUBCLASSES),
        "split": ["train"] * 8,
    })
    assert observed_label_surfaces(full)["correspondence"] == (
        CANONICAL_CORRESPONDENCE_SUBCLASSES)


def test_fixture_parity_corporate_record_head():
    docs = build_documents(fixture_rows())
    # observed keys in sanctioned config order (officer_certificate < bylaws)
    assert observed_label_surfaces(docs)["corporate_record"] == (
        "officer_certificate", "bylaws")


def test_contract_and_merger_heads_stay_canonical():
    docs = build_documents(fixture_rows())
    surfaces = observed_label_surfaces(docs)
    # sanctioned canon heads regardless of the (sparse) fixture observation
    assert surfaces["contract"] == CONTRACT_SUBTYPE_KEYS + ("other",)
    assert surfaces["merger_agreement"] == MAUD_CONSIDERATION_TYPES
    observed_contract = set(docs[docs["doc_type"] == "contract"]["subclass"])
    assert observed_contract <= set(surfaces["contract"])
    assert len(observed_contract) < len(surfaces["contract"])


def test_label_maps_use_observed_surfaces():
    docs = build_documents(fixture_rows())
    surfaces = observed_label_surfaces(docs)
    maps = label_maps(docs)
    for cls in surfaces:
        assert maps[cls]["labels"] == list(surfaces[cls]), cls
    # zero-row enum labels are never trainable classes (#66/#67)
    assert "voicemail" not in maps["correspondence"]["labels"]
    assert "other" not in maps["correspondence"]["labels"]
    assert "certificate_of_formation" not in maps["corporate_record"]["labels"]
    # derivation notes: observed-vs-canonical + pinned revision
    assert "surface=observed-gt" in maps["correspondence"]["note"]
    assert "surface=canonical-enum" in maps["contract"]["note"]
    assert f"corpus_revision={FINETUNE_REVISION}" in maps["contract"]["note"]
    # id2label/label2id exact inverses on an observed head
    head = maps["correspondence"]
    assert {v: k for k, v in head["id2label"].items()} == head["label2id"]
    # weights cover only labels present in train rows, all inside the head
    assert set(maps["correspondence"]["weights"]) <= set(head["labels"])


@pytest.mark.fullcorpus
@pytest.mark.skipif(
    not (DATA_DIR / "parquet" / "ground_truth" / "train").exists(),
    reason="local snapshot absent (data/parquet) — fetch via training/build_dataset.py",
)
def test_full_corpus_surfaces_match_sanctioned_sets():
    """#75 surface-drift guardrail on the real 3,302-row corpus.

    Any divergence from the tracker-sanctioned head vocabs (#66/#67/#68)
    fails CI and is quoted verbatim by the assertion message.
    """
    docs = build_documents(load_corpus_rows())
    surfaces = observed_label_surfaces(docs)
    expected = {
        "correspondence": CANONICAL_CORRESPONDENCE_SUBCLASSES,
        "insurance_claim": INSURANCE_SUBCLASSES,
        "corporate_record": OBSERVED_CORPORATE_RECORD_SUBCLASSES,
        "contract": CONTRACT_SUBTYPE_KEYS + ("other",),
        "merger_agreement": MAUD_CONSIDERATION_TYPES,
    }
    for cls, exp in expected.items():
        assert surfaces[cls] == exp, (
            f"SURFACE DRIFT #{cls}: observed={surfaces[cls]} "
            f"expected={exp}")
    # issue specifics, asserted explicitly for readable CI failures
    assert "voicemail" not in surfaces["correspondence"]
    assert "other" not in surfaces["correspondence"]
    assert "certificate_of_formation" not in surfaces["corporate_record"]
    # label maps must expose exactly those observed heads
    maps = label_maps(docs)
    for cls in expected:
        assert maps[cls]["labels"] == list(surfaces[cls]), cls
