"""Label normalization + label-map tests (port of corpus-eda test_prep.py).

Pins the DMR-066 normalization contract: every corpus contract surface
resolves to its canonical CUAD key, other classes resolve case-folded, and
unresolvable values become ``other``.  Runs on fixtures only — no snapshot.
"""
from __future__ import annotations

from conftest import fixture_rows
from mailroom_ml.config import DOC_TYPES
from mailroom_ml.dataset import build_documents, class_weights
from mailroom_ml.labels import (
    CONTRACT_SUBTYPE_KEYS,
    MAUD_CONSIDERATION_TYPES,
    SUBCLASS_BY_CLASS,
    label_maps,
    normalize_subclass,
)


def test_normalize_subclass_contract_surfaces():
    """Every corpus contract surface resolves to its canonical key (DMR-066)."""
    cases = {
        "Affiliate_Agreements": "affiliate",
        "Agency Agreements": "agency",
        "Co_Branding": "co_branding",
        "Collaboration": "collaboration",
        "Consulting Agreements": "consulting",
        "Development": "development",
        "Distributor": "distributor",
        "Endorsement": "endorsement",
        "Franchise": "franchise",
        "Hosting": "hosting",
        "IP": "ip",
        "Joint Venture": "joint_venture",
        "Joint Venture _ Filing": "joint_venture",
        "License_Agreements": "license",
        "Maintenance": "maintenance",
        "Manufacturing": "manufacturing",
        "Marketing": "marketing",
        "Non_Compete_Non_Solicit": "non_compete_no_solicit",
        "Outsourcing": "outsourcing",
        "Promotion": "promotion",
        "Reseller": "reseller",
        "Service": "service",
        "Sponsorship": "sponsorship",
        "Strategic Alliance": "strategic_alliance",
        "Supply": "supply",
        "Transportation": "transportation",
    }
    for surface, expected in cases.items():
        assert normalize_subclass("contract", surface) == expected, surface


def test_normalize_subclass_other_classes():
    assert normalize_subclass("merger_agreement", "all_cash") == "all_cash"
    assert normalize_subclass("merger_agreement", "All Cash") == "all_cash"
    assert normalize_subclass("merger_agreement", "bogus") == "other"
    assert normalize_subclass("corporate_record", "bylaws") == "bylaws"
    assert normalize_subclass("correspondence", "email") == "email"
    assert normalize_subclass("insurance_claim", "outpatient") == "outpatient"
    assert normalize_subclass("insurance_claim", "") == "other"
    assert normalize_subclass("contract", None) == "other"
    # unknown doc_type: no enum -> always "other"
    assert normalize_subclass("bogus_class", "email") == "other"


def test_vocabularies_vendored_shape():
    """Canonical table shapes pinned by the committed corpus-eda build."""
    assert DOC_TYPES == (
        "contract", "merger_agreement", "corporate_record", "correspondence",
        "insurance_claim",
    )
    assert len(CONTRACT_SUBTYPE_KEYS) == 25
    assert MAUD_CONSIDERATION_TYPES == (
        "all_cash", "all_stock", "mixed_cash_stock",
        "mixed_cash_stock_election", "other",
    )
    for cls in DOC_TYPES:
        assert "other" in SUBCLASS_BY_CLASS[cls] or cls == "insurance_claim"
        assert SUBCLASS_BY_CLASS[cls][-1] == ("other" if cls != "insurance_claim" else "auto")


def test_label_maps_and_weights():
    docs = build_documents(fixture_rows())
    maps = label_maps(docs)
    assert maps["doc_type"]["labels"] == list(DOC_TYPES) + ["unknown"]
    assert maps["contract"]["labels"][0] == "affiliate"
    w = maps["doc_type"]["weights"]
    # inverse-frequency invariant: sum over classes of weight*count == total
    counts = docs[docs["split"] == "train"]["doc_type"].value_counts().to_dict()
    assert abs(sum(w[k] * counts[k] for k in w) - sum(counts.values())) < 1e-9
    # weights cover every training label
    for cls in DOC_TYPES:
        assert set(maps[cls]["weights"]) <= set(maps[cls]["labels"])
    # id2label/label2id are exact inverses of each other
    for head in maps.values():
        assert {v: k for k, v in head["id2label"].items()} == head["label2id"]


def test_class_weights_train_only():
    docs = build_documents(fixture_rows())
    weights = class_weights(docs, "doc_type")
    # zero-weight classes absent; fixture frames weights only over train draws
    for w in weights.values():
        assert w > 0
    assert set(weights) - set(DOC_TYPES) == set()
    # deterministic
    assert class_weights(docs, "doc_type") == weights


def test_json_serializable_maps():
    import json
    docs = build_documents(fixture_rows())
    round_tripped = json.loads(json.dumps(label_maps(docs), sort_keys=True))
    assert round_tripped["doc_type"]["labels"] == list(DOC_TYPES) + ["unknown"]
