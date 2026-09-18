"""Canonical label vocabularies + normalization (port of ``modernbert/prep.py``).

Vendored from the Digital Mailroom's llm-dojo-scoring package (DMR-066),
the SAME tables + normalization the sandbox corpus uses — this standalone
repo never imports the dojo package (Mailroom-Corpus-EDA commit cf096fa):

- ``llm_dojo_scoring/config.py``      CONTRACT_SUBTYPES / SUBTYPE_ALIASES /
                                       MAUD_CONSIDERATION_TYPES
- ``llm_dojo_scoring/corpus.py``      DOC_TYPE_SUBCLASSES /
                                       normalize_corpus_subclass
- ``llm_dojo_scoring/equivalences.py`` normalize_subtype / normalize_doc_subclass

The corpus GT surfaces (CORPUS_SUBCLASS_SURFACES) all resolve through these
tables; tests pin every surface value to its canonical key.
"""
from __future__ import annotations

import re
from typing import Any

from mailroom_ml.config import DOC_TYPES

__all__ = [
    "DOC_TYPES",
    "CONTRACT_SUBTYPE_KEYS",
    "SUBTYPE_ALIASES",
    "MAUD_CONSIDERATION_TYPES",
    "SUBCLASS_BY_CLASS",
    "CONTRACT_SUBTYPE_LABELS",
    "normalize_subclass",
    "label_maps",
]

# ---------------------------------------------------------------------------
# Canonical vocabularies
# ---------------------------------------------------------------------------

# 25 CUAD families (canonical snake_case keys) — dojo CONTRACT_SUBTYPE_KEYS.
CONTRACT_SUBTYPE_KEYS: tuple[str, ...] = (
    "affiliate", "agency", "co_branding", "collaboration", "consulting",
    "development", "distributor", "endorsement", "franchise", "hosting",
    "ip", "joint_venture", "license", "maintenance", "manufacturing",
    "marketing", "non_compete_no_solicit", "outsourcing", "promotion",
    "reseller", "service", "sponsorship", "strategic_alliance", "supply",
    "transportation",
)

# CUAD folder-name aliases -> canonical key — dojo SUBTYPE_ALIASES.
SUBTYPE_ALIASES: dict[str, str] = {
    "affiliate_agreements": "affiliate",
    "affiliate_agreement": "affiliate",
    "agency_agreements": "agency",
    "co_branding": "co_branding",
    "collaboration": "collaboration",
    "consulting_agreements": "consulting",
    "development": "development",
    "distributor": "distributor",
    "endorsement": "endorsement",
    "endorsement_agreement": "endorsement",
    "franchise": "franchise",
    "hosting": "hosting",
    "ip": "ip",
    "joint_venture": "joint_venture",
    "joint_venture_filing": "joint_venture",
    "license_agreements": "license",
    "maintenance": "maintenance",
    "manufacturing": "manufacturing",
    "marketing": "marketing",
    "non_compete_non_solicit": "non_compete_no_solicit",
    "outsourcing": "outsourcing",
    "promotion": "promotion",
    "reseller": "reseller",
    "service": "service",
    "sponsorship": "sponsorship",
    "strategic_alliance": "strategic_alliance",
    "supply": "supply",
    "transportation": "transportation",
}

# MAUD Type-of-Consideration keys — dojo MAUD_CONSIDERATION_TYPES.
MAUD_CONSIDERATION_TYPES: tuple[str, ...] = (
    "all_cash", "all_stock", "mixed_cash_stock", "mixed_cash_stock_election",
    "other",
)

# Canonical subclass keys per doc_type — dojo DOC_TYPE_SUBCLASSES (corpus
# classes only; the corpus ships a subset of the full enum).
SUBCLASS_BY_CLASS: dict[str, tuple[str, ...]] = {
    "contract": CONTRACT_SUBTYPE_KEYS + ("other",),
    "merger_agreement": MAUD_CONSIDERATION_TYPES,
    "corporate_record": (
        "bylaws", "articles_of_incorporation", "certificate_of_formation",
        "charter_amendment", "powers_of_attorney", "subsidiary_list",
        "rights_instrument", "indenture", "board_resolution",
        "officer_certificate", "other",
    ),
    "correspondence": (
        "email", "memo", "letter", "notice", "demand", "attorney_demand",
        "press_release", "meeting_request", "other",
    ),
    "insurance_claim": ("carrier", "inpatient", "outpatient", "pde",
                        "property", "auto"),
}

# Canonical label -> human label (contract families) for the card + id2label.
CONTRACT_SUBTYPE_LABELS: dict[str, str] = {
    "affiliate": "Affiliate Agreement",
    "agency": "Agency Agreement",
    "co_branding": "Co-Branding Agreement",
    "collaboration": "Collaboration / Cooperation Agreement",
    "consulting": "Consulting Agreement",
    "development": "Development Agreement",
    "distributor": "Distributor Agreement",
    "endorsement": "Endorsement Agreement",
    "franchise": "Franchise Agreement",
    "hosting": "Hosting Agreement",
    "ip": "IP Agreement",
    "joint_venture": "Joint Venture Agreement",
    "license": "License Agreement",
    "maintenance": "Maintenance Agreement",
    "manufacturing": "Manufacturing Agreement",
    "marketing": "Marketing Agreement",
    "non_compete_no_solicit": "Non-Compete / No-Solicit / Non-Disparagement Agreement",
    "outsourcing": "Outsourcing Agreement",
    "promotion": "Promotion Agreement",
    "reseller": "Reseller Agreement",
    "service": "Service Agreement",
    "sponsorship": "Sponsorship Agreement",
    "strategic_alliance": "Strategic Alliance Agreement",
    "supply": "Supply Agreement",
    "transportation": "Transportation Agreement",
}

_ALIAS_KEY_RE = re.compile(r"[^a-z0-9]")


def normalize_subclass(doc_type: str, value: Any) -> str:
    """Canonical subclass key for ``value`` under ``doc_type``.

    Replicates ``llm_dojo_scoring.corpus.normalize_corpus_subclass``
    (DMR-066): contract surfaces resolve through the CUAD alias table
    (case/separator folded); every other class through case-folded exact
    match against its canonical enum; unresolvable values become ``other``.
    """
    if value is None:
        return "other"
    raw = str(value).strip()
    if not raw:
        return "other"
    if doc_type == "contract":
        key = _ALIAS_KEY_RE.sub("", raw.lower())
        if not key:
            return "other"
        if key in CONTRACT_SUBTYPE_KEYS:
            return key
        aliases = {_ALIAS_KEY_RE.sub("", k): v for k, v in SUBTYPE_ALIASES.items()}
        if key in aliases:
            return aliases[key]
        for subtype in CONTRACT_SUBTYPE_KEYS:
            norm_label = _ALIAS_KEY_RE.sub("", CONTRACT_SUBTYPE_LABELS[subtype].lower())
            if key == norm_label or key.startswith(norm_label[:8]):
                return subtype
        return "other"
    allowed = SUBCLASS_BY_CLASS.get(doc_type, ())
    if raw in allowed:
        return raw
    key = _ALIAS_KEY_RE.sub("", raw.lower())
    for candidate in allowed:
        if key == _ALIAS_KEY_RE.sub("", candidate.lower()):
            return candidate
    return "other"


def label_maps(docs_df) -> dict[str, dict[str, Any]]:
    """Per-head id2label/label2id/weights for the hierarchical classifier.

    Builds the label contract consumed by the training loop + inference
    harness:

    - ``doc_type`` head: the 5 corpus classes + ``unknown`` (inference-only,
      zero training rows — OOD/abstention bucket).
    - one subclass head per doc_type, label vocabulary = the class's
      ``SUBCLASS_BY_CLASS`` enum.

    Weights are inverse-frequency over the TRAIN split only (computed by
    ``mailroom_ml.dataset.class_weights`` — imported lazily to keep the
    dataset module's import graph acyclic).
    """
    from mailroom_ml.dataset import class_weights

    heads: dict[str, dict[str, Any]] = {}
    doc_labels = list(DOC_TYPES) + ["unknown"]  # unknown = inference-only
    heads["doc_type"] = {
        "labels": doc_labels,
        "id2label": {i: k for i, k in enumerate(doc_labels)},
        "label2id": {k: i for i, k in enumerate(doc_labels)},
        "weights": class_weights(docs_df, "doc_type"),
        "note": "unknown is inference-time only (zero training rows)",
    }
    for cls in DOC_TYPES:
        labels = list(SUBCLASS_BY_CLASS[cls])
        heads[cls] = {
            "labels": labels,
            "id2label": {i: k for i, k in enumerate(labels)},
            "label2id": {k: i for i, k in enumerate(labels)},
            "weights": class_weights(docs_df[docs_df["doc_type"] == cls], "subclass"),
            "note": "fires only when doc_type predicts this class",
        }
    return heads
