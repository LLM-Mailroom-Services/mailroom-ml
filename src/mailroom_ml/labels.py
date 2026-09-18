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

HEAD VOCABS ARE OBSERVED, NOT ENUMERATED (mailroom-issues #66/#67/#68/#75,
see ``observed_label_surfaces``): per-class training heads are derived from
the pinned GT at build time (train+validation rows only — the held-out test
split never shapes a head).  ``SUBCLASS_BY_CLASS`` remains the CANONICAL
normalization reference (what ``normalize_subclass`` resolves through), not
the head vocabulary; a zero-row enum label (``certificate_of_formation``,
``voicemail``) is not a trainable class.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from mailroom_ml.config import (
    CANONICAL_CORRESPONDENCE_SUBCLASSES,
    DOC_TYPES,
    FINETUNE_REVISION,
    INSURANCE_SUBCLASSES,
    OBSERVED_CORPORATE_RECORD_SUBCLASSES,
)

__all__ = [
    "DOC_TYPES",
    "CONTRACT_SUBTYPE_KEYS",
    "SUBTYPE_ALIASES",
    "MAUD_CONSIDERATION_TYPES",
    "SUBCLASS_BY_CLASS",
    "CONTRACT_SUBTYPE_LABELS",
    "normalize_subclass",
    "observed_label_surfaces",
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

    Note: ``other`` is emitted for unresolvable values even in classes whose
    OBSERVED head has no ``other`` token (e.g. correspondence) — such a row
    is a guard-failure candidate per #75 (surface-drift parity test), never
    an invented label; the canonical normalization is unchanged.
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


def observed_label_surfaces(docs: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Tracker-sanctioned per-class head vocabs, derived from the pinned GT.

    Mechanism: per doc_type, the subclass keys observed over rows where
    ``split != "test"`` — the held-out test split must never shape a head
    vocab (mailroom-issues #66/#67/#75).  With the sanctioned exceptions
    where the tracker keeps the FULL canonical enum as the head
    (config.py, corpus revision ``FINETUNE_REVISION``):

    - ``contract`` — CUAD canon ``CONTRACT_SUBTYPE_KEYS + ("other",)``:
      ``other`` is the CUAD fallback token for unseen families at
      inference time, independent of whether the corpus observes it;
    - ``merger_agreement`` — MAUD canon ``MAUD_CONSIDERATION_TYPES`` (5);
    - ``corporate_record`` / ``correspondence`` / ``insurance_claim`` — the
      OBSERVED set only: zero-row enum labels (``certificate_of_formation``,
      ``voicemail``) and ``other``-by-fiat are NOT trainable classes.  An
      unresolved row in a class whose observed head has no ``other`` token
      is a guard failure (#75), never an invented label.

    Determinism/order: observed keys are ordered by their sanctioned
    config tuple when one exists (the tracker-reviewed corpus order, which
    pins id2label indices), with any drift keys NOT in the sanction —
    the #75 failures — appended in sorted order; contract and
    merger_agreement follow their canonical tuple order.
    """
    active = docs[docs["split"] != "test"]
    observed: dict[str, set[str]] = {}
    for cls, grp in active.groupby("doc_type", sort=True):
        observed[cls] = set(grp["subclass"])
    sanctioned_order = {
        "correspondence": CANONICAL_CORRESPONDENCE_SUBCLASSES,
        "corporate_record": OBSERVED_CORPORATE_RECORD_SUBCLASSES,
        "insurance_claim": INSURANCE_SUBCLASSES,
    }
    surfaces: dict[str, tuple[str, ...]] = {}
    for cls in DOC_TYPES:
        if cls == "contract":
            surfaces[cls] = CONTRACT_SUBTYPE_KEYS + ("other",)
        elif cls == "merger_agreement":
            surfaces[cls] = MAUD_CONSIDERATION_TYPES
        else:
            obs = observed.get(cls, set())
            order = sanctioned_order[cls]
            ordered = tuple(k for k in order if k in obs)
            drift = tuple(sorted(obs - set(order)))  # unsanctioned keys -> drift failure
            surfaces[cls] = ordered + drift
    return surfaces


def label_maps(docs_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-head id2label/label2id/weights for the hierarchical classifier.

    Builds the label contract consumed by the training loop + inference
    harness:

    - ``doc_type`` head: the 5 corpus classes + ``unknown`` (inference-only,
      zero training rows — OOD/abstention bucket).
    - one subclass head per class, label vocabulary = the OBSERVED surface
      from ``observed_label_surfaces`` (train+validation rows of the pinned
      GT — never the full canonical enum, never the held-out test split;
      issues #66/#67/#68/#75).

    Weights are inverse-frequency over the TRAIN split only (computed by
    ``mailroom_ml.dataset.class_weights`` — imported lazily to keep the
    dataset module's import graph acyclic).  Each head carries a ``note``
    documenting its derivation source (observed-vs-canonical) and the
    corpus revision.
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
    surfaces = observed_label_surfaces(docs_df)
    for cls in DOC_TYPES:
        labels = list(surfaces[cls])
        surface_source = (
            "canonical-enum" if cls in ("contract", "merger_agreement")
            else "observed-gt"
        )
        heads[cls] = {
            "labels": labels,
            "id2label": {i: k for i, k in enumerate(labels)},
            "label2id": {k: i for i, k in enumerate(labels)},
            "weights": class_weights(docs_df[docs_df["doc_type"] == cls], "subclass"),
            "note": (
                f"fires only when doc_type predicts this class; surface="
                f"{surface_source} (observed over split != test), "
                f"corpus_revision={FINETUNE_REVISION}"
            ),
        }
    return heads
