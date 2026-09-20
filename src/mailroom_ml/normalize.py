"""Intake clerk — vendored ``deterministic_normalize`` (DMR-066 pattern).

Vendored byte-faithfully from the Digital Mailroom's
``llm_dojo_scoring.intake`` (the SAME clerk ``llm-mailroom.agents.intake``
runs as ``apply_intake`` before the sorter/classifier).  This standalone repo
never imports the dojo package (mirrors ``mailroom_ml/labels.py``).

WHY THIS EXISTS (2026-09-20 pre-flight audit): the published training set
consumed the corpus ``doc_text`` RAW, while the pipeline feeds the classifier
``deterministic_normalize(doc_text)``.  Training on raw text (BOM, CRLF,
3+ blank runs, C0 controls, hyphen wraps) is a train/inference skew — the
model sees inputs the pipeline will never produce.  The build now runs this
clerk so training input is byte-representative of inference input.

Keep in sync with ``llm_dojo_scoring/intake.py`` and
``The-Mailroom/mailroom_ui/intake_normalize.py`` (three copies, one behavior).
"""
from __future__ import annotations

import re
import unicodedata

__all__ = ["deterministic_normalize", "INTAKE_PREP_STEPS"]

#: Ordered prep steps the clerk runs (mirrors the dojo constant).
INTAKE_PREP_STEPS: tuple[str, ...] = (
    "nfc_normalize",
    "newline_unify",
    "nbsp_to_space",
    "strip_zero_width",
    "control_chars",
    "hyphen_unwrap",
    "collapse_blank_runs",
    "collapse_horizontal_space",
    "trim_edges",
)

_ZW = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)
_MULTI_BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_MULTI_SPACE = re.compile(r"[^\S\n]{2,}")
_HYPHEN_WRAP = re.compile(r"(?<=[A-Za-z])-\n(?=[A-Za-z])")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def deterministic_normalize(text: str) -> tuple[str, dict]:
    """Whitespace / hyphen / NBSP clerk. Returns ``(cleaned, stats)``.

    Stats keys match mailroom: ``raw_chars``, ``cleaned_chars``,
    ``collapsed_blank_runs``, ``hyphen_unwraps``, ``changed``.
    """
    raw_chars = len(text or "")
    if not text:
        return "", {
            "raw_chars": 0,
            "cleaned_chars": 0,
            "collapsed_blank_runs": 0,
            "hyphen_unwraps": 0,
            "changed": False,
        }
    original = text
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2028", "\n").replace("\u2029", "\n")
    text = text.translate(_ZW)
    text = _CTRL.sub(" ", text)
    hyphen_unwraps = len(_HYPHEN_WRAP.findall(text))
    text = _HYPHEN_WRAP.sub("", text)
    collapsed_blank = len(_MULTI_BLANK.findall(text))
    text = _MULTI_BLANK.sub("\n\n", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.rstrip()
        if stripped.lstrip().startswith("|") and stripped.rstrip().endswith("|"):
            lines.append(stripped)
        else:
            lines.append(_MULTI_SPACE.sub(" ", stripped).strip() if stripped else "")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    cleaned = "\n".join(lines)
    return cleaned, {
        "raw_chars": raw_chars,
        "cleaned_chars": len(cleaned),
        "collapsed_blank_runs": collapsed_blank,
        "hyphen_unwraps": hyphen_unwraps,
        "changed": cleaned != original,
    }