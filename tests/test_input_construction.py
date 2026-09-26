"""Input-construction v1/v2 tests (#29). No torch."""
from __future__ import annotations

import pytest

from mailroom_ml.windows import decorate_window, window_document


def test_decorate_window_v1_matches_published_format():
    assert decorate_window("Title", "body") == "Title\n\nbody"
    assert decorate_window("", "body") == "body"
    assert decorate_window("Title", "body", version="v1") == "Title\n\nbody"


def test_decorate_window_v2_tagged_prefix():
    text = decorate_window(
        "Merger", "WHEREAS the parties", version="v2",
        filename="deal.pdf", window_index=2)
    assert text.startswith("[FILE_NAME] deal.pdf\n")
    assert "[TITLE] Merger\n" in text
    assert "[WINDOW_INDEX] 2\n\n" in text
    assert text.endswith("WHEREAS the parties")
    # v1 never emits the tags
    v1 = decorate_window("Merger", "WHEREAS the parties", filename="deal.pdf")
    assert "[FILE_NAME]" not in v1


def test_decorate_window_unknown_version_is_loud():
    with pytest.raises(ValueError, match="unknown input construction"):
        decorate_window("t", "b", version="v9")


def test_window_document_unknown_version_is_loud():
    with pytest.raises(ValueError, match="unknown input construction"):
        window_document("t", "b", version="v9")
