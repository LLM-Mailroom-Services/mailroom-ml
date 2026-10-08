"""score_blind_pool (#26): unscored rows are counted, never passed off as
scored; an all-unscored run fails.  Hermetic: the bundle is stubbed."""
from __future__ import annotations

import pandas as pd

import training.eval.score_blind_pool as sbp


def _run(tmp_path, monkeypatch, classify):
    pool = tmp_path / "pool.parquet"
    pd.DataFrame([{"filename": "a", "title": "t", "doc_text": "x"},
                  {"filename": "b", "title": "t", "doc_text": "y"}]
                 ).to_parquet(pool)
    monkeypatch.setattr(sbp, "load_bundle", lambda _ckpt: object())
    monkeypatch.setattr(sbp, "classify_document", classify)
    out = tmp_path / "out.parquet"
    rc = sbp.main(["--pool", str(pool), "--out", str(out)])
    return rc, (pd.read_parquet(out) if out.exists() else None)


def test_oversize_rows_are_not_marked_scored(tmp_path, monkeypatch):
    def classify(_b, _t, text, **_k):
        if text == "x":
            return {"status": "ok", "reason": "oversize_chars"}
        return {"status": "ok", "calibrated_confidence": 0.9,
                "subclass_confidence": 0.8, "agreement": 1.0}

    rc, df = _run(tmp_path, monkeypatch, classify)
    assert rc == 0
    by = df.set_index("filename")
    assert not by.loc["a", "scored"]
    assert by.loc["a", "score_reason"] == "oversize_chars"
    assert by.loc["b", "scored"]


def test_all_rows_failing_exits_nonzero(tmp_path, monkeypatch):
    def classify(*_a, **_k):
        raise RuntimeError("broken bundle")

    rc, df = _run(tmp_path, monkeypatch, classify)
    assert rc == 3
    assert set(df["score_reason"]) == {"error:RuntimeError"}
