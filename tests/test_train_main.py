"""End-to-end ``main()`` seams of the trainer, hermetic and CPU-only.

Drives ``training.train.train_modernbert.main`` with a tiny stand-in backbone /
tokenizer (``transformers.AutoModel`` / ``AutoTokenizer`` patched) and an
in-memory dataset: no network, no model download, a few seconds.  Pins the
orchestration fixes that unit tests cannot see — selected-epoch promotion
BEFORE ``--eval-test``, promotion never clobbering logs/resume state, the
ungated-push refusal, the nothing-trained resume flag and eval-mode test
scoring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("scipy.optimize")
pytest.importorskip("pandas")

import torch.nn as nn  # noqa: E402
import transformers  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from training.train import train_modernbert as tm  # noqa: E402

pytestmark = pytest.mark.train

MAX_LEN = 8


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.emb = nn.Embedding(64, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(Path(path) / "model.safetensors"))


class _FakeAutoModel:
    @staticmethod
    def from_pretrained(src, **_kw):
        model = _FakeBackbone()
        weights = Path(str(src)) / "model.safetensors"
        if weights.is_file():
            model.load_state_dict(load_file(str(weights)))
        return model


class _FakeTokenizer:
    def __call__(self, texts, *, padding=None, truncation=None, max_length=8,
                 return_tensors=None):
        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:max_length]):
                ids[i, j] = 1 + ord(ch) % 60
        return {"input_ids": ids, "attention_mask": (ids > 0).long()}

    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(*_a, **_kw):
        return _FakeTokenizer()


def _head(labels, weights, **extra):
    cfg = {"labels": labels,
           "label2id": {lab: i for i, lab in enumerate(labels)},
           "id2label": {str(i): lab for i, lab in enumerate(labels)},
           "weights": weights, "note": "", **extra}
    return cfg


_MAPS = {
    "doc_type": _head(["contract", "insurance_claim", "unknown"],
                      {"contract": 1.0, "insurance_claim": 1.0},
                      inference_only=["unknown"], routing_only=["unknown"]),
    "contract": _head(["service", "license"],
                      {"service": 1.0, "license": 1.0},
                      inference_only=[], routing_only=[]),
    "insurance_claim": _head(["auto", "home"], {"auto": 1.0, "home": 1.0},
                             inference_only=[], routing_only=[]),
}


def _windows(n_per: int, tag: str) -> list[dict]:
    rows = []
    for i in range(n_per):
        rows.append({"text": f"contract {i} service lease",
                     "doc_type": "contract",
                     "subclass": "service" if i % 2 == 0 else "license",
                     "filename": f"{tag}-c{i}.txt"})
        rows.append({"text": f"claim {i} policy damage",
                     "doc_type": "insurance_claim",
                     "subclass": "auto" if i % 2 == 0 else "home",
                     "filename": f"{tag}-i{i}.txt"})
    return rows


def _docs() -> list[dict]:
    return [{"title": r["filename"], "doc_text": r["text"],
             "doc_type": r["doc_type"], "subclass": r["subclass"],
             "filename": r["filename"]} for r in _windows(2, "test")]


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "labels.json").write_text(json.dumps(_MAPS))
    monkeypatch.setattr(transformers, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)
    data = {"train": _windows(4, "tr"), "validation": _windows(2, "va"),
            "test": _docs()}
    monkeypatch.setattr(tm, "load_dataset",
                        lambda _d, split: [dict(r) for r in data[split]])
    def _window(title, text, **_kw):
        state["in_test"] = True  # only the held-out loop windows documents
        events.append(("window", dict(_kw)))
        return [text]

    monkeypatch.setattr(tm, "window_document", _window)
    monkeypatch.delenv("MAILROOM_ML_CHECKPOINT_VOLUME", raising=False)
    events: list[tuple] = []
    pushes: list[tuple] = []
    state = {"in_test": False}
    orig_load = tm._load_epoch_weights

    def _spy_load(*a, **kw):
        events.append(("load_selected",))
        return orig_load(*a, **kw)

    monkeypatch.setattr(tm, "_load_epoch_weights", _spy_load)
    monkeypatch.setattr(
        tm, "_push_to_hub",
        lambda output, repo, **kw: pushes.append((repo, kw)))
    orig_fwd = tm.HierarchicalClassifier.forward

    def _spy_fwd(self, input_ids, attention_mask):
        if state["in_test"]:
            events.append(("test_forward", self.training))
        return orig_fwd(self, input_ids, attention_mask)

    monkeypatch.setattr(tm.HierarchicalClassifier, "forward", _spy_fwd)

    def run(output: Path, *extra: str) -> int:
        argv = ["train_modernbert.py", "--data", str(stage),
                "--output", str(output), "--model", "fake",
                "--epochs", "2", "--batch-size", "2", "--grad-accum", "1",
                "--max-length", str(MAX_LEN), "--log-every", "1",
                "--prefetch-batches", "0", "--lr", "1e-3",
                "--subclass-min-train-rows", "0", "--seed", "3", *extra]
        monkeypatch.setattr(sys, "argv", argv)
        return tm.main()

    def select(epoch: int):
        """Pin the selection decision (the rule itself is unit-tested)."""
        def fake(events_, subclass_heads, *, select_on_subclass, prior=None):
            if epoch == 0:
                return dict(tm._NO_SELECTION), float("-inf")
            return ({"epoch": epoch, "macro_f1": 0.9, "ece": 0.01,
                     "subclass_objective": 0.5}, 0.9)
        monkeypatch.setattr(tm, "_select_epoch", fake)

    return SimpleNamespace(run=run, select=select, events=events,
                           pushes=pushes, tmp=tmp_path)


def test_eval_test_scores_the_selected_epoch_weights(harness):
    """--eval-test used to score the in-memory FINAL epoch while the promoted
    weights were the selected epoch's.  The selected archive is now loaded
    into the live model before the test loop, and promotion leaves the logs
    and resume state of the run alone."""
    harness.select(1)
    out = harness.tmp / "latest"
    assert harness.run(out, "--eval-test") == 0

    kinds = [e[0] for e in harness.events]
    assert "load_selected" in kinds and "test_forward" in kinds
    assert kinds.index("load_selected") < kinds.index("test_forward")
    # the trainer's own test windowing honours --input-construction (and
    # hands the windower the filename the v2 prefix needs)
    windows = [e[1] for e in harness.events if e[0] == "window"]
    assert windows and all(w["version"] == "v1" and w["filename"]
                           for w in windows)
    # test scoring ran in eval mode (dropout off)
    assert all(e[1] is False for e in harness.events if e[0] == "test_forward")

    run_id = json.loads((out / "summary.json").read_text())["run_id"]
    archive = harness.tmp / "runs" / f"{run_id}-e1"
    assert archive.is_dir()
    # shipped weights == the selected epoch's archive, not the final epoch's
    assert (out / "heads.pt").read_bytes() == (archive / "heads.pt").read_bytes()
    final = harness.tmp / "runs" / f"{run_id}-e2"
    assert (out / "heads.pt").read_bytes() != (final / "heads.pt").read_bytes()

    summary = json.loads((out / "summary.json").read_text())
    assert summary["checkpoint_selection"]["epoch"] == 1
    assert summary["test_metrics"]["n_docs"] == 4
    assert summary["nothing_trained"] is False
    # promotion must not truncate the run-wide logs to the epoch-1 snapshot
    rows = [json.loads(ln) for ln in
            (out / "epoch_metrics.jsonl").read_text().splitlines() if ln]
    assert [r["epoch"] for r in rows if r.get("event") == "epoch"] == [1, 2]
    assert (out / "train_steps.jsonl").stat().st_size > (
        archive / "train_steps.jsonl").stat().st_size


def test_missing_final_archive_leaves_output_not_resumable(harness,
                                                           monkeypatch):
    """Promotion repoints the resume manifest at the final epoch's archive;
    with that archive missing, the final-epoch optimizer/scheduler/resume
    state left in `output` must not be paired with the promoted weights."""
    orig_copytree = tm.shutil.copytree

    def _copytree(src, dst, *a, **kw):
        if str(dst).endswith("-e2"):
            return dst  # simulate a failed/deleted final-epoch archive
        return orig_copytree(src, dst, *a, **kw)

    monkeypatch.setattr(tm.shutil, "copytree", _copytree)
    harness.select(1)
    out = harness.tmp / "latest"
    assert harness.run(out) == 0
    run_id = json.loads((out / "summary.json").read_text())["run_id"]
    assert not (harness.tmp / "runs" / f"{run_id}-e2").exists()
    assert (out / "heads.pt").read_bytes() == (
        harness.tmp / "runs" / f"{run_id}-e1" / "heads.pt").read_bytes()
    for name in ("optimizer.pt", "scheduler.pt", "resume.json",
                 "resume_manifest.json"):
        assert not (out / name).exists(), name


def test_push_refused_when_no_epoch_met_the_gate(harness):
    harness.select(0)
    out = harness.tmp / "latest"
    assert harness.run(out, "--push-to-hub", "org/model") == 3
    assert harness.pushes == []
    # the explicit override publishes the ungated final epoch
    out2 = harness.tmp / "other" / "latest"
    assert harness.run(out2, "--push-to-hub", "org/model",
                       "--push-ungated") == 0
    assert [p[0] for p in harness.pushes] == ["org/model"]
    assert harness.pushes[0][1]["selected_epoch"] == 0


def test_push_proceeds_when_gate_met(harness):
    harness.select(2)
    out = harness.tmp / "latest"
    assert harness.run(out, "--push-to-hub", "org/model") == 0
    assert [p[0] for p in harness.pushes] == ["org/model"]
    assert harness.pushes[0][1]["selected_epoch"] == 2


def test_resume_past_the_last_epoch_trains_nothing_and_evals_in_eval_mode(
        harness):
    """A --resume already at --epochs trains nothing: summary flags it (the
    deploy layer skips export/promote/push), and the eval-only path must put
    the freshly built heads in eval mode before the test loop."""
    harness.select(2)
    first = harness.tmp / "first"
    assert harness.run(first) == 0
    harness.events.clear()
    second = harness.tmp / "second"
    assert harness.run(second, "--resume", str(first), "--eval-test") == 0
    summary = json.loads((second / "summary.json").read_text())
    assert summary["nothing_trained"] is True
    assert summary["epochs_trained_this_run"] == 0
    fwd = [e for e in harness.events if e[0] == "test_forward"]
    assert fwd and all(e[1] is False for e in fwd)
