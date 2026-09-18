"""Training-layer tests: CLI surface, metric math, conditional loss, stage loader.

Hermetic by design: no network, no GPU, no model downloads, no Hub writes.
Marked ``train``; the module-level ``importorskip`` keeps the core suite
green without the heavy extras.  The trainer imports ``transformers``/
``datasets``/``huggingface_hub`` lazily inside its functions, so exercising
the CLI + math never touches the Hub.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mailroom_ml.config import (  # noqa: E402
    MAX_TOKENS,
    MODEL_ID,
    RUNS_DIR,
    TRAINING_DATA_REPO,
)
from training.train_modernbert import (  # noqa: E402
    build_parser,
    ece,
    fit_temperature,
    head_loss,
    load_dataset,
    macro_f1,
)

pytestmark = pytest.mark.train


class _StubHeads(nn.Module):
    """Deterministic stand-in for HierarchicalClassifier: fixed per-head logits.

    ``forward`` returns the configured logit map verbatim — the pure-logic
    seam for testing the conditional loss structure without a backbone.
    """

    def __init__(self, logits_by_head: dict[str, torch.Tensor]):
        super().__init__()
        self.logits_by_head = {k: v.clone() for k, v in logits_by_head.items()}

    def forward(self, input_ids, attention_mask):
        return {k: v for k, v in self.logits_by_head.items()}


# -- CLI surface ---------------------------------------------------------------

def test_cli_accepts_documented_and_extra_flags():
    ap = build_parser()
    ns = ap.parse_args([
        "--data", "org/repo", "--output", "/tmp/x", "--epochs", "3",
        "--batch-size", "8", "--grad-accum", "2", "--lr", "1e-5",
        "--seed", "7", "--push-to-hub", "org/model", "--eval-test",
        "--max-length", "512", "--limit", "64", "--warmup-frac", "0.1",
        "--model", "answerdotai/ModernBERT-base",
    ])
    assert ns.data == "org/repo"
    assert ns.output == Path("/tmp/x")
    assert ns.epochs == 3
    assert ns.batch_size == 8
    assert ns.grad_accum == 2
    assert ns.lr == 1e-5
    assert ns.seed == 7
    assert ns.push_to_hub == "org/model"
    assert ns.eval_test is True
    assert ns.max_length == 512
    assert ns.limit == 64
    assert ns.warmup_frac == 0.1
    assert ns.model == "answerdotai/ModernBERT-base"


def test_cli_defaults_match_deploy_contract():
    """The deploy-pinned surface parses with the documented defaults."""
    ns = build_parser().parse_args([])
    assert ns.data == TRAINING_DATA_REPO
    assert ns.output == RUNS_DIR / "latest"
    assert ns.seed == 42
    assert ns.lr == 2e-5
    assert ns.epochs == 5
    assert ns.batch_size == 16
    assert ns.grad_accum == 2
    assert ns.push_to_hub == ""
    assert ns.eval_test is False
    # allowed extras keep predecessor parity without breaking the pinned set
    assert ns.max_length == MAX_TOKENS
    assert ns.limit == 0
    assert ns.warmup_frac == 0.06
    assert ns.model == MODEL_ID


# -- metric math --------------------------------------------------------------

def test_fit_temperature_recovers_known_scale():
    """Logits from softmax(z / T_true) fit back to T_true within tolerance."""
    pytest.importorskip("scipy.optimize")  # presence guard (lazy import)
    rng = np.random.RandomState(0)
    z = rng.normal(size=(4000, 5))
    t_true = 1.7
    p = np.exp(z / t_true)
    p /= p.sum(axis=1, keepdims=True)
    labels = torch.tensor([rng.choice(5, p=p[i]) for i in range(len(p))])
    t_hat = fit_temperature(torch.tensor(z), labels)
    assert abs(t_hat - t_true) < 0.25, t_hat


def test_ece_hand_computed():
    # perfectly confident and correct -> ECE ~0
    lg = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    assert ece(lg, torch.tensor([0, 1])) == pytest.approx(0.0, abs=1e-4)
    # confident + half wrong -> ~0.5 (all mass in the top bin)
    lg2 = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    assert ece(lg2, torch.tensor([0, 1])) == pytest.approx(0.5, abs=1e-3)
    # mid-bin case: conf = sigmoid(1), acc 3/4 -> ECE = |0.75 - sigmoid(1)|
    lg3 = torch.tensor([[1.0, 0.0]] * 4)
    assert ece(lg3, torch.tensor([0, 0, 0, 1])) == pytest.approx(
        abs(0.75 - 1 / (1 + np.exp(-1))))


def test_macro_f1_hand_computed():
    lg = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    assert macro_f1(lg, torch.tensor([0, 1])) == 1.0
    # both rows predict class 0; class 1 never recovered -> macro 1/3
    lg2 = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    assert macro_f1(lg2, torch.tensor([0, 1])) == pytest.approx(1 / 3)


# -- conditional loss ---------------------------------------------------------

def test_head_loss_conditional_subclass_structure():
    """Subclass CE fires only on each class's own rows — 0 contribution elsewhere."""
    device = torch.device("cpu")
    heads = {
        "doc_type": {"label2id": {"contract": 0, "insurance_claim": 1},
                     "weight": torch.ones(2)},
        "contract": {"label2id": {"service": 0, "license": 1},
                     "weight": torch.ones(2)},
    }
    batch = {
        "input_ids": torch.zeros(4, 8, dtype=torch.long),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
        "doc_type": torch.tensor([0, 0, 1, 1]),
        "subclass": torch.tensor([0, 1, 0, 0]),
        "filename": ["a.txt", "b.txt", "c.txt", "d.txt"],
    }
    lgs = {
        "doc_type": torch.tensor([[5.0, 0.0], [5.0, 0.0],
                                  [5.0, 0.0], [5.0, 0.0]]),
        "contract": torch.tensor([[5.0, 0.0], [0.0, 5.0],
                                  [0.0, 0.0], [0.0, 0.0]]),
    }
    loss, logits = head_loss(_StubHeads(lgs), batch, heads, device)
    sel = batch["doc_type"] == 0
    expected = (
        F.cross_entropy(logits["doc_type"], batch["doc_type"],
                        weight=heads["doc_type"]["weight"])
        + F.cross_entropy(logits["contract"][sel], batch["subclass"][sel],
                          weight=heads["contract"]["weight"])
    )
    assert loss.item() == pytest.approx(expected.item())
    # garbage logits on non-matching rows must contribute exactly nothing
    garbage = dict(lgs)
    garbage["contract"] = lgs["contract"].clone()
    garbage["contract"][2:] = torch.tensor([[100.0, 0.0], [-100.0, 0.0]])
    loss2, _ = head_loss(_StubHeads(garbage), batch, heads, device)
    assert loss2.item() == pytest.approx(loss.item())


# -- local stage loader -------------------------------------------------------

def _write_parquet(root, cfg: str, split: str, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = root / "parquet" / cfg / split
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),
                   d / f"{split}-00000-of-00001.parquet")


def test_local_stage_loader_reads_committed_layout(tmp_path):
    """A stage dir in the committed layout resolves train/val/test rows."""
    stage = tmp_path / "stage"
    _write_parquet(stage, "windows", "train", [
        {"filename": "a.txt", "window_index": 0, "n_windows": 1,
         "text": "a window", "doc_type": "contract", "subclass": "service",
         "split": "train", "window_tokens": 5},
        {"filename": "b.txt", "window_index": 0, "n_windows": 1,
         "text": "b window", "doc_type": "insurance_claim",
         "subclass": "auto", "split": "train", "window_tokens": 5},
    ])
    _write_parquet(stage, "windows", "validation", [
        {"filename": "c.txt", "window_index": 0, "n_windows": 1,
         "text": "c window", "doc_type": "contract", "subclass": "license",
         "split": "validation", "window_tokens": 5},
    ])
    _write_parquet(stage, "documents", "test", [
        {"filename": "z.txt", "title": "Z", "doc_text": "body",
         "doc_type": "insurance_claim", "subclass": "carrier",
         "split": "test"},
    ])
    (stage / "labels.json").write_text(json.dumps({
        "doc_type": {"labels": ["contract", "insurance_claim", "unknown"]},
        "contract": {"labels": ["service", "license"]},
        "insurance_claim": {"labels": ["auto", "carrier"]},
    }, sort_keys=True))

    train = load_dataset(str(stage), "train")
    assert [r["filename"] for r in train] == ["a.txt", "b.txt"]
    assert {"text", "doc_type", "subclass", "filename"} <= set(train[0])
    assert train[1]["doc_type"] == "insurance_claim"

    val = load_dataset(str(stage), "validation")
    assert [r["filename"] for r in val] == ["c.txt"]

    test = load_dataset(str(stage), "test")
    assert [r["filename"] for r in test] == ["z.txt"]
    assert {"title", "doc_text", "doc_type", "subclass"} <= set(test[0])


def test_local_stage_loader_missing_split(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(str(tmp_path), "train")

