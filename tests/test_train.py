"""Training-layer tests: CLI surface, metric math, conditional loss, stage loader.

Hermetic by design: no network, no GPU, no model downloads, no Hub writes.
Marked ``train``; the module-level ``importorskip`` keeps the core suite
green without the heavy extras.  The trainer imports ``transformers``/
``datasets``/``huggingface_hub`` lazily inside its functions, so exercising
the CLI + math never touches the Hub.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mailroom_ml.config import (  # noqa: E402
    MAX_TOKENS,
    MODEL_ID,
    RUNS_DIR,
    TRAINING_DATA_REPO,
)
from training.train_modernbert import (  # noqa: E402
    HierarchicalClassifier,
    LossConfig,
    _apply_resume,
    _apply_subclass_support_threshold,
    _scheduler_plan,
    build_parser,
    ece,
    ece_calibrated,
    fit_temperature,
    head_loss,
    load_dataset,
    macro_f1,
    make_batches,
    save_checkpoint,
    train_epoch,
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
    assert ns.resume is None


def test_cli_accepts_resume_flag():
    ns = build_parser().parse_args(["--resume", "/checkpoints/latest"])
    assert ns.resume == Path("/checkpoints/latest")


# -- resume-from-checkpoint ---------------------------------------------------

class _FakeSave:
    """Stand-in for save_pretrained on backbone/tokenizer (hermetic)."""

    def __init__(self, name: str):
        self.name = name

    def save_pretrained(self, d: Path) -> None:
        (Path(d) / self.name).write_text("x")


def _tiny_model():
    return SimpleNamespace(
        backbone=_FakeSave("backbone.txt"),
        heads=nn.ModuleDict({"doc_type": nn.Linear(4, 2)}),
    )


def _tiny_optim_sched(model, lr: float = 1e-3):
    opt = torch.optim.AdamW(model.heads.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    return opt, sched


def test_resume_state_roundtrip(tmp_path):
    """save_checkpoint with optimizer state -> --resume restores counters,
    model weights, and the optimizer's moment estimates exactly."""
    out = tmp_path / "ckpt"
    model = _tiny_model()
    opt, sched = _tiny_optim_sched(model)
    # one real step so AdamW carries non-zero exp_avg state
    loss = model.heads["doc_type"](torch.zeros(2, 4)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    save_checkpoint(out, model, _FakeSave("tokenizer.txt"),
                    {"doc_type": {"labels": ["a", "b"]}}, {}, [], {},
                    {"run_id": "r1"}, optimizer=opt, scheduler=sched,
                    epoch=1, steps_done=10)
    assert (out / "optimizer.pt").is_file()
    assert (out / "scheduler.pt").is_file()
    assert json.loads((out / "resume.json").read_text()) == {
        "epoch": 1, "steps_done": 10, "steps_done_micro": 0, "run_id": "r1"}

    # fresh model/optimizer/scheduler — resume must restore everything
    model2 = _tiny_model()
    opt2, sched2 = _tiny_optim_sched(model2)
    state = _apply_resume(out, model2, opt2, sched2, torch.device("cpu"),
                          n_train_rows=20, batch_size=4, grad_accum=2)
    assert state["start_epoch"] == 2
    assert state["steps_done"] == 10
    # model weights restored from heads.pt
    for (n1, p1), (_, p2) in zip(
            model.heads.named_parameters(), model2.heads.named_parameters(),
            strict=True):
        assert torch.equal(p1.detach(), p2.detach()), n1
    # optimizer moments restored
    s1, s2 = opt.state_dict()["state"], opt2.state_dict()["state"]
    assert s1.keys() == s2.keys()
    for k in s1:
        assert s1[k]["step"] == s2[k]["step"]
        assert torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"])
        assert torch.equal(s1[k]["exp_avg_sq"], s2[k]["exp_avg_sq"])


def test_resume_without_optimizer_state(tmp_path):
    """Bundles from the pre-resume era (no optimizer.pt/resume.json) resume
    from summary.json counters with a reconstructed optimizer/scheduler."""
    out = tmp_path / "ckpt"
    out.mkdir()
    torch.save({"doc_type": nn.Linear(4, 2).state_dict()}, out / "heads.pt")
    (out / "summary.json").write_text(json.dumps({
        "epochs_run": 1,
        "epochs": [{"epoch": 1, "loss": 2.0, "val_loss": 1.5}],
        "checkpoint_selection": {"epoch": 1, "macro_f1": 0.8, "ece": 0.04},
    }))
    model = _tiny_model()
    opt, sched = _tiny_optim_sched(model)
    state = _apply_resume(out, model, opt, sched, torch.device("cpu"),
                          n_train_rows=20, batch_size=4, grad_accum=2)
    assert state["start_epoch"] == 2
    # 5 micro-batches/epoch, grad-accum 2 -> 3 OPTIMIZER steps/epoch (the
    # 2026-09-20 audit fixed the micro-batch/optimizer-step mismatch)
    assert state["steps_done"] == 3
    assert state["steps_done_micro"] == 5
    assert state["best_val"] == 1.5
    assert state["stale"] == 0
    assert state["selected"]["epoch"] == 1
    assert sched.last_epoch == 3             # scheduler stepped into position
    assert len(state["events"]) == 1


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
    """Subclass CE fires only on each class's own rows — 0 contribution
    elsewhere; the blended loss weights doc_type by lambda_dt."""
    device = torch.device("cpu")
    heads = {
        "doc_type": {"label2id": {"contract": 0, "insurance_claim": 1},
                     "labels": ["contract", "insurance_claim"],
                     "weights": {"contract": 1.0, "insurance_claim": 1.0}},
        "contract": {"label2id": {"service": 0, "license": 1},
                     "labels": ["service", "license"],
                     "weights": {"service": 1.0, "license": 1.0}},
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
    cfg = LossConfig(lambda_dt=0.65)
    loss, logits = head_loss(_StubHeads(lgs), batch, heads, device, cfg)
    sel = batch["doc_type"] == 0
    dt_ce = F.cross_entropy(logits["doc_type"], batch["doc_type"])
    sc_ce = F.cross_entropy(logits["contract"][sel], batch["subclass"][sel])
    expected = 0.65 * dt_ce + 0.35 * sc_ce
    assert loss.item() == pytest.approx(expected.item())
    # garbage logits on non-matching rows must contribute exactly nothing
    garbage = dict(lgs)
    garbage["contract"] = lgs["contract"].clone()
    garbage["contract"][2:] = torch.tensor([[100.0, 0.0], [-100.0, 0.0]])
    loss2, _ = head_loss(_StubHeads(garbage), batch, heads, device, cfg)
    assert loss2.item() == pytest.approx(loss.item())


def test_head_loss_lambda_extremes():
    """lambda_dt=1.0 -> doc_type only; 0.0 -> subclass mean only."""
    device = torch.device("cpu")
    heads = {
        "doc_type": {"label2id": {"contract": 0, "insurance_claim": 1},
                     "labels": ["contract", "insurance_claim"],
                     "weights": {"contract": 1.0, "insurance_claim": 1.0}},
        "contract": {"label2id": {"service": 0, "license": 1},
                     "labels": ["service", "license"],
                     "weights": {"service": 1.0, "license": 1.0}},
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
    only_dt, _ = head_loss(_StubHeads(lgs), batch, heads, device,
                           LossConfig(lambda_dt=1.0))
    only_sc, _ = head_loss(_StubHeads(lgs), batch, heads, device,
                           LossConfig(lambda_dt=0.0))
    dt_ce = F.cross_entropy(lgs["doc_type"], batch["doc_type"])
    sc_ce = F.cross_entropy(lgs["contract"][batch["doc_type"] == 0],
                            batch["subclass"][batch["doc_type"] == 0])
    assert only_dt.item() == pytest.approx(dt_ce.item())
    assert only_sc.item() == pytest.approx(sc_ce.item())


def test_loss_config_weight_modes():
    """sqrt-inverse tames rare-class amplification; cap clamps; none = 1.0."""
    weights = {"common": 0.5, "rare": 12.0}
    labels = ["common", "rare"]
    inv = LossConfig(weight_mode="inverse", weight_cap=100.0)
    assert inv.transform_weights(weights, labels).tolist() == [0.5, 12.0]
    sqrt = LossConfig(weight_mode="sqrt-inverse", weight_cap=100.0)
    got = sqrt.transform_weights(weights, labels).tolist()
    assert got[1] == pytest.approx(math.sqrt(12.0))
    capped = LossConfig(weight_mode="inverse", weight_cap=5.0)
    assert capped.transform_weights(weights, labels).tolist() == [0.5, 5.0]
    none = LossConfig(weight_mode="none")
    assert none.transform_weights(weights, labels).tolist() == [1.0, 1.0]


def test_scheduler_plan_optimizer_step_units():
    """total_steps/warmup are in OPTIMIZER steps — the old micro-batch units
    made the 6% warmup consume ~48% of the run (grad-accum 8)."""
    total, warmup = _scheduler_plan(4573, 4, 8, 2, 0.06)
    assert total == 286          # ceil(1143/8)=143 opt-steps/epoch x 2
    assert warmup == 17          # 6% of 286, not 6% of 2286 micro-batches
    total2, warmup2 = _scheduler_plan(100, 16, 2, 3, 0.1)
    assert total2 == 12          # ceil(7/2)=4 opt-steps/epoch x 3
    assert warmup2 == 1


def test_train_epoch_flushes_stale_gradients():
    """10 micro-batches at grad-accum 4 -> 2 full steps + 1 flush step;
    the trailing accumulation is optimized, not leaked into the next epoch."""
    device = torch.device("cpu")

    class _Tiny2Arg(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, input_ids, attention_mask):
            z = self.lin(input_ids.float())
            return {"doc_type": z, "a": z[:, :1]}

    model = _Tiny2Arg()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    steps = []

    class _SpyOpt:
        def __init__(self, inner):
            self.inner = inner

        def step(self):
            steps.append(1)
            self.inner.step()

        def zero_grad(self):
            self.inner.zero_grad()

        def __getattr__(self, name):
            return getattr(self.inner, name)

    spy = _SpyOpt(opt)
    heads = {
        "doc_type": {"label2id": {"a": 0, "b": 1}, "labels": ["a", "b"],
                     "weights": {"a": 1.0, "b": 1.0}},
        "a": {"label2id": {"a": 0}, "labels": ["a"],
              "weights": {"a": 1.0}},
    }
    rows = [{"input_ids": torch.zeros(4, dtype=torch.long),
             "attention_mask": torch.ones(4, dtype=torch.long),
             "doc_type": "a", "subclass": "a", "filename": f"f{i}.txt"}
            for i in range(10)]
    batches = list(make_batches(rows, 1, False, heads, device))
    mean, endpoint, n_micro, n_opt = train_epoch(
        model, iter(batches), spy, sched, heads, device, 4, loss_cfg=LossConfig())
    assert n_micro == 10
    assert n_opt == 3            # 2 full + 1 flush
    assert len(steps) == 3
    assert 0.0 < endpoint < mean  # endpoint excludes the early high-loss steps


def test_mlp_heads_forward_shape():
    """--mlp-heads builds the ModernBERT recipe heads (SiLU MLP + dropout)."""
    class _FakeBackbone(nn.Module):
        config = type("C", (), {"hidden_size": 8})()

        def forward(self, input_ids, attention_mask):
            return type("O", (), {"last_hidden_state": input_ids.unsqueeze(1)})()

    model = HierarchicalClassifier(_FakeBackbone(), {"doc_type": 3, "x": 2},
                                   head_kind="mlp", head_dropout=0.1)
    assert isinstance(model.heads["doc_type"], nn.Sequential)
    out = model(torch.zeros(2, 8, dtype=torch.long),
                torch.ones(2, 8, dtype=torch.long))
    assert out["doc_type"].shape == (2, 3)
    assert out["x"].shape == (2, 2)
    # linear heads unchanged by default
    lin = HierarchicalClassifier(_FakeBackbone(), {"doc_type": 3})
    assert isinstance(lin.heads["doc_type"], nn.Linear)


def test_macro_f1_observed_only():
    """observed_only excludes zero-row classes (e.g. inference-only
    `unknown`) from the macro average — the old full-vocab average deflated
    every epoch's score."""
    lg = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    lab = torch.tensor([0, 1])  # class 2 has zero label rows
    full = macro_f1(lg, lab)
    obs = macro_f1(lg, lab, observed_only=True)
    assert full == pytest.approx(2 / 3)   # 3 classes, class 2 F1 = 0
    assert obs == 1.0                     # 2 observed classes, both perfect


def test_ece_calibrated_matches_scaled_logits():
    """ece_calibrated(logits, T) == ece(logits / T) — the number the
    selection gate compares against."""
    rng = np.random.RandomState(3)
    lg = torch.tensor(rng.normal(size=(500, 4)))
    lab = torch.tensor(rng.randint(0, 4, size=500))
    t = 2.5
    assert ece_calibrated(lg, lab, t) == pytest.approx(
        ece(lg / t, lab), abs=1e-6)


def test_subclass_support_threshold_remaps_and_drops():
    """Classes below the floor remap to `other` when present; otherwise the
    rows leave train AND val and the head vocabulary shrinks."""
    train = [
        {"doc_type": "contract", "subclass": "service", "filename": "a"},
        {"doc_type": "contract", "subclass": "service", "filename": "b"},
        {"doc_type": "contract", "subclass": "service", "filename": "c"},
        {"doc_type": "contract", "subclass": "service", "filename": "d"},
        {"doc_type": "contract", "subclass": "license", "filename": "e"},
        {"doc_type": "contract", "subclass": "license", "filename": "f"},
        {"doc_type": "contract", "subclass": "license", "filename": "g"},
        {"doc_type": "contract", "subclass": "license", "filename": "h"},
        {"doc_type": "contract", "subclass": "rare", "filename": "i"},
        {"doc_type": "insurance_claim", "subclass": "auto", "filename": "j"},
        {"doc_type": "insurance_claim", "subclass": "auto", "filename": "k"},
        {"doc_type": "insurance_claim", "subclass": "auto", "filename": "l"},
        {"doc_type": "insurance_claim", "subclass": "auto", "filename": "m"},
        {"doc_type": "insurance_claim", "subclass": "tiny", "filename": "n"},
    ]
    val = [
        {"doc_type": "contract", "subclass": "rare", "filename": "v1"},
        {"doc_type": "insurance_claim", "subclass": "tiny", "filename": "v2"},
    ]
    maps = {
        "doc_type": {"labels": ["contract", "insurance_claim"]},
        "contract": {"labels": ["service", "license", "rare", "other"],
                     "label2id": {"service": 0, "license": 1, "rare": 2,
                                  "other": 3},
                     "id2label": {"0": "service", "1": "license",
                                  "2": "rare", "3": "other"},
                     "weights": {"service": 1.0, "license": 1.0,
                                 "rare": 1.0, "other": 1.0}},
        "insurance_claim": {"labels": ["auto", "tiny"],
                            "label2id": {"auto": 0, "tiny": 1},
                            "id2label": {"0": "auto", "1": "tiny"},
                            "weights": {"auto": 1.0, "tiny": 1.0}},
    }
    tr, va, new_maps, info = _apply_subclass_support_threshold(
        train, val, maps, min_rows=4)
    # contract.rare (1 row) remapped to `other`; insurance_claim.tiny (1 row)
    # has no `other` -> dropped entirely (train + val)
    assert info["remapped"] == {"contract": ["rare"]}
    assert info["dropped"] == {"insurance_claim": ["tiny"]}
    assert info["dropped_val_rows"] == 1
    assert all(r["subclass"] == "other" for r in tr
               if r["doc_type"] == "contract" and r["filename"] == "i")
    assert all(r["doc_type"] != "insurance_claim" or r["subclass"] != "tiny"
               for r in tr)
    assert all(r["doc_type"] != "insurance_claim" or r["subclass"] != "tiny"
               for r in va)
    assert new_maps["insurance_claim"]["labels"] == ["auto"]
    assert new_maps["insurance_claim"]["label2id"] == {"auto": 0}
    # contract.rare was remapped to `other` — zero rows remain, so it leaves
    # the head vocabulary (a zero-row class would deflate macro-F1)
    assert new_maps["contract"]["labels"] == ["service", "license", "other"]
    # weights rebuilt over surviving rows
    assert new_maps["insurance_claim"]["weights"]["auto"] == pytest.approx(1.0)


def test_cli_accepts_audit_lever_flags():
    ns = build_parser().parse_args([
        "--loss-lambda-dt", "0.7", "--label-smoothing", "0.05",
        "--weight-mode", "sqrt-inverse", "--weight-cap", "8",
        "--mlp-heads", "--head-dropout", "0.2",
        "--freeze-backbone-epochs", "1", "--early-stop-patience", "3",
        "--weight-decay", "0.05", "--betas", "0.9,0.98", "--eps", "1e-6",
        "--subclass-min-train-rows", "12",
    ])
    assert ns.loss_lambda_dt == 0.7
    assert ns.label_smoothing == 0.05
    assert ns.weight_mode == "sqrt-inverse"
    assert ns.weight_cap == 8
    assert ns.mlp_heads is True
    assert ns.head_dropout == 0.2
    assert ns.freeze_backbone_epochs == 1
    assert ns.early_stop_patience == 3
    assert ns.weight_decay == 0.05
    assert ns.betas == "0.9,0.98"
    assert ns.eps == 1e-6
    assert ns.subclass_min_train_rows == 12


# -- local stage loader -------------------------------------------------------

def _write_parquet(root, cfg: str, split: str, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = root / "data" / cfg / split
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

