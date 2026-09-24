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

from conftest import fixture_rows  # noqa: E402
from mailroom_ml.config import (  # noqa: E402
    MAX_TOKENS,
    MODEL_ID,
    RUNS_DIR,
    TRAINING_DATA_REPO,
    TRAINING_DATA_REVISION,
)
from mailroom_ml.dataset import build_documents  # noqa: E402
from mailroom_ml.labels import label_maps  # noqa: E402
from training.train_modernbert import (  # noqa: E402
    DOC_TYPE_GATE_TOL,
    ECE_BUDGET,
    HierarchicalClassifier,
    LossConfig,
    _apply_resume,
    _apply_subclass_support_plan,
    _apply_subclass_support_threshold,
    _hub_data_glob,
    _scheduler_plan,
    _select_epoch,
    _selection_snapshot,
    _subclass_label,
    _subclass_objective,
    _summary,
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
    # provenance carry: this legacy bundle has no run_id/wall -> safe defaults
    assert state["run_id"] is None
    assert state["prior_wall_s"] == 0.0


def test_resume_carries_original_run_identity(tmp_path):
    """Provenance: an eval-only / continued resume keeps the ORIGINAL run's
    run_id + accumulated wall so the artifact summary is not rewritten as a
    fresh zero-second run (2026-09-20 recovery wrinkle)."""
    out = tmp_path / "ckpt"
    out.mkdir()
    torch.save({"doc_type": nn.Linear(4, 2).state_dict()}, out / "heads.pt")
    (out / "summary.json").write_text(json.dumps({
        "run_id": "20260920-173810",
        "training_wall_s": 22600.0,
        "epochs_run": 2,
        "epochs": [{"epoch": 1, "loss": 2.0, "val_loss": 1.5},
                   {"epoch": 2, "loss": 1.0, "val_loss": 1.0}],
        "checkpoint_selection": {"epoch": 2, "macro_f1": 0.9, "ece": 0.02},
    }))
    model = _tiny_model()
    opt, sched = _tiny_optim_sched(model)
    state = _apply_resume(out, model, opt, sched, torch.device("cpu"),
                          n_train_rows=20, batch_size=4, grad_accum=2)
    assert state["run_id"] == "20260920-173810"
    assert state["prior_wall_s"] == 22600.0


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


def test_transform_weights_device_honored():
    """Weights must land on the requested device: F.cross_entropy rejects a
    CPU weight against CUDA logits (the first GPU smoke crashed on this)."""
    cfg = LossConfig(weight_mode="inverse", weight_cap=10.0)
    dev = torch.device("cpu")
    w = cfg.transform_weights({"a": 2.0}, ["a"], device=dev)
    assert w.device == dev


def test_hub_data_glob_embeds_revision():
    """The pinned revision MUST be in the hf:// path (`@<rev>`): the
    `revision=` kwarg is ignored for hf:// globs, so a bare URL silently
    loads the stale (pre-clean) stage — measured 2026-09-20 (4573 vs 4499)."""
    g_train = _hub_data_glob(TRAINING_DATA_REPO, "train")
    assert f"@{TRAINING_DATA_REVISION}" in g_train
    assert g_train.endswith("/data/windows/train/*.parquet")
    g_test = _hub_data_glob(TRAINING_DATA_REPO, "test")
    assert g_test.endswith("/data/documents/test/*.parquet")
    assert f"@{TRAINING_DATA_REVISION}" in g_test


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


def test_inference_only_class_zero_support_excluded_and_default_weight():
    """#116: contract ``other`` (and doc_type ``unknown``) is present in the
    head's ``labels`` but absent from ``weights`` — it is recorded in
    ``inference_only`` and excluded from the observed macro-F1 average.  Its
    CE weight silently defaults to 1.0 (``transform_weights`` uses
    ``weights.get(label, 1.0)`` and it has no positive rows), the asymmetry
    that surprises every macro-F1 reader."""
    maps = label_maps(build_documents(fixture_rows()))
    contract = maps["contract"]
    assert "other" in contract["labels"]
    assert "other" in contract["inference_only"]
    assert "other" not in contract["weights"]
    # observed-only scoring over a head whose labels exceed its observed GT:
    # the zero-support class never enters the denominator
    lg = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    lab = torch.tensor([0, 1])  # class index 2 = the zero-support token
    assert macro_f1(lg, lab, observed_only=True) == 1.0
    assert macro_f1(lg, lab) == pytest.approx(2 / 3)
    # the missing weight defaults to 1.0 (silent — the trap)
    w = LossConfig().transform_weights(contract["weights"], ["other"])
    assert w.item() == pytest.approx(1.0)


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


def test_support_plan_mirrors_train_val_on_heldout_split():
    """The support floor must reach the held-out split too: a remapped
    subclass becomes `other` (kept) and a dropped one leaves the split — the
    2026-09-20 run crashed because test kept `affiliate` while the contract
    head vocabulary no longer did (KeyError in the test eval)."""
    info = {"remapped": {"contract": ["rare"]},
            "dropped": {"insurance_claim": ["tiny"]}}
    test = [
        {"doc_type": "contract", "subclass": "rare", "filename": "t1"},
        {"doc_type": "contract", "subclass": "service", "filename": "t2"},
        {"doc_type": "insurance_claim", "subclass": "tiny", "filename": "t3"},
    ]
    out = _apply_subclass_support_plan(test, info)
    assert [r["filename"] for r in out] == ["t1", "t2"]
    assert out[0]["subclass"] == "other"          # remapped, row kept
    assert test[0]["subclass"] == "rare"          # input not mutated


def test_subclass_label_falls_back_to_other_else_unscorable():
    heads = {
        "contract": {"label2id": {"service": 0, "other": 1}},
        "insurance_claim": {"label2id": {"auto": 0}},
    }
    assert _subclass_label(heads, "contract", "service") == 0
    # unknown in a head that HAS `other` -> the deployment fail-open target
    assert _subclass_label(heads, "contract", "affiliate") == 1
    # unknown in a head WITHOUT `other` -> unscorable (leaves the denominator)
    assert _subclass_label(heads, "insurance_claim", "tiny") is None


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



def test_train_epoch_aborts_on_nonfinite_loss():
    """Divergence guard (2026-09-20 audit R6): a NaN loss must abort the run
    before a NaN checkpoint can be saved/pushed."""
    device = torch.device("cpu")

    class _NanModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, input_ids, attention_mask):
            z = self.lin(input_ids.float())
            return {"doc_type": z * float("nan"), "a": z[:, :1]}

    model = _NanModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    heads = {
        "doc_type": {"label2id": {"a": 0, "b": 1}, "labels": ["a", "b"],
                     "weights": {"a": 1.0, "b": 1.0}},
        "a": {"label2id": {"a": 0}, "labels": ["a"], "weights": {"a": 1.0}},
    }
    rows = [{"input_ids": torch.zeros(4, dtype=torch.long),
             "attention_mask": torch.ones(4, dtype=torch.long),
             "doc_type": "a", "subclass": "a", "filename": "f.txt"}]
    batches = list(make_batches(rows, 1, False, heads, device))
    with pytest.raises(RuntimeError, match="non-finite loss"):
        train_epoch(model, iter(batches), opt, sched, heads, device, 1,
                    loss_cfg=LossConfig())


def test_summary_records_hyperparameters_and_test_metrics():
    """R2/R3: the summary carries the full hyperparameter set and the held-out
    test metrics, and flags whether the selection gate was met."""
    from training.train_modernbert import _summary

    args = SimpleNamespace(
        data="repo", model=MODEL_ID, seed=42, epochs=3, batch_size=4,
        grad_accum=8, lr=2e-5, loss_lambda_dt=0.65, label_smoothing=0.05,
        weight_mode="sqrt-inverse", weight_cap=10.0, mlp_heads=True,
        head_dropout=0.1, subclass_min_train_rows=12, early_stop_patience=2,
        weight_decay=0.01, betas="0.9,0.999", eps=1e-8, max_length=8192,
        freeze_backbone_epochs=0, eval_test=True, push_to_hub="",
        resume="", output="/tmp/x", limit=0, log_every=50, max_steps=0,
    )
    events = [{"epoch": 1, "loss": 1.0, "lr": 2e-5, "epoch_wall_s": 10.0}]
    selected = {"epoch": 1, "macro_f1": 0.7, "ece": 0.04, "ece_raw": 0.2}
    s = _summary("rid", args, "cpu", events, selected, {"doc_type": 0.5},
                 100.0, 1, {"n_docs": 323, "doc_type_acc": 0.8})
    assert s["hyperparameters"]["loss_lambda_dt"] == 0.65
    assert s["hyperparameters"]["mlp_heads"] is True
    assert s["hyperparameters"]["subclass_min_train_rows"] == 12
    assert s["test_metrics"]["doc_type_acc"] == 0.8
    assert s["checkpoint_selection"]["gate_met"] is True
    # gate not met -> flagged
    s2 = _summary("rid", args, "cpu", events,
                  {"epoch": 0, "macro_f1": -1.0, "ece": None}, {}, 1.0, 1)
    assert s2["checkpoint_selection"]["gate_met"] is False
    assert s2["test_metrics"] == {}


def test_summary_carries_per_head_ece_and_exclusion_policy():
    """#107: the selection snapshot carries the per-head calibrated ECE
    sidecar and the derived head-exclusion policy — the deployment gate's
    data source.  Heads over the budget are flagged excluded; heads under
    it are not; a selection without the sidecar yields an empty policy."""
    from training.train_modernbert import ECE_BUDGET, _summary

    args = SimpleNamespace(
        data="repo", model=MODEL_ID, seed=42, epochs=3, batch_size=4,
        grad_accum=8, lr=2e-5, loss_lambda_dt=0.65, label_smoothing=0.05,
        weight_mode="sqrt-inverse", weight_cap=10.0, mlp_heads=True,
        head_dropout=0.1, subclass_min_train_rows=12, early_stop_patience=2,
        weight_decay=0.01, betas="0.9,0.999", eps=1e-8, max_length=8192,
        freeze_backbone_epochs=0, eval_test=True, push_to_hub="",
        resume="", output="/tmp/x", limit=0, log_every=50, max_steps=0,
    )
    selected = {
        "epoch": 2, "macro_f1": 0.9, "ece": 0.02, "ece_raw": 0.16,
        "per_head_ece_calibrated": {
            "doc_type": 0.02, "contract": 0.09, "correspondence": 0.03,
        },
    }
    s = _summary("rid", args, "cpu", [], selected, {}, 1.0, 2)
    sel = s["checkpoint_selection"]
    assert sel["per_head_ece_calibrated"] == {
        "doc_type": 0.02, "contract": 0.09, "correspondence": 0.03}
    policy = sel["head_exclusion_policy"]
    assert policy["budget"] == ECE_BUDGET
    assert policy["excluded"]["contract"] == {
        "excluded": True, "ece_calibrated": 0.09}
    assert policy["excluded"]["doc_type"] == {
        "excluded": False, "ece_calibrated": 0.02}
    assert policy["excluded"]["correspondence"]["excluded"] is False
    # pre-#107 selection (no sidecar) -> empty policy, still self-describing
    s_old = _summary("rid", args, "cpu", [],
                     {"epoch": 1, "macro_f1": 0.7, "ece": 0.04}, {}, 1.0, 1)
    old_policy = s_old["checkpoint_selection"]["head_exclusion_policy"]
    assert old_policy["excluded"] == {}


# ---------------------------------------------------------------------------
# #112 M9a-U4 lexicographic checkpoint selection (extracted pure seam)
# ---------------------------------------------------------------------------

def _epoch_val(dt: float, objective: float, *, ece: float = 0.01,
               head: str = "contract") -> dict:
    """A minimal ``val`` dict as the selection rule consumes it.

    The subclass objective is carried both as the pre-computed
    ``subclass_objective`` (what ``main`` sets before calling) and as the
    single head's observed macro-F1 (what ``_selection_snapshot`` recomputes
    from), so the two must agree.
    """
    return {
        "doc_type_macro_f1_observed": dt,
        "doc_type_ece_calibrated": ece,
        "doc_type_ece": ece,
        f"{head}_macro_f1_observed": objective,
        f"{head}_ece_calibrated": ece,
        "subclass_objective": objective,
    }


def test_select_epoch_prefers_better_subclass_within_doc_type_tolerance():
    """The #112 rule: a slightly lower doc_type macro-F1 (within the
    DOC_TYPE_GATE_TOL band) does not veto an epoch with a better subclass
    objective — run-3's 0.9245 -> 0.9227, 0.26 -> 0.30 case."""
    selected = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                "subclass_objective": -1.0}
    best_doc_type = float("-inf")
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.9245, 0.26), 1, selected, best_doc_type, ["contract"],
        select_on_subclass=True)
    assert selected["epoch"] == 1
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.9227, 0.30), 2, selected, best_doc_type, ["contract"],
        select_on_subclass=True)
    assert selected["epoch"] == 2
    assert selected["subclass_objective"] == 0.30
    assert best_doc_type == 0.9245


def test_select_epoch_rejects_doc_type_regression_beyond_tolerance():
    """A higher subclass objective cannot buy back a doc_type regression
    beyond DOC_TYPE_GATE_TOL below the ECE-eligible floor."""
    selected = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                "subclass_objective": -1.0}
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.9245, 0.26), 1, selected, float("-inf"), ["contract"],
        select_on_subclass=True)
    assert 0.006 > DOC_TYPE_GATE_TOL  # the regression under test is rejected
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.9245 - 0.006, 0.40), 2, selected, best_doc_type,
        ["contract"], select_on_subclass=True)
    assert selected["epoch"] == 1
    assert selected["subclass_objective"] == 0.26


def test_select_epoch_requires_doc_type_ece_within_budget():
    """An epoch over the calibrated doc_type ECE budget is ineligible even
    with a better subclass objective — and never raises the floor."""
    selected = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                "subclass_objective": -1.0}
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.90, 0.20), 1, selected, float("-inf"), ["contract"],
        select_on_subclass=True)
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.95, 0.90, ece=ECE_BUDGET + 0.01), 2, selected,
        best_doc_type, ["contract"], select_on_subclass=True)
    assert selected["epoch"] == 1
    assert selected["subclass_objective"] == 0.20
    assert best_doc_type == 0.90  # ineligible epoch never raises the floor


def test_select_on_subclass_default_true_and_legacy_flag():
    """Default is the lexicographic rule; --no-select-on-subclass restores
    the legacy doc_type-only rule, where a higher-obj/lower-dt epoch does
    not displace the higher-dt one."""
    assert build_parser().parse_args([]).select_on_subclass is True
    assert build_parser().parse_args(
        ["--no-select-on-subclass"]).select_on_subclass is False

    selected = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                "subclass_objective": -1.0}
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.9245, 0.26), 1, selected, float("-inf"), ["contract"],
        select_on_subclass=False)
    assert selected["epoch"] == 1
    selected, best_doc_type = _select_epoch(
        _epoch_val(0.9227, 0.30), 2, selected, best_doc_type, ["contract"],
        select_on_subclass=False)
    assert selected["epoch"] == 1  # legacy rule ignores the subclass gain
    assert selected["subclass_objective"] == 0.26


def test_subclass_objective_ignores_absent_heads_and_empty_is_zero():
    val = {"contract_macro_f1_observed": 0.4,
           "correspondence_macro_f1_observed": 0.6}
    assert _subclass_objective(
        val, ["contract", "correspondence", "missing"]) == pytest.approx(0.5)
    assert _subclass_objective({}, ["contract"]) == 0.0
    assert _subclass_objective(val, []) == 0.0


def test_selection_snapshot_carries_subclass_objective_and_per_head():
    """The snapshot's ``per_head_macro_f1_observed`` currently INCLUDES
    ``doc_type`` (every ``*_macro_f1_observed`` key on the epoch event), while
    ``subclass_objective`` averages only the named subclass heads."""
    val = {
        "doc_type_macro_f1_observed": 0.9,
        "doc_type_ece_calibrated": 0.02,
        "doc_type_ece": 0.10,
        "contract_macro_f1_observed": 0.5,
        "contract_ece_calibrated": 0.03,
        "correspondence_macro_f1_observed": 0.7,
        "correspondence_ece_calibrated": 0.04,
    }
    snap = _selection_snapshot(val, 3, ["contract", "correspondence"])
    assert snap["epoch"] == 3
    assert snap["macro_f1"] == 0.9
    assert snap["ece"] == 0.02
    assert snap["ece_raw"] == 0.10
    assert snap["subclass_objective"] == pytest.approx(0.6)
    assert snap["per_head_macro_f1_observed"] == {
        "contract": 0.5, "correspondence": 0.7, "doc_type": 0.9}
    assert snap["per_head_ece_calibrated"] == {
        "contract": 0.03, "correspondence": 0.04, "doc_type": 0.02}


def test_summary_records_lexicographic_selection_metadata():
    """With ``select_on_subclass`` the summary's rule string names the
    lexicographic rule and the two new selection keys are populated."""
    args = SimpleNamespace(
        data="repo", model=MODEL_ID, seed=42, epochs=3, batch_size=4,
        grad_accum=8, lr=2e-5, loss_lambda_dt=0.65, label_smoothing=0.05,
        weight_mode="sqrt-inverse", weight_cap=10.0, mlp_heads=True,
        head_dropout=0.1, subclass_min_train_rows=12, early_stop_patience=2,
        weight_decay=0.01, betas="0.9,0.999", eps=1e-8, max_length=8192,
        freeze_backbone_epochs=0, eval_test=True, push_to_hub="",
        resume="", output="/tmp/x", limit=0, log_every=50, max_steps=0,
        select_on_subclass=True,
    )
    selected = {
        "epoch": 2, "macro_f1": 0.9, "ece": 0.02, "ece_raw": 0.16,
        "subclass_objective": 0.61,
        "per_head_macro_f1_observed": {"doc_type": 0.9, "contract": 0.5},
    }
    s = _summary("rid", args, "cpu", [], selected, {}, 1.0, 2)
    sel = s["checkpoint_selection"]
    assert "lexicographic" in sel["rule"]
    assert sel["subclass_objective"] == 0.61
    assert sel["per_head_macro_f1_observed"] == {
        "doc_type": 0.9, "contract": 0.5}


def test_epoch_val_dict_includes_selection_keys():
    """Keys the trainer writes before ``_select_epoch`` (#139 contract)."""
    val = {
        "doc_type_macro_f1_observed": 0.9,
        "doc_type_ece": 0.10,
        "doc_type_ece_calibrated": 0.02,
        "contract_macro_f1_observed": 0.5,
        "contract_ece_calibrated": 0.03,
        "subclass_objective": 0.5,
    }
    snap = _selection_snapshot(val, 2, ["contract"])
    assert snap["macro_f1"] == 0.9
    assert snap["ece"] == 0.02
    assert snap["subclass_objective"] == 0.5


def test_summary_temperatures_reflect_passed_temps():
    """Promotion must pass selected-epoch temps into ``_summary`` (#130)."""
    args = SimpleNamespace(
        data="repo", model=MODEL_ID, seed=42, epochs=3, batch_size=4,
        grad_accum=8, lr=2e-5, loss_lambda_dt=0.65, label_smoothing=0.05,
        weight_mode="sqrt-inverse", weight_cap=10.0, mlp_heads=True,
        head_dropout=0.1, subclass_min_train_rows=12, early_stop_patience=2,
        weight_decay=0.01, betas="0.9,0.999", eps=1e-8, max_length=8192,
        freeze_backbone_epochs=0, eval_test=True, push_to_hub="",
        resume="", output="/tmp/x", limit=0, log_every=50, max_steps=0,
        select_on_subclass=False,
    )
    selected = {"epoch": 1, "macro_f1": 0.9, "ece": 0.02, "ece_raw": 0.1}
    s_sel = _summary("rid", args, "cpu", [], selected,
                     {"doc_type": 1.1}, 1.0, 1)
    s_final = _summary("rid", args, "cpu", [], selected,
                       {"doc_type": 9.9}, 1.0, 1)
    assert s_sel["temperatures"]["doc_type"] == 1.1
    assert s_final["temperatures"]["doc_type"] == 9.9


def test_resume_restores_best_doc_type_gate_floor(tmp_path):
    """#112: --resume re-derives the doc_type gate floor as the best observed
    doc_type macro-F1 among ECE-eligible prior epochs; no eligible event
    leaves the gate open (``-inf``)."""
    out = tmp_path / "ckpt"
    out.mkdir()
    torch.save({"doc_type": nn.Linear(4, 2).state_dict()}, out / "heads.pt")
    (out / "summary.json").write_text(json.dumps({
        "epochs_run": 3,
        "epochs": [
            {"epoch": 1, "loss": 2.0, "val_loss": 2.0,
             "doc_type_macro_f1_observed": 0.80,
             "doc_type_ece_calibrated": 0.02},   # eligible
            {"epoch": 2, "loss": 1.5, "val_loss": 1.5,
             "doc_type_macro_f1_observed": 0.95,
             "doc_type_ece_calibrated": 0.09},   # over budget -> ignored
            {"epoch": 3, "loss": 1.0, "val_loss": 1.0,
             "doc_type_macro_f1_observed": 0.90,
             "doc_type_ece_calibrated": ECE_BUDGET},  # eligible (boundary)
        ],
        "checkpoint_selection": {"epoch": 3, "macro_f1": 0.90, "ece": 0.05},
    }))
    model = _tiny_model()
    opt, sched = _tiny_optim_sched(model)
    state = _apply_resume(out, model, opt, sched, torch.device("cpu"),
                          n_train_rows=20, batch_size=4, grad_accum=2)
    assert state["best_doc_type"] == pytest.approx(0.90)

    # legacy bundle: no per-event doc_type metrics -> open gate
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save({"doc_type": nn.Linear(4, 2).state_dict()}, legacy / "heads.pt")
    (legacy / "summary.json").write_text(json.dumps({
        "epochs_run": 1,
        "epochs": [{"epoch": 1, "loss": 2.0, "val_loss": 1.5}],
        "checkpoint_selection": {"epoch": 1, "macro_f1": 0.8, "ece": 0.04},
    }))
    model2 = _tiny_model()
    opt2, sched2 = _tiny_optim_sched(model2)
    state2 = _apply_resume(legacy, model2, opt2, sched2, torch.device("cpu"),
                           n_train_rows=20, batch_size=4, grad_accum=2)
    assert state2["best_doc_type"] == float("-inf")
