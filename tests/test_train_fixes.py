"""Regression tests for the trainer correctness pass (prefetch, resume, -100
labels, input-construction guard, selected-epoch promotion helpers, ...).

Hermetic like ``test_train.py``: no network, no GPU, no model downloads.  The
end-to-end ``main()`` seams live in ``test_train_main.py``.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from training.train.train_modernbert import (  # noqa: E402
    CE_IGNORE_INDEX,
    LossConfig,
    _align_checkpoint_every,
    _apply_backbone_freeze,
    _check_input_construction,
    _epoch_shuffle_rng,
    _fit_temperatures_from_logits,
    _prefetch_batches,
    _promote_epoch_files,
    _push_to_hub,
    _remaining_micro_budget,
    _stage_input_construction,
    _summary,
    ece,
    evaluate,
    fit_temperature,
    head_loss,
    macro_f1,
    make_batches,
    train_epoch,
)

pytestmark = pytest.mark.train

CPU = torch.device("cpu")


# -- prefetch worker must not swallow its exception ---------------------------

def test_prefetch_reraises_producer_exception():
    """The old ``finally: put(sentinel)`` swallowed a collate failure and the
    consumer just saw a SHORT epoch (exit 0).  The error must surface."""
    def producer():
        yield "b0"
        yield "b1"
        raise ValueError("collate blew up")

    got = []
    with pytest.raises(ValueError, match="collate blew up"):
        for item in _prefetch_batches(producer(), 2):
            got.append(item)
    assert got == ["b0", "b1"]  # everything produced before the failure


def test_make_batches_prefetch_surfaces_bad_label():
    heads = {"doc_type": {"label2id": {"a": 0}, "labels": ["a"],
                          "weights": {"a": 1.0}},
             "a": {"label2id": {"x": 0}, "labels": ["x"], "weights": {"x": 1.0}}}
    rows = [{"input_ids": torch.zeros(4, dtype=torch.long),
             "attention_mask": torch.ones(4, dtype=torch.long),
             "doc_type": "a", "subclass": "x", "filename": "ok.txt"},
            {"input_ids": torch.zeros(4, dtype=torch.long),
             "attention_mask": torch.ones(4, dtype=torch.long),
             "doc_type": "a", "subclass": "NOT-IN-HEAD", "filename": "bad.txt"}]
    with pytest.raises(KeyError):
        list(make_batches(rows, 1, False, heads, CPU, prefetch_batches=2))


# -- per-epoch deterministic shuffle -----------------------------------------

def _order(epoch: int, seed: int = 42, n: int = 12) -> list[str]:
    heads = {"doc_type": {"label2id": {"a": 0}, "labels": ["a"],
                          "weights": {"a": 1.0}},
             "a": {"label2id": {"x": 0}, "labels": ["x"], "weights": {"x": 1.0}}}
    rows = [{"input_ids": torch.zeros(2, dtype=torch.long),
             "attention_mask": torch.ones(2, dtype=torch.long),
             "doc_type": "a", "subclass": "x", "filename": f"f{i}"}
            for i in range(n)]
    out: list[str] = []
    for b in make_batches(rows, 1, True, heads, CPU,
                          shuffle_rng=_epoch_shuffle_rng(seed, epoch)):
        out += b["filename"]
    return out


def test_epoch_shuffle_depends_only_on_seed_and_epoch():
    """A resumed epoch k must replay the ORIGINAL run's epoch-k permutation:
    it cannot depend on how much global RNG state earlier epochs consumed."""
    random.seed(0)
    first = _order(epoch=3)
    random.seed(999)
    random.random()  # perturb the global stream like earlier epochs would
    assert _order(epoch=3) == first
    assert _order(epoch=2) != first          # epochs differ
    assert _order(epoch=3, seed=7) != first  # seeds differ
    assert sorted(first) == sorted(f"f{i}" for i in range(12))


# -- accumulation alignment ---------------------------------------------------

class _SpyOpt:
    def __init__(self, inner):
        self.inner = inner
        self.steps = 0

    def step(self):
        self.steps += 1
        self.inner.step()

    def zero_grad(self):
        self.inner.zero_grad()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _epoch_fixture(n_rows: int):
    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 2)

        def forward(self, input_ids, attention_mask):
            z = self.lin(input_ids.float())
            return {"doc_type": z, "a": z[:, :1]}

    model = _Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    heads = {"doc_type": {"label2id": {"a": 0, "b": 1}, "labels": ["a", "b"],
                          "weights": {"a": 1.0, "b": 1.0}},
             "a": {"label2id": {"a": 0}, "labels": ["a"],
                   "weights": {"a": 1.0}}}
    rows = [{"input_ids": torch.ones(4, dtype=torch.long),
             "attention_mask": torch.ones(4, dtype=torch.long),
             "doc_type": "a", "subclass": "a", "filename": f"f{i}.txt"}
            for i in range(n_rows)]
    return model, opt, sched, heads, rows


def test_resumed_epoch_flush_uses_absolute_micro_index():
    """skip_micro=2, 10 micro-batches, accum 4: absolute boundaries fall at
    4 and 8, and micro-batches 9-10 are a trailing partial window that MUST be
    flushed (3 steps).  The old flush keyed on the post-skip count n=8 (8 % 4
    == 0), so it never flushed and the 2 leftover gradients leaked into the
    next epoch's first step."""
    model, opt, sched, heads, rows = _epoch_fixture(10)
    spy = _SpyOpt(opt)
    batches = list(make_batches(rows, 1, False, heads, CPU))
    _, _, n_micro, n_opt = train_epoch(
        model, iter(batches), spy, sched, heads, CPU, 4,
        loss_cfg=LossConfig(), skip_micro=2)
    assert n_micro == 8
    assert n_opt == 3 and spy.steps == 3
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for p in model.parameters())  # nothing left accumulated


def test_resumed_epoch_does_not_flush_an_aligned_tail():
    """skip_micro=3, 12 micro-batches, accum 4: boundaries at 4, 8, 12 and no
    leftover.  The old ``n % accum`` (9 % 4 == 1) fired a spurious extra
    zero-gradient optimizer step."""
    model, opt, sched, heads, rows = _epoch_fixture(12)
    spy = _SpyOpt(opt)
    batches = list(make_batches(rows, 1, False, heads, CPU))
    _, _, n_micro, n_opt = train_epoch(
        model, iter(batches), spy, sched, heads, CPU, 4,
        loss_cfg=LossConfig(), skip_micro=3)
    assert n_micro == 9
    assert n_opt == 3 and spy.steps == 3


def test_checkpoint_cadence_rounds_up_to_accum_boundary():
    assert _align_checkpoint_every(4, 8) == 8      # the smoke: log_every=4
    assert _align_checkpoint_every(50, 8) == 56
    assert _align_checkpoint_every(16, 8) == 16
    assert _align_checkpoint_every(50, 1) == 50
    assert _align_checkpoint_every(0, 8) == 0


# -- freeze-backbone schedule -------------------------------------------------

def test_backbone_freeze_follows_the_epoch_number_not_an_edge():
    model = SimpleNamespace(backbone=nn.Linear(3, 3))
    # epochs 1..N frozen
    assert _apply_backbone_freeze(model, 1, 2) is True
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert _apply_backbone_freeze(model, 2, 2) is None     # already frozen
    # unfreeze at N+1
    assert _apply_backbone_freeze(model, 3, 2) is False
    assert all(p.requires_grad for p in model.backbone.parameters())
    # a --resume that STARTS past N+1 (frozen at startup in the old code and
    # only unfrozen on the epoch == N+1 edge, which it never visits)
    for p in model.backbone.parameters():
        p.requires_grad = False
    assert _apply_backbone_freeze(model, 5, 2) is False
    assert all(p.requires_grad for p in model.backbone.parameters())
    # flag off: never touches the backbone
    for p in model.backbone.parameters():
        p.requires_grad = False
    assert _apply_backbone_freeze(model, 5, 0) is None
    assert not any(p.requires_grad for p in model.backbone.parameters())


# -- --max-steps budget on resume --------------------------------------------

def test_remaining_micro_budget_distinguishes_unlimited_from_exhausted():
    assert _remaining_micro_budget(0, 123) is None      # flag off: unlimited
    assert _remaining_micro_budget(24, 10) == 14
    assert _remaining_micro_budget(24, 24) == 0         # spent -> STOP
    assert _remaining_micro_budget(24, 40) == 0         # (0 as a step_limit
    # would have meant unlimited to train_epoch — the bug)


# -- routing-only (-100) labels ----------------------------------------------

def test_fit_temperature_ignores_ignore_index_rows():
    """``log_probs[arange, -100]`` raised IndexError for any head narrower
    than 100 classes (every real subclass head)."""
    pytest.importorskip("scipy.optimize")
    rng = torch.Generator().manual_seed(0)
    lg = torch.randn(40, 5, generator=rng)
    lab = torch.randint(0, 5, (40,), generator=rng)
    lab2 = lab.clone()
    lab2[::4] = CE_IGNORE_INDEX
    got = fit_temperature(lg, lab2)  # must not raise
    assert 0.05 <= got <= 10.0
    keep = lab2 != CE_IGNORE_INDEX
    assert got == pytest.approx(fit_temperature(lg[keep], lab2[keep]))


def test_metrics_skip_ignored_rows():
    lg = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    full = torch.tensor([0, 1, CE_IGNORE_INDEX])
    assert macro_f1(lg, full) == macro_f1(lg[:2], full[:2]) == 1.0
    assert ece(lg, full) == pytest.approx(ece(lg[:2], full[:2]))
    temps = _fit_temperatures_from_logits(
        {"h": [lg]}, {"h": [torch.tensor([0, CE_IGNORE_INDEX, CE_IGNORE_INDEX])]})
    assert temps == {"h": 1.0}  # <2 scorable rows -> uncalibrated, no crash


def test_weighted_mean_all_ignored_head_is_finite():
    """F.cross_entropy(weighted mean) over only-ignored targets is 0/0 = NaN,
    which the divergence guard turned into an aborted run."""
    heads = {
        "doc_type": {"label2id": {"contract": 0}, "labels": ["contract"],
                     "weights": {"contract": 1.0}},
        "contract": {"label2id": {"service": 0, "license": 1},
                     "labels": ["service", "license"],
                     "weights": {"service": 1.0, "license": 1.0}},
    }
    batch = {"input_ids": torch.zeros(2, 8, dtype=torch.long),
             "attention_mask": torch.ones(2, 8, dtype=torch.long),
             "doc_type": torch.zeros(2, dtype=torch.long),
             "subclass": torch.tensor([CE_IGNORE_INDEX, CE_IGNORE_INDEX]),
             "filename": ["a", "b"]}

    class _Stub(nn.Module):
        def forward(self, input_ids, attention_mask):
            return {"doc_type": torch.zeros(2, 1),
                    "contract": torch.tensor([[1.0, 0.0]] * 2)}

    loss, _ = head_loss(_Stub(), batch, heads, CPU,
                        LossConfig(lambda_dt=0.5,
                                   subclass_loss_norm="weighted-mean"))
    assert torch.isfinite(loss)


def test_evaluate_drops_routing_only_rows_from_subclass_metrics():
    """Contract ``other`` GT rows are -100: they must leave the subclass head's
    arrays (no -100 in the temperature fit) and the subclass doc denominator."""
    maps = {"doc_type": {"trainable_id2label": {"0": "contract"}}}
    heads = {
        "doc_type": {"label2id": {"contract": 0}, "labels": ["contract"],
                     "weights": {"contract": 1.0}},
        "contract": {"label2id": {"service": 0, "license": 1},
                     "labels": ["service", "license"],
                     "weights": {"service": 1.0, "license": 1.0}},
    }
    batch = {"input_ids": torch.zeros(4, 8, dtype=torch.long),
             "attention_mask": torch.ones(4, 8, dtype=torch.long),
             "doc_type": torch.zeros(4, dtype=torch.long),
             "subclass": torch.tensor([0, 1, CE_IGNORE_INDEX,
                                       CE_IGNORE_INDEX]),
             "filename": ["a", "b", "c", "d"]}

    class _Stub(nn.Module):
        def forward(self, input_ids, attention_mask):
            return {"doc_type": torch.zeros(4, 1),
                    "contract": torch.tensor([[5.0, 0.0], [0.0, 5.0],
                                              [5.0, 0.0], [5.0, 0.0]])}

    metrics, lgs, labs = evaluate(_Stub(), [batch], heads, maps, CPU)
    lab = torch.cat(labs["contract"])
    assert lab.tolist() == [0, 1]
    assert torch.cat(lgs["contract"]).shape == (2, 2)
    assert metrics["contract_window_acc"] == 1.0
    assert metrics["contract_macro_f1_observed"] == 1.0
    # 4 docs correct on doc_type; 2 unscorable subclass docs leave the
    # denominator; both scorable ones are right
    assert metrics["subclass_doc_acc"] == 1.0
    temps = _fit_temperatures_from_logits(lgs, labs)  # no IndexError
    assert set(temps) == {"doc_type", "contract"}


# -- input-construction guard -------------------------------------------------

def test_input_construction_must_match_the_prebuilt_windows():
    _check_input_construction("v1", "v1")
    with pytest.raises(SystemExit, match="prebuilt as v1"):
        _check_input_construction("v2", "v1")


def test_stage_input_construction_default_and_declared(tmp_path):
    assert _stage_input_construction("org/repo") == "v1"      # published pin
    stage = tmp_path / "stage"
    stage.mkdir()
    assert _stage_input_construction(str(stage)) == "v1"
    (stage / "dataset_info.json").write_text(
        json.dumps({"input_construction": "v2"}))
    assert _stage_input_construction(str(stage)) == "v2"
    (stage / "dataset_info.json").write_text("not json")
    assert _stage_input_construction(str(stage)) == "v1"


# -- selected-epoch promotion -------------------------------------------------

def test_promotion_copies_weights_and_calibration_only(tmp_path):
    """Promotion used to copy EVERY archive file over the output, clobbering
    the complete epoch logs and the resume state with truncated mid-run
    copies.  Only weights + per-epoch calibration artifacts may move."""
    archive, out = tmp_path / "run-e1", tmp_path / "latest"
    archive.mkdir()
    out.mkdir()
    for name in ("model.safetensors", "heads.pt", "temperatures.json",
                 "ood_probe.json", "train_counts.json", "labels.json",
                 "config.json", "epoch_metrics.jsonl", "epoch_metrics.tsv",
                 "train_steps.jsonl", "test_steps.jsonl", "resume.json",
                 "resume_manifest.json", "optimizer.pt", "scheduler.pt",
                 "summary.json", "tokenizer.json"):
        (archive / name).write_text(f"EPOCH1:{name}")
        (out / name).write_text(f"FINAL:{name}")
    copied = _promote_epoch_files(archive, out)
    assert {"model.safetensors", "heads.pt", "temperatures.json",
            "ood_probe.json"} <= set(copied)
    for name in ("model.safetensors", "heads.pt", "temperatures.json",
                 "ood_probe.json"):
        assert (out / name).read_text() == f"EPOCH1:{name}"
    for name in ("epoch_metrics.jsonl", "epoch_metrics.tsv",
                 "train_steps.jsonl", "test_steps.jsonl", "resume.json",
                 "resume_manifest.json", "optimizer.pt", "scheduler.pt",
                 "summary.json", "tokenizer.json"):
        assert (out / name).read_text() == f"FINAL:{name}", name


# -- hub push -----------------------------------------------------------------

def test_push_to_hub_ignores_resume_state(monkeypatch, tmp_path):
    calls = {}

    class _Api:
        def create_repo(self, repo, **kw):
            calls["create"] = (repo, kw)

        def upload_folder(self, **kw):
            calls["upload"] = kw

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    _push_to_hub(tmp_path, "org/model", epochs=3, selected_epoch=2)
    assert calls["create"][0] == "org/model"
    up = calls["upload"]
    assert up["repo_id"] == "org/model"
    assert set(up["ignore_patterns"]) == {"optimizer.pt", "scheduler.pt",
                                          "resume.json"}
    assert "selected epoch 2" in up["commit_message"]


# -- summary flag -------------------------------------------------------------

def test_summary_flags_nothing_trained():
    args = SimpleNamespace(
        data="repo", model="m", seed=1, epochs=3, batch_size=4, grad_accum=8,
        lr=2e-5, select_on_subclass=True, input_construction="v1")
    sel = {"epoch": 2, "macro_f1": 0.9, "ece": 0.02}
    cold = _summary("rid", args, "cpu", [], sel, {}, 1.0, 3, epochs_trained=0)
    assert cold["nothing_trained"] is True
    assert cold["epochs_trained_this_run"] == 0
    warm = _summary("rid", args, "cpu", [], sel, {}, 1.0, 3, epochs_trained=2)
    assert warm["nothing_trained"] is False
    legacy = _summary("rid", args, "cpu", [], sel, {}, 1.0, 3)
    assert legacy["nothing_trained"] is False  # unknown is not "nothing"
