"""Regression tests for the deploy-layer correctness pass.

Hermetic like ``test_deploy.py``: every test is ``importorskip``-guarded on the
deploy/serve extras it needs and nothing here contacts Modal or the Hub (the
Modal functions are exercised through ``Function.local`` with the subprocess,
Hub client and Volume replaced by stand-ins).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.deploy


def _need_modal():
    return pytest.importorskip("modal")


def _need_serve():
    _need_modal()
    pytest.importorskip("fastapi")


# -- serve_app: head predictions + no-truncation encoding ---------------------

def test_serve_head_prediction_uses_trainable_labels_not_routing_only():
    """The ONNX heads emit one logit per TRAINABLE label.  ``_head_prediction``
    indexed the full ``id2label`` (doc_type ``unknown``, contract ``other``)
    and, iterating over its length, hit ``IndexError`` -> HTTP 500 on every
    request."""
    _need_serve()
    import numpy as np

    from deploy import serve_app

    maps = {
        "doc_type": {
            "labels": ["contract", "insurance_claim", "unknown"],
            "id2label": {"0": "contract", "1": "insurance_claim",
                         "2": "unknown"},
            "routing_only": ["unknown"],
            "trainable_id2label": {"0": "contract", "1": "insurance_claim"},
        },
        "contract": {
            "labels": ["license", "service", "other"],
            "id2label": {"0": "license", "1": "service", "2": "other"},
            "routing_only": ["other"],
            "trainable_id2label": {"0": "license", "1": "service"},
        },
    }
    dt = serve_app._head_prediction(np.array([0.1, 2.0], dtype=np.float32),
                                    maps, "doc_type")
    assert dt["label"] == "insurance_claim"
    assert set(dt["logits"]) == {"contract", "insurance_claim"}
    sc = serve_app._head_prediction(np.array([3.0, 0.5], dtype=np.float32),
                                    maps, "contract")
    assert sc["label"] == "license" and set(sc["logits"]) == {"license",
                                                              "service"}


def test_serve_trainable_map_matches_the_library_normalizer():
    """serve_app re-implements the trainable map dependency-free (the lean
    image has no pandas, which ``mailroom_ml.labels`` imports) — pin it to
    ``normalize_label_maps`` for legacy bundles that lack the fields."""
    _need_serve()
    pytest.importorskip("pandas")
    import copy

    from deploy import serve_app
    from mailroom_ml.labels import normalize_label_maps

    legacy = {
        "doc_type": {"labels": ["contract", "unknown"]},
        "contract": {"labels": ["license", "service", "other"]},
        "correspondence": {"labels": ["email", "letter"]},
    }
    lib = normalize_label_maps(copy.deepcopy(legacy))
    for head, cfg in legacy.items():
        got = serve_app._trainable_id2label(cfg, head)
        want = {int(k): v for k, v in lib[head]["trainable_id2label"].items()}
        assert got == want, head


class _Enc:
    def __init__(self, n: int):
        self.ids = list(range(1, n + 1))


class _Tok:
    def __init__(self, lengths: list[int]):
        self.lengths = lengths

    def encode_batch(self, texts):
        return [_Enc(self.lengths[i]) for i in range(len(texts))]


def test_serve_encode_pads_to_batch_not_max_and_never_truncates():
    _need_serve()
    from deploy import serve_app

    state = {"tokenizer": _Tok([5, 3]), "pad_id": 9}
    ids, mask = serve_app._encode(["a", "b"], state)
    assert ids.shape == (2, 5) == mask.shape  # not (2, MAX_LENGTH)
    assert ids[1].tolist() == [1, 2, 3, 9, 9]
    assert mask[1].tolist() == [1, 1, 1, 0, 0]

    over = {"tokenizer": _Tok([4, serve_app.MAX_LENGTH + 1]), "pad_id": 0}
    with pytest.raises(serve_app.TextTooLongError, match=r"texts\[1\]"):
        serve_app._encode(["ok", "huge"], over)


# -- onnx_export: serving sidecars --------------------------------------------

def test_onnx_export_copies_serving_sidecars(tmp_path, monkeypatch):
    """The bundle only carried labels/tokenizer/config, so ``load_bundle`` on
    the ONNX dir got T=1, no head exclusions, no OOD probe, no thresholds and
    empty support counts."""
    torch = pytest.importorskip("torch")
    from deploy import onnx_export

    src, out = tmp_path / "pt", tmp_path / "onnx"
    src.mkdir()
    (src / "labels.json").write_text("{}")
    for name in onnx_export.SERVING_SIDECARS:
        if name != "routing_thresholds.json":  # absent -> simply not copied
            (src / name).write_text(json.dumps({"from": name}))

    monkeypatch.setattr(
        onnx_export, "build_reference_model",
        lambda _d: (torch.nn.Linear(2, 2), {"doc_type": {}}))
    monkeypatch.setattr(
        torch.onnx, "export",
        lambda _m, _a, path, **_kw: Path(path).write_bytes(b"onnx"))
    onnx_export.export_onnx(src, out, quantize=False)

    assert (out / "model.onnx").is_file()
    for name in ("temperatures.json", "summary.json", "train_counts.json",
                 "ood_probe.json"):
        assert json.loads((out / name).read_text()) == {"from": name}
    assert not (out / "routing_thresholds.json").exists()


# -- onnx_parity_check: int8 is gated against the fp32 graph ------------------

def test_int8_agreement_is_measured_against_the_fp32_onnx_graph():
    pytest.importorskip("onnxruntime")
    import numpy as np

    from deploy.onnx_parity_check import _int8_vs_fp32

    fp32 = {"doc_type": np.array([[2.0, 0.0], [0.0, 3.0]])}
    same = {"doc_type": np.array([[1.9, 0.1], [0.2, 2.8]])}
    flip = {"doc_type": np.array([[0.0, 1.0], [0.0, 3.0]])}
    agree, drift = _int8_vs_fp32(fp32, same, require=True, n_samples=2)
    assert agree == {"doc_type": 1.0} and drift["doc_type"] == pytest.approx(0.2)
    with pytest.raises(AssertionError, match="fp32 graph"):
        _int8_vs_fp32(fp32, flip, require=True, n_samples=2)
    agree, _ = _int8_vs_fp32(fp32, flip, require=False, n_samples=2)
    assert agree == {"doc_type": 0.5}


def test_run_parity_compares_int8_with_fp32_not_pytorch(tmp_path, monkeypatch):
    """int8 == fp32 ONNX but both differ in argmax from the PyTorch reference
    (fp32 gate loosened): the documented ``--require-int8`` (vs fp32) passes;
    the old code compared int8 with PyTorch and raised."""
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    transformers = pytest.importorskip("transformers")
    import numpy as np

    from deploy import onnx_export, onnx_parity_check

    (tmp_path / "model.onnx").write_bytes(b"")
    (tmp_path / "model_quantized.onnx").write_bytes(b"")
    n = len(onnx_parity_check._SAMPLE_TEXTS)
    ref = torch.tensor([[5.0, 0.0]] * n)  # PyTorch: argmax 0
    onnx_logits = np.array([[0.0, 5.0]] * n, dtype=np.float32)  # ONNX: argmax 1

    class _Sess:
        def __init__(self, *_a, **_kw):
            pass

        def run(self, _names, _feeds):
            return [onnx_logits]

        def get_outputs(self):
            return [SimpleNamespace(name="logits_doc_type")]

    monkeypatch.setattr(ort, "InferenceSession", _Sess)
    monkeypatch.setattr(
        onnx_export, "build_reference_model",
        lambda _d: (lambda input_ids, attention_mask: {"doc_type": ref}, {}))

    class _Tok:
        @staticmethod
        def from_pretrained(*_a, **_kw):
            return lambda texts, **_k: {
                "input_ids": torch.zeros(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long)}

    monkeypatch.setattr(transformers, "AutoTokenizer", _Tok)
    res = onnx_parity_check.run_parity(
        tmp_path, tmp_path, tolerance=1e9, require_int8_agreement=True,
        max_length=4)
    assert res["argmax_agreement_int8"] == {"doc_type": 1.0}


# -- modal_app: push after parity, only when gated; nothing-trained ----------

def _summary(*, gate_met: bool, nothing_trained: bool = False,
             epoch: int = 2) -> dict:
    return {"nothing_trained": nothing_trained,
            "checkpoint_selection": {"gate_met": gate_met, "epoch": epoch}}


def test_push_decision_requires_gate_and_a_real_run():
    _need_modal()
    from deploy import modal_app

    assert modal_app._push_decision(_summary(gate_met=True), "o/m") == (True, "")
    ok, why = modal_app._push_decision(_summary(gate_met=False, epoch=0), "o/m")
    assert not ok and "gate not met" in why
    ok, why = modal_app._push_decision(
        _summary(gate_met=True, nothing_trained=True), "o/m")
    assert not ok and why == "nothing trained"
    assert modal_app._push_decision(_summary(gate_met=True), "")[0] is False
    # a missing gate key is NOT a pass
    assert modal_app._push_decision({}, "o/m")[0] is False
    ok, why = modal_app._push_decision(_summary(gate_met=True), "o/m",
                                       parity_ok=False)
    assert ok is False and "parity" in why


@pytest.fixture()
def fake_train_env(tmp_path, monkeypatch):
    """Run ``modal_app.train`` locally against stand-ins (no Modal, no Hub)."""
    _need_modal()
    hub = pytest.importorskip("huggingface_hub")
    from deploy import modal_app

    calls: list[tuple] = []
    state = {"summary": _summary(gate_met=True)}

    class _Api:
        def dataset_info(self, _repo, revision=None):
            return SimpleNamespace(sha=str(revision) + "0")

        def create_repo(self, repo, **kw):
            calls.append(("create_repo", repo))

        def upload_folder(self, **kw):
            calls.append(("upload", kw))

    def _run(cmd, check=False):
        script = Path(cmd[1]).name
        calls.append((script, list(cmd)))
        if script == "train_modernbert.py":
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "heads.pt").write_bytes(b"w")
            (out / "summary.json").write_text(json.dumps(state["summary"]))
        return SimpleNamespace(returncode=0)

    class _Vol:
        def commit(self):
            calls.append(("commit",))

    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(hub, "HfApi", _Api)
    monkeypatch.setattr(modal_app.subprocess, "run", _run)
    monkeypatch.setattr(modal_app, "CHECKPOINT_MOUNT", str(tmp_path / "ckpt"))
    monkeypatch.setattr(modal_app, "checkpoint_vol", _Vol())
    return SimpleNamespace(modal_app=modal_app, calls=calls, state=state,
                           ckpt=tmp_path / "ckpt")


def _names(calls):
    return [c[0] for c in calls]


def test_train_pushes_after_parity_and_not_via_the_trainer(fake_train_env):
    env = fake_train_env
    res = env.modal_app.train.local(push_to_hub="org/model", eval_test=False)
    trainer_cmd = next(c[1] for c in env.calls if c[0] == "train_modernbert.py")
    assert "--push-to-hub" not in trainer_cmd
    names = _names(env.calls)
    # trainer -> export -> parity -> (promote) -> upload
    assert names.index("train_modernbert.py") < names.index("onnx_export.py") \
        < names.index("onnx_parity_check.py") < names.index("upload")
    up = next(c[1] for c in env.calls if c[0] == "upload")
    assert up["repo_id"] == "org/model"
    assert {"optimizer.pt", "scheduler.pt", "resume.json"} <= set(
        up["ignore_patterns"])
    assert res["pushed"] is True and res["promoted"] is True
    assert (env.ckpt / "latest" / "heads.pt").is_file()


def test_train_does_not_push_an_ungated_run(fake_train_env):
    env = fake_train_env
    env.state["summary"] = _summary(gate_met=False, epoch=0)
    res = env.modal_app.train.local(push_to_hub="org/model", eval_test=False)
    assert "upload" not in _names(env.calls)
    assert res["pushed"] is False and "gate not met" in res["skip_reason"]


def test_train_skips_export_promote_push_when_nothing_trained(fake_train_env):
    env = fake_train_env
    env.state["summary"] = _summary(gate_met=True, nothing_trained=True)
    res = env.modal_app.train.local(push_to_hub="org/model", eval_test=False,
                                    resume="/checkpoints/runs/old")
    names = _names(env.calls)
    assert "onnx_export.py" not in names and "upload" not in names
    assert not (env.ckpt / "latest").exists()
    assert res["promoted"] is False and res["pushed"] is False
    assert res["skip_reason"] == "nothing trained"
    assert res["export_onnx"] is False


def test_train_smoke_never_pushes_or_promotes(fake_train_env):
    env = fake_train_env
    res = env.modal_app.train.local(push_to_hub="org/model", max_steps=24,
                                    eval_test=False)
    names = _names(env.calls)
    assert "upload" not in names and "onnx_export.py" not in names
    assert res["pushed"] is False and not (env.ckpt / "latest").exists()


def test_resume_help_points_at_run_dirs_not_latest():
    _need_modal()
    from deploy.spawn_train import build_parser

    help_text = build_parser().format_help()
    flat = " ".join(help_text.split())
    assert "/checkpoints/runs/<run-id>" in flat
    assert "latest/ only ever holds the last SUCCESSFUL run" in flat


# -- spawn_train: smoke checkpoint cadence ------------------------------------

def test_smoke_checkpoint_cadence_matches_a_real_run():
    """The trainer's checkpoint cadence defaults to ``--log-every``; the smoke
    pins log_every=4, so its 24-micro-batch timing was dominated by checkpoint
    I/O a real run (log_every 50) never pays at that rate."""
    _need_modal()
    from deploy import modal_app, spawn_train

    extra = spawn_train.smoke_trainer_extra()
    assert extra == ["--checkpoint-every=50"]
    assert spawn_train.REAL_CHECKPOINT_EVERY > spawn_train.SMOKE_LOG_EVERY
    cmd = modal_app._build_train_cmd(
        epochs=1, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False, max_steps=spawn_train.SMOKE_STEPS,
        log_every=spawn_train.SMOKE_LOG_EVERY, trainer_extra=extra)
    assert cmd[-1] == "--checkpoint-every=50"


# -- eval_app / eval_tracing --------------------------------------------------

def test_eval_cli_prints_non_json_stdout(monkeypatch, capsys):
    """Captured stdout was only echoed on failure: a non-JSON run showed the
    operator nothing, and a JSON run none of its progress lines."""
    _need_modal()
    from deploy import eval_app

    table = "doc_type acc 0.91\nsubclass acc 0.77\n"
    monkeypatch.setattr(
        eval_app.subprocess, "run",
        lambda cmd, **_kw: SimpleNamespace(returncode=0, stdout=table,
                                           stderr=""))
    eval_app._run_one_eval_cli(["python", "eval.py"])
    assert "doc_type acc 0.91" in capsys.readouterr().out

    report = {"n_docs": 3, "doc_type_accuracy": 0.9}
    blob = "progress line 1\nprogress line 2\n" + json.dumps(report, indent=2)
    monkeypatch.setattr(
        eval_app.subprocess, "run",
        lambda cmd, **_kw: SimpleNamespace(returncode=0, stdout=blob,
                                           stderr=""))
    out = eval_app._run_one_eval_cli(["python", "eval.py", "--json"])
    printed = capsys.readouterr().out
    assert "progress line 2" in printed and '"n_docs"' not in printed
    assert out["report"]["n_docs"] == 3


def test_eval_app_docstring_names_the_real_flag():
    _need_modal()
    from deploy import eval_app

    doc = eval_app.__doc__
    assert "--module runs/<run-id> --as-json" in doc
    assert "--module runs/<run-id> --json\n" not in doc


def test_phoenix_tracer_also_archives_spans_to_the_local_file(tmp_path):
    """``_eval_use_file`` was set on the Phoenix tracer and never read, so
    ``otel_spans.jsonl`` was empty whenever Phoenix was reachable."""
    from deploy import eval_tracing

    class _OtelSpan:
        def __init__(self):
            self.attrs = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

    class _Ctx:
        def __init__(self, span):
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, *a):
            return False

    class _PhoenixTracer:
        _eval_use_file = True

        def start_as_current_span(self, name):
            return _Ctx(_OtelSpan())

    eval_tracing._SPANS.clear()
    with eval_tracing.eval_span(_PhoenixTracer(), "modernbert.modal_eval",
                                subset="test"):
        pass
    assert [s["name"] for s in eval_tracing._SPANS] == ["modernbert.modal_eval"]
    assert eval_tracing._SPANS[0]["attributes"] == {"subset": "test"}
    path = eval_tracing.flush_tracing(None, out_path=tmp_path / "otel.jsonl")
    assert path is not None and len(path.read_text().splitlines()) == 1

    # file-less Phoenix tracer: no mirror
    class _PhoenixOnly(_PhoenixTracer):
        _eval_use_file = False

    eval_tracing._SPANS.clear()
    with eval_tracing.eval_span(_PhoenixOnly(), "x"):
        pass
    assert eval_tracing._SPANS == []


def test_flush_tracing_provider_flush_is_optional():
    from deploy import eval_tracing

    class _Provider:
        def __init__(self):
            self.calls = []

        def force_flush(self):
            self.calls.append("flush")

        def shutdown(self):
            self.calls.append("shutdown")

    prov = _Provider()
    tracer = SimpleNamespace(sandbox_provider=prov)
    eval_tracing.flush_tracing(tracer, flush_provider=False)
    assert prov.calls == []
    eval_tracing.flush_tracing(tracer)
    assert prov.calls == ["flush", "shutdown"]


def test_decode_adjust_sweep_flushes_the_provider_once_at_the_end(
        tmp_path, monkeypatch):
    _need_modal()
    from deploy import eval_app

    monkeypatch.setenv("MODERNBERT_TRACE_SINK", "none")
    monkeypatch.chdir(tmp_path)
    reports = {d: {"n_docs": 1} for d in ("0.25", "0.5", "0.75")}
    monkeypatch.setattr(
        eval_app, "run_eval",
        SimpleNamespace(remote=lambda **_kw: {"reports": reports}))
    flushes: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        eval_app, "_write_eval_json",
        lambda dest, report, **kw: flushes.append(
            (Path(dest).name, kw["flush_provider"])))
    eval_app.main(module="runs/r1", as_json=True, out=str(tmp_path),
                  decode_adjust_sweep="0.25,0.5,0.75")
    assert [f for _, f in flushes] == [False, False, True]
