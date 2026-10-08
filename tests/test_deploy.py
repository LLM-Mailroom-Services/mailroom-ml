"""Deploy-layer tests: lightweight, skipif-guarded, **no network calls**.

Coverage:

- ``modal`` importable → both Modal apps construct (names, volumes, image),
  the documented trainer CLI surface is produced exactly by
  ``_build_train_cmd``, and the deployed GPU/timeout constants match the
  runbook (string GPU API per the 1.6.0 docs).
- ONNX parity is wired behind the ``serve`` marker and skips when
  torch/onnxruntime/transformers or the exported artifacts are absent, so the
  core suite stays green without any of these deps.

Everything here is asserted against the *documented* modal surface (object
types, constructor signatures, module constants) — private ``Image``/``Function``
internals are not stable across SDK releases and are deliberately not touched.

Run the full deploy suite:

    uv run --extra dev --extra deploy python -m pytest tests/test_deploy.py -v

Artifact parity (needs a real export; see deploy/README.md):

    uv run --extra dev --extra train --extra serve python -m pytest -m serve -v
"""
from __future__ import annotations

import inspect
import os

import pytest


def _need_modal():
    """Skip guard: modal is a deploy-time dependency only."""
    return pytest.importorskip("modal")


def _need_serve():
    """Skip guard: the serving app needs the serve extra (fastapi).

    Returns the ``modal`` module (the tests assert ``modal.App`` /
    ``modal.Secret`` types) — it used to return ``fastapi`` by mistake, which
    failed ``AttributeError: module 'fastapi' has no attribute 'App'``
    wherever both were installed.
    """
    modal = _need_modal()
    pytest.importorskip("fastapi")
    return modal


def _expected_train_flags() -> list[str]:
    """The documented training/train/train_modernbert.py CLI surface."""
    return [
        "--data", "--output", "--epochs", "--batch-size",
        "--grad-accum", "--lr", "--seed", "--push-to-hub", "--eval-test",
    ]


# -- training app -------------------------------------------------------------
def test_train_app_constructs() -> None:
    modal = _need_modal()

    from deploy import modal_app

    assert isinstance(modal_app.app, modal.App)
    assert modal_app.APP_NAME == "mailroom-ml-train"
    assert modal_app.CHECKPOINT_VOLUME_NAME == "modernbert-checkpoints"
    assert isinstance(modal_app.image, modal.Image)
    assert isinstance(modal_app.train, modal.Function)
    # deployed configuration (documented in deploy/README.md):
    assert modal_app.TRAIN_GPU == "L4"
    assert modal_app.TRAIN_TIMEOUT_S == 8 * 60 * 60
    assert modal_app.TRAIN_STARTUP_TIMEOUT_S == 10 * 60
    # compute guardrails (2026-09-19: 1-vCPU default starved the GPU — the
    # trainer must never run on the default 0.125-core request again; the
    # cost pass cut 8->2 cores / 16->10 GiB — billed on request, and the
    # per-step CPU work is ~0.5s vs ~6s of GPU work):
    assert modal_app.TRAIN_CPU >= 2
    assert modal_app.TRAIN_MEMORY_MIB >= 10240
    assert modal_app.TRAIN_RETRIES == 0


def test_train_app_uses_current_documented_surface() -> None:
    modal = _need_modal()

    import deploy.modal_app as modal_app

    # App tags (1.2.0+) and volumes are constructor metadata in the 1.6.0 SDK.
    app_params = inspect.signature(modal.App.__init__).parameters
    assert "tags" in app_params and "volumes" in app_params
    assert modal_app.app.name == "mailroom-ml-train"
    # the deployed GPU is the string constant asserted elsewhere — prove the
    # decorator actually binds it (keeps TRAIN_GPU honest without private APIs)
    assert "gpu=TRAIN_GPU" in inspect.getsource(modal_app)


def test_train_cmd_builds_exact_cli() -> None:
    _need_modal()

    from deploy import modal_app

    cmd = modal_app._build_train_cmd(
        epochs=3, batch_size=8, grad_accum=4, lr=3e-5, seed=7,
        push_to_hub="Lucius-Morningstar/mailroom-modernbert-classifier",
        eval_test=True,
    )
    assert cmd[0] == modal_app.sys.executable
    assert cmd[1] == "/root/training/train/train_modernbert.py"
    assert cmd[2:4] == ["--data", modal_app.TRAINING_DATA_REPO]
    assert cmd[4:6] == ["--output", "/checkpoints/latest"]
    flags = cmd[2::2]  # flag positions: index 0 is the executable, so 2, 4, 6…
    for flag in _expected_train_flags():
        assert flag in flags, f"missing {flag}"
    assert cmd[-3:] == ["--push-to-hub",
                        "Lucius-Morningstar/mailroom-modernbert-classifier",
                        "--eval-test"]


def test_train_cmd_omits_optional_flags_when_turned_off() -> None:
    _need_modal()

    from deploy import modal_app

    cmd = modal_app._build_train_cmd(
        epochs=1, batch_size=1, grad_accum=1, lr=1e-5, seed=1,
        push_to_hub="", eval_test=False,
    )
    assert "--push-to-hub" not in cmd
    assert "--eval-test" not in cmd
    assert set(cmd) == {
        modal_app.sys.executable, "/root/training/train/train_modernbert.py",
        "--data", modal_app.TRAINING_DATA_REPO, "--output",
        "/checkpoints/latest", "--epochs", "1", "--batch-size", "--grad-accum", "--lr", "1e-05", "--seed",
    }


def test_train_app_sources_are_bundled() -> None:
    """The local sources add_local_dir copies must exist at deploy time."""
    _need_modal()

    from deploy import modal_app

    root = modal_app.ROOT
    assert (root / "src" / "mailroom_ml" / "config.py").is_file()
    assert (root / "training").is_dir()
    assert (root / "configs").is_dir()
    # the trainer entrypoint the image wraps
    assert modal_app.TRAINER_SCRIPT == "/root/training/train/train_modernbert.py"


def test_train_cmd_threads_log_every() -> None:
    """The smoke passes --log-every so step lines land inside the 24-micro-
    batch cadence window; the real run must NOT get the flag (trainer default
    50 leaves the exact documented CLI byte-identical)."""
    _need_modal()

    from deploy import modal_app

    smoke = modal_app._build_train_cmd(
        epochs=1, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False, max_steps=24, log_every=4)
    assert smoke[-2:] == ["--log-every", "4"]
    real = modal_app._build_train_cmd(
        epochs=2, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False)
    assert "--log-every" not in real


def test_train_cmd_threads_resume_flag() -> None:
    """--resume reaches the trainer only when a bundle dir is given."""
    _need_modal()

    from deploy import modal_app

    resumed = modal_app._build_train_cmd(
        epochs=2, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False, resume="/checkpoints/latest")
    assert resumed[-2:] == ["--resume", "/checkpoints/latest"]
    fresh = modal_app._build_train_cmd(
        epochs=2, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False)
    assert "--resume" not in fresh


def test_train_cmd_threads_trainer_extra() -> None:
    """Audit-lever flags pass through verbatim (2026-09-20: loss rebalance,
    label smoothing, MLP heads, freeze, subclass support threshold)."""
    _need_modal()

    from deploy import modal_app

    cmd = modal_app._build_train_cmd(
        epochs=4, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False,
        trainer_extra=["--label-smoothing=0.05", "--mlp-heads",
                       "--freeze-backbone-epochs", "1",
                       "--subclass-min-train-rows", "12"])
    assert cmd[-6:] == ["--label-smoothing=0.05", "--mlp-heads",
                        "--freeze-backbone-epochs", "1",
                        "--subclass-min-train-rows", "12"]
    plain = modal_app._build_train_cmd(
        epochs=2, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False)
    assert "--label-smoothing=0.05" not in plain


def test_train_cmd_smoke_output_never_clobbers_latest() -> None:
    """The smoke probe writes to /checkpoints/smoke-<ts>, never latest/ —
    latest/ must stay the real checkpoint the resume path reads."""
    _need_modal()

    from deploy import modal_app

    smoke = modal_app._build_train_cmd(
        epochs=1, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False, max_steps=24, log_every=4,
        output="/checkpoints/smoke-20260920-000000")
    out_flag = smoke.index("--output")
    assert smoke[out_flag + 1].startswith("/checkpoints/smoke-")
    real = modal_app._build_train_cmd(
        epochs=2, batch_size=4, grad_accum=8, lr=2e-5, seed=42,
        push_to_hub="", eval_test=False)
    assert real[real.index("--output") + 1] == "/checkpoints/latest"


# -- spawn launcher: budget guard (pure logic, no network) -------------------
@pytest.mark.deploy
def test_spawn_budget_estimate() -> None:
    pytest.importorskip("modal")
    """Estimate formula: (epochs x steps/epoch x s/step + startup) / 3600 x $/h.
    Verified against the exact arithmetic with fixed inputs, plus the trip
    condition: a healthy Option A-shaped cadence (2 epochs, 750 steps/epoch @
    6s/step) must stay under the $4.32 ceiling while a slow/oversized run
    (5 epochs @ 30s/step) must exceed it. (The true steps/epoch comes from the
    smoke's train-windows measurement — never guessed here.)"""
    from deploy import spawn_train as st

    est = st._estimate_run_cost_usd(epochs=2, steps_per_epoch=750,
                                    sec_per_step=6.0,
                                    usd_per_hour=st.ALL_IN_USD_PER_HOUR,
                                    startup_overhead_s=60 * 30)
    hours = (2 * 750 * 6.0 + 1800) / 3600
    assert est == hours * st.ALL_IN_USD_PER_HOUR
    assert est < st.BUDGET_CEILING_USD, f"healthy Option A est {est:.2f} tripped"
    # a slow/misconfigured run must trip the ceiling
    slow = st._estimate_run_cost_usd(epochs=5, steps_per_epoch=3000,
                                     sec_per_step=30.0)
    assert slow > st.BUDGET_CEILING_USD


@pytest.mark.deploy
def test_spawn_budget_all_in_rate_matches_pricing_page() -> None:
    pytest.importorskip("modal")
    """2026-09-19 pricing page: L4 $0.000222/s, CPU $0.0000131/core/s,
    memory $0.00000222/GiB/s -> L4+2 cores+10GiB = $0.9734/h."""
    from deploy import spawn_train as st

    assert st.ALL_IN_USD_PER_HOUR == round(
        (0.000222 + 2 * 0.0000131 + 10 * 0.00000222) * 3600, 6)


@pytest.mark.deploy
def test_spawn_cadence_parser(tmp_path) -> None:
    pytest.importorskip("modal")
    """`modal app logs --timestamps` lines -> median seconds/step, and the
    trainer's windows line -> dataset size."""
    from deploy import spawn_train as st

    lines = [
        "2026-09-19 15:00:00 [trainer] windows: train 3000 / validation 300",
        "2026-09-19 15:00:06   step 4 loss 2.3100",
        "2026-09-19 15:00:12   step 8 loss 2.2400",
        "2026-09-19 15:00:18   step 12 loss 2.1900",
        "2026-09-19 15:00:24   step 16 loss 2.1500",
        "2026-09-19 15:00:30   step 20 loss 2.1000",
    ]
    assert st._sec_per_step_from_lines(lines) == 1.5  # 6s / 4 steps
    assert st._train_windows_from_lines(lines) == 3000
    assert st._steps_per_epoch(3000, batch_size=4) == 750
    # non-timestamped lines (no --timestamps) must not mis-parse
    assert st._sec_per_step_from_lines(["step 4 loss 2.3", "step 8 loss 2.2"]) is None


@pytest.mark.deploy
def test_spawn_metrics_roundtrip(tmp_path, monkeypatch) -> None:
    pytest.importorskip("modal")
    """Smoke metrics persist + reload through the configured path (env seam)."""
    from deploy import spawn_train as st

    monkeypatch.setenv(st.SMOKE_METRICS_ENV, str(tmp_path / "m.json"))
    # the default path may hold REAL smoke metrics from a prior smoke run on
    # this machine — pin it to the tmp tree so the test is hermetic.
    monkeypatch.setattr(st, "SMOKE_METRICS_DEFAULT", tmp_path / "default.json")
    p = st._save_smoke_metrics({"sec_per_step": 1.5, "train_windows": 3000})
    assert p.is_file()
    assert st._load_smoke_metrics()["train_windows"] == 3000
    monkeypatch.delenv(st.SMOKE_METRICS_ENV)
    assert st._load_smoke_metrics() == {}


# -- fallback serving app -----------------------------------------------------
def test_serve_app_constructs() -> None:
    modal = _need_serve()

    from deploy import serve_app

    assert isinstance(serve_app.app, modal.App)
    assert serve_app.APP_NAME == "mailroom-ml-serve"
    # both endpoints are Modal web functions
    assert isinstance(serve_app.predict, modal.Function)
    assert isinstance(serve_app.health, modal.Function)
    # CPU-only fallback: no GPU anywhere in this app (image declared only)
    assert serve_app.image is not None


def test_serve_app_secrets_require_token() -> None:
    """Token source: deploy-time env (dict secret) or a loud named-secret
    requirement — never a tokenless deployment."""
    modal = _need_serve()

    from deploy import serve_app

    old = os.environ.pop("SERVE_API_TOKEN", None)
    try:
        named = serve_app._serve_secrets()
        assert len(named) == 1 and isinstance(named[0], modal.Secret)
        os.environ["SERVE_API_TOKEN"] = "test-token"
        local = serve_app._serve_secrets()
        assert len(local) == 1 and isinstance(local[0], modal.Secret)
    finally:
        if old is not None:
            os.environ["SERVE_API_TOKEN"] = old
        else:
            os.environ.pop("SERVE_API_TOKEN", None)


def test_serve_model_dir_resolution() -> None:
    _need_serve()

    from deploy import serve_app

    # deterministic: explicit env override wins; no artifacts/volume in a bare
    # checkout → clear error (the runbook lays out both provisioning paths).
    old = os.environ.pop("SERVE_MODEL_DIR", None)
    try:
        with pytest.raises(FileNotFoundError):
            serve_app._resolve_model_dir()
        os.environ["SERVE_MODEL_DIR"] = "/tmp/whatever"
        assert serve_app._resolve_model_dir() == "/tmp/whatever"
    finally:
        if old is not None:
            os.environ["SERVE_MODEL_DIR"] = old


def test_serve_subclass_abstains_on_unknown_doc_type() -> None:
    """#121: doc_type ``unknown`` has no subclass head — must not KeyError."""
    _need_serve()
    import numpy as np

    from deploy import serve_app

    maps = {
        "doc_type": {
            "id2label": {"0": "contract", "1": "unknown"},
            "label2id": {"contract": 0, "unknown": 1},
        },
    }
    logits_by_head = {
        "logits_doc_type": np.array([[0.0, 5.0]], dtype=np.float32),
    }
    assert serve_app._subclass_prediction(
        logits_by_head, maps, "unknown", 0) is None
    logits_by_head["logits_contract"] = np.array([[3.0, 0.0]], dtype=np.float32)
    maps["contract"] = {"id2label": {"0": "nda", "1": "license"},
                          "label2id": {"nda": 0, "license": 1}}
    pred = serve_app._subclass_prediction(
        logits_by_head, maps, "contract", 0)
    assert pred is not None and pred["label"] == "nda"


# -- ONNX parity wiring -------------------------------------------------------
def test_onnx_parity_importable_and_wired() -> None:
    """The parity gate module imports cleanly and carries the serve marker."""
    import deploy.onnx_parity_check as parity

    assert parity._TOLERANCE_FP32 == 1e-4
    assert callable(parity.run_parity)
    marker = getattr(parity.test_onnx_pytorch_logits_parity, "pytestmark", None)
    assert marker is not None
    assert any(getattr(m, "name", None) == "serve" for m in marker)


@pytest.mark.serve
def test_onnx_parity_executes_or_skips() -> None:
    """Real parity run when artifacts + deps exist; otherwise skip — never
    network, never fail the core suite."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from deploy import onnx_parity_check as parity

    pytorch_dir, onnx_dir = parity._DEFAULT_PYTORCH_DIR, parity._DEFAULT_ONNX_DIR
    if not (pytorch_dir / "heads.pt").is_file() or \
            not (onnx_dir / "model.onnx").is_file():
        pytest.skip("artifacts not exported — run deploy/onnx_export.py first")
    result = parity.run_parity(
        pytorch_dir, onnx_dir, tolerance=1e-4, require_int8_agreement=True)
    assert result["pass"] is True
    if result.get("argmax_agreement_int8"):
        assert all(v == 1.0 for v in result["argmax_agreement_int8"].values())


def test_head_from_state_reconstructs_linear_and_mlp() -> None:
    """#101: heads.pt is self-describing — single-Linear heads (``weight``/
    ``bias``) and MLP heads (``0.*``/``3.*``: Linear, SiLU, Dropout, Linear —
    the run-2 ``--mlp-heads`` artifacts) must both reconstruct and load."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from deploy.onnx_export import _head_from_state

    hidden = 8
    linear_state = {"weight": torch.randn(3, hidden), "bias": torch.randn(3)}
    lin = _head_from_state(linear_state, hidden)
    assert isinstance(lin, nn.Linear)
    assert lin.out_features == 3
    lin.load_state_dict(linear_state)

    mlp_state = {
        "0.weight": torch.randn(hidden, hidden), "0.bias": torch.randn(hidden),
        "3.weight": torch.randn(4, hidden), "3.bias": torch.randn(4),
    }
    mlp = _head_from_state(mlp_state, hidden)
    assert isinstance(mlp, nn.Sequential)
    assert len(mlp) == 4  # Linear, SiLU, Dropout, Linear
    assert mlp[-1].out_features == 4
    mlp.load_state_dict(mlp_state)
    # eval-mode forward: dropout is identity, output shape is the head's
    x = torch.randn(2, hidden)
    assert mlp(x).shape == (2, 4)


def test_export_onnx_and_parity_fail_loud(monkeypatch) -> None:
    """#19: a failed ONNX export must not promote ``/checkpoints/latest``."""
    _need_modal()
    from deploy import modal_app

    def _fail(cmd, check=False):  # noqa: ARG001
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()

    monkeypatch.setattr(modal_app.subprocess, "run", _fail)
    with pytest.raises(RuntimeError, match="latest/ not updated"):
        modal_app._export_onnx_and_parity("/checkpoints/runs/x")


def test_promote_latest_is_atomic(tmp_path) -> None:
    """#19: promotion is a rename, not an in-place overwrite of latest/."""
    _need_modal()
    from deploy import modal_app

    src = tmp_path / "run"
    src.mkdir()
    (src / "heads.pt").write_bytes(b"ok")
    dest = tmp_path / "latest"
    dest.mkdir()
    (dest / "old").write_text("keep-until-rename")
    modal_app._promote_latest(src, dest)
    assert (dest / "heads.pt").read_bytes() == b"ok"
    assert not (dest / "old").exists()


def test_train_defaults_export_onnx_on() -> None:
    """``modal_app.train`` is a ``modal.Function`` once decorated, and
    ``inspect.signature`` rejects it (TypeError) — read the default from the
    source instead of any private Modal attribute."""
    _need_modal()
    import ast

    from deploy import modal_app

    fn = next(n for n in ast.parse(inspect.getsource(modal_app)).body
              if isinstance(n, ast.FunctionDef) and n.name == "train")
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) \
        + list(fn.args.defaults) + list(fn.args.kw_defaults)
    default = dict(zip((a.arg for a in args), defaults, strict=False))["export_onnx"]
    assert isinstance(default, ast.Constant) and default.value is True


def test_spawn_train_forwards_export_onnx() -> None:
    _need_modal()
    from deploy.spawn_train import build_parser

    ns = build_parser().parse_args(["--no-export-onnx"])
    assert ns.export_onnx is False
    assert build_parser().parse_args([]).export_onnx is True


def test_eval_app_parses_mixed_stdout_json() -> None:
    _need_modal()
    from deploy.eval_app import _parse_eval_report_stdout

    report = {"n_docs": 1323, "eval_subset": "heldout-plus", "doc_type_accuracy": 0.9}
    blob = (
        "[eval_modernbert] pulling repo\n"
        + __import__("json").dumps(report, sort_keys=True, indent=2)
    )
    parsed = _parse_eval_report_stdout(blob)
    assert parsed["n_docs"] == 1323
    assert parsed["eval_subset"] == "heldout-plus"


def test_eval_app_decode_logit_adjust_cli_default_zero() -> None:
    _need_modal()
    from deploy.eval_app import _eval_cli_cmd, _parse_decode_adjust_sweep

    cmd = _eval_cli_cmd(
        "runs/20261006-021245", 0, 42, "test",
        selective_risk=False, as_json=True)
    assert "--subclass-decode-logit-adjust" in cmd
    assert cmd[cmd.index("--subclass-decode-logit-adjust") + 1] == "0.0"
    cmd_d = _eval_cli_cmd(
        "runs/20261006-021245", 0, 42, "test",
        selective_risk=False, as_json=True,
        subclass_decode_logit_adjust=0.5)
    assert cmd_d[cmd_d.index("--subclass-decode-logit-adjust") + 1] == "0.5"
    assert _parse_decode_adjust_sweep("0.25,0.5,0.75,1.0") == [
        "0.25", "0.5", "0.75", "1.0"]
