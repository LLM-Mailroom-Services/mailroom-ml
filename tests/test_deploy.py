"""Deploy-layer tests: lightweight, skipif-guarded, **no network calls**.

Coverage:

- ``modal`` importable → both Modal apps construct (names, volumes, image),
  the documented trainer CLI surface is produced exactly by
  ``_build_train_cmd``, and the deployed GPU/timeout constants match the
  runbook (string GPU API per the 1.5.5 docs).
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


def _expected_train_flags() -> list[str]:
    """The documented training/train_modernbert.py CLI surface."""
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

    # App tags (1.2.0+) and volumes are constructor metadata in the 1.5.5 SDK.
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
    assert cmd[1] == "/root/training/train_modernbert.py"
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
        modal_app.sys.executable, "/root/training/train_modernbert.py",
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
    assert modal_app.TRAINER_SCRIPT == "/root/training/train_modernbert.py"


# -- fallback serving app -----------------------------------------------------
def test_serve_app_constructs() -> None:
    modal = _need_modal()

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
    modal = _need_modal()

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
    _need_modal()

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


# -- ONNX parity wiring -------------------------------------------------------
def test_onnx_parity_importable_and_wired() -> None:
    """The parity gate module imports cleanly and carries the serve marker."""
    import deploy.onnx_parity_check as parity

    assert parity._TOLERANCE_FP32 == 1e-4
    assert callable(parity.run_parity)
    marker = getattr(parity.test_onnx_pytorch_logits_parity, "pytestmark", None)
    assert marker is not None
    assert any(getattr(m, "name", None) == "serve" for m in marker)


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
    result = parity.run_parity(pytorch_dir, onnx_dir, tolerance=1e-4)
    assert result["pass"] is True
