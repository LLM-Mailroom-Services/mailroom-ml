"""Logits-parity check: PyTorch reference vs ONNX export.

Compares every head's logits between the reconstructed PyTorch model
(artifacts/pytorch/model) and the ONNX session (artifacts/onnx/model) on fixed
sample inputs — **no network access**: tokenizer comes from the artifact
bundle, texts are hardcoded.

Gates (max |PyTorch − ONNX| over all heads and positions):

    model.onnx            (fp32 export)      atol 1e-4  ← THE export contract
    model_quantized.onnx  (int8 dynamic)     argmax agreement + drift are
                                             measured and REPORTED; pass
                                             ``--require-int8`` to fail the
                                             gate when any head's argmax flips.
                                             (int8 weight-quantization error is
                                             weight-dependent — a trained
                                             model's confident margins survive;
                                             random-weight fixtures do not.)

Usage:

    uv run python deploy/onnx_parity_check.py
    uv run python deploy/onnx_parity_check.py \\
        --pytorch-dir artifacts/pytorch/model --onnx-dir artifacts/onnx/model \\
        --require-int8

As a pytest test (marker "serve", self-skips when artifacts or deps absent):

    uv run python -m pytest -m serve tests/test_deploy.py -v
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python deploy/onnx_parity_check.py`
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mailroom_ml.config import MAX_TOKENS  # noqa: E402

_DEFAULT_PYTORCH_DIR = ROOT / "artifacts" / "pytorch" / "model"
_DEFAULT_ONNX_DIR = ROOT / "artifacts" / "onnx" / "model"

# Deterministic sample inputs (mailroom-flavored, no I/O).
_SAMPLE_TEXTS = [
    "Exhibit A: Second Amended and Restated Software License Agreement "
    "between the parties, effective as of January 1, 2024.",
    "NOTICE OF AGENT'S ACTION REGARDING INSURANCE CLAIM FOR PROPERTY DAMAGE, "
    "Policy No. PL-22901, submitted for review and reimbursement.",
    "MINUTES of the special meeting of the Board of Directors, held by "
    "written consent; matters voted on and resolutions adopted.",
    "Dear Counsel: Re: Merger Agreement dated March 15, 2023 — closing "
    "condition checklist and pre-closing deliverables.",
    "MEMORANDUM — internal correspondence regarding vendor onboarding "
    "procedures and account approvals.",
]

_TOLERANCE_FP32 = 1e-4   # the contractual fp32 export gate

try:
    import pytest  # noqa: F401 — only the pytest test path needs it
except ImportError:  # CLI use without the dev extra
    pytest = None  # type: ignore[assignment]


def run_parity(pytorch_dir: Path, onnx_dir: Path, *, tolerance: float,
               require_int8_agreement: bool = False,
               max_length: int = MAX_TOKENS) -> dict:
    """Compare PyTorch vs ONNX logits per head; raises AssertionError on failure.

    - ``model.onnx`` (fp32) must match the PyTorch reference within
      ``tolerance`` (1e-4 default) — the export correctness contract.
    - ``model_quantized.onnx`` (int8, when present) argmax agreement + logit
      drift are measured and returned; only when ``require_int8_agreement`` is
      set does an argmax flip fail the gate.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoTokenizer

    from deploy.onnx_export import build_reference_model

    model, _ = build_reference_model(pytorch_dir)
    tokenizer = AutoTokenizer.from_pretrained(pytorch_dir)

    enc = tokenizer(_SAMPLE_TEXTS, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    with torch.no_grad():
        ref_logits = model(input_ids=ids, attention_mask=mask)

    feeds = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}

    def _run(path: Path) -> dict[str, np.ndarray]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        outs = sess.run(None, feeds)
        # graph output names carry the logits_ prefix; align with torch keys
        return {name.removeprefix("logits_"): arr
                for name, arr in zip([o.name for o in sess.get_outputs()],
                                     outs, strict=True)}

    fp32_path = onnx_dir / "model.onnx"
    assert fp32_path.is_file(), f"{fp32_path} missing — run deploy/onnx_export.py"
    ort_fp32 = _run(fp32_path)

    diffs: dict[str, float] = {}
    assert sorted(ort_fp32) == sorted(ref_logits), (
        f"head mismatch: onnx {sorted(ort_fp32)} vs pytorch {sorted(ref_logits)}"
    )
    for head, ref in sorted(ref_logits.items()):
        got = torch.as_tensor(ort_fp32[head])
        diff = float((ref - got).abs().max())
        diffs[head] = diff
        assert diff <= tolerance, (
            f"head {head!r} fp32 logits diverge: max |pt - onnx| = {diff:.3e} "
            f"> tolerance {tolerance:.1e}"
        )

    result: dict = {
        "onnx_file": str(fp32_path),
        "tolerance": tolerance,
        "sample_texts": len(_SAMPLE_TEXTS),
        "max_length": max_length,
        "max_abs_diff_by_head_fp32": diffs,
        "argmax_agreement_int8": None,
        "max_abs_diff_by_head_int8": None,
        "pass": True,
    }

    quant = onnx_dir / "model_quantized.onnx"
    if quant.is_file():
        ort_int8 = _run(quant)
        agree: dict[str, float] = {}
        drift: dict[str, float] = {}
        for head in sorted(ref_logits):
            ref_np = ref_logits[head].numpy()
            q = ort_int8[head]
            agree[head] = float(np.mean(ref_np.argmax(-1) == q.argmax(-1)))
            drift[head] = float(np.abs(ref_np - q).max())
            if require_int8_agreement and agree[head] < 1.0:
                raise AssertionError(
                    f"head {head!r} int8 decision parity broken: argmax "
                    f"agreement {agree[head]:.3f} < 1.0 over "
                    f"{len(_SAMPLE_TEXTS)} samples — re-export or skip int8"
                )
        result["onnx_file"] = str(quant)
        result["argmax_agreement_int8"] = agree
        result["max_abs_diff_by_head_int8"] = drift

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pytorch-dir", type=Path, default=_DEFAULT_PYTORCH_DIR)
    ap.add_argument("--onnx-dir", type=Path, default=_DEFAULT_ONNX_DIR)
    ap.add_argument("--tolerance", type=float, default=_TOLERANCE_FP32,
                    help="fp32 export gate, max |pt - onnx| (default 1e-4)")
    ap.add_argument("--require-int8", action="store_true",
                    help="fail when any int8 argmax flips vs the fp32 graph")
    ap.add_argument("--max-length", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    if not (args.pytorch_dir / "heads.pt").is_file():
        raise SystemExit(f"reference checkpoint missing under {args.pytorch_dir} "
                         "— see deploy/README.md, 'Copy a checkpoint locally'")
    if not (args.onnx_dir / "model.onnx").is_file():
        raise SystemExit(f"ONNX artifact missing under {args.onnx_dir} "
                         "— run deploy/onnx_export.py first")

    result = run_parity(args.pytorch_dir, args.onnx_dir,
                        tolerance=args.tolerance,
                        require_int8_agreement=args.require_int8,
                        max_length=args.max_length)
    print(f"fp32 export parity PASS (tolerance {args.tolerance:.1e}):")
    for head, diff in result["max_abs_diff_by_head_fp32"].items():
        print(f"  {head:>22s}  max|pt-onnx| = {diff:.3e}")
    if result["argmax_agreement_int8"] is not None:
        print(f"int8 decision parity PASS (argmax agreement 1.0 over "
              f"{result['sample_texts']} samples); logit drift reported:")
        for head, drift in result["max_abs_diff_by_head_int8"].items():
            print(f"  {head:>22s}  max|pt-int8| = {drift:.3e}  "
                  f"(agreement {result['argmax_agreement_int8'][head]:.3f})")
    return 0


def test_onnx_pytorch_logits_parity() -> None:
    """Pytest wiring of the parity gate; skipif-guarded, no network."""
    pytest.importorskip("onnxruntime")  # type: ignore[union-attr]
    pytest.importorskip("transformers")  # type: ignore[union-attr]

    pytorch_dir = Path(os.environ.get("PARITY_PYTORCH_DIR", _DEFAULT_PYTORCH_DIR))
    onnx_dir = Path(os.environ.get("PARITY_ONNX_DIR", _DEFAULT_ONNX_DIR))
    if not (pytorch_dir / "heads.pt").is_file() or \
            not (onnx_dir / "model.onnx").is_file():
        pytest.skip("artifacts not exported — run deploy/onnx_export.py first "
                    "(see deploy/README.md)")  # type: ignore[union-attr]

    tolerance = float(os.environ.get("PARITY_TOLERANCE", _TOLERANCE_FP32))
    require_int8 = os.environ.get("PARITY_REQUIRE_INT8", "1") != "0"
    run_parity(pytorch_dir, onnx_dir, tolerance=tolerance,
               require_int8_agreement=require_int8)


if pytest is not None:
    test_onnx_pytorch_logits_parity = pytest.mark.serve(
        test_onnx_pytorch_logits_parity)


if __name__ == "__main__":
    raise SystemExit(main())
