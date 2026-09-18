"""Logits-parity check: PyTorch reference vs ONNX export.

Compares every head's logits between the reconstructed PyTorch model
(artifacts/pytorch/model) and the ONNX session (artifacts/onnx/model) on fixed
sample inputs — **no network access**: tokenizer comes from the artifact
bundle, texts are hardcoded.

Tolerances (max |PyTorch − ONNX| over all heads and positions):

    model.onnx            (fp32 export)      atol 1e-4  ← the export gate
    model_quantized.onnx  (int8 dynamic)     atol 5e-3  (int8 weight
                                     quantization shifts logits; the semantic
                                     gate for the quantized artifact uses a
                                     looser bound, documented in the runbook)

Both can be overridden with ``--tolerance``.

Usage:

    uv run --extra train --extra serve python deploy/onnx_parity_check.py
    uv run --extra train --extra serve python deploy/onnx_parity_check.py \\
        --pytorch-dir artifacts/pytorch/model --onnx-dir artifacts/onnx/model

As a pytest test (marker "serve", self-skips when artifacts or deps absent):

    uv run --extra train --extra serve python -m pytest -m serve \\
        tests/test_deploy.py -v
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

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

_TOLERANCE_FP32 = 1e-4   # the contractual export gate
_TOLERANCE_INT8 = 5e-3   # int8 weight quantization drift budget

try:
    import pytest  # noqa: F401 — only the pytest test path needs it
except ImportError:  # CLI use without the dev extra
    pytest = None  # type: ignore[assignment]


def _default_tolerance(onnx_dir: Path) -> float:
    return (_TOLERANCE_INT8 if (onnx_dir / "model_quantized.onnx").is_file()
            else _TOLERANCE_FP32)


def run_parity(pytorch_dir: Path, onnx_dir: Path, *, tolerance: float,
               max_length: int = 512) -> dict:
    """Compare PyTorch vs ONNX logits per head; raises AssertionError on failure."""
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

    quant = onnx_dir / "model_quantized.onnx"
    onnx_file = str(quant if quant.is_file() else onnx_dir / "model.onnx")
    sess = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
    ort_logits = dict(zip([o.name for o in sess.get_outputs()], outs))

    diffs: dict[str, float] = {}
    assert sorted(ort_logits) == sorted(ref_logits), (
        f"head mismatch: onnx {sorted(ort_logits)} vs pytorch {sorted(ref_logits)}"
    )
    for head, ref in sorted(ref_logits.items()):
        got = torch.as_tensor(ort_logits[f"logits_{head}"])
        diff = float((ref - got).abs().max())
        diffs[head] = diff
        assert diff <= tolerance, (
            f"head {head!r} logits diverge: max |pt - onnx| = {diff:.3e} "
            f"> tolerance {tolerance:.1e}"
        )

    return {
        "onnx_file": onnx_file,
        "tolerance": tolerance,
        "sample_texts": len(_SAMPLE_TEXTS),
        "max_length": max_length,
        "max_abs_diff_by_head": diffs,
        "pass": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pytorch-dir", type=Path, default=_DEFAULT_PYTORCH_DIR)
    ap.add_argument("--onnx-dir", type=Path, default=_DEFAULT_ONNX_DIR)
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="override tolerance (default: 1e-4 fp32 / 5e-3 int8)")
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    if not (args.pytorch_dir / "heads.pt").is_file():
        raise SystemExit(f"reference checkpoint missing under {args.pytorch_dir} "
                         "— see deploy/README.md, 'Copy a checkpoint locally'")
    if not (args.onnx_dir / "model.onnx").is_file():
        raise SystemExit(f"ONNX artifact missing under {args.onnx_dir} "
                         "— run deploy/onnx_export.py first")

    tolerance = args.tolerance or _default_tolerance(args.onnx_dir)
    result = run_parity(args.pytorch_dir, args.onnx_dir, tolerance=tolerance,
                        max_length=args.max_length)
    print(f"parity PASS (tolerance {tolerance:.1e}):")
    for head, diff in result["max_abs_diff_by_head"].items():
        print(f"  {head:>22s}  max|pt-onnx| = {diff:.3e}")
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

    tolerance = float(os.environ.get("PARITY_TOLERANCE",
                                     _default_tolerance(onnx_dir)))
    run_parity(pytorch_dir, onnx_dir, tolerance=tolerance)


if pytest is not None:
    test_onnx_pytorch_logits_parity = pytest.mark.serve(
        test_onnx_pytorch_logits_parity)


if __name__ == "__main__":
    raise SystemExit(main())