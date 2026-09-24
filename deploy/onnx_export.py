"""Export the fine-tuned hierarchical ModernBERT checkpoint to ONNX.

Turns the trainer's output layout (a ``save_pretrained`` ModernBERT backbone +
``heads.pt`` + ``labels.json`` — the cf096fa hierarchical-head structure) into
the serving artifact bundle:

    artifacts/onnx/model/
        model.onnx             fp32 export (dynamic batch+sequence axes)
        model_quantized.onnx   int8 dynamic-quantized (same graph contract)
        labels.json            label maps (copied, for serve/parity)
        tokenizer.json / tokenizer_config.json / config.json ...

Graph contract (stable — serve_app.py and onnx_parity_check.py depend on it):

    inputs:      input_ids      int64[batch, seq]
                 attention_mask int64[batch, seq]
    outputs:     logits_<head>  float32[batch, num_labels] for every head in
                 labels.json (doc_type + per-class subclass heads), one output
                 per head, ordered by sorted head name.

Why not ``optimum-cli export onnx`` / ``ORTModelForSequenceClassification``?

Both optimum paths assume a standard ``AutoModelForSequenceClassification``
checkpoint. Our artifact is a shared encoder + per-class conditional heads
stored outside ``config.json`` (``heads.pt``); the optimum exporter would
re-initialize random head weights and silently export a broken model. The
robust route for a custom architecture is ``torch.onnx.export`` with explicit
dynamic axes (documented API, stable since torch 1.x) plus
``onnxruntime.quantization.quantize_dynamic`` for int8 weights (the same
quantizer family optimum's ``ORTQuantizer`` wraps — see the optimum quicktour,
https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model,
for the standard-checkpoint path with ``optimum-cli export onnx --quantize int8``).

Requires the train/serve extras (torch, transformers, onnxruntime). Exporting
with torch >= 2.14 additionally needs the ONNX toolchain wheel (``onnx`` /
``onnxscript`` — the 2.14 exporter imports it):

    uv sync --extra train --extra serve
    uv pip install onnx onnxscript    # torch >= 2.14 export/quantize toolchain

Then:

    uv run python deploy/onnx_export.py \\
        --pytorch-dir artifacts/pytorch/model \\
        --out-dir artifacts/onnx/model

Notes on the exporter choice (verified live 2026-09-18, torch 2.14.0):

- torch.onnx.export now defaults to the Dynamo exporter (``dynamo=True``),
  which currently emits an **invalid** graph for this architecture
  (``Split`` with a removed ``num_outputs`` attribute → onnx.checker /
  onnxruntime both reject it). We therefore pass ``dynamo=False`` to use the
  stable TorchScript-tracer exporter (opset 17, dynamic batch+sequence axes;
  graph validated with onnx.checker + onnxruntime before quantization).
- torch's legacy exporter is still the default-arg-free stable path for custom
  encoder graphs; revisit Dynamo when its Split emission is fixed.

Verify parity afterwards:

    uv run python deploy/onnx_parity_check.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn as nn

from mailroom_ml.labels import normalize_label_maps

ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_PYTORCH_DIR = ROOT / "artifacts" / "pytorch" / "model"
_DEFAULT_OUT_DIR = ROOT / "artifacts" / "onnx" / "model"


class HierarchicalClassifier(nn.Module):
    """Reference reconstruction of the cf096fa hierarchical head model.

    Shared ModernBERT backbone + one ``nn.Linear`` head per label space; the
    pooled representation is the first-token (<s>) hidden state, cast to fp32
    (the committed train.py recipe).
    """

    def __init__(self, base_model, head_sizes: dict[str, int]) -> None:
        super().__init__()
        self.backbone = base_model
        self.heads = nn.ModuleDict(
            {name: nn.Linear(base_model.config.hidden_size, n, bias=True)
             for name, n in sorted(head_sizes.items())}
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0].float()
        return {name: head(pooled) for name, head in self.heads.items()}


def _head_from_state(state: dict, hidden: int) -> nn.Module:
    """Reconstruct a head module from its state dict (linear vs MLP).

    The trainer's ``head_kind`` is not recorded in the bundle, but the
    state dict is self-describing: single-Linear heads save ``weight``/
    ``bias``; MLP heads (``--mlp-heads``, the ModernBERT recipe) save
    ``0.weight``/``0.bias``/``3.weight``/``3.bias`` (Sequential: Linear,
    SiLU, Dropout, Linear — train_modernbert.HierarchicalClassifier).
    Dropout has no parameters; eval-mode it is identity, so the
    reconstruction uses 0.0.
    """
    if any(k.startswith("0.") for k in state):
        return nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, state["3.weight"].shape[0]),
        )
    return nn.Linear(hidden, state["weight"].shape[0], bias=True)


def build_reference_model(pytorch_dir: Path):
    """Load the trainer output and rebuild the trained model (fp32, eval mode).

    Returns ``(model, label_maps)`` where label_maps is the ``labels.json``
    content (per-head ``label2id`` / ``id2label`` / ``labels`` / ``weights``).
    """
    maps = json.loads((pytorch_dir / "labels.json").read_text())
    maps = normalize_label_maps(maps)
    head_sizes = {name: len(cfg["trainable_labels"]) for name, cfg in maps.items()}
    from transformers import AutoModel  # heavy dep: import lazily

    base = AutoModel.from_pretrained(pytorch_dir, torch_dtype=torch.float32)
    model = HierarchicalClassifier(base, head_sizes)
    # heads.pt is saved by the trainer as {"<head>": <head state dict>} —
    # linear or MLP depending on --mlp-heads; rebuild each head from its
    # own state dict shape (2026-09-21: run-2 artifacts use MLP heads).
    heads_state = torch.load(pytorch_dir / "heads.pt", map_location="cpu",
                             weights_only=True)
    for name, state in heads_state.items():
        head = _head_from_state(state, base.config.hidden_size)
        head.load_state_dict(state)
        model.heads[name] = head
    model.eval()
    return model, maps


class _ExportWrapper(nn.Module):
    """Maps the dict-of-heads forward to a single ordered tuple for ONNX."""

    def __init__(self, model: HierarchicalClassifier, head_order: list[str]) -> None:
        super().__init__()
        self.model = model
        self.head_order = head_order

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return tuple(logits[name] for name in self.head_order)


def export_onnx(
    pytorch_dir: Path,
    out_dir: Path,
    *,
    quantize: bool = True,
    opset: int = 17,
    dynamo: bool = False,
    dummy_seq_len: int = 64,
    max_seq_len: int = 8192,
) -> dict:
    """torch.onnx.export (dynamic batch+seq) + optional int8 quantize.

    ``dynamo=False`` pins the legacy TorchScript-tracer exporter: the torch
    2.14 Dynamo exporter currently emits an invalid ``Split`` node for this
    architecture (module docstring). Writes ``model.onnx`` (+
    ``model_quantized.onnx`` when quantize) and copies ``labels.json`` +
    tokenizer files so the bundle is self-contained.
    """
    import torch

    model, maps = build_reference_model(pytorch_dir)
    head_order = sorted(maps)
    wrapper = _ExportWrapper(model, head_order)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_input_ids = torch.zeros(1, dummy_seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, dummy_seq_len, dtype=torch.long)

    input_names = ["input_ids", "attention_mask"]
    output_names = [f"logits_{name}" for name in head_order]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        **{f"logits_{name}": {0: "batch"} for name in head_order},
    }

    fp32_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_mask),
        fp32_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=dynamo,
    )

    meta = {
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "exported_from": str(pytorch_dir),
        "heads": head_order,
        "opset": opset,
        "exporter": "torchscript-tracer" if not dynamo else "dynamo",
        "dynamic_axes": dynamic_axes,
        "max_seq_len": max_seq_len,
        "quantized": False,
    }

    quant_path = None
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quant_path = out_dir / "model_quantized.onnx"
        quantize_dynamic(str(fp32_path), str(quant_path),
                         weight_type=QuantType.QInt8)
        meta["quantized"] = True

        import onnxruntime as ort

        meta["onnxruntime_version"] = ort.__version__

    # self-contained bundle: label maps + tokenizer assets + model config
    shutil.copy2(pytorch_dir / "labels.json", out_dir / "labels.json")
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "config.json", "vocab.txt"):
        src = pytorch_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
    (out_dir / "export_meta.json").write_text(
        json.dumps(meta, sort_keys=True, indent=2))

    # Drop stale external-data sidecars from older exports (self-contained graphs
    # no longer reference *.onnx.data — mailroom-issues #145).
    stale_data = out_dir / "model.onnx.data"
    if stale_data.is_file():
        stale_data.unlink()

    sizes = {p.name: p.stat().st_size for p in out_dir.glob("*.onnx")}
    return {"output_dir": str(out_dir), "heads": head_order, "sizes": sizes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pytorch-dir", type=Path, default=_DEFAULT_PYTORCH_DIR,
                    help="trainer output: backbone files + heads.pt + labels.json")
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
                    help="artifact bundle (artifacts/onnx/model by default)")
    ap.add_argument("--no-quantize", action="store_true",
                    help="skip the int8 quantize step (fp32 model.onnx only)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--max-seq-len", type=int, default=8192,
                    help="ModernBERT native context; recorded in export_meta.json")
    args = ap.parse_args()

    if not (args.pytorch_dir / "heads.pt").is_file():
        raise SystemExit(
            f"no heads.pt under {args.pytorch_dir} — expected the trainer output "
            "(backbone files + heads.pt + labels.json). Copy one from the "
            "checkpoint Volume: `modal volume get modernbert-checkpoints latest/ ...`"
        )
    result = export_onnx(args.pytorch_dir, args.out_dir,
                         quantize=not args.no_quantize, opset=args.opset,
                         max_seq_len=args.max_seq_len)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
