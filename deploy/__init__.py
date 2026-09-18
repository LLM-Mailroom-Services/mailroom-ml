"""mailroom-ml deploy layer.

- modal_app.py:        Modal training app (mailroom-ml-train)
- serve_app.py:        fallback ONNX int8 serving app (mailroom-ml-serve)
- onnx_export.py:      torch.onnx.export + int8 quantization of the checkpoint
- onnx_parity_check.py: PyTorch-vs-ONNX logits parity gate (pytest marker "serve")
- README.md:           deployment runbook
"""