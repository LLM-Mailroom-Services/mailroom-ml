"""mailroom-ml — the Digital Mailroom's dedicated ML environment.

Hosts the ModernBERT ingest fast-path: dataset preparation (from the
``mailroom-finetune`` working copy of the pinned corpus), hierarchical
fine-tuning, calibration + selective-risk analysis, routing policy, ONNX CPU
serving, and the constrained synthetic-data program.

See ``docs/intake-classifier-combined-plan.md`` for the governing plan and
``governance/TASKS.md`` for the card board.
"""

__version__ = "0.1.0"
