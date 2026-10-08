#!/usr/bin/env python3
"""Fine-tune the hierarchical ModernBERT-base classifier (mailroom-ml trainer).

The documented training entrypoint of the mailroom-ml deploy layer:
``deploy/modal_app.py`` invokes this script as ``sys.executable
/root/training/train/train_modernbert.py --data <repo> --output
/checkpoints/runs/<run-id> --epochs N --batch-size N --grad-accum N --lr F
--seed N [--eval-test]`` (it does not pass ``--push-to-hub``: the app pushes
after ONNX export + parity).

Port of the committed predecessor (``Mailroom-Corpus-EDA @ cf096fa``,
``modernbert/train.py``), adapted to the mailroom-ml package:

- **Data**: the published ``config.TRAINING_DATA_REPO`` set (windows parquet
  for train/validation, document-level parquet for the held-out test) OR a
  local stage dir (``data/modernbert_training/stage`` layout).  The deploy
  layer validates the pinned revision pre-GPU and exports it as
  ``TRAINING_DATA_REVISION`` env, which is honored here (falling back to the
  committed ``config`` pin when unset — never train on an unpinned dataset).
- **Model**: one shared ModernBERT-base backbone + 6 heads (``doc_type`` with
  5 classes + ``unknown``, plus one subclass head per class) — the pipeline's
  existing conditional structure.  Heads are keyed by head name from the
  ``labels.json`` sidecar (the data-driven single source of truth).
- **Loss**: class-weighted cross-entropy per head (weights from
  ``labels.json``, inverse-frequency over the train split); the subclass
  head for class *c* fires only on rows whose doc_type is *c*.
- **Recipe (plan §7)**: AdamW bf16 (fp32 CPU), LR 2e-5, linear schedule with
  6% warmup, grad clip 1.0, grad-accum, seeded (random/np/torch + cudnn
  deterministic); early stop patience 2 on validation loss.
- **Eval**: per-head window metrics (acc, macro-F1, ECE) + document-level
  plurality-vote metrics (the sorter's merge); hardening seam recorded in the
  printed table + ``summary.json``: lexicographic checkpoint selection
  (#112 M9a-U4, ``--select-on-subclass`` default on) — among epochs whose
  calibrated doc_type ECE <= 0.05 and whose observed doc_type macro-F1 is
  within ``DOC_TYPE_GATE_TOL`` of the BEST eligible doc_type macro-F1 of the
  whole run, pick the one maximizing the mean observed subclass macro-F1;
  ``--no-select-on-subclass`` restores the legacy doc_type-only rule
  (both recorded, not enforced as an exit).  The selected epoch's weights are
  promoted into ``--output`` BEFORE ``--eval-test``, so the test metrics
  describe the shipped weights.
- **Calibration (plan §8)**: per-head temperature scaling on validation
  logits (scipy ``minimize_scalar``, bounded (0.05, 10.0)); heads with < 2
  rows or < 2 unique classes in val stay at T = 1.0.
- **Checkpoint**: backbone + tokenizer + ``heads.pt`` + ``labels.json``
  (sidecar copy) + ``temperatures.json`` + ``train_counts.json`` (authentic
  per-(doc_type, subclass) train-row counts — the routing gate's
  ``ROUTE_MIN_AUTHENTIC_SUPPORT`` data source) + ``summary.json`` (+ optional
  Hub push via ``--push-to-hub``, refused when no epoch met the selection
  gate unless ``--push-ungated``; the Modal app pushes after ONNX parity
  instead).
- **Test gate** (``--eval-test``): report-only — held-out document accuracy
  via the committed windower at eval time; the P0 thresholds (doc_type >=
  0.95 / subclass >= 0.75) belong to the eval harness, NOT this trainer:
  exit 0 unless a real error.

Usage:
    .venv/bin/python training/train/train_modernbert.py --epochs 5 --output data/modernbert_training/runs/run1
    .venv/bin/python training/train/train_modernbert.py --push-to-hub Lucius-Morningstar/mailroom-modernbert-classifier
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mailroom_ml.config import (
    DOC_TYPE_GATE_TOL,
    ECE_BUDGET,
    INPUT_CONSTRUCTION_VERSION,
    MAX_TOKENS,
    MODEL_ID,
    RUNS_DIR,
    TRAINING_DATA_REPO,
    TRAINING_DATA_REVISION,
    WINDOW_OVERLAP_TOKENS,
)
from mailroom_ml.labels import attach_trainable_fields, normalize_label_maps
from mailroom_ml.logit_adjust import log_prior_values
from mailroom_ml.windows import window_document

CE_IGNORE_INDEX = -100

DEFAULT_DATA = TRAINING_DATA_REPO
DEFAULT_OUTPUT = RUNS_DIR / "latest"


def _hub_revision() -> str | None:
    """Dataset revision for Hub pulls: deploy env wins, else the config pin."""
    return os.environ.get("TRAINING_DATA_REVISION") or TRAINING_DATA_REVISION


class HierarchicalClassifier(nn.Module):
    """Shared ModernBERT backbone + per-head classifiers.

    ``head_kind="linear"`` keeps the original single-linear heads;
    ``head_kind="mlp"`` uses the ModernBERT classification recipe (hidden
    SiLU MLP + dropout — what FlexBertForSequenceClassification ships).
    """

    def __init__(self, base_model, head_sizes: dict[str, int],
                 head_kind: str = "linear", head_dropout: float = 0.1):
        super().__init__()
        self.backbone = base_model
        hidden = base_model.config.hidden_size
        heads: dict[str, nn.Module] = {}
        for name, n in sorted(head_sizes.items()):
            if head_kind == "mlp":
                heads[name] = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Dropout(head_dropout),
                    nn.Linear(hidden, n),
                )
            else:
                heads[name] = nn.Linear(hidden, n, bias=True)
        self.heads = nn.ModuleDict(heads)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # ModernBERT has no pooler — use the <s> first-token embedding
        # (RoBERTa-style; the model card's classification recipe). Cast to
        # float32: the backbone runs bf16, the heads are fp32.
        pooled = out.last_hidden_state[:, 0].float()
        return {name: head(pooled) for name, head in self.heads.items()}


def _hub_data_glob(data: str, split: str) -> str:
    """The pinned ``hf://`` glob for a Hub dataset split.

    The revision MUST be embedded in the path as ``@<rev>``: the ``revision=``
    kwarg is IGNORED for ``hf://`` data_files globs, so a bare URL silently
    resolves ``main`` from the local/stale cache — measured 2026-09-20: a bare
    glob + revision kwarg loaded 4573 windows (the pre-clean, leaky stage)
    while the pinned revision holds 4499. ``@<rev>`` is honored.
    """
    cfg = "windows" if split != "test" else "documents"
    return f"hf://datasets/{data}@{_hub_revision()}/data/{cfg}/{split}/*.parquet"


def load_dataset(data: str, split: str) -> list[dict]:
    """Rows from the windows config (train/validation) or documents (test).

    A local path resolves the committed stage layout
    (``data/{windows|documents}/{split}/*.parquet``); anything else is an
    HF dataset repo id resolved through the published ``data`` folders via
    ``hf://`` data_files globs, honoring ``TRAINING_DATA_REVISION`` (deploy
    env or the committed config pin).
    """
    local = Path(data)
    if local.exists():
        import pandas as pd

        cfg = "windows" if split != "test" else "documents"
        files = sorted((local / "data" / cfg / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {cfg}/{split} parquet under {local}")
        frames = [pd.read_parquet(f) for f in files]
        return pd.concat(frames, ignore_index=True).to_dict("records")
    # Remote Hub repo: resolve the published data/<cfg>/<split> folders
    # directly via hf:// data_files globs (robust regardless of how the
    # datasets-server indexes the repo). The pinned revision is embedded in
    # the URL by _hub_data_glob — see that docstring for why.
    from datasets import load_dataset as hf_load_dataset

    glob = _hub_data_glob(data, split)
    ds = hf_load_dataset("parquet", split=split, data_files={split: glob})
    return [dict(r) for r in ds]


def tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    """Tokenize window text; rows carry doc_type/subclass labels.

    Fixed ``max_length`` padding (see make_batches — the dynamic-padding
    variant measured ~15% faster but hung mid-epoch-2 on the L4).
    """
    enc = tokenizer([r["text"] for r in rows], padding="max_length",
                    truncation=True, max_length=max_length, return_tensors="pt")
    return [{
        "input_ids": enc["input_ids"][i],
        "attention_mask": enc["attention_mask"][i],
        "doc_type": r["doc_type"],
        "subclass": r["subclass"],
        "filename": r["filename"],
    } for i, r in enumerate(rows)]


@dataclass
class LossConfig:
    """Loss-shaping knobs (2026-09-20 audit: loss rebalance + calibration).

    - ``lambda_dt``: doc_type share of the loss. The old summed loss gave
      doc_type 1/(1+n_subclass_heads) of the gradient — the subclass heads
      dominated the shared backbone. Weighted blend::
          loss = λ · CE_dt + (1-λ) · mean(CE_subclass_heads)
    - ``label_smoothing``: applied to the doc_type head only (calibration
      lever).
    - ``subclass_label_smoothing``: applied to each subclass CE head
      independently (#22). Default 0.0 preserves hard targets. Does not
      affect the doc_type head (``--label-smoothing`` stays separate).
    - ``weight_mode``: "inverse" (labels.json inverse-frequency, as before),
      "sqrt-inverse" (sqrt of the stored weights ≈ inverse-sqrt frequency —
      tames the rare-class amplification), "none" (uniform).
    - ``weight_cap``: clamp class weights to [1/cap, cap] after the mode
      transform (default 10× — a rare class never out-weights a common one
      by more than an order of magnitude).
    - ``subclass_loss_norm``: how each subclass head's weighted CE is
      reduced (#112 follow-up, 2026-09-28). "count" (default) divides the
      weighted sum by the head's row count in the micro-batch.
      "weighted-mean" is ``F.cross_entropy``'s default, which divides by the
      sum of the rows' class weights: with batch 4 a head usually sees 1-2
      of its own rows per micro-batch, so the class weights cancel (exactly,
      for one row) and ``--weight-mode`` / ``--weight-cap`` barely reach the
      subclass heads. That is why run-2 inverse, run-3 sqrt-inverse and
      M9a Arm B (inverse, cap 20) all left the contract head collapsed.
      Inverse-frequency weights average about 1.0 over the train rows
      (exactly 1.0 per document before the cap), so "count" keeps the
      expected loss scale. doc_type keeps the weighted mean.
    - ``subclass_logit_adjust``: logit-adjusted CE for the subclass heads
      (Menon et al. 2021): add ``tau * log(prior)`` to each head's logits
      inside the loss only, so rare classes need a larger margin to win in
      training and the raw logits used at inference are balanced. Priors
      come from the stored inverse-frequency weights (prior ∝ 1/w). Uses no
      per-row weights, so it cannot over-amplify a handful of rare rows the
      way a capped weight can. 0.0 (default) = off; pair 1.0 with
      ``weight_mode="none"``. Inference does **not** apply this term
      (argmax uses raw logits). To retune ``tau_eff`` on a trained
      checkpoint without a new run, eval
      ``--subclass-decode-logit-adjust d`` with ``tau_eff = tau_train - d``.
    """
    lambda_dt: float = 0.65
    label_smoothing: float = 0.0
    subclass_label_smoothing: float = 0.0
    weight_mode: str = "inverse"
    weight_cap: float = 10.0
    subclass_loss_norm: str = "count"
    subclass_logit_adjust: float = 0.0

    def transform_weights(self, weights: dict[str, float],
                          labels: list[str],
                          device: torch.device | None = None) -> torch.Tensor:
        out = []
        for label in labels:
            w = weights.get(label, 1.0)
            if self.weight_mode == "sqrt-inverse":
                w = math.sqrt(max(w, 1e-6))
            elif self.weight_mode == "none":
                w = 1.0
            out.append(min(max(w, 1.0 / self.weight_cap), self.weight_cap))
        # device is required on CUDA: F.cross_entropy's weight must live on
        # the same device as the logits (CPU-built weights crashed the first
        # GPU smoke, 2026-09-20).
        return torch.tensor(out, dtype=torch.float32, device=device)


def log_prior(weights: dict[str, float], labels: list[str],
              device: torch.device | None = None) -> torch.Tensor:
    """Per-label log train prior from inverse-frequency weights.

    ``class_weights`` stores ``N / (K * n_k)``, so ``n_k ∝ 1 / w_k``. A label
    with no stored weight has no train support; it gets the smallest prior
    seen, so the adjustment never favours it. Shared with decode-time
    adjust (``mailroom_ml.logit_adjust``) so residual ``tau`` is exact.
    """
    vals = log_prior_values(weights, labels)
    return torch.tensor(vals, dtype=torch.float32, device=device)


def param_groups(model, lr: float,
                 subclass_head_lr: float | None = None) -> list[dict]:
    """AdamW param groups: one group, or subclass heads on their own LR.

    The heads start from random init but shared the encoder's fine-tuning
    LR (2e-5). In M9a the imbalanced subclass heads gave a near-constant
    answer per head (Arm A: ``supply`` for 27/41 contracts, ``indenture``
    for 38/38 corporate records), i.e. they barely left init.
    ``subclass_head_lr`` (e.g. 1e-3) trains them faster; the backbone and
    the doc_type head keep ``lr``. The warmup/decay schedule scales both.
    """
    if not subclass_head_lr:
        return [{"params": list(model.parameters()), "lr": lr}]
    sub = [p for name, head in model.heads.items() if name != "doc_type"
           for p in head.parameters()]
    sub_ids = {id(p) for p in sub}
    rest = [p for p in model.parameters() if id(p) not in sub_ids]
    return [{"params": rest, "lr": lr},
            {"params": sub, "lr": subclass_head_lr}]


def _trainer_heads(maps: dict) -> dict:
    """Runtime head view: trainable label ids + routing-only set."""
    out: dict = {}
    for name, cfg in maps.items():
        routing_only = set(cfg.get("routing_only", ()))
        out[name] = {
            "labels": list(cfg["trainable_labels"]),
            "label2id": dict(cfg["trainable_label2id"]),
            "weights": cfg["weights"],
            "routing_only": routing_only,
            "inference_only": set(cfg.get("inference_only", ())),
        }
    return out


def _head_sizes(maps: dict) -> dict[str, int]:
    return {name: len(cfg["trainable_labels"]) for name, cfg in maps.items()}


def _label_id(head: dict, label: str) -> int:
    if label in head.get("routing_only", ()):
        return CE_IGNORE_INDEX
    return head["label2id"][label]


def head_loss(model, batch, heads, device,
              cfg: LossConfig | None = None) -> tuple[torch.Tensor, dict]:
    """doc_type CE on every row + subclass CE on each class's own rows.

    Subclass heads are taken from the ``heads`` config (built from
    ``labels.json`` — the data-driven single source of truth), so each class
    head contributes only through its own rows: 0 contribution elsewhere.
    """
    cfg = cfg or LossConfig()
    logits = model(batch["input_ids"], batch["attention_mask"])
    dt_ce = F.cross_entropy(
        logits["doc_type"], batch["doc_type"],
        weight=cfg.transform_weights(heads["doc_type"]["weights"],
                                     heads["doc_type"]["labels"],
                                     device=logits["doc_type"].device),
        label_smoothing=cfg.label_smoothing,
        ignore_index=CE_IGNORE_INDEX)
    sc_ces: list[torch.Tensor] = []
    for cls in heads:
        if cls == "doc_type":
            continue
        sel = batch["doc_type"] == heads["doc_type"]["label2id"][cls]
        if sel.any():
            target = batch["subclass"][sel]
            weight = cfg.transform_weights(heads[cls]["weights"],
                                           heads[cls]["labels"],
                                           device=logits[cls].device)
            # routing-only rows only (contract `other` -> CE_IGNORE_INDEX):
            # nothing to learn.  Checked for BOTH reductions — the
            # weighted-mean path used to hit 0/0 -> NaN here, which the
            # divergence guard then turned into an aborted run.
            n_valid = int((target != CE_IGNORE_INDEX).sum())
            if n_valid == 0:
                continue
            sc_logits = logits[cls][sel]
            if cfg.subclass_logit_adjust:
                sc_logits = sc_logits + cfg.subclass_logit_adjust * log_prior(
                    heads[cls]["weights"], heads[cls]["labels"],
                    device=sc_logits.device)
            if cfg.subclass_loss_norm == "weighted-mean":
                sc_ces.append(F.cross_entropy(
                    sc_logits, target, weight=weight,
                    label_smoothing=cfg.subclass_label_smoothing,
                    ignore_index=CE_IGNORE_INDEX))
                continue
            sc_ces.append(F.cross_entropy(
                sc_logits, target, weight=weight,
                label_smoothing=cfg.subclass_label_smoothing,
                ignore_index=CE_IGNORE_INDEX, reduction="sum") / n_valid)
    if sc_ces:
        loss = cfg.lambda_dt * dt_ce + (1.0 - cfg.lambda_dt) * torch.stack(
            sc_ces).mean()
    else:
        loss = dt_ce
    return loss, logits


def _collate_batch(sel: list[dict], heads, device: torch.device,
                   non_blocking: bool) -> dict:
    to = {"non_blocking": non_blocking} if device.type == "cuda" else {}
    return {
        "input_ids": torch.stack([r["input_ids"] for r in sel]).to(device, **to),
        "attention_mask": torch.stack([r["attention_mask"] for r in sel]).to(
            device, **to),
        "doc_type": torch.tensor(
            [_label_id(heads["doc_type"], r["doc_type"]) for r in sel],
            device=device),
        "subclass": torch.tensor(
            [_label_id(heads[r["doc_type"]], r["subclass"]) for r in sel],
            device=device),
        "filename": [r["filename"] for r in sel],
    }


def _prefetch_batches(it, depth: int):
    """Overlap CPU collate/H2D with GPU forward (pre-tokenized rows, no DataLoader).

    A producer exception is captured and RE-RAISED in the consumer: the old
    ``finally: put(sentinel)`` swallowed it, so a collate failure (e.g. a
    label missing from a head map) silently truncated the epoch and the run
    exited 0 with a short, mis-scored epoch.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    _sentinel = object()
    failure: list[BaseException] = []

    def _worker() -> None:
        try:
            for batch in it:
                q.put(batch)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            failure.append(exc)
        finally:
            q.put(_sentinel)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is _sentinel:
            if failure:
                raise failure[0]
            return
        yield item


def _epoch_shuffle_rng(seed: int, epoch: int) -> random.Random:
    """Deterministic per-epoch shuffle stream.

    The old ``random.shuffle`` drew from the process-global RNG seeded once at
    start, so a ``--resume``d epoch k saw a different permutation than the
    original run's epoch k (and mid-epoch ``skip_micro`` skipped the WRONG
    rows).  Keying the stream on ``(seed, epoch)`` makes epoch k's order
    independent of how many epochs ran before it.  A string seed is hashed
    (sha512) by ``random.Random`` -> stable across processes.
    """
    return random.Random(f"{seed}:{epoch}")


def make_batches(rows, batch_size: int, shuffle: bool, heads, device,
                 *, prefetch_batches: int = 0, non_blocking: bool = False,
                 shuffle_rng: random.Random | None = None):
    """Yield batches (rows are pre-padded to max_length by tokenize_rows).

    FIXED padding (not dynamic): the dynamic-padding variant (pad to the
    batch's longest row) measured only ~15% faster — 75% of windows sit
    near the 8,192 cap, so most shuffled batches still pad near-full — and
    the variable-length sdpa + gradient-checkpointing path hung mid-epoch-2
    on the L4 (2026-09-19). Fixed padding is the proven-stable config.

    ``shuffle_rng`` (see ``_epoch_shuffle_rng``) makes the permutation a pure
    function of (seed, epoch); without it the global RNG is used.
    """
    idx = list(range(len(rows)))
    if shuffle:
        (shuffle_rng or random).shuffle(idx)

    def _produce():
        for i in range(0, len(idx), batch_size):
            sel = [rows[j] for j in idx[i:i + batch_size]]
            yield _collate_batch(sel, heads, device, non_blocking)

    base = _produce()
    if prefetch_batches > 0:
        yield from _prefetch_batches(base, prefetch_batches)
    else:
        yield from base


def _fmt_duration(seconds: float | None) -> str:
    """Human-readable duration for ETA / wall lines (e.g. ``1h 02m 03s``)."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _append_jsonl(path: Path, row: dict) -> None:
    """Append one machine-parseable JSON object (durable step/epoch log)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


class TrainProgress:
    """Step cadence tracker: throughput + ETA remaining wall time.

    ETA = mean sec/micro-step (EMA) × remaining planned micro-steps. Early
    stop may finish sooner; the printed ETA is an upper-bound estimate for
    the requested epoch budget.
    """

    def __init__(self, *, total_micro_planned: int, batch_size: int,
                 step_log: Path | None = None, run_t0: float | None = None,
                 micro_done_before: int = 0):
        self.total_micro_planned = max(0, int(total_micro_planned))
        self.batch_size = max(1, int(batch_size))
        self.step_log = step_log
        self.run_t0 = run_t0 if run_t0 is not None else time.time()
        self.micro_done = max(0, int(micro_done_before))
        self._last_t = self.run_t0
        self._ema_sec_per_step: float | None = None

    def note_window(self, *, n_steps: int, epoch: int, epochs: int,
                    step_in_epoch: int, loss: float,
                    n_samples: int) -> dict:
        """Record a window of ``n_steps`` micro-batches ending at this log."""
        n_steps = max(1, int(n_steps))
        now = time.time()
        dt = max(1e-6, now - self._last_t)
        self._last_t = now
        self.micro_done += n_steps
        sec_per = dt / n_steps
        if self._ema_sec_per_step is None:
            self._ema_sec_per_step = sec_per
        else:
            self._ema_sec_per_step = 0.2 * sec_per + 0.8 * self._ema_sec_per_step
        steps_per_sec = 1.0 / self._ema_sec_per_step
        samples_per_sec = steps_per_sec * max(1, n_samples)
        remaining = max(0, self.total_micro_planned - self.micro_done)
        eta_s = remaining * self._ema_sec_per_step
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "event": "step",
            "epoch": epoch,
            "epochs": epochs,
            "step": step_in_epoch,
            "micro_done": self.micro_done,
            "micro_planned": self.total_micro_planned,
            "loss": round(float(loss), 6),
            "steps_per_sec": round(steps_per_sec, 4),
            "samples_per_sec": round(samples_per_sec, 2),
            "eta_s": round(eta_s, 1),
            "eta": _fmt_duration(eta_s),
            "wall_s": round(now - self.run_t0, 1),
        }
        if self.step_log is not None:
            _append_jsonl(self.step_log, row)
        return row

    def bump_unlogged(self, n: int) -> None:
        """Advance counters for trailing steps with no printed log line."""
        n = max(0, int(n))
        if n == 0:
            return
        now = time.time()
        dt = max(1e-6, now - self._last_t)
        self._last_t = now
        self.micro_done += n
        sec_per = dt / n
        if self._ema_sec_per_step is None:
            self._ema_sec_per_step = sec_per
        else:
            self._ema_sec_per_step = 0.2 * sec_per + 0.8 * self._ema_sec_per_step

    def eta_s(self) -> float | None:
        if self._ema_sec_per_step is None:
            return None
        remaining = max(0, self.total_micro_planned - self.micro_done)
        return remaining * self._ema_sec_per_step


def _epoch_metrics_tsv_header() -> str:
    return "\t".join([
        "epoch", "train_loss", "train_loss_endpoint", "val_loss",
        "doc_type_macro_f1_obs", "doc_type_ece", "doc_type_ece_cal",
        "subclass_objective", "selected", "gate_met",
        "checkpoint", "archive", "epoch_wall_s", "eta_remaining",
    ])


def _write_epoch_metrics(epoch_jsonl: Path, epoch_tsv: Path, row: dict) -> None:
    """Durable epoch table: JSONL (full) + TSV (operator glance)."""
    _append_jsonl(epoch_jsonl, row)
    write_header = not epoch_tsv.exists()
    with epoch_tsv.open("a", encoding="utf-8") as fh:
        if write_header:
            fh.write(_epoch_metrics_tsv_header() + "\n")
        fh.write("\t".join([
            str(row.get("epoch", "")),
            str(row.get("loss", "")),
            str(row.get("loss_endpoint", "")),
            str(row.get("val_loss", "")),
            str(row.get("doc_type_macro_f1_observed", "")),
            str(row.get("doc_type_ece", "")),
            str(row.get("doc_type_ece_calibrated", "")),
            str(row.get("subclass_objective", "")),
            str(row.get("selected_this_epoch", "")),
            str(row.get("gate_met", "")),
            str(row.get("checkpoint", "")),
            str(row.get("archive", "")),
            str(row.get("epoch_wall_s", "")),
            str(row.get("eta_remaining", "")),
        ]) + "\n")


def train_epoch(model, batches, optimizer, scheduler, heads, device,
                grad_accum: int, step_limit: int = 0,
                log_every: int = 50,
                loss_cfg: LossConfig | None = None,
                progress: TrainProgress | None = None,
                epoch: int = 0, epochs: int = 0,
                skip_micro: int = 0,
                on_micro_step=None) -> tuple[float, float, int, int]:
    """One epoch. Returns (mean loss, endpoint loss, micro-steps, opt-steps).

    Guardrails:
    - ``log_every``: per-step progress line so a stalled/starved container is
      visible in ``modal app logs`` within seconds instead of at epoch end.
      When ``progress`` is set, lines also include steps/sec, samples/sec, ETA.
    - ``step_limit``: cap micro-batches for the pre-flight smoke run (smoke
      exercises forward+backward+optimizer without burning an epoch).
    - Endpoint loss: the mean over the last ``grad_accum`` micro-batches —
      the honest "where did the epoch END" number. The whole-epoch mean
      buries convergence signal under warmup + early high-loss steps (the
      old run's train≫val gap was mostly this measurement artifact).
    - Stale-gradient flush: when the epoch's micro-batch count is not a
      multiple of ``grad_accum``, the trailing micro-batches used to
      accumulate gradients that were never optimized AND contaminated the
      next epoch's first step. A partial optimizer step now flushes them.
    - Absolute accumulation index: the optimizer-step boundary AND the
      trailing flush both key on the micro-batch's ABSOLUTE index in the
      epoch (``step``), never on the post-``skip_micro`` count ``n`` — a
      mid-epoch resume whose skip is not a multiple of ``grad_accum`` used
      to flush at the wrong place (mis-sized accumulation windows).
    """
    cfg = loss_cfg or LossConfig()
    model.train()
    total, n = 0.0, 0
    opt_steps = 0
    window: list[float] = []
    since_log = 0
    skip_micro = max(0, int(skip_micro))
    last_abs = 0  # absolute 1-based index of the last micro-batch trained on
    for step, batch in enumerate(batches):
        if step < skip_micro:
            continue
        loss, _ = head_loss(model, batch, heads, device, cfg)
        # divergence guard (2026-09-20 audit R6): a non-finite loss would
        # otherwise propagate NaN weights into a checkpoint that gets pushed.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at micro-step {step + 1} "
                f"({loss.item()}) — aborting before a NaN checkpoint is saved")
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_steps += 1
        total += loss.item()
        n += 1
        last_abs = step + 1
        since_log += 1
        window.append(loss.item())
        if len(window) > grad_accum:
            window.pop(0)
        if (step + 1) % log_every == 0:
            n_samples = int(batch["input_ids"].shape[0])
            if progress is not None:
                row = progress.note_window(
                    n_steps=since_log, epoch=epoch, epochs=epochs,
                    step_in_epoch=step + 1, loss=loss.item(),
                    n_samples=n_samples)
                print(
                    f"  step {step + 1} loss {loss.item():.4f} "
                    f"{row['steps_per_sec']:.2f} steps/s "
                    f"{row['samples_per_sec']:.1f} samples/s "
                    f"ETA {_fmt_duration(row['eta_s'])}",
                    flush=True)
            else:
                print(f"  step {step + 1} loss {loss.item():.4f}", flush=True)
            since_log = 0
        if on_micro_step is not None:
            on_micro_step(step + 1, opt_steps, loss.item())
        if step_limit and (step + 1) - skip_micro >= step_limit:
            break
    if progress is not None and since_log > 0:
        progress.bump_unlogged(since_log)
    if last_abs % grad_accum != 0:  # flush trailing accumulation (partial step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        opt_steps += 1
    endpoint = sum(window) / max(1, len(window))
    return total / max(1, n), endpoint, n, opt_steps


@torch.no_grad()
def evaluate(model, batches, heads, maps, device,
             loss_cfg: LossConfig | None = None) -> tuple[dict, dict, dict]:
    """Per-head window metrics + document-level plurality-vote metrics.

    The document vote mirrors the sorter's merge: doc_type by plurality over
    windows; subclass by plurality over windows whose doc_type vote is the
    winning class (the pipeline's conditional structure).  Windows whose
    doc_type vote falls on the inference-only ``unknown`` class carry no
    subclass vote (the sorter routes them to the LLM instead) — counted as a
    doc_type miss when the label differs, never a crash.

    Returns ``(metrics, logits_by_head, labels_by_head)`` — the logits feed
    temperature fitting so the selection gate can use the CALIBRATED ECE
    (the old gate compared raw T=1 ECE, which no overconfident head can
    pass — the gate was structurally unreachable).
    """
    cfg = loss_cfg or LossConfig()
    model.eval()
    logits_by_head: dict[str, list] = defaultdict(list)
    labels_by_head: dict[str, list] = defaultdict(list)
    doc_votes: dict[str, list] = defaultdict(list)  # fn -> [(dt_pred, sc_pred)]
    doc_labels: dict[str, tuple] = {}
    total_loss = 0.0
    n_batches = 0
    for batch in batches:
        loss, logits = head_loss(model, batch, heads, device, cfg)
        total_loss += loss.item()
        n_batches += 1
        dt_preds = logits["doc_type"].argmax(-1)
        sc_preds = {name: logits[name].argmax(-1)
                    for name in heads if name != "doc_type"}
        for name, lg in logits.items():
            if name == "doc_type":
                lab = batch["doc_type"]
                keep = torch.ones_like(lab, dtype=torch.bool)
            else:
                # subclass heads are scored only on their own class's rows
                lab = batch["subclass"]
                keep = batch["doc_type"] == heads["doc_type"]["label2id"][name]
            # routing-only GT rows (contract `other`) carry CE_IGNORE_INDEX:
            # they are not scorable for a trainable head, and a -100 label
            # would corrupt every metric (and index out of range in the
            # temperature fit), so they leave the per-head arrays here.
            keep = keep & (lab != CE_IGNORE_INDEX)
            logits_by_head[name].append(lg[keep].cpu())
            labels_by_head[name].append(lab[keep].cpu())
        for i, fn in enumerate(batch["filename"]):
            dt_p = dt_preds[i].item()
            cls = maps["doc_type"]["trainable_id2label"][str(dt_p)]
            sc_p = sc_preds[cls][i].item() if cls in sc_preds else None
            doc_votes[fn].append((dt_p, sc_p))
            doc_labels[fn] = (batch["doc_type"][i].item(),
                              batch["subclass"][i].item())
    metrics = {"val_loss": total_loss / max(1, n_batches)}
    for name in sorted(logits_by_head):
        lg = torch.cat(logits_by_head[name])
        lab = torch.cat(labels_by_head[name])
        metrics[f"{name}_window_acc"] = round(
            (lg.argmax(-1) == lab).float().mean().item(), 4)
        metrics[f"{name}_ece"] = round(ece(lg, lab), 4)
        metrics[f"{name}_macro_f1"] = round(macro_f1(lg, lab), 4)
        metrics[f"{name}_macro_f1_observed"] = round(
            macro_f1(lg, lab, observed_only=True), 4)
    dt_correct = sc_correct = sc_unscorable = 0
    for fn, votes in doc_votes.items():
        dt_label, sc_label = doc_labels[fn]
        dt_pred = Counter(v[0] for v in votes).most_common(1)[0][0]
        if dt_pred == dt_label:
            dt_correct += 1
            if sc_label == CE_IGNORE_INDEX:
                # routing-only GT subclass (contract `other`): the head cannot
                # predict it, so the doc leaves the subclass denominator
                sc_unscorable += 1
                continue
            cond = [v[1] for v in votes if v[0] == dt_pred and v[1] is not None]
            if not cond:
                continue
            sc_pred = Counter(cond).most_common(1)[0][0]
            if sc_pred == sc_label:  # both ids in head `cls`'s space
                sc_correct += 1
    metrics["doc_type_doc_acc"] = round(dt_correct / max(1, len(doc_votes)), 4)
    metrics["subclass_doc_acc"] = round(
        sc_correct / max(1, dt_correct - sc_unscorable), 4)
    return metrics, logits_by_head, labels_by_head


def _drop_ignored(logits: torch.Tensor,
                  labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop ``CE_IGNORE_INDEX`` (routing-only) rows before any metric/fit.

    A ``-100`` label is "not scorable", not "class -100": left in, it counts
    as a miss in ECE/F1 and indexes out of range in the temperature fit.
    """
    keep = labels != CE_IGNORE_INDEX
    if bool(keep.all()):
        return logits, labels
    return logits[keep], labels[keep]


def _ece_from_probs(probs: torch.Tensor, labels: torch.Tensor,
                    n_bins: int = 10) -> float:
    """ECE binning over already-softmaxed probabilities (shared core)."""
    probs, labels = _drop_ignored(probs, labels)
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo)
        if sel.sum() == 0:
            continue
        total += (sel.sum().item() / len(conf)) * abs(
            correct[sel].mean().item() - conf[sel].mean().item())
    return total


def ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Raw (T=1) Expected Calibration Error over equal-width bins."""
    return _ece_from_probs(F.softmax(logits, dim=-1), labels, n_bins)


def ece_calibrated(logits: torch.Tensor, labels: torch.Tensor,
                   temperature: float, n_bins: int = 10) -> float:
    """ECE after temperature scaling — the number the deployment gate
    should compare against (the old gate used raw T=1 ECE, which is
    structurally unreachable for any overconfident head)."""
    return _ece_from_probs(F.softmax(logits / temperature, dim=-1),
                           labels, n_bins)


def macro_f1(logits: torch.Tensor, labels: torch.Tensor,
             observed_only: bool = False) -> float:
    logits, labels = _drop_ignored(logits, labels)
    preds = logits.argmax(-1)
    n_classes = logits.shape[1]
    f1s = []
    for c in range(n_classes):
        if observed_only and (labels == c).sum().item() == 0:
            continue  # zero-row classes (e.g. inference-only `unknown`)
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _subclass_objective(val: dict, subclass_heads: list[str]) -> float:
    """Mean observed macro-F1 across the subclass heads (#112 M9a-U4).

    The lexicographic selection objective: higher = the conditional subclass
    heads are collectively better.  Heads absent from ``val`` are ignored; an
    empty head set scores 0.0 (no subclass evidence).
    """
    vals = [val[f"{h}_macro_f1_observed"] for h in subclass_heads
            if f"{h}_macro_f1_observed" in val]
    return float(np.mean(vals)) if vals else 0.0


def _selection_snapshot(val: dict, epoch: int, subclass_heads: list[str]) -> dict:
    """The selected-epoch record persisted under ``checkpoint_selection``.

    Carries the legacy fields (epoch, doc_type macro-F1/ECE, per-head
    calibrated-ECE sidecar) plus the #112 per-head observed macro-F1 and the
    subclass objective the lexicographic rule maximized.
    """
    return {
        "epoch": epoch,
        "macro_f1": val["doc_type_macro_f1_observed"],
        "ece": val["doc_type_ece_calibrated"],
        "ece_raw": val.get("doc_type_ece"),
        "subclass_objective": _subclass_objective(val, subclass_heads),
        "per_head_macro_f1_observed": {
            name[:-len("_macro_f1_observed")]: val[name]
            for name in sorted(val)
            if name.endswith("_macro_f1_observed")},
        "per_head_ece_calibrated": {
            name[:-len("_ece_calibrated")]: val[name]
            for name in sorted(val)
            if name.endswith("_ece_calibrated")},
    }


_NO_SELECTION: dict = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                       "subclass_objective": -1.0}


def _select_epoch(events: list[dict], subclass_heads: list[str], *,
                  select_on_subclass: bool,
                  prior: dict | None = None) -> tuple[dict, float]:
    """The lexicographic checkpoint-selection decision (#112 M9a-U4).

    Pure seam: given EVERY epoch's validation event (``events`` — each the
    ``val`` dict plus ``epoch``, including the already-computed
    ``subclass_objective``), return ``(selected snapshot, best_doc_type)``.

    The decision is recomputed over the whole history, not folded in epoch by
    epoch.  The gate is "observed doc_type macro-F1 no more than
    ``DOC_TYPE_GATE_TOL`` below the BEST ECE-eligible doc_type macro-F1 of the
    run" — that is the rule written into ``summary.json``.  The previous
    incremental fold only compared an epoch against the best seen SO FAR, so
    a later epoch that raised the best left an earlier selection sitting
    below ``best - DOC_TYPE_GATE_TOL`` and the shipped checkpoint violated
    its own rule.

    With ``select_on_subclass``: among epochs with CALIBRATED doc_type ECE
    ``<= ECE_BUDGET`` and doc_type macro-F1 ``>= best - DOC_TYPE_GATE_TOL``,
    pick the max subclass objective (ties -> the earliest epoch).  Otherwise
    the legacy doc_type-only rule: the max doc_type macro-F1 among
    ECE-eligible epochs (ties -> the earliest).  No ECE-eligible epoch ->
    the ``epoch 0`` sentinel (gate not met) and a ``-inf`` floor.

    ``prior`` is a resumed run's recorded selection: a legacy bundle whose
    epoch events lack the per-epoch selection keys still competes through
    its snapshot (a ``None`` objective counts as ``-1.0``).  ``events`` is
    never mutated.
    """
    cands: list[dict] = []
    keyed_epochs: set[int] = set()
    for ev in events:
        if ev.get("doc_type_macro_f1_observed") is None \
                or ev.get("doc_type_ece_calibrated") is None:
            continue  # legacy event without selection keys
        keyed_epochs.add(ev["epoch"])
        if ev["doc_type_ece_calibrated"] <= ECE_BUDGET:
            snap = _selection_snapshot(ev, ev["epoch"], subclass_heads)
            if ev.get("subclass_objective") is not None:
                snap["subclass_objective"] = ev["subclass_objective"]
            cands.append(snap)
    if prior and prior.get("epoch") and prior.get("macro_f1") is not None \
            and prior["epoch"] not in keyed_epochs:
        legacy = dict(prior)
        if legacy.get("subclass_objective") is None:
            legacy["subclass_objective"] = -1.0
        cands.append(legacy)
    if not cands:
        return dict(_NO_SELECTION), float("-inf")
    cands.sort(key=lambda c: c["epoch"])
    best_doc_type = max(c["macro_f1"] for c in cands)
    if select_on_subclass:
        pool = [c for c in cands
                if c["macro_f1"] >= best_doc_type - DOC_TYPE_GATE_TOL]
        key = "subclass_objective"
    else:
        pool = cands
        key = "macro_f1"
    chosen = pool[0]
    for c in pool[1:]:
        if c[key] > chosen[key]:
            chosen = c
    return chosen, best_doc_type


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Platt-style temperature scaling: T minimizing NLL on validation."""
    from scipy.optimize import minimize_scalar

    logits, labels = _drop_ignored(logits.detach(), labels)
    lg, lab = logits.float().numpy(), labels.numpy()

    def nll(t: float) -> float:
        if t <= 1e-3:
            return 1e9
        z = lg / t
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(lab)), lab].mean()

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def _commit_checkpoint_volume() -> None:
    """Persist volume writes so a killed run keeps its per-epoch checkpoints.

    Only meaningful inside the Modal container (env set by
    ``deploy/modal_app.py``): the trainer's ``/checkpoints`` writes are
    uncommitted volume changes until ``commit()`` — a kill/timeout before
    the app-level commit would silently lose every epoch. A failed commit
    never fails training (the app-level commit still runs on success).
    """
    name = os.environ.get("MAILROOM_ML_CHECKPOINT_VOLUME")
    if not name:
        return
    try:
        import modal  # noqa: PLC0415 — modal is only present in the container

        modal.Volume.from_name(name).commit()
        print(f"[trainer] volume committed: {name}", flush=True)
    except Exception as exc:  # noqa: BLE001 — commit must never kill training
        print(f"[trainer] volume commit failed: {exc}", flush=True)


def _write_resume_manifest(
        output: Path, *, checkpoint_dir: Path, epoch: int,
        step_in_epoch: int, steps_done: int, steps_done_micro: int,
        run_id: str, epoch_complete: bool) -> None:
    """Operator-facing pointer: latest resumable bundle + exact CLI flags."""
    output.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "epoch_complete": epoch_complete,
        "steps_done": steps_done,
        "steps_done_micro": steps_done_micro,
        "run_id": run_id,
        "resume_flag": f"--resume {checkpoint_dir.resolve()}",
    }
    (output / "resume_manifest.json").write_text(
        json.dumps(row, sort_keys=True, indent=2))


def save_checkpoint(output: Path, model, tokenizer, maps, heads, train_rows,
                    temps: dict, summary: dict, *,
                    optimizer=None, scheduler=None, epoch: int = 0,
                    steps_done: int = 0, steps_done_micro: int = 0,
                    step_in_epoch: int = 0, epoch_complete: bool = False) -> None:
    """Write the full checkpoint bundle (backbone + heads + sidecars).

    When ``optimizer`` is given, the optimizer/scheduler state and a
    ``resume.json`` counter file are written too, so a later ``--resume`` can
    continue training from this exact point (not just reload weights).
    ``steps_done`` counts OPTIMIZER steps (the scheduler's unit — the
    2026-09-20 audit fixed the micro-batch/optimizer-step mismatch);
    ``steps_done_micro`` is the micro-batch count for --max-steps smoke
    accounting.  Mid-epoch saves set ``epoch_complete=False`` and a non-zero
    ``step_in_epoch`` so ``--resume`` can skip already-trained micro-batches.
    """
    output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.save({name: head.state_dict() for name, head in model.heads.items()},
               output / "heads.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), output / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), output / "scheduler.pt")
        resume_body = {
            "epoch": epoch,
            "steps_done": steps_done,
            "steps_done_micro": steps_done_micro,
            "step_in_epoch": step_in_epoch,
            "epoch_complete": epoch_complete,
            "run_id": summary.get("run_id", ""),
        }
        (output / "resume.json").write_text(
            json.dumps(resume_body, sort_keys=True, indent=2))
        _write_resume_manifest(
            output, checkpoint_dir=output, epoch=epoch,
            step_in_epoch=step_in_epoch, steps_done=steps_done,
            steps_done_micro=steps_done_micro,
            run_id=summary.get("run_id", ""), epoch_complete=epoch_complete)
    (output / "labels.json").write_text(
        json.dumps(maps, sort_keys=True, indent=2))
    # authentic-support sidecar: per (doc_type, subclass) train-row counts —
    # the ROUTE_MIN_AUTHENTIC_SUPPORT gate's data source (inference.py reads
    # train_counts.json; absent sidecar -> gate fails open to the LLM path).
    # Counts reflect the rows actually trained on (post --limit).
    train_counts: dict[str, dict[str, int]] = {}
    for r in train_rows:
        train_counts.setdefault(r["doc_type"], {}).setdefault(r["subclass"], 0)
        train_counts[r["doc_type"]][r["subclass"]] += 1
    (output / "train_counts.json").write_text(
        json.dumps(train_counts, sort_keys=True, indent=2))
    (output / "temperatures.json").write_text(
        json.dumps(temps, sort_keys=True, indent=2))
    (output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2))


def _write_ood_probe(output: Path, val_logits: dict) -> None:
    """Fit the energy OOD probe on validation doc_type logits (#18).

    Validation only — never the held-out test (plan D11).  Missing or
    empty logits leave any existing sidecar untouched.
    """
    rows = val_logits.get("doc_type") if val_logits else None
    if not rows:
        return
    stacked = torch.cat(rows).detach().cpu().numpy()
    if stacked.size == 0:
        return
    from mailroom_ml.ood import fit_ood_probe, write_ood_probe

    write_ood_probe(output / "ood_probe.json", fit_ood_probe(stacked))


def _apply_resume(resume_dir: Path, model, optimizer, scheduler, device,
                  n_train_rows: int, batch_size: int,
                  grad_accum: int) -> dict:
    """Load a checkpoint bundle for ``--resume``; returns the run state.

    Handles bundles saved before optimizer state existed (the 2026-09-19
    per-epoch checkpoints): the optimizer is reconstructed fresh and the
    scheduler is stepped to the right position. Counters come from
    ``resume.json`` when present, else ``summary.json``'s ``epochs_run``.
    ``steps_done`` is in OPTIMIZER steps (the scheduler's unit — the
    2026-09-20 audit fixed the micro-batch/optimizer-step mismatch).
    """
    heads_state = torch.load(resume_dir / "heads.pt", map_location=device)
    for name, sd in heads_state.items():
        if name not in model.heads:
            raise RuntimeError(f"checkpoint head {name!r} not in the model")
        model.heads[name].load_state_dict(sd)
    resume_json = resume_dir / "resume.json"
    micro_skip = 0
    if resume_json.exists():
        rj = json.loads(resume_json.read_text())
        epoch_done, steps_done = rj["epoch"], rj["steps_done"]
        steps_done_micro = rj.get("steps_done_micro", 0)
        step_in_epoch = int(rj.get("step_in_epoch", 0))
        epoch_complete = bool(rj.get("epoch_complete", step_in_epoch == 0))
        if step_in_epoch > 0 and not epoch_complete:
            micro_skip = step_in_epoch
    else:
        summary_path = resume_dir / "summary.json"
        epoch_done = (json.loads(summary_path.read_text()).get("epochs_run", 0)
                      if summary_path.exists() else 0)
        steps_done = epoch_done * math.ceil(
            math.ceil(n_train_rows / batch_size) / grad_accum)
        steps_done_micro = epoch_done * math.ceil(n_train_rows / batch_size)
    opt_path = resume_dir / "optimizer.pt"
    sched_path = resume_dir / "scheduler.pt"
    opt_state = (torch.load(opt_path, map_location=device)
                 if opt_path.exists() else None)
    if opt_state is not None and len(opt_state["param_groups"]) != len(
            optimizer.param_groups):
        # The bundle was saved with a different param-group layout (e.g. a
        # pre-``--subclass-head-lr`` single-group run resumed with the flag,
        # or the reverse). Loading would raise, and the saved LambdaLR
        # base_lrs would drop a group's LR. Keep the fresh optimizer and
        # step the scheduler into position, as for a legacy bundle.
        print(f"resume: optimizer has {len(optimizer.param_groups)} param "
              f"group(s), checkpoint has {len(opt_state['param_groups'])}; "
              "starting fresh optimizer/scheduler state")
        opt_state = None
        sched_path = resume_dir / "_no_scheduler_state"
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)
    if sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
    else:
        for _ in range(steps_done):
            scheduler.step()
    events: list[dict] = []
    selected: dict = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                      "subclass_objective": -1.0}
    temps: dict[str, float] = {}
    prior_run_id: str | None = None
    prior_wall_s = 0.0
    summary_path = resume_dir / "summary.json"
    if summary_path.exists():
        sj = json.loads(summary_path.read_text())
        events = sj.get("epochs", [])
        sel = sj.get("checkpoint_selection", {})
        if sel.get("epoch"):
            selected = sel
        temps = sj.get("temperatures", {})
        # provenance: the resumed epochs belong to the ORIGINAL run — carry its
        # identity + accumulated wall so an eval-only resume does not rewrite
        # the artifact's summary as a fresh zero-second run.
        prior_run_id = sj.get("run_id")
        prior_wall_s = float(sj.get("training_wall_s") or 0.0)
    best_val = float("inf")
    stale = 0
    for e in events:
        if e["val_loss"] < best_val:
            best_val = e["val_loss"]
            stale = 0
        else:
            stale += 1
    # #112 doc_type gate floor: the best observed doc_type macro-F1 among
    # epochs whose calibrated ECE cleared the budget (legacy events lack the
    # key -> ignored, so the resumed run starts with an open gate).
    best_doc_type = float("-inf")
    for e in events:
        dt = e.get("doc_type_macro_f1_observed")
        ece = e.get("doc_type_ece_calibrated")
        if dt is not None and ece is not None and ece <= ECE_BUDGET:
            best_doc_type = max(best_doc_type, dt)
    start_epoch = epoch_done if micro_skip else epoch_done + 1
    return {"start_epoch": start_epoch, "steps_done": steps_done,
            "steps_done_micro": steps_done_micro, "micro_skip": micro_skip,
            "events": events, "selected": selected, "best_val": best_val,
            "stale": stale, "temps": temps,
            "run_id": prior_run_id, "prior_wall_s": prior_wall_s,
            "best_doc_type": best_doc_type}


def _scheduler_plan(n_rows: int, batch_size: int, grad_accum: int,
                    epochs: int, warmup_frac: float) -> tuple[int, int]:
    """Optimizer-step schedule plan -> (total_steps, warmup_steps).

    The old code computed ``total_steps`` in MICRO-batch units
    (ceil(n/batch) × epochs) while ``scheduler.step()`` fires once per
    OPTIMIZER step — with grad-accum 8 the 6% warmup consumed ~48% of the
    run and the LR never decayed (ended at ~0.93×peak). Effective training
    was ~1 full-LR epoch. Both counts here are in optimizer steps.
    """
    micro_per_epoch = math.ceil(n_rows / batch_size)
    opt_per_epoch = math.ceil(micro_per_epoch / grad_accum)
    total = opt_per_epoch * epochs
    warmup = max(1, int(total * warmup_frac))
    return total, warmup


def _apply_subclass_support_threshold(train_rows: list[dict],
                                      val_rows: list[dict], maps: dict,
                                      min_rows: int) -> tuple[list, list, dict, dict]:
    """Drop/merge subclass classes with < ``min_rows`` train windows.

    Data-side diagnosis (2026-09-20): contract/correspondence heads carry
    3-doc classes and val cells of n=1 — macro-F1 there is coin-flip noise.
    Classes below the support floor are remapped to the head's ``other``
    class when one exists (the deployment's fail-open route), else dropped
    from the head vocabulary entirely (their rows leave train AND val —
    counted in the returned info for honesty). ``maps`` is rebuilt so the
    checkpoint's labels.json reflects the reduced vocabulary and
    train_counts.json stays the authentic-support source.

    ``other`` may be routing-only (contract ``other``, #116): it is in
    ``labels`` but not in the trainable decision space, and ``_label_id``
    maps it to ``CE_IGNORE_INDEX``.  A remap into it is intentional — the
    rows stay in train/val for the doc_type head and the test split scores
    them as ``subclass_unscorable`` (route-to-LLM) — so the subclass loss
    skips them and every metric/temperature fit must tolerate
    ``CE_IGNORE_INDEX`` labels (``_drop_ignored``).  Dropping the rows
    instead would also erase their doc_type signal.
    """
    info: dict = {"remapped": {}, "dropped": {}, "dropped_val_rows": 0}
    if min_rows <= 0:
        return train_rows, val_rows, maps, info
    for cls, cfg in list(maps.items()):
        if cls == "doc_type":
            continue
        counts = Counter(r["subclass"] for r in train_rows
                         if r["doc_type"] == cls)
        # 'other' is the head's catch-all and the deployment's fail-open
        # target — never a remap/drop SOURCE (it may be the remap TARGET).
        low = sorted(s for s, c in counts.items()
                     if c < min_rows and s != "other")
        if not low:
            continue
        low_set = set(low)
        has_other = "other" in cfg["label2id"] and "other" not in low_set
        for s in low:
            if has_other:
                info["remapped"].setdefault(cls, []).append(s)
                for r in train_rows:
                    if r["doc_type"] == cls and r["subclass"] == s:
                        r["subclass"] = "other"
                for r in val_rows:
                    if r["doc_type"] == cls and r["subclass"] == s:
                        r["subclass"] = "other"
            else:
                info["dropped"].setdefault(cls, []).append(s)
                train_rows = [r for r in train_rows
                              if not (r["doc_type"] == cls
                                      and r["subclass"] == s)]
                kept_val: list[dict] = []
                for r in val_rows:
                    if r["doc_type"] == cls and r["subclass"] == s:
                        info["dropped_val_rows"] += 1
                    else:
                        kept_val.append(r)
                val_rows = kept_val
        keep = [lab for lab in cfg["labels"] if lab not in low_set]
        if len(keep) == len(cfg["labels"]):
            continue
        new_counts = Counter(r["subclass"] for r in train_rows
                             if r["doc_type"] == cls)
        total = sum(new_counts.values())
        new_cfg = {
            "labels": keep,
            "label2id": {lab: i for i, lab in enumerate(keep)},
            "id2label": {str(i): lab for i, lab in enumerate(keep)},
            # zero-support labels stay ABSENT from the weights, exactly as
            # in ``label_maps`` (#116): a stored 1.0 would read as "supported
            # at the average prior" in ``log_prior_values`` instead of the
            # smallest-prior floor.  K counts the supported classes only,
            # like ``class_weights``.
            "weights": {
                lab: total / (len(new_counts) * new_counts[lab])
                for lab in keep if new_counts[lab]},
            "inference_only": [lab for lab in keep if lab not in new_counts],
            "routing_only": list(cfg.get("routing_only", ())),
            "note": cfg.get("note", ""),
        }
        attach_trainable_fields(new_cfg)
        maps[cls] = new_cfg
    return train_rows, val_rows, maps, info


def _apply_subclass_support_plan(rows: list[dict], info: dict) -> list[dict]:
    """Apply the train-derived support floor to ANOTHER split.

    ``_apply_subclass_support_threshold`` mutates only train/val, so the
    held-out test split kept its original subclass labels while the head
    vocabularies were rebuilt without them — the 2026-09-20 run crashed at the
    test eval with ``KeyError: 'affiliate'`` (a remapped contract subclass).
    This mirrors the train/val treatment exactly: remapped subclasses become
    the head's ``other`` (kept); dropped subclasses leave the split.
    """
    remapped = info.get("remapped", {})
    dropped = info.get("dropped", {})
    out: list[dict] = []
    for r in rows:
        cls, sc = r["doc_type"], r["subclass"]
        if sc in dropped.get(cls, ()):
            continue
        if sc in remapped.get(cls, ()):
            r = {**r, "subclass": "other"}
        out.append(r)
    return out


def _subclass_label(heads: dict, cls: str, subclass: str) -> int | None:
    """Resolve a row's subclass to the head's vocabulary.

    Unknown subclasses fall back to the head's ``other`` (the deployment's
    fail-open target) when it exists; ``None`` means the row is unscorable —
    its class was dropped from the vocabulary — and must leave the metric
    denominator rather than crash the run.
    """
    label2id = heads[cls]["label2id"]
    if subclass in heads[cls].get("routing_only", ()):
        return None
    if subclass in label2id:
        return label2id[subclass]
    if "other" in label2id:
        return label2id["other"]
    return None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="HF repo id or local stage dir")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--push-to-hub", default="",
                    help="model repo id to push the checkpoint to (refused "
                         "when no epoch cleared the selection gate unless "
                         "--push-ungated; the Modal app does not use this "
                         "flag — it pushes after ONNX export + parity)")
    ap.add_argument("--push-ungated", action="store_true",
                    help="allow --push-to-hub when no epoch met the "
                         "selection gate (selected_epoch == 0: uncalibrated "
                         "final-epoch weights)")
    ap.add_argument("--eval-test", action="store_true",
                    help="run the held-out test split through the trained "
                         "model (report-only P0 gate)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test: cap train rows (val = max(1, limit//4), "
                         "test docs = limit); 0 = all")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="pre-flight smoke: cap micro-batches across the run "
                         "(0 = unlimited); stops after the epoch containing "
                         "the cap")
    ap.add_argument("--log-every", type=int, default=50,
                    help="print a per-step loss line every N micro-batches "
                         "(visibility guardrail for long GPU runs)")
    ap.add_argument("--prefetch-batches", type=int, default=2,
                    help="overlap batch collate/H2D with GPU via a prefetch "
                         "thread (0=off). Rows are pre-tokenized in RAM — "
                         "this replaces DataLoader num_workers for this trainer")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="cudnn.benchmark=True (faster conv/alg picks; breaks "
                         "bitwise reproducibility with --seed)")
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="disable activation checkpointing on CUDA (faster "
                         "when VRAM allows; default on for 22 GB L4 @ 8192)")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="mid-epoch resume checkpoint every N micro-batches "
                         "(0 = same as --log-every; rounded UP to a "
                         "--grad-accum multiple so a resume restarts on an "
                         "optimizer boundary; writes resume.json under "
                         "--output)")
    ap.add_argument("--model", default=MODEL_ID,
                    help="backbone model id (default: the committed pin)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="checkpoint bundle dir to resume from — continues "
                         "at the next epoch or mid-epoch when resume.json "
                         "has epoch_complete=false (optimizer/scheduler "
                         "state when present)")
    # ---- 2026-09-20 audit levers (loss rebalance + regularization) --------
    ap.add_argument("--loss-lambda-dt", type=float, default=0.65,
                    help="doc_type share of the blended loss; the old summed "
                         "loss starved doc_type to 1/(1+n_heads) of the "
                         "gradient (default 0.65)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing on the doc_type head only "
                         "(calibration lever; 0.05 recommended). Does not "
                         "touch subclass CE — see --subclass-label-smoothing")
    ap.add_argument("--subclass-label-smoothing", type=float, default=0.0,
                    help="label smoothing on subclass CE heads only "
                         "(#22; default 0.0 keeps hard targets). Independent "
                         "of --label-smoothing (doc_type). Interacts with "
                         "--weight-mode: smoothing is applied after class "
                         "weights inside F.cross_entropy")
    ap.add_argument("--input-construction", choices=["v1", "v2"],
                    default="v1",
                    help="window decoration version (#29). v1 = title + "
                         "blank line + body (published Hub pin). v2 = "
                         "[FILE_NAME]/[TITLE]/[WINDOW_INDEX] prefix. The "
                         "trainer reads PREBUILT train/validation windows, so "
                         "it errors out unless this matches the stage's "
                         "construction; it also drives the --eval-test "
                         "windowing. Never mix v2 windows into the v1 Hub "
                         "revision")
    ap.add_argument("--weight-mode", choices=["inverse", "sqrt-inverse",
                                              "none"], default="inverse",
                    help="class-weight transform: inverse (labels.json), "
                         "sqrt-inverse (tames rare-class amplification), "
                         "none (uniform)")
    ap.add_argument("--weight-cap", type=float, default=10.0,
                    help="clamp class weights to [1/cap, cap] after the "
                         "mode transform")
    ap.add_argument("--subclass-loss-norm", choices=["count",
                                                     "weighted-mean"],
                    default="count",
                    help="subclass CE reduction: count (weighted sum / "
                         "head rows, so class weights take effect on small "
                         "per-head micro-batches) or weighted-mean (the "
                         "pre-2026-09-28 behavior, where weights largely "
                         "cancel)")
    ap.add_argument("--subclass-head-lr", type=float, default=None,
                    help="separate AdamW LR for the subclass heads (e.g. "
                         "1e-3); unset = share --lr with the backbone. The "
                         "doc_type head keeps --lr")
    ap.add_argument("--subclass-logit-adjust", type=float, default=0.0,
                    help="logit-adjusted CE on subclass heads: add tau * "
                         "log(train prior) to their logits in the loss only "
                         "(0 = off; 1.0 with --weight-mode none is the "
                         "standard setting)")
    ap.add_argument("--mlp-heads", action="store_true",
                    help="use the ModernBERT classification recipe heads "
                         "(hidden SiLU MLP + dropout) instead of linear")
    ap.add_argument("--head-dropout", type=float, default=0.1,
                    help="dropout inside MLP heads (--mlp-heads only)")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="freeze the backbone for the first N epochs "
                         "(heads always train); unfreeze at epoch N+1")
    ap.add_argument("--early-stop-patience", type=int, default=2,
                    help="stop after this many epochs without val_loss "
                         "improvement")
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay")
    ap.add_argument("--betas", default="0.9,0.999",
                    help="AdamW betas (comma-separated)")
    ap.add_argument("--eps", type=float, default=1e-8,
                    help="AdamW epsilon")
    ap.add_argument("--subclass-min-train-rows", type=int, default=0,
                    help="drop/merge subclass classes with fewer train "
                         "windows than this (remap to `other` when present, "
                         "else drop the rows; 0 = off)")
    # ---- #112 M9a-U4 lexicographic checkpoint selection -------------------
    ap.add_argument("--select-on-subclass",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="checkpoint selection: among doc_type-gate-eligible "
                         "epochs pick the one maximizing the mean observed "
                         "subclass macro-F1 (lexicographic). "
                         "--no-select-on-subclass restores the legacy "
                         "doc_type-only rule (default: on)")
    return ap


def _fit_temperatures_from_logits(logits_by_head: dict[str, list],
                                  labels_by_head: dict[str, list]) -> dict:
    """Per-head temperature scaling on validation logits (plan §8).

    Fits from the logits already collected by ``evaluate`` — no second
    forward pass. Heads with < 2 rows or < 2 unique labels stay at T = 1.0
    (uncalibratable).
    """
    temps: dict[str, float] = {}
    for name in logits_by_head:
        lg, lab = _drop_ignored(torch.cat(logits_by_head[name]),
                                torch.cat(labels_by_head[name]))
        if len(lab) < 2 or len(set(lab.tolist())) < 2:
            temps[name] = 1.0  # too few rows to fit T — leave uncalibrated
        else:
            temps[name] = fit_temperature(lg, lab)
    return temps


def _summary(run_id: str, args, device, events: list[dict], selected: dict,
             temps: dict, wall: float, epochs_run: int,
             test_metrics: dict | None = None, *,
             epochs_trained: int | None = None) -> dict:
    """Run summary — the artifact's self-describing record.

    Carries the FULL hyperparameter set (2026-09-20 audit R3: the artifact
    could not be tied back to its config) and the held-out test metrics
    (R2: they were printed to stdout only and lost).  #107: the selection
    block also carries the per-head calibrated ECE sidecar + the derived
    head-exclusion policy — the deployment gate (inference.py) consumes it
    to route LLM when a subclass head's calibration never cleared the
    budget.

    ``epochs_trained`` is the number of epochs THIS process trained (vs
    ``epochs_run``, the cumulative count including resumed ones); ``0``
    flags ``nothing_trained`` — a ``--resume`` already at/after ``--epochs``
    — so the deploy layer does not export/promote/push it as a new run.
    """
    per_head_ece = selected.get("per_head_ece_calibrated", {})
    if not per_head_ece:
        # Legacy-record fallback: trainer images that predate the nested
        # sidecar still record the per-head calibrated ECE as flat
        # ``<head>_ece_calibrated`` keys on the selected epoch event —
        # derive the sidecar from those (same numbers the selection rule
        # used), so the artifact always ships an exclusion policy.
        per_head_ece = {
            name[:-len("_ece_calibrated")]: round(v, 6)
            for name, v in sorted(selected.items())
            if name.endswith("_ece_calibrated")
            and isinstance(v, int | float)  # not the nested sidecar dict itself
        }
        if per_head_ece:
            per_head_ece = {k: float(v) for k, v in per_head_ece.items()}
    return {
        "run_id": run_id,
        "data": args.data,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "epochs_run": epochs_run,
        "epochs_trained_this_run": epochs_trained,
        "nothing_trained": epochs_trained == 0,
        "epochs_requested": args.epochs,
        "training_wall_s": round(wall, 1),
        "hyperparameters": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in sorted(vars(args).items())
        },
        "checkpoint_selection": {
            "rule": ("lexicographic: best val subclass objective (mean of the "
                     "five observed subclass macro-F1) among epochs whose "
                     "observed doc_type macro-F1 is within "
                     f"{DOC_TYPE_GATE_TOL} of the BEST ECE-eligible doc_type "
                     "macro-F1 of the run and whose CALIBRATED doc_type ECE "
                     f"<= {ECE_BUDGET}"
                     if getattr(args, "select_on_subclass", False) else
                     "best val doc_type macro-F1 (observed classes) with "
                     f"CALIBRATED doc_type ECE <= {ECE_BUDGET}"),
            "epoch": selected["epoch"],
            "macro_f1": selected["macro_f1"],
            "ece": selected["ece"],
            "ece_raw": selected.get("ece_raw"),
            "subclass_objective": selected.get("subclass_objective"),
            "per_head_macro_f1_observed": selected.get(
                "per_head_macro_f1_observed", {}),
            "gate_met": bool(selected["epoch"] > 0),
            "per_head_ece_calibrated": per_head_ece,
            "head_exclusion_policy": {
                "rule": ("exclude a subclass head from the fast path when "
                         f"its calibrated ECE exceeds {ECE_BUDGET} — the "
                         "same budget the doc_type selection gate enforces"),
                "budget": ECE_BUDGET,
                "excluded": {
                    name: {"excluded": ece > ECE_BUDGET,
                           "ece_calibrated": ece}
                    for name, ece in sorted(per_head_ece.items())
                },
            },
        },
        "test_metrics": test_metrics or {},
        "epochs": events,
        "temperatures": {k: round(v, 3) for k, v in temps.items()},
        "input_construction": getattr(args, "input_construction", "v1"),
    }


# Files a selected-epoch promotion copies over the final-epoch output: model
# weights + the per-epoch calibration artifacts that were fitted on THAT
# epoch's validation logits.  Deliberately an allowlist — logs
# (epoch_metrics.*, train_steps.jsonl, test_steps.jsonl) and resume state
# (optimizer.pt, scheduler.pt, resume.json, resume_manifest.json) belong to
# the run as a whole, and the archive's copies of them are TRUNCATED/stale
# snapshots from mid-run.
_PROMOTE_EXACT = frozenset({
    "heads.pt", "temperatures.json", "ood_probe.json", "train_counts.json",
    "labels.json", "config.json", "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
})
_PROMOTE_GLOBS = ("*.safetensors", "pytorch_model*.bin")

# Hub upload hygiene shared by every push path (the Modal app mirrors this):
# optimizer/scheduler/resume state is training-internal (2026-09-20 audit
# R10: it doubled repo size).
HUB_IGNORE_PATTERNS = ["optimizer.pt", "scheduler.pt", "resume.json"]


def _promote_epoch_files(src: Path, dst: Path) -> list[str]:
    """Copy the selected epoch's weights + calibration artifacts into ``dst``.

    Never touches logs or resume state (see ``_PROMOTE_EXACT``).  Returns the
    copied file names (sorted) for the operator log.
    """
    copied: list[str] = []
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        if f.name in _PROMOTE_EXACT or any(f.match(g) for g in _PROMOTE_GLOBS):
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)
    return copied


def _load_epoch_weights(model, src: Path, device, dtype) -> None:
    """Load a saved epoch bundle's backbone + heads into the live ``model``.

    ``--eval-test`` must score the weights that SHIP (the selected epoch), not
    whatever the last epoch left in memory.
    """
    from transformers import AutoModel

    base = AutoModel.from_pretrained(src, torch_dtype=dtype,
                                     attn_implementation="sdpa")
    model.backbone = base.to(device)
    heads_state = torch.load(src / "heads.pt", map_location=device)
    for name, sd in heads_state.items():
        model.heads[name].load_state_dict(sd)
    model.eval()


def _apply_backbone_freeze(model, epoch: int, freeze_epochs: int) -> bool | None:
    """Set backbone ``requires_grad`` for ``epoch``; the backbone is frozen for
    epochs ``1..freeze_epochs`` and trains from ``freeze_epochs + 1`` on.

    Applied per epoch from the epoch number (not as a one-shot at start plus
    an ``epoch == N + 1`` edge): a ``--resume`` that starts past ``N + 1``
    never saw the edge and trained with a frozen backbone forever.  Returns
    the new frozen state when it CHANGED, else None.
    """
    if freeze_epochs <= 0:
        return None
    want_frozen = epoch <= freeze_epochs
    params = list(model.backbone.parameters())
    if not params or all(p.requires_grad == (not want_frozen) for p in params):
        return None  # already in the wanted state
    for p in params:
        p.requires_grad = not want_frozen
    return want_frozen


def _remaining_micro_budget(max_steps: int, steps_done_micro: int) -> int | None:
    """Micro-batches left under ``--max-steps``.

    ``None`` = unlimited (flag off).  ``0`` = EXHAUSTED: ``train_epoch``
    reads ``step_limit=0`` as "no limit", so a resume with the budget already
    spent used to train a full unlimited epoch; the caller must stop instead.
    """
    if not max_steps:
        return None
    return max(0, max_steps - steps_done_micro)


def _align_checkpoint_every(every: int, grad_accum: int) -> int:
    """Round the mid-epoch checkpoint cadence UP to a ``grad_accum`` multiple.

    A checkpoint between optimizer boundaries would snapshot with a
    half-accumulated gradient that ``resume`` cannot restore (gradients are
    not saved), so the resumed accumulation window is short.  Saving only at
    boundaries makes the skip count whole accumulation windows.
    """
    if every <= 0:
        return every
    ga = max(1, grad_accum)
    return -(-every // ga) * ga


def _stage_input_construction(data: str) -> str:
    """Input-construction version the prebuilt training WINDOWS were built with.

    The trainer never re-windows train/validation: it reads the stage's
    prebuilt parquet.  The published pin (and every local stage the repo
    builds) is v1; a stage dir may declare otherwise via an
    ``input_construction`` key in its ``dataset_info.json``.
    """
    local = Path(data)
    info = local / "dataset_info.json"
    if local.exists() and info.is_file():
        try:
            declared = json.loads(info.read_text()).get("input_construction")
        except (OSError, ValueError):
            declared = None
        if declared:
            return str(declared)
    return INPUT_CONSTRUCTION_VERSION


def _check_input_construction(requested: str, stage: str) -> None:
    """Refuse to train a model for one input construction on windows built
    with another.

    ``--input-construction`` used to be recorded in ``summary.json`` and
    nothing else, so ``--input-construction v2`` trained on v1 windows while
    inference then decorated v2 prefixes for a v1-trained model.
    """
    if requested != stage:
        raise SystemExit(
            f"--input-construction {requested} requested but the training "
            f"windows are prebuilt as {stage}: the trainer does not re-window "
            "train/validation, so the model would learn one construction and "
            f"be served another. Rebuild the stage with {requested} windows "
            "(and a new pinned revision) first.")


def _push_to_hub(output: Path, repo: str, *, epochs: int,
                 selected_epoch: int) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(output), repo_id=repo, repo_type="model",
        ignore_patterns=HUB_IGNORE_PATTERNS,
        commit_message=f"ModernBERT hierarchical classifier "
                       f"(epochs {epochs}, selected epoch {selected_epoch})")
    print(f"pushed: https://huggingface.co/{repo}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    # fail before any model/data load (and any GPU minute)
    _check_input_construction(args.input_construction,
                              _stage_input_construction(args.data))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn_fast = args.cudnn_benchmark or os.environ.get(
        "TRAINER_CUDNN_BENCHMARK", "").strip() in ("1", "true", "yes")
    if cudnn_fast and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[trainer] cudnn deterministic={torch.backends.cudnn.deterministic} "
              f"benchmark={torch.backends.cudnn.benchmark} "
              f"prefetch_batches={args.prefetch_batches}", flush=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # bf16 on CUDA only — CPU bf16 is emulated and pathologically slow
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # sdpa: fused memory-efficient attention — eager attention materializes
    # fp32 QK^T scores (~12.9 GB/layer at 8,192 context) and OOMs even at
    # batch 4 on a 22 GB L4; sdpa never materializes the scores (no extra dep).
    # Gradient checkpointing on CUDA: retained activations across 22 layers
    # (fp32 rotary casts + QKV) still OOM the L4 in training mode (22.9 GB at
    # batch 4); checkpointing recomputes them in backward -> 3.3 GB peak.
    if args.resume:
        # Resume: the backbone + labels come from the checkpoint bundle, not
        # the base model id — the bundle is the self-consistent source.
        resume_dir = Path(args.resume)
        if not (resume_dir / "heads.pt").is_file():
            raise SystemExit(f"--resume: no heads.pt under {resume_dir}")
        labels_path = resume_dir / "labels.json"
        if not labels_path.is_file():
            raise SystemExit(f"--resume: no labels.json under {resume_dir}")
        base = AutoModel.from_pretrained(resume_dir, torch_dtype=dtype,
                                         attn_implementation="sdpa")
    else:
        base = AutoModel.from_pretrained(args.model, torch_dtype=dtype,
                                         attn_implementation="sdpa")
        # head configs from the published labels.json
        local_data = Path(args.data)
        if local_data.exists():
            labels_path = local_data / "labels.json"
        else:
            from huggingface_hub import hf_hub_download

            labels_path = Path(hf_hub_download(args.data, "labels.json",
                                               repo_type="dataset",
                                               revision=_hub_revision()))
    if device.type == "cuda" and not args.no_gradient_checkpointing:
        base.gradient_checkpointing_enable()
    elif device.type == "cuda":
        print("[trainer] gradient checkpointing OFF (higher VRAM, faster step)",
              flush=True)
    base = base.to(device)

    maps = normalize_label_maps(json.loads(labels_path.read_text()))

    # rows load BEFORE the model: the subclass support threshold reshapes
    # the head vocabularies (and the model's head sizes) from the data
    raw_train = load_dataset(args.data, "train")
    raw_val = load_dataset(args.data, "validation")
    if args.limit:
        raw_train = raw_train[:args.limit]
        raw_val = raw_val[:max(1, args.limit // 4)]
    raw_train, raw_val, maps, support_info = _apply_subclass_support_threshold(
        raw_train, raw_val, maps, args.subclass_min_train_rows)
    if support_info["remapped"] or support_info["dropped"]:
        print(f"subclass support threshold ({args.subclass_min_train_rows}): "
              f"remapped {support_info['remapped']} "
              f"dropped {support_info['dropped']} "
              f"(val rows dropped: {support_info['dropped_val_rows']})",
              flush=True)

    head_sizes = _head_sizes(maps)
    model = HierarchicalClassifier(
        base, head_sizes,
        head_kind="mlp" if args.mlp_heads else "linear",
        head_dropout=args.head_dropout).to(device)

    heads = _trainer_heads(maps)

    train_rows = tokenize_rows(raw_train, tokenizer, args.max_length)
    val_rows = tokenize_rows(raw_val, tokenizer, args.max_length)
    print(f"windows: train {len(train_rows)} / validation {len(val_rows)}",
          flush=True)

    # scheduler plan in OPTIMIZER steps (the old micro-batch units made the
    # 6% warmup consume ~48% of the run and the LR never decayed)
    total_steps, warmup = _scheduler_plan(len(train_rows), args.batch_size,
                                          args.grad_accum, args.epochs,
                                          args.warmup_frac)
    betas = tuple(float(b) for b in args.betas.split(","))
    optimizer = torch.optim.AdamW(
        param_groups(model, args.lr, args.subclass_head_lr),
        lr=args.lr, betas=betas, eps=args.eps,
        weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_cfg = LossConfig(lambda_dt=args.loss_lambda_dt,
                          label_smoothing=args.label_smoothing,
                          subclass_label_smoothing=args.subclass_label_smoothing,
                          weight_mode=args.weight_mode,
                          weight_cap=args.weight_cap,
                          subclass_loss_norm=args.subclass_loss_norm,
                          subclass_logit_adjust=args.subclass_logit_adjust)

    best_val = float("inf")
    stale = 0
    events: list[dict] = []
    selected: dict = {"epoch": 0, "macro_f1": -1.0, "ece": None,
                      "subclass_objective": -1.0}
    best_doc_type = float("-inf")  # #112 doc_type gate floor (best eligible)
    prior_selected: dict | None = None  # a resumed run's recorded selection
    start_epoch = 1
    if args.resume:
        resume_state = _apply_resume(Path(args.resume), model, optimizer,
                                     scheduler, device, len(train_rows),
                                     args.batch_size, args.grad_accum)
        start_epoch = resume_state["start_epoch"]
        steps_done = resume_state["steps_done"]
        steps_done_micro = resume_state["steps_done_micro"]
        micro_skip = resume_state.get("micro_skip", 0)
        events = resume_state["events"]
        selected = resume_state["selected"]
        best_val = resume_state["best_val"]
        stale = resume_state["stale"]
        temps = resume_state["temps"]
        best_doc_type = resume_state.get("best_doc_type", best_doc_type)
        prior_selected = dict(selected) if selected.get("epoch") else None
        skip_note = (f", skip {micro_skip} micro-batches in epoch "
                     f"{start_epoch}") if micro_skip else ""
        print(f"resumed from {args.resume}: continuing at epoch "
              f"{start_epoch} (opt-steps {steps_done}, "
              f"prior epochs {len(events)}{skip_note})", flush=True)
    val_logits: dict = {}
    t0 = time.time()
    if not args.resume:
        steps_done = 0
        steps_done_micro = 0
        micro_skip = 0
        temps: dict[str, float] = {}
    checkpoint_every = _align_checkpoint_every(
        args.checkpoint_every if args.checkpoint_every > 0 else args.log_every,
        args.grad_accum)
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if args.resume and resume_state.get("run_id"):
        # provenance: a resumed run keeps the ORIGINAL training run's identity
        # (its epochs are that run's; an eval-only resume must not rewrite it)
        run_id = resume_state["run_id"]
    runs_dir = args.output.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    step_log = args.output / "train_steps.jsonl"
    epoch_jsonl = args.output / "epoch_metrics.jsonl"
    epoch_tsv = args.output / "epoch_metrics.tsv"
    micro_per_epoch = math.ceil(len(train_rows) / max(1, args.batch_size))
    if args.max_steps:
        total_micro_planned = min(
            args.max_steps,
            micro_per_epoch * max(0, args.epochs - start_epoch + 1)
            + steps_done_micro)
    else:
        total_micro_planned = micro_per_epoch * args.epochs
    progress = TrainProgress(
        total_micro_planned=total_micro_planned,
        batch_size=args.batch_size,
        step_log=step_log,
        run_t0=t0,
        micro_done_before=steps_done_micro,
    )
    test_step_log = args.output / "test_steps.jsonl"
    print(
        f"[trainer] durable logs: steps={step_log} epochs_jsonl={epoch_jsonl} "
        f"epochs_tsv={epoch_tsv} test_steps={test_step_log} "
        f"archives={runs_dir}/<run_id>-eN "
        f"planned_micro={total_micro_planned} (ETA on each step line) "
        f"mid_epoch_ckpt_every={checkpoint_every}",
        flush=True)
    last_manifest_ckpt = args.output.resolve()
    epochs_trained = 0  # epochs trained by THIS process (0 => nothing_trained)

    def _maybe_mid_epoch_checkpoint(step_in_epoch: int, n_opt: int,
                                    skipped: int) -> None:
        # ``step_in_epoch`` is the ABSOLUTE micro-batch index in the epoch
        # (what a later resume skips); ``skipped`` of those were trained in a
        # previous process and are already inside ``steps_done_micro``.
        nonlocal last_manifest_ckpt
        if checkpoint_every <= 0:
            return
        if step_in_epoch % checkpoint_every != 0:
            return
        partial_summary = _summary(
            run_id, args, device, events, selected, temps,
            time.time() - t0, len(events), epochs_trained=epochs_trained)
        save_checkpoint(
            args.output, model, tokenizer, maps, heads, train_rows,
            temps, partial_summary, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, steps_done=steps_done + n_opt,
            steps_done_micro=steps_done_micro + step_in_epoch - skipped,
            step_in_epoch=step_in_epoch, epoch_complete=False)
        last_manifest_ckpt = args.output.resolve()
        print(f"[trainer] mid-epoch resume checkpoint "
              f"(epoch {epoch} step {step_in_epoch}): {args.output}",
              flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        remaining = _remaining_micro_budget(args.max_steps, steps_done_micro)
        if remaining == 0:
            # budget already spent (resume of a capped run): train_epoch reads
            # step_limit=0 as UNLIMITED, so stop here instead
            print(f"max-steps ({args.max_steps}) already reached at resume; "
                  "nothing to train", flush=True)
            break
        frozen = _apply_backbone_freeze(model, epoch,
                                        args.freeze_backbone_epochs)
        if frozen is True:
            print(f"backbone frozen for the first "
                  f"{args.freeze_backbone_epochs} epoch(s); heads always "
                  "train", flush=True)
        elif frozen is False:
            print(f"backbone unfrozen at epoch {epoch}", flush=True)
        prev_selected_epoch = selected.get("epoch", 0)
        epoch_skip = micro_skip if epoch == start_epoch and micro_skip else 0
        if epoch_skip:
            print(f"[trainer] skipping first {epoch_skip} micro-batches "
                  f"in epoch {epoch} (--resume mid-epoch)", flush=True)
            micro_skip = 0
        _batch_kw = dict(prefetch_batches=args.prefetch_batches,
                         non_blocking=(device.type == "cuda"))

        def _on_micro(step_in_epoch: int, n_opt_in_epoch: int,
                      _loss: float, _skipped: int = epoch_skip) -> None:
            _maybe_mid_epoch_checkpoint(step_in_epoch, n_opt_in_epoch,
                                        _skipped)

        loss, loss_endpoint, n_micro, n_opt = train_epoch(
            model, make_batches(train_rows, args.batch_size, True,
                                heads, device, **_batch_kw,
                                shuffle_rng=_epoch_shuffle_rng(
                                    args.seed, epoch)),
            optimizer, scheduler, heads, device,
            args.grad_accum, step_limit=remaining or 0,
            log_every=args.log_every, loss_cfg=loss_cfg,
            progress=progress, epoch=epoch, epochs=args.epochs,
            skip_micro=epoch_skip,
            on_micro_step=_on_micro if checkpoint_every > 0 else None)
        steps_done += n_opt
        steps_done_micro += n_micro
        epochs_trained += 1
        val, val_logits, val_labels = evaluate(
            model, make_batches(val_rows, args.batch_size, False,
                                heads, device), heads, maps, device,
            loss_cfg)
        # temperatures from the eval pass's logits (no second forward pass);
        # the selection gate uses the CALIBRATED ECE — the old raw-T=1 gate
        # was structurally unreachable for any overconfident head
        temps = _fit_temperatures_from_logits(val_logits, val_labels)
        # calibrated ECE for EVERY head (was doc_type only) — the subclass
        # heads are the deployment-critical ones for the conditional route
        for name in val_logits:
            val[f"{name}_ece_calibrated"] = round(
                ece_calibrated(torch.cat(val_logits[name]),
                               torch.cat(val_labels[name]), temps[name]), 4)
        # hardening seam: lexicographic checkpoint selection (#112 M9a-U4).
        # The doc_type gate is preserved (CALIBRATED doc_type ECE <= budget;
        # observed doc_type macro-F1 not regressing beyond DOC_TYPE_GATE_TOL
        # below the BEST ECE-eligible doc_type of the whole run, re-derived
        # every epoch so a later, better epoch can retire an earlier
        # selection that no longer satisfies the rule).  Among gate-eligible
        # epochs the subclass objective (mean of the five observed subclass
        # macro-F1) decides, so a subclass-focused epoch is no longer thrown
        # away when doc_type merely holds.  --no-select-on-subclass restores
        # the legacy doc_type-only rule verbatim.
        # #107: the selection snapshot carries the per-head calibrated ECE
        # sidecar — the artifact's head-exclusion policy derives from it, so
        # the deployment gate can exclude subclass heads whose calibration
        # never cleared the budget.
        subclass_heads = sorted(name for name in val_logits
                                if name != "doc_type")
        val["subclass_objective"] = round(
            _subclass_objective(val, subclass_heads), 4)
        selected, best_doc_type = _select_epoch(
            [*events, {"epoch": epoch, **val}], subclass_heads,
            select_on_subclass=args.select_on_subclass, prior=prior_selected)
        selected_this_epoch = selected.get("epoch") == epoch
        epoch_wall_s = round(time.time() - epoch_t0, 1)
        eta_rem = progress.eta_s()
        event = {"epoch": epoch, "loss": round(loss, 4),
                 "loss_endpoint": round(loss_endpoint, 4),
                 "lr": round(scheduler.get_last_lr()[0], 8),
                 "epoch_wall_s": epoch_wall_s,
                 "micro_steps": n_micro, "opt_steps": n_opt,
                 "steps_per_sec": round(
                     n_micro / max(1e-6, epoch_wall_s), 4),
                 "samples_per_sec": round(
                     (n_micro * args.batch_size) / max(1e-6, epoch_wall_s), 2),
                 "eta_remaining_s": None if eta_rem is None else round(eta_rem, 1),
                 "eta_remaining": _fmt_duration(eta_rem),
                 "selected_this_epoch": selected_this_epoch,
                 "gate_met": bool(selected.get("epoch", 0) > 0),
                 "selection_epoch": selected.get("epoch", 0),
                 **val}
        events.append(event)
        sc_bits = " ".join(
            f"{h}_f1={val.get(f'{h}_macro_f1_observed')}"
            for h in subclass_heads)
        print(
            f"epoch {epoch}/{args.epochs} loss {loss:.4f} "
            f"(endpoint {loss_endpoint:.4f}) "
            f"val_loss {val['val_loss']:.4f} "
            f"doc_type_acc {val['doc_type_window_acc']} "
            f"doc_acc {val['doc_type_doc_acc']} "
            f"macro_f1 {val['doc_type_macro_f1_observed']} "
            f"ece {val['doc_type_ece']} "
            f"ece_cal {val['doc_type_ece_calibrated']} "
            f"subclass_obj {val['subclass_objective']} "
            f"[{sc_bits}] "
            f"selected={'yes' if selected_this_epoch else 'no'} "
            f"(best_e={selected.get('epoch', 0)}) "
            f"wall {epoch_wall_s}s ETA {_fmt_duration(eta_rem)}",
            flush=True)
        # per-epoch checkpoint: save + archive + volume commit so a kill or
        # timeout never loses more than the in-flight epoch (2026-09-19: a
        # cancelled run lost everything because the only save happened at
        # the very end). Temperatures are fitted per epoch so any epoch's
        # checkpoint is deployment-usable.
        print("temperatures:", {k: round(v, 3) for k, v in temps.items()},
              flush=True)
        save_checkpoint(args.output, model, tokenizer, maps, heads, train_rows,
                        temps, _summary(run_id, args, device, events, selected,
                                        temps, time.time() - t0, epoch,
                                        epochs_trained=epochs_trained),
                        optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, steps_done=steps_done,
                        steps_done_micro=steps_done_micro,
                        step_in_epoch=0, epoch_complete=True)
        _write_ood_probe(args.output, val_logits)
        # smoke (--max-steps) never archives: the run is a cadence probe, not
        # a checkpoint family — keep runs/ for real epochs only.
        archive_path = ""
        if not args.max_steps:
            epoch_archive = runs_dir / f"{run_id}-e{epoch}"
            if epoch_archive.exists():
                shutil.rmtree(epoch_archive)
            shutil.copytree(args.output, epoch_archive)
            archive_path = str(epoch_archive)
            print(f"[trainer] epoch {epoch} checkpoint archived: "
                  f"{epoch_archive}", flush=True)
        event["checkpoint"] = str(args.output)
        event["archive"] = archive_path
        print(f"[trainer] checkpoint written: {args.output}"
              + (f" archive={archive_path}" if archive_path else ""),
              flush=True)
        _write_epoch_metrics(epoch_jsonl, epoch_tsv, {
            "ts": datetime.now(UTC).isoformat(),
            "event": "epoch",
            "run_id": run_id,
            **event,
            "prev_selected_epoch": prev_selected_epoch,
            "per_head_macro_f1_observed": {
                h: val.get(f"{h}_macro_f1_observed") for h in
                ["doc_type", *subclass_heads]
            },
            "per_head_ece_calibrated": {
                h: val.get(f"{h}_ece_calibrated") for h in
                ["doc_type", *subclass_heads]
            },
        })
        _commit_checkpoint_volume()
        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break
        if args.max_steps and steps_done_micro >= args.max_steps:
            print(f"max-steps reached ({args.max_steps}); smoke run complete",
                  flush=True)
            break
    wall = time.time() - t0
    if args.resume:
        wall += resume_state.get("prior_wall_s", 0.0)
    print(f"training wall: {wall:.1f}s", flush=True)

    # selection enforcement (2026-09-20 audit R1): the shipped artifact must be
    # the SELECTED epoch, not merely the last one.  Promotion happens BEFORE
    # the held-out test eval so --eval-test scores the weights that actually
    # ship (it used to score the in-memory FINAL epoch and then attach those
    # metrics to a summary describing the selected one).
    selected_epoch = selected["epoch"]
    last_epoch = len(events)
    promoted = False
    if selected_epoch > 0 and selected_epoch != last_epoch:
        src = runs_dir / f"{run_id}-e{selected_epoch}"
        if src.is_dir():
            copied = _promote_epoch_files(src, args.output)
            sel_temps_path = src / "temperatures.json"
            if sel_temps_path.is_file():
                temps = json.loads(sel_temps_path.read_text(encoding="utf-8"))
            # live model := shipped weights (the final-epoch weights in memory
            # are not what the summary/selection describe)
            _load_epoch_weights(model, args.output, device, dtype)
            promoted = True
            print(f"[trainer] promoted selected epoch {selected_epoch} "
                  f"weights into {args.output} ({', '.join(copied)})",
                  flush=True)
            # optimizer.pt/scheduler.pt/resume.json in `output` still describe
            # the FINAL epoch: point the resume manifest at its archive so a
            # --resume never pairs them with the promoted weights.
            final_archive = runs_dir / f"{run_id}-e{last_epoch}"
            if final_archive.is_dir():
                _write_resume_manifest(
                    args.output, checkpoint_dir=final_archive,
                    epoch=last_epoch, step_in_epoch=0, steps_done=steps_done,
                    steps_done_micro=steps_done_micro, run_id=run_id,
                    epoch_complete=True)
        else:
            print(f"[trainer] WARNING: selected epoch {selected_epoch} "
                  f"archive missing ({src}); latest/ holds the final epoch",
                  flush=True)
    elif selected_epoch == 0:
        print(f"[trainer] WARNING: selection gate NOT met by any epoch "
              f"(calibrated doc_type ECE never <= {ECE_BUDGET}); latest/ "
              f"holds the final epoch — treat this artifact as UNCALIBRATED",
              flush=True)

    # held-out test eval BEFORE the final save so the metrics land in the
    # summary (2026-09-20 audit R2: they were stdout-only and lost).
    test_metrics: dict = {}
    if args.eval_test:
        # eval mode: an eval-only --resume never entered the epoch loop, and
        # freshly built heads start in train mode (live MLP-head dropout)
        model.eval()
        test_docs = load_dataset(args.data, "test")
        if args.limit:
            test_docs = test_docs[:args.limit]
        # the support floor reshaped the head vocabularies from train/val, so
        # the test split must share that vocabulary (2026-09-20 crash:
        # KeyError 'affiliate' — a remapped contract subclass still labelled
        # in test). Rows whose class was dropped leave the split, mirroring
        # train/val, and any residual unknown falls back to the head's `other`.
        test_docs = _apply_subclass_support_plan(test_docs, support_info)
        dt_correct = sc_correct = sc_scorable = sc_unscorable = 0
        n = len(test_docs)
        test_t0 = time.time()
        log_every = max(1, int(args.log_every))

        def _log_test_progress(doc_done: int, *, final: bool = False) -> None:
            acc = round(dt_correct / doc_done, 4) if doc_done else None
            sc_acc = (
                round(sc_correct / sc_scorable, 4) if sc_scorable else None
            )
            row = {
                "ts": datetime.now(UTC).isoformat(),
                "event": "test_complete" if final else "test_step",
                "doc_done": doc_done,
                "doc_planned": n,
                "doc_type_acc": acc,
                "subclass_acc_conditional": sc_acc,
                "doc_type_correct": dt_correct,
                "subclass_correct": sc_correct,
                "subclass_scorable": sc_scorable,
                "subclass_unscorable": sc_unscorable,
                "wall_s": round(time.time() - test_t0, 1),
            }
            if final:
                row["n_docs"] = n
            _append_jsonl(test_step_log, row)
            sc_bit = f" subclass_cond {sc_acc}" if sc_acc is not None else ""
            print(
                f"  test doc {doc_done}/{n} doc_type_acc {acc}{sc_bit}",
                flush=True,
            )

        for i, r in enumerate(test_docs):
            wins = window_document(r["title"], r["doc_text"],
                                   max_tokens=args.max_length,
                                   overlap=WINDOW_OVERLAP_TOKENS,
                                   version=args.input_construction,
                                   filename=r.get("filename", ""))
            enc = tokenizer(wins, padding="max_length", truncation=True,
                            max_length=args.max_length, return_tensors="pt")
            with torch.no_grad():
                lg = model(enc["input_ids"].to(device),
                           enc["attention_mask"].to(device))
            dt_votes = Counter(lg["doc_type"].argmax(-1).tolist())
            dt_pred = dt_votes.most_common(1)[0][0]
            dt_label = heads["doc_type"]["label2id"][r["doc_type"]]
            if dt_pred == dt_label:
                dt_correct += 1
                cls = maps["doc_type"]["trainable_id2label"][str(dt_pred)]
                if cls in heads:  # unknown (inference-only) carries no head
                    sc_votes = Counter(lg[cls].argmax(-1).tolist())
                    sc_pred = sc_votes.most_common(1)[0][0]
                    sc_label = _subclass_label(heads, cls, r["subclass"])
                    if sc_label is None:
                        sc_unscorable += 1
                    else:
                        sc_scorable += 1
                        if sc_pred == sc_label:
                            sc_correct += 1
            doc_done = i + 1
            if doc_done % log_every == 0 or doc_done == n:
                _log_test_progress(doc_done, final=(doc_done == n))
        test_metrics = {
            "n_docs": n,
            "doc_type_acc": round(dt_correct / n, 4) if n else None,
            "subclass_acc_conditional": (
                round(sc_correct / sc_scorable, 4) if sc_scorable else None),
            "doc_type_correct": dt_correct,
            "subclass_correct": sc_correct,
            "subclass_scorable": sc_scorable,
            "subclass_unscorable": sc_unscorable,
        }
        print(f"test metrics: {test_metrics}", flush=True)

    summary = _summary(run_id, args, device, events, selected, temps, wall,
                       len(events), test_metrics,
                       epochs_trained=epochs_trained)

    if promoted:
        # weights/calibration already promoted from the selected epoch's
        # archive; only the summary (now carrying the test metrics of those
        # weights) is rewritten — optimizer/resume state stays untouched.
        (args.output / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2))
    else:
        # final save (reuses the last epoch's temperatures — no extra val pass)
        save_checkpoint(args.output, model, tokenizer, maps, heads, train_rows,
                        temps, summary,
                        optimizer=optimizer, scheduler=scheduler,
                        epoch=len(events), steps_done=steps_done,
                        steps_done_micro=steps_done_micro,
                        step_in_epoch=0, epoch_complete=True)
        _write_ood_probe(args.output, val_logits)

    print(f"checkpoint saved: {args.output}", flush=True)
    for p in sorted(args.output.iterdir()):
        print(f"  {p.name}", flush=True)
    print(f"selection: {'lexicographic subclass objective' if args.select_on_subclass else 'best val doc_type macro-F1 (observed) s.t. calibrated ECE'}"
          f" -> epoch {selected['epoch']} "
          f"(macro_f1 {selected['macro_f1']}, ece {selected['ece']}, "
          f"subclass_obj {selected.get('subclass_objective')})",
          flush=True)
    print(f"run summary: device={device} seed={args.seed} "
          f"epochs={len(events)} wall={wall:.1f}s data={args.data}",
          flush=True)

    if args.push_to_hub:
        if selected_epoch == 0 and not args.push_ungated:
            print("[trainer] REFUSING --push-to-hub: no epoch met the "
                  "selection gate (selected_epoch == 0), so the checkpoint is "
                  "the UNCALIBRATED final epoch. Re-run with --push-ungated "
                  "to publish it anyway.", flush=True)
            return 3
        _push_to_hub(args.output, args.push_to_hub, epochs=args.epochs,
                     selected_epoch=selected_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

