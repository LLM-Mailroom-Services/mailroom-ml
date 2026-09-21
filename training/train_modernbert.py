#!/usr/bin/env python3
"""Fine-tune the hierarchical ModernBERT-base classifier (mailroom-ml trainer).

The documented training entrypoint of the mailroom-ml deploy layer:
``deploy/modal_app.py`` invokes this script as ``sys.executable
/root/training/train_modernbert.py --data <repo> --output /checkpoints/latest
--epochs N --batch-size N --grad-accum N --lr F --seed N [--push-to-hub
REPO] [--eval-test]``.

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
  printed table + ``summary.json``: best val macro-F1 s.t. ECE <= 0.05 (the
  plan's deployment gate — recorded, not enforced as an exit).
- **Calibration (plan §8)**: per-head temperature scaling on validation
  logits (scipy ``minimize_scalar``, bounded (0.05, 10.0)); heads with < 2
  rows or < 2 unique classes in val stay at T = 1.0.
- **Checkpoint**: backbone + tokenizer + ``heads.pt`` + ``labels.json``
  (sidecar copy) + ``temperatures.json`` + ``train_counts.json`` (authentic
  per-(doc_type, subclass) train-row counts — the routing gate's
  ``ROUTE_MIN_AUTHENTIC_SUPPORT`` data source) + ``summary.json`` (+ optional
  Hub push via ``--push-to-hub``).
- **Test gate** (``--eval-test``): report-only — held-out document accuracy
  via the committed windower at eval time; the P0 thresholds (doc_type >=
  0.95 / subclass >= 0.75) belong to the eval harness, NOT this trainer:
  exit 0 unless a real error.

Usage:
    .venv/bin/python training/train_modernbert.py --epochs 5 --output data/modernbert_training/runs/run1
    .venv/bin/python training/train_modernbert.py --push-to-hub Lucius-Morningstar/mailroom-modernbert-classifier
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
    MAX_TOKENS,
    MODEL_ID,
    RUNS_DIR,
    TRAINING_DATA_REPO,
    TRAINING_DATA_REVISION,
    WINDOW_OVERLAP_TOKENS,
)
from mailroom_ml.windows import window_document

DEFAULT_DATA = TRAINING_DATA_REPO
DEFAULT_OUTPUT = RUNS_DIR / "latest"
ECE_BUDGET = 0.05  # plan §8 deployment gate — selection constraint, not an exit


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


def _pad_right(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad a 1-D tensor with zeros to ``target_len`` (no-op when at or
    over length). Used for dynamic-padding batches; pads are masked out by
    the attention mask, so the embedding of token 0 never contributes."""
    return F.pad(t, (0, max(0, target_len - t.shape[0])))


@dataclass
class LossConfig:
    """Loss-shaping knobs (2026-09-20 audit: loss rebalance + calibration).

    - ``lambda_dt``: doc_type share of the loss. The old summed loss gave
      doc_type 1/(1+n_subclass_heads) of the gradient — the subclass heads
      dominated the shared backbone. Weighted blend::
          loss = λ · CE_dt + (1-λ) · mean(CE_subclass_heads)
    - ``label_smoothing``: applied to the doc_type head only (calibration
      lever; subclass heads keep hard targets).
    - ``weight_mode``: "inverse" (labels.json inverse-frequency, as before),
      "sqrt-inverse" (sqrt of the stored weights ≈ inverse-sqrt frequency —
      tames the rare-class amplification), "none" (uniform).
    - ``weight_cap``: clamp class weights to [1/cap, cap] after the mode
      transform (default 10× — a rare class never out-weights a common one
      by more than an order of magnitude).
    """
    lambda_dt: float = 0.65
    label_smoothing: float = 0.0
    weight_mode: str = "inverse"
    weight_cap: float = 10.0

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
        label_smoothing=cfg.label_smoothing)
    sc_ces: list[torch.Tensor] = []
    for cls in heads:
        if cls == "doc_type":
            continue
        sel = batch["doc_type"] == heads["doc_type"]["label2id"][cls]
        if sel.any():
            sc_ces.append(F.cross_entropy(
                logits[cls][sel], batch["subclass"][sel],
                weight=cfg.transform_weights(heads[cls]["weights"],
                                             heads[cls]["labels"],
                                             device=logits[cls].device)))
    if sc_ces:
        loss = cfg.lambda_dt * dt_ce + (1.0 - cfg.lambda_dt) * torch.stack(
            sc_ces).mean()
    else:
        loss = dt_ce
    return loss, logits


def _pad_right(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad a 1-D tensor with zeros to ``target_len`` (no-op when at or
    over length). Used for dynamic-padding batches; pads are masked out by
    the attention mask, so the embedding of token 0 never contributes."""
    return F.pad(t, (0, max(0, target_len - t.shape[0])))


def make_batches(rows, batch_size: int, shuffle: bool, heads, device):
    """Yield batches (rows are pre-padded to max_length by tokenize_rows).

    FIXED padding (not dynamic): the dynamic-padding variant (pad to the
    batch's longest row) measured only ~15% faster — 75% of windows sit
    near the 8,192 cap, so most shuffled batches still pad near-full — and
    the variable-length sdpa + gradient-checkpointing path hung mid-epoch-2
    on the L4 (2026-09-19). Fixed padding is the proven-stable config.
    """
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = [rows[j] for j in idx[i:i + batch_size]]
        yield {
            "input_ids": torch.stack([r["input_ids"] for r in sel]).to(device),
            "attention_mask": torch.stack([r["attention_mask"] for r in sel]).to(device),
            "doc_type": torch.tensor([heads["doc_type"]["label2id"][r["doc_type"]]
                                      for r in sel], device=device),
            "subclass": torch.tensor([heads[r["doc_type"]]["label2id"][r["subclass"]]
                                      for r in sel], device=device),
            "filename": [r["filename"] for r in sel],
        }


def train_epoch(model, batches, optimizer, scheduler, heads, device,
                grad_accum: int, step_limit: int = 0,
                log_every: int = 50,
                loss_cfg: LossConfig | None = None) -> tuple[float, float, int, int]:
    """One epoch. Returns (mean loss, endpoint loss, micro-steps, opt-steps).

    Guardrails:
    - ``log_every``: per-step progress line so a stalled/starved container is
      visible in ``modal app logs`` within seconds instead of at epoch end.
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
    """
    cfg = loss_cfg or LossConfig()
    model.train()
    total, n = 0.0, 0
    opt_steps = 0
    window: list[float] = []
    for step, batch in enumerate(batches):
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
        window.append(loss.item())
        if len(window) > grad_accum:
            window.pop(0)
        if (step + 1) % log_every == 0:
            print(f"  step {step + 1} loss {loss.item():.4f}", flush=True)
        if step_limit and (step + 1) >= step_limit:
            break
    if n % grad_accum != 0:  # flush trailing accumulation (partial step)
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
            logits_by_head[name].append(lg.cpu())
            if name == "doc_type":
                labels_by_head[name].append(batch["doc_type"].cpu())
            else:
                # subclass heads are scored only on their own class's rows
                sel = batch["doc_type"] == heads["doc_type"]["label2id"][name]
                logits_by_head[name][-1] = lg[sel].cpu()
                labels_by_head[name].append(batch["subclass"][sel].cpu())
        for i, fn in enumerate(batch["filename"]):
            dt_p = dt_preds[i].item()
            cls = maps["doc_type"]["id2label"][str(dt_p)]
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
    dt_correct = sc_correct = 0
    for fn, votes in doc_votes.items():
        dt_label, sc_label = doc_labels[fn]
        dt_pred = Counter(v[0] for v in votes).most_common(1)[0][0]
        if dt_pred == dt_label:
            dt_correct += 1
            cls = maps["doc_type"]["id2label"][str(dt_pred)]
            cond = [v[1] for v in votes if v[0] == dt_pred and v[1] is not None]
            if not cond:
                continue
            sc_pred = Counter(cond).most_common(1)[0][0]
            if sc_pred == sc_label:  # both ids in head `cls`'s space
                sc_correct += 1
    metrics["doc_type_doc_acc"] = round(dt_correct / max(1, len(doc_votes)), 4)
    metrics["subclass_doc_acc"] = round(sc_correct / max(1, dt_correct), 4)
    return metrics, logits_by_head, labels_by_head


def _ece_from_probs(probs: torch.Tensor, labels: torch.Tensor,
                    n_bins: int = 10) -> float:
    """ECE binning over already-softmaxed probabilities (shared core)."""
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


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Platt-style temperature scaling: T minimizing NLL on validation."""
    from scipy.optimize import minimize_scalar

    lg, lab = logits.detach().float().numpy(), labels.numpy()

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


def save_checkpoint(output: Path, model, tokenizer, maps, heads, train_rows,
                    temps: dict, summary: dict, *,
                    optimizer=None, scheduler=None, epoch: int = 0,
                    steps_done: int = 0, steps_done_micro: int = 0) -> None:
    """Write the full checkpoint bundle (backbone + heads + sidecars).

    When ``optimizer`` is given, the optimizer/scheduler state and a
    ``resume.json`` counter file are written too, so a later ``--resume`` can
    continue training from this exact point (not just reload weights).
    ``steps_done`` counts OPTIMIZER steps (the scheduler's unit — the
    2026-09-20 audit fixed the micro-batch/optimizer-step mismatch);
    ``steps_done_micro`` is the micro-batch count for --max-steps smoke
    accounting.
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
        (output / "resume.json").write_text(json.dumps({
            "epoch": epoch,
            "steps_done": steps_done,
            "steps_done_micro": steps_done_micro,
            "run_id": summary.get("run_id", ""),
        }, sort_keys=True, indent=2))
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
    if resume_json.exists():
        rj = json.loads(resume_json.read_text())
        epoch_done, steps_done = rj["epoch"], rj["steps_done"]
        steps_done_micro = rj.get("steps_done_micro", 0)
    else:
        summary_path = resume_dir / "summary.json"
        epoch_done = (json.loads(summary_path.read_text()).get("epochs_run", 0)
                      if summary_path.exists() else 0)
        steps_done = epoch_done * math.ceil(
            math.ceil(n_train_rows / batch_size) / grad_accum)
        steps_done_micro = epoch_done * math.ceil(n_train_rows / batch_size)
    opt_path = resume_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    sched_path = resume_dir / "scheduler.pt"
    if sched_path.exists():
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
    else:
        for _ in range(steps_done):
            scheduler.step()
    events: list[dict] = []
    selected: dict = {"epoch": 0, "macro_f1": -1.0, "ece": None}
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
    return {"start_epoch": epoch_done + 1, "steps_done": steps_done,
            "steps_done_micro": steps_done_micro, "events": events,
            "selected": selected, "best_val": best_val,
            "stale": stale, "temps": temps,
            "run_id": prior_run_id, "prior_wall_s": prior_wall_s}


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
        new_cfg = {
            "labels": keep,
            "label2id": {lab: i for i, lab in enumerate(keep)},
            "id2label": {str(i): lab for i, lab in enumerate(keep)},
        }
        new_counts = Counter(r["subclass"] for r in train_rows
                             if r["doc_type"] == cls)
        total = sum(new_counts.values())
        new_cfg["weights"] = {
            lab: (total / (len(keep) * new_counts[lab]) if new_counts[lab]
                  else 1.0)
            for lab in keep}
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
                    help="model repo id to push the checkpoint to")
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
    ap.add_argument("--model", default=MODEL_ID,
                    help="backbone model id (default: the committed pin)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="checkpoint bundle dir to resume from — continues "
                         "at the next epoch (optimizer/scheduler state when "
                         "present; else reconstructed from the counters)")
    # ---- 2026-09-20 audit levers (loss rebalance + regularization) --------
    ap.add_argument("--loss-lambda-dt", type=float, default=0.65,
                    help="doc_type share of the blended loss; the old summed "
                         "loss starved doc_type to 1/(1+n_heads) of the "
                         "gradient (default 0.65)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing on the doc_type head only "
                         "(calibration lever; 0.05 recommended)")
    ap.add_argument("--weight-mode", choices=["inverse", "sqrt-inverse",
                                              "none"], default="inverse",
                    help="class-weight transform: inverse (labels.json), "
                         "sqrt-inverse (tames rare-class amplification), "
                         "none (uniform)")
    ap.add_argument("--weight-cap", type=float, default=10.0,
                    help="clamp class weights to [1/cap, cap] after the "
                         "mode transform")
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
        lg = torch.cat(logits_by_head[name])
        lab = torch.cat(labels_by_head[name])
        if len(lab) < 2 or len(set(lab.tolist())) < 2:
            temps[name] = 1.0  # too few rows to fit T — leave uncalibrated
        else:
            temps[name] = fit_temperature(lg, lab)
    return temps


def _summary(run_id: str, args, device, events: list[dict], selected: dict,
             temps: dict, wall: float, epochs_run: int,
             test_metrics: dict | None = None) -> dict:
    """Run summary — the artifact's self-describing record.

    Carries the FULL hyperparameter set (2026-09-20 audit R3: the artifact
    could not be tied back to its config) and the held-out test metrics
    (R2: they were printed to stdout only and lost).  #107: the selection
    block also carries the per-head calibrated ECE sidecar + the derived
    head-exclusion policy — the deployment gate (inference.py) consumes it
    to route LLM when a subclass head's calibration never cleared the
    budget.
    """
    per_head_ece = selected.get("per_head_ece_calibrated", {})
    return {
        "run_id": run_id,
        "data": args.data,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "epochs_run": epochs_run,
        "epochs_requested": args.epochs,
        "training_wall_s": round(wall, 1),
        "hyperparameters": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in sorted(vars(args).items())
        },
        "checkpoint_selection": {
            "rule": ("best val doc_type macro-F1 (observed classes) with "
                     f"CALIBRATED doc_type ECE <= {ECE_BUDGET}"),
            "epoch": selected["epoch"],
            "macro_f1": selected["macro_f1"],
            "ece": selected["ece"],
            "ece_raw": selected.get("ece_raw"),
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
    }


def main() -> int:
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

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
    if device.type == "cuda":
        base.gradient_checkpointing_enable()
    base = base.to(device)

    maps = json.loads(labels_path.read_text())

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

    head_sizes = {name: len(cfg["labels"]) for name, cfg in maps.items()}
    model = HierarchicalClassifier(
        base, head_sizes,
        head_kind="mlp" if args.mlp_heads else "linear",
        head_dropout=args.head_dropout).to(device)

    heads = {}
    for name, cfg in maps.items():
        heads[name] = {
            "label2id": cfg["label2id"],
            "labels": cfg["labels"],
            "weights": cfg["weights"],
        }

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=betas, eps=args.eps,
                                  weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"backbone frozen for the first {args.freeze_backbone_epochs} "
              f"epoch(s); heads always train", flush=True)

    loss_cfg = LossConfig(lambda_dt=args.loss_lambda_dt,
                          label_smoothing=args.label_smoothing,
                          weight_mode=args.weight_mode,
                          weight_cap=args.weight_cap)

    best_val = float("inf")
    stale = 0
    events: list[dict] = []
    selected: dict = {"epoch": 0, "macro_f1": -1.0, "ece": None}
    start_epoch = 1
    if args.resume:
        resume_state = _apply_resume(Path(args.resume), model, optimizer,
                                     scheduler, device, len(train_rows),
                                     args.batch_size, args.grad_accum)
        start_epoch = resume_state["start_epoch"]
        steps_done = resume_state["steps_done"]
        steps_done_micro = resume_state["steps_done_micro"]
        events = resume_state["events"]
        selected = resume_state["selected"]
        best_val = resume_state["best_val"]
        stale = resume_state["stale"]
        temps = resume_state["temps"]
        print(f"resumed from {args.resume}: continuing at epoch "
              f"{start_epoch} (opt-steps {steps_done}, "
              f"prior epochs {len(events)})", flush=True)
    t0 = time.time()
    if not args.resume:
        steps_done = 0
        steps_done_micro = 0
        temps: dict[str, float] = {}
    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if args.resume and resume_state.get("run_id"):
        # provenance: a resumed run keeps the ORIGINAL training run's identity
        # (its epochs are that run's; an eval-only resume must not rewrite it)
        run_id = resume_state["run_id"]
    runs_dir = args.output.parent / "runs"
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()
        if args.freeze_backbone_epochs > 0 \
                and epoch == args.freeze_backbone_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print(f"backbone unfrozen at epoch {epoch}", flush=True)
        remaining = max(0, args.max_steps - steps_done_micro) \
            if args.max_steps else 0
        loss, loss_endpoint, n_micro, n_opt = train_epoch(
            model, make_batches(train_rows, args.batch_size, True,
                                heads, device),
            optimizer, scheduler, heads, device,
            args.grad_accum, step_limit=remaining,
            log_every=args.log_every, loss_cfg=loss_cfg)
        steps_done += n_opt
        steps_done_micro += n_micro
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
        # hardening seam: select the best val macro-F1 (observed classes)
        # s.t. the CALIBRATED ECE is acceptable (recorded, not an exit).
        # #107: the selection snapshot carries the per-head calibrated ECE
        # sidecar — the artifact's head-exclusion policy derives from it, so
        # the deployment gate can exclude subclass heads whose calibration
        # never cleared the budget.
        if val["doc_type_macro_f1_observed"] > selected["macro_f1"] \
                and val["doc_type_ece_calibrated"] <= ECE_BUDGET:
            selected = {"epoch": epoch,
                        "macro_f1": val["doc_type_macro_f1_observed"],
                        "ece": val["doc_type_ece_calibrated"],
                        "ece_raw": val["doc_type_ece"],
                        "per_head_ece_calibrated": {
                            name: val[f"{name}_ece_calibrated"]
                            for name in sorted(val_logits)}}
        events.append({"epoch": epoch, "loss": round(loss, 4),
                       "loss_endpoint": round(loss_endpoint, 4),
                       "lr": round(scheduler.get_last_lr()[0], 8),
                       "epoch_wall_s": round(time.time() - epoch_t0, 1),
                       **val})
        print(f"epoch {epoch}/{args.epochs} loss {loss:.4f} "
              f"(endpoint {loss_endpoint:.4f}) "
              f"val_loss {val['val_loss']:.4f} "
              f"doc_type_acc {val['doc_type_window_acc']} "
              f"doc_acc {val['doc_type_doc_acc']} "
              f"macro_f1 {val['doc_type_macro_f1_observed']} "
              f"ece {val['doc_type_ece']} "
              f"ece_cal {val['doc_type_ece_calibrated']}", flush=True)
        # per-epoch checkpoint: save + archive + volume commit so a kill or
        # timeout never loses more than the in-flight epoch (2026-09-19: a
        # cancelled run lost everything because the only save happened at
        # the very end). Temperatures are fitted per epoch so any epoch's
        # checkpoint is deployment-usable.
        print("temperatures:", {k: round(v, 3) for k, v in temps.items()},
              flush=True)
        save_checkpoint(args.output, model, tokenizer, maps, heads, train_rows,
                        temps, _summary(run_id, args, device, events, selected,
                                        temps, time.time() - t0, epoch),
                        optimizer=optimizer, scheduler=scheduler,
                        epoch=epoch, steps_done=steps_done,
                        steps_done_micro=steps_done_micro)
        # smoke (--max-steps) never archives: the run is a cadence probe, not
        # a checkpoint family — keep runs/ for real epochs only.
        if not args.max_steps:
            epoch_archive = runs_dir / f"{run_id}-e{epoch}"
            shutil.copytree(args.output, epoch_archive)
            print(f"[trainer] epoch {epoch} checkpoint archived: "
                  f"{epoch_archive}", flush=True)
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

    # held-out test eval BEFORE the final save so the metrics land in the
    # summary (2026-09-20 audit R2: they were stdout-only and lost).
    test_metrics: dict = {}
    if args.eval_test:
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
        for r in test_docs:
            wins = window_document(r["title"], r["doc_text"],
                                   max_tokens=args.max_length,
                                   overlap=WINDOW_OVERLAP_TOKENS)
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
                cls = maps["doc_type"]["id2label"][str(dt_pred)]
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
        n = len(test_docs)
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
                       len(events), test_metrics)

    # final save (reuses the last epoch's temperatures — no extra val pass)
    save_checkpoint(args.output, model, tokenizer, maps, heads, train_rows,
                    temps, summary,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch=len(events), steps_done=steps_done,
                    steps_done_micro=steps_done_micro)

    # selection enforcement (2026-09-20 audit R1): the pushed artifact must be
    # the SELECTED epoch, not merely the last one. Promote the selected
    # epoch's weights into latest/ when it differs from the final epoch.
    selected_epoch = selected["epoch"]
    if selected_epoch > 0 and selected_epoch != len(events):
        src = runs_dir / f"{run_id}-e{selected_epoch}"
        if src.is_dir():
            for f in src.iterdir():
                if f.is_file() and f.name != "summary.json":
                    shutil.copy2(f, args.output / f.name)
            (args.output / "summary.json").write_text(
                json.dumps(summary, sort_keys=True, indent=2))
            print(f"[trainer] promoted selected epoch {selected_epoch} "
                  f"weights into {args.output}", flush=True)
        else:
            print(f"[trainer] WARNING: selected epoch {selected_epoch} "
                  f"archive missing ({src}); latest/ holds the final epoch",
                  flush=True)
    elif selected_epoch == 0:
        print(f"[trainer] WARNING: selection gate NOT met by any epoch "
              f"(calibrated doc_type ECE never <= {ECE_BUDGET}); latest/ "
              f"holds the final epoch — treat this artifact as UNCALIBRATED",
              flush=True)

    print(f"checkpoint saved: {args.output}", flush=True)
    for p in sorted(args.output.iterdir()):
        print(f"  {p.name}", flush=True)
    print(f"selection: best val macro-F1 (observed) s.t. calibrated "
          f"ECE <= {ECE_BUDGET} -> epoch {selected['epoch']} "
          f"(macro_f1 {selected['macro_f1']}, ece {selected['ece']})",
          flush=True)
    print(f"run summary: device={device} seed={args.seed} "
          f"epochs={len(events)} wall={wall:.1f}s data={args.data}",
          flush=True)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        # optimizer/scheduler/resume state is training-internal — never push
        # it to the model repo (2026-09-20 audit R10: it doubled repo size).
        api.upload_folder(
            folder_path=str(args.output), repo_id=args.push_to_hub,
            repo_type="model",
            ignore_patterns=["optimizer.pt", "scheduler.pt", "resume.json"],
            commit_message=f"ModernBERT hierarchical classifier "
                           f"(epochs {args.epochs}, selected epoch "
                           f"{selected_epoch})")
        print(f"pushed: https://huggingface.co/{args.push_to_hub}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

