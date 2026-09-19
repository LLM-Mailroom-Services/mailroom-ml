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
import time
from collections import Counter, defaultdict
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
    """Shared ModernBERT backbone + per-head linear classifiers."""

    def __init__(self, base_model, head_sizes: dict[str, int]):
        super().__init__()
        self.backbone = base_model
        self.heads = nn.ModuleDict(
            {name: nn.Linear(base_model.config.hidden_size, n, bias=True)
             for name, n in sorted(head_sizes.items())})

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # ModernBERT has no pooler — use the <s> first-token embedding
        # (RoBERTa-style; the model card's classification recipe). Cast to
        # float32: the backbone runs bf16, the heads are fp32.
        pooled = out.last_hidden_state[:, 0].float()
        return {name: head(pooled) for name, head in self.heads.items()}


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
    # datasets-server indexes the repo).
    from datasets import load_dataset as hf_load_dataset

    cfg = "windows" if split != "test" else "documents"
    glob = f"hf://datasets/{data}/data/{cfg}/{split}/*.parquet"
    ds = hf_load_dataset("parquet", split=split, data_files={split: glob},
                         revision=_hub_revision())
    return [dict(r) for r in ds]


def tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    """Tokenize window text; rows carry doc_type/subclass labels.

    Rows are tokenized with truncation but NO padding — make_batches pads
    to the longest row in each batch (dynamic padding: p50 train window is
    3,845 tokens vs the 8,192 cap; fixed padding doubles GPU work).
    """
    enc = tokenizer([r["text"] for r in rows], padding=False,
                    truncation=True, max_length=max_length, return_tensors="np")
    return [{
        "input_ids": torch.from_numpy(enc["input_ids"][i]),
        "attention_mask": torch.from_numpy(enc["attention_mask"][i]),
        "doc_type": r["doc_type"],
        "subclass": r["subclass"],
        "filename": r["filename"],
    } for i, r in enumerate(rows)]


def class_weight_tensor(weights: dict[str, float], labels: list[str],
                        device) -> torch.Tensor:
    return torch.tensor([weights.get(label, 1.0) for label in labels],
                        dtype=torch.float32, device=device)


def make_batches(rows, batch_size: int, shuffle: bool, heads, device):
    """Yield batches; input_ids/attention_mask are pad-to-longest in batch.

    Dynamic padding (no fixed 8,192 pad): train windows measure p50 = 3,845
    tokens, so a fixed max_length pad wastes ~2x the GPU work — measured
    wall time at 8,192 padding was 6.1 s/step on an L4. Padding to the
    batch's longest row (attention-masked) cuts the epoch wall time roughly
    in half; truncation still never exceeds MAX_TOKENS.
    """
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = [rows[j] for j in idx[i:i + batch_size]]
        max_len = max(r["input_ids"].shape[0] for r in sel)
        # pad right to the batch's longest row (0-fill; masked out by the
        # attention mask, so the embedding of token 0 never contributes)
        pad_to = max_len  # bind outside the closure (ruff B023)

        def _pad(t):
            return F.pad(t, (0, pad_to - t.shape[0]))
        yield {
            "input_ids": torch.stack([_pad(r["input_ids"]) for r in sel]).to(device),
            "attention_mask": torch.stack(
                [_pad(r["attention_mask"]) for r in sel]).to(device),
            "doc_type": torch.tensor([heads["doc_type"]["label2id"][r["doc_type"]]
                                      for r in sel], device=device),
            "subclass": torch.tensor([heads[r["doc_type"]]["label2id"][r["subclass"]]
                                      for r in sel], device=device),
            "filename": [r["filename"] for r in sel],
        }


def head_loss(model, batch, heads, device) -> torch.Tensor:
    """doc_type CE on every row + subclass CE on each class's own rows.

    Subclass heads are taken from the ``heads`` config (built from
    ``labels.json`` — the data-driven single source of truth), so each class
    head contributes only through its own rows: 0 contribution elsewhere.
    """
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = F.cross_entropy(logits["doc_type"], batch["doc_type"],
                           weight=heads["doc_type"]["weight"])
    for cls in heads:
        if cls == "doc_type":
            continue
        sel = batch["doc_type"] == heads["doc_type"]["label2id"][cls]
        if sel.any():
            loss = loss + F.cross_entropy(
                logits[cls][sel], batch["subclass"][sel],
                weight=heads[cls]["weight"])
    return loss, logits


def train_epoch(model, batches, optimizer, scheduler, heads, device,
                grad_accum: int, step_limit: int = 0,
                log_every: int = 50) -> tuple[float, int]:
    """One epoch. Returns (mean loss, steps taken).

    Guardrails:
    - ``log_every``: per-step progress line so a stalled/starved container is
      visible in ``modal app logs`` within seconds instead of at epoch end.
    - ``step_limit``: cap micro-batches for the pre-flight smoke run (smoke
      exercises forward+backward+optimizer without burning an epoch).
    """
    model.train()
    total, n = 0.0, 0
    for step, batch in enumerate(batches):
        loss, _ = head_loss(model, batch, heads, device)
        (loss / grad_accum).backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        total += loss.item()
        n += 1
        if (step + 1) % log_every == 0:
            print(f"  step {step + 1} loss {loss.item():.4f}", flush=True)
        if step_limit and (step + 1) >= step_limit:
            break
    return total / max(1, n), n


@torch.no_grad()
def evaluate(model, batches, heads, maps, device) -> dict:
    """Per-head window metrics + document-level plurality-vote metrics.

    The document vote mirrors the sorter's merge: doc_type by plurality over
    windows; subclass by plurality over windows whose doc_type vote is the
    winning class (the pipeline's conditional structure).  Windows whose
    doc_type vote falls on the inference-only ``unknown`` class carry no
    subclass vote (the sorter routes them to the LLM instead) — counted as a
    doc_type miss when the label differs, never a crash.
    """
    model.eval()
    logits_by_head: dict[str, list] = defaultdict(list)
    labels_by_head: dict[str, list] = defaultdict(list)
    doc_votes: dict[str, list] = defaultdict(list)  # fn -> [(dt_pred, sc_pred)]
    doc_labels: dict[str, tuple] = {}
    total_loss = 0.0
    n_batches = 0
    for batch in batches:
        loss, logits = head_loss(model, batch, heads, device)
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
    return metrics


def ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    probs = F.softmax(logits, dim=-1)
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


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(-1)
    n_classes = logits.shape[1]
    f1s = []
    for c in range(n_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


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
    return ap


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
    base = AutoModel.from_pretrained(args.model, torch_dtype=dtype,
                                     attn_implementation="sdpa")
    if device.type == "cuda":
        base.gradient_checkpointing_enable()
    base = base.to(device)

    # head configs from the published labels.json
    local_data = Path(args.data)
    if local_data.exists():
        labels_path = local_data / "labels.json"
    else:
        from huggingface_hub import hf_hub_download

        labels_path = Path(hf_hub_download(args.data, "labels.json",
                                           repo_type="dataset",
                                           revision=_hub_revision()))
    maps = json.loads(labels_path.read_text())
    head_sizes = {name: len(cfg["labels"]) for name, cfg in maps.items()}
    model = HierarchicalClassifier(base, head_sizes).to(device)

    heads = {}
    for name, cfg in maps.items():
        heads[name] = {
            "label2id": cfg["label2id"],
            "weight": class_weight_tensor(cfg["weights"], cfg["labels"], device),
        }

    train_rows = tokenize_rows(load_dataset(args.data, "train"), tokenizer,
                               args.max_length)
    val_rows = tokenize_rows(load_dataset(args.data, "validation"), tokenizer,
                             args.max_length)
    if args.limit:
        train_rows = train_rows[:args.limit]
        val_rows = val_rows[:max(1, args.limit // 4)]
    print(f"windows: train {len(train_rows)} / validation {len(val_rows)}",
          flush=True)

    n_steps = math.ceil(len(train_rows) / args.batch_size)
    total_steps = n_steps * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, 1.0 - (step - warmup) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    stale = 0
    events: list[dict] = []
    selected: dict = {"epoch": 0, "macro_f1": -1.0, "ece": None}
    t0 = time.time()
    steps_done = 0
    for epoch in range(1, args.epochs + 1):
        remaining = max(0, args.max_steps - steps_done) if args.max_steps else 0
        loss, n = train_epoch(model, make_batches(train_rows, args.batch_size, True,
                                                  heads, device),
                              optimizer, scheduler, heads, device,
                              args.grad_accum, step_limit=remaining,
                              log_every=args.log_every)
        steps_done += n
        val = evaluate(model, make_batches(val_rows, args.batch_size, False,
                                           heads, device), heads, maps, device)
        # hardening seam: select the best val macro-F1 s.t. ECE is acceptable
        # (the plan's deployment gate — recorded, not enforced as an exit)
        if val["doc_type_macro_f1"] > selected["macro_f1"] \
                and val["doc_type_ece"] <= ECE_BUDGET:
            selected = {"epoch": epoch,
                        "macro_f1": val["doc_type_macro_f1"],
                        "ece": val["doc_type_ece"]}
        events.append({"epoch": epoch, "loss": round(loss, 4), **val})
        print(f"epoch {epoch}/{args.epochs} loss {loss:.4f} "
              f"val_loss {val['val_loss']:.4f} "
              f"doc_type_acc {val['doc_type_window_acc']} "
              f"doc_acc {val['doc_type_doc_acc']} "
              f"macro_f1 {val['doc_type_macro_f1']} "
              f"ece {val['doc_type_ece']}", flush=True)
        if val["val_loss"] < best_val:
            best_val = val["val_loss"]
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                print(f"early stop at epoch {epoch}", flush=True)
                break
        if args.max_steps and steps_done >= args.max_steps:
            print(f"max-steps reached ({args.max_steps}); smoke run complete",
                  flush=True)
            break
    wall = time.time() - t0
    print(f"training wall: {wall:.1f}s", flush=True)

    # temperature scaling per head on validation logits
    temps: dict[str, float] = {}
    val_logits: dict[str, list] = defaultdict(list)
    val_labels: dict[str, list] = defaultdict(list)
    with torch.no_grad():
        for batch in make_batches(val_rows, args.batch_size, False, heads, device):
            lg = model(batch["input_ids"], batch["attention_mask"])
            for name, t in lg.items():
                if name == "doc_type":
                    val_logits[name].append(t.cpu())
                    val_labels[name].append(batch["doc_type"].cpu())
                else:
                    sel = batch["doc_type"] == heads["doc_type"]["label2id"][name]
                    val_logits[name].append(t[sel].cpu())
                    val_labels[name].append(batch["subclass"][sel].cpu())
    for name in val_logits:
        lg = torch.cat(val_logits[name])
        lab = torch.cat(val_labels[name])
        if len(lab) < 2 or len(set(lab.tolist())) < 2:
            temps[name] = 1.0  # too few rows to fit T — leave uncalibrated
        else:
            temps[name] = fit_temperature(lg, lab)
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()},
          flush=True)

    # save checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    torch.save({name: head.state_dict() for name, head in model.heads.items()},
               args.output / "heads.pt")
    (args.output / "labels.json").write_text(
        json.dumps(maps, sort_keys=True, indent=2))
    # authentic-support sidecar: per (doc_type, subclass) train-row counts —
    # the ROUTE_MIN_AUTHENTIC_SUPPORT gate's data source (inference.py reads
    # train_counts.json; absent sidecar -> gate fails open to the LLM path).
    # Counts reflect the rows actually trained on (post --limit).
    train_counts: dict[str, dict[str, int]] = {}
    for r in train_rows:
        train_counts.setdefault(r["doc_type"], {}).setdefault(r["subclass"], 0)
        train_counts[r["doc_type"]][r["subclass"]] += 1
    (args.output / "train_counts.json").write_text(
        json.dumps(train_counts, sort_keys=True, indent=2))
    (args.output / "temperatures.json").write_text(
        json.dumps(temps, sort_keys=True, indent=2))
    summary = {
        "data": args.data,
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "epochs_run": len(events),
        "epochs_requested": args.epochs,
        "training_wall_s": round(wall, 1),
        "checkpoint_selection": {
            "rule": f"best val doc_type macro-F1 with doc_type ECE <= {ECE_BUDGET}",
            "epoch": selected["epoch"],
            "macro_f1": selected["macro_f1"],
            "ece": selected["ece"],
        },
        "epochs": events,
        "temperatures": {k: round(v, 3) for k, v in temps.items()},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2))
    print(f"checkpoint saved: {args.output}", flush=True)
    for p in sorted(args.output.iterdir()):
        print(f"  {p.name}", flush=True)
    print(f"selection: best val macro-F1 s.t. ECE <= {ECE_BUDGET} -> "
          f"epoch {selected['epoch']} (macro_f1 {selected['macro_f1']}, "
          f"ece {selected['ece']})", flush=True)
    print(f"run summary: device={device} seed={args.seed} "
          f"epochs={len(events)} wall={wall:.1f}s data={args.data}",
          flush=True)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(args.output), repo_id=args.push_to_hub,
                          repo_type="model",
                          commit_message=f"ModernBERT hierarchical classifier "
                                         f"(epochs {args.epochs})")
        print(f"pushed: https://huggingface.co/{args.push_to_hub}", flush=True)

    if args.eval_test:
        # held-out test: window each document at eval time (documents config
        # carries full text), plurality-vote the windows per head
        test_docs = load_dataset(args.data, "test")
        if args.limit:
            test_docs = test_docs[:args.limit]
        dt_correct = sc_correct = 0
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
                    if sc_pred == heads[cls]["label2id"][r["subclass"]]:
                        sc_correct += 1
        n = len(test_docs)
        if n:
            print(f"test doc_type acc: {dt_correct}/{n} = {dt_correct / n:.4f}")
            print(f"test subclass acc (conditional): {sc_correct}/{dt_correct} "
                  f"= {sc_correct / max(1, dt_correct):.4f}")
        else:
            print("test doc_type acc: 0/0 (no test documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

