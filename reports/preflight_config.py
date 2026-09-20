#!/usr/bin/env python3
"""Config pre-flight: scheduler plan, effective class weights, support floor.

Computes, on the REBUILT dataset, exactly what the trainer will do at launch:
  A. scheduler plan (old buggy units vs corrected optimizer-step units)
  B. effective per-head class weights under the chosen loss levers
  C. subclass support-floor effect (which classes remap/drop) + val adequacy
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "modernbert_training" / "stage"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from train_modernbert import LossConfig, _scheduler_plan  # noqa: E402


def load(cfg: str, split: str) -> pd.DataFrame:
    p = STAGE / "data" / cfg / split
    return pd.concat([pd.read_parquet(f) for f in sorted(p.glob("*.parquet"))],
                     ignore_index=True)


def main() -> int:
    labels = json.loads((STAGE / "labels.json").read_text())
    train_w = load("windows", "train")
    val_w = load("windows", "validation")
    n_train = len(train_w)
    batch, accum = 4, 8
    out: dict = {"n_train_windows": n_train, "batch_size": batch,
                 "grad_accum": accum}

    # ---- A. scheduler plan --------------------------------------------------
    plans = {}
    for epochs in (2, 3):
        total, warmup = _scheduler_plan(n_train, batch, accum, epochs, 0.06)
        micro = math.ceil(n_train / batch)
        opt_per_epoch = math.ceil(micro / accum)
        # OLD (buggy) units: total in micro-batches, warmup fraction of that
        old_total = micro * epochs
        old_warmup = max(1, int(old_total * 0.06))
        plans[epochs] = {
            "micro_per_epoch": micro,
            "opt_per_epoch": opt_per_epoch,
            "correct_total_opt_steps": total,
            "correct_warmup_opt_steps": warmup,
            "correct_warmup_pct_of_run": round(100 * warmup / total, 1),
            "OLD_buggy_total_steps_micro": old_total,
            "OLD_buggy_warmup_opt_steps": old_warmup,
            "OLD_warmup_pct_of_epoch1": round(100 * old_warmup / opt_per_epoch, 1),
            "lr_at_end_pct_of_peak": round(
                100 * (1 - (total - warmup) / total), 1),
        }
    out["A_scheduler_plan"] = plans

    # ---- B. effective class weights ----------------------------------------
    for mode, cap in (("inverse", 10.0), ("sqrt-inverse", 10.0)):
        cfg = LossConfig(weight_mode=mode, weight_cap=cap)
        per_head = {}
        for head, spec in labels.items():
            labels_list = spec["labels"]
            w = spec.get("weights", {})
            eff = cfg.transform_weights(w, labels_list).tolist()
            pairs = sorted(zip(labels_list, eff, strict=False),
                           key=lambda kv: -kv[1])
            per_head[head] = {
                "n_classes": len(labels_list),
                "max_weight": round(max(eff), 3),
                "min_weight": round(min(eff), 3),
                "ratio": round(max(eff) / max(min(eff), 1e-9), 2),
                "top3": [(k, round(v, 2)) for k, v in pairs[:3]],
                "bottom3": [(k, round(v, 2)) for k, v in pairs[-3:]],
            }
        out[f"B_weights_{mode}_cap{int(cap)}"] = per_head

    # ---- C. subclass support floor + val adequacy --------------------------
    min_rows = 12
    floors = {}
    for head, spec in labels.items():
        if head == "doc_type":
            continue
        counts = Counter(train_w[train_w["doc_type"] == head]["subclass"])
        val_counts = Counter(val_w[val_w["doc_type"] == head]["subclass"])
        low = sorted(s for s, c in counts.items()
                     if c < min_rows and s != "other")
        has_other = "other" in spec["label2id"] and "other" not in set(low)
        floors[head] = {
            "n_classes": len(spec["labels"]),
            "low_support_classes": low,
            "remap_to_other": has_other,
            "val_cells_lt5": sum(1 for s in spec["labels"]
                                 if val_counts.get(s, 0) < 5),
            "val_cells_zero": sum(1 for s in spec["labels"]
                                  if val_counts.get(s, 0) == 0),
            "classes_after_floor": len(spec["labels"]) - (
                0 if has_other else len(low)),
        }
    out["C_support_floor_min12"] = floors

    print(json.dumps(out, indent=2, default=str))
    (ROOT / "reports" / "preflight_config.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
