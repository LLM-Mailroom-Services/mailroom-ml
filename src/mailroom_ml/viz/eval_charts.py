"""Per-document-type charts for ModernBERT eval JSON (no torch).

Renders the reports hub's classifier views as static SVG next to the markdown
reports, from the same ``eval_*.json`` artifacts ``write_eval_report.py``
reads:

- doc-type recall per class (one bar per run)
- row-normalised doc-type confusion matrix
- subclass accuracy per head, with the collapse signal (share of the head's
  predictions on its single most common subclass, against the most common
  true subclass's share)
- true vs predicted subclass counts per head
- selective risk: accuracy and coverage against the confidence threshold
- per-head ECE against the 0.05 exclusion line
- doc-type accuracy by window count, and a surrogate ALE of log2(windows)
  on P(doc type correct) with a 90% bootstrap band

Charts that need ``per_doc`` rows are skipped for older evals (run-3) that do
not carry them. Everything is computed from the JSON; nothing is hand-entered.

    python -m mailroom_ml.viz.eval_charts reports/eval_<tag>.json [more.json ...] \\
        --label "Arm B" [--label ...] --out reports/charts/<tag>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from mailroom_ml.viz import svgcharts as sc

CLASSES = ["contract", "corporate_record", "correspondence", "insurance_claim", "merger_agreement"]
LABEL = {"contract": "Contract", "corporate_record": "Corporate record", "correspondence": "Correspondence",
         "insurance_claim": "Insurance claim", "merger_agreement": "Merger agreement"}
ECE_EXCLUDE = 0.05
WINDOW_BUCKETS = [("1 window", 1, 1), ("2–3", 2, 3), ("4–8", 4, 8), ("9+", 9, 10**9)]


def confusion(report: dict) -> dict[str, Counter]:
    """``per_stratum_confusion`` ("true->pred": n) as {true: Counter(pred)}."""
    out: dict[str, Counter] = {c: Counter() for c in CLASSES}
    for key, n in (report.get("per_stratum_confusion") or {}).items():
        t, p = key.split("->")
        out.setdefault(t, Counter())[p] += int(n)
    return out


def recall_by_class(report: dict) -> dict[str, float | None]:
    conf = confusion(report)
    return {c: (conf[c][c] / sum(conf[c].values()) if sum(conf[c].values()) else None) for c in CLASSES}


def subclass_stats(report: dict) -> dict[str, dict]:
    """Per head, over rows whose doc type was right (the scorable set)."""
    out = {}
    for c in CLASSES:
        rows = [r for r in report.get("per_doc") or [] if r.get("gt_doc_type") == c and r.get("dt_correct")
                and r.get("gt_subclass") is not None and r.get("pred_subclass") is not None]
        if not rows:
            continue
        pred, true = Counter(r["pred_subclass"] for r in rows), Counter(r["gt_subclass"] for r in rows)
        out[c] = {"n": len(rows), "correct": sum(bool(r.get("sc_correct")) for r in rows),
                  "pred": pred, "true": true,
                  "top_pred": pred.most_common(1)[0], "top_true": true.most_common(1)[0]}
    return out


def window_stats(report: dict) -> list[dict]:
    rows = report.get("per_doc") or []
    out = []
    for lab, lo, hi in WINDOW_BUCKETS:
        sel = [r for r in rows if lo <= int(r.get("n_windows") or 1) <= hi]
        out.append({"label": lab, "n": len(sel), "correct": sum(bool(r["dt_correct"]) for r in sel)})
    return out


def windows_ale(report: dict) -> dict | None:
    """Logistic-surrogate ALE of log2(n_windows) on P(doc type correct), OOD flag as a covariate."""
    from mailroom_ml.viz import ale

    rows = [{"y": 1.0 if r["dt_correct"] else 0.0, "lw": math.log2(max(int(r.get("n_windows") or 1), 1)),
             "ood": 1.0 if r.get("ood_flag") else 0.0} for r in report.get("per_doc") or []]
    if len(rows) < 30 or len({r["y"] for r in rows}) < 2:
        return None
    try:
        return ale.ale(rows, "y", ["lw", "ood"], "lw", kind="logit", boot=200)
    except ValueError:
        return None


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def render(runs: list[tuple[str, dict]]) -> dict[str, str]:
    """``runs`` = [(label, eval_json)], oldest first; the last run is the subject of single-run charts.

    Returns {filename: svg}.
    """
    charts: dict[str, str] = {}
    cats = [LABEL[c] for c in CLASSES]
    older = [sc.GREY, sc.ORANGE, sc.GREEN, "#8a5cd6", "#c7457d"]
    colors = [older[i % len(older)] for i in range(len(runs) - 1)] + [sc.BLUE]  # newest run in blue
    series = [(lab, colors[i], [recall_by_class(r)[c] for c in CLASSES]) for i, (lab, r) in enumerate(runs)]
    n = runs[-1][1].get("n_docs")
    charts["doc_type_recall.svg"] = sc.grouped_bars(
        "Document-type recall by class", cats, series, ymax=1.0, tick_fmt=lambda t: f"{t:.1f}",
        subtitle=f"Share of each true class routed to it · {', '.join(lab for lab, _ in runs)} · n = {n} documents")

    label, rep = runs[-1]
    conf = confusion(rep)
    charts["confusion.svg"] = sc.heatmap(
        f"Doc-type confusion matrix · {label}", [LABEL[c] for c in CLASSES], [LABEL[c] for c in CLASSES],
        [[conf[t][p] for p in CLASSES] for t in CLASSES],
        subtitle="Rows = true class, columns = predicted; shade = share of the row (blue diagonal, red errors)")

    sr = rep.get("selective_risk") or {}
    rows = sr.get("rows") or []
    if rows:
        th = [r["threshold"] for r in rows]
        pick = sr.get("recommended_threshold")
        # the pick is a float read back from JSON: match by tolerance (the
        # nearest row within 1e-6), never ``==`` — a 1-ulp difference would
        # report "no threshold met the budget" for a threshold that was met.
        at = None
        if pick is not None:
            near = min(rows, key=lambda r: abs(float(r["threshold"]) - float(pick)))
            if abs(float(near["threshold"]) - float(pick)) <= 1e-6:
                at = near
        sub = (f"Pick {pick:.2f}: {_pct(at['coverage'])} of windows accepted at {at['accuracy'] * 100:.1f}% accuracy"
               if at else "No threshold met the error budget")
        charts["selective_risk.svg"] = sc.lines(
            f"Calibration and selective risk · {label}",
            [{"name": "accuracy of accepted windows", "color": sc.BLUE, "x": th, "y": [r["accuracy"] for r in rows]},
             {"name": "coverage (share accepted)", "color": sc.ORANGE, "x": th, "y": [r["coverage"] for r in rows]},
             {"name": "Wilson lower bound", "color": sc.GREY, "x": th, "y": [r.get("wilson_lower_acc") for r in rows],
              "dash": "4 3"}],
            subtitle=sub, x_label="confidence threshold", y_range=(0.0, 1.0), y_fmt=lambda t: f"{t:.1f}",
            x_fmt=lambda t: f"{t:.2f}",
            vlines=[(pick, "pick")] if pick is not None else [])

    ece = rep.get("head_ece") or {}
    if ece:
        heads = ["doc_type"] + [c for c in CLASSES if c in ece]
        wc = (rep.get("window_calibration") or {}).get("ece")
        charts["head_ece.svg"] = sc.grouped_bars(
            f"Per-head calibration error · {label}", ["doc type" if h == "doc_type" else LABEL[h] for h in heads],
            [("ECE", sc.BLUE, [ece.get(h) for h in heads])], fmt=lambda v: f"{v:.3f}", tick_fmt=lambda t: f"{t:.2f}",
            ref=(ECE_EXCLUDE, f"exclusion line {ECE_EXCLUDE}"),
            subtitle=f"Expected calibration error per head; window-level ECE {wc}" if wc is not None else "")

    subs = subclass_stats(rep)
    if subs:
        have = [c for c in CLASSES if c in subs]
        charts["subclass_collapse.svg"] = sc.grouped_bars(
            f"Subclass accuracy and collapse signal · {label}", [LABEL[c] for c in have],
            [("subclass accuracy", sc.BLUE, [subs[c]["correct"] / subs[c]["n"] for c in have]),
             ("share of predictions on one subclass", sc.ORANGE, [subs[c]["top_pred"][1] / subs[c]["n"] for c in have]),
             ("most common true subclass's share", sc.GREY, [subs[c]["top_true"][1] / subs[c]["n"] for c in have])],
            ymax=1.0, tick_fmt=lambda t: f"{t:.1f}", ref=(0.75, "P0 subclass gate 0.75"),
            subtitle="Documents whose doc type was right. Orange far above grey = the head collapsed onto one answer")
        for c in have:
            s = subs[c]
            labs = sorted(set(s["pred"]) | set(s["true"]), key=lambda k: (-(s["pred"][k] + s["true"][k]), k))[:12]
            charts[f"subclass_{c}.svg"] = sc.hbars(
                f"{LABEL[c]} subclasses: true vs predicted · {label}", [k.replace("_", " ") for k in labs],
                [("true", sc.GREY, [s["true"][k] for k in labs]), ("predicted", sc.BLUE, [s["pred"][k] for k in labs])],
                subtitle=f"{s['correct']}/{s['n']} right · top prediction '{s['top_pred'][0]}' takes "
                         f"{s['top_pred'][1]}/{s['n']}")

    ws = window_stats(rep)
    if any(w["n"] for w in ws):
        charts["accuracy_by_windows.svg"] = sc.grouped_bars(
            f"Doc-type accuracy by document length · {label}", [f"{w['label']} (n={w['n']})" for w in ws],
            [("doc-type accuracy", sc.BLUE, [w["correct"] / w["n"] if w["n"] else None for w in ws])],
            ymax=1.0, fmt=lambda v: f"{v * 100:.1f}%", tick_fmt=_pct, subtitle="Documents bucketed by 8k-token window count")
    a = windows_ale(rep)
    if a:
        charts["ale_windows.svg"] = sc.lines(
            f"ALE: document length on P(doc type correct) · {label}",
            [{"name": "ALE (90% bootstrap band)", "color": sc.BLUE, "x": a["x"], "y": a["ale"], "lo": a["lo"], "hi": a["hi"]}],
            subtitle=f"Logistic surrogate on log2(windows) + OOD flag, n = {a['n']}, cross-validated "
                     f"{a['fit']['metric']} {a['fit']['value']}. Flat = length does not explain errors",
            x_label="windows (log scale)", y_label="Δ P(correct)", rug=a["rug"],
            x_fmt=lambda t: str(round(2 ** t)), y_fmt=lambda t: f"{t * 100:+.0f} pt")
    return charts


CAPTIONS = {
    "doc_type_recall.svg": "Document-type recall by class",
    "confusion.svg": "Doc-type confusion matrix",
    "subclass_collapse.svg": "Subclass accuracy and collapse signal",
    "selective_risk.svg": "Calibration and selective risk",
    "head_ece.svg": "Per-head calibration error",
    "accuracy_by_windows.svg": "Doc-type accuracy by document length",
    "ale_windows.svg": "ALE of document length on P(doc type correct)",
}


def index_md(runs: list[tuple[str, str]], charts: dict[str, str]) -> str:
    """Markdown gallery embedding every chart, in a fixed order."""
    lines = ["# Eval charts", "",
             "Generated by `python -m mailroom_ml.viz.eval_charts` from:", ""]
    lines += [f"- {lab}: `{path}`" for lab, path in runs]
    lines.append("")
    order = list(CAPTIONS) + sorted(k for k in charts if k.startswith("subclass_") and k not in CAPTIONS)
    for name in order:
        if name in charts:
            cap = CAPTIONS.get(name) or f"{LABEL[name[len('subclass_'):-4]]} subclasses: true vs predicted"
            lines += [f"## {cap}", "", f"![{cap}]({name})", ""]
    return "\n".join(lines)


def write(eval_paths: list[str], labels: list[str], out: Path) -> list[Path]:
    runs = [(lab, json.loads(Path(p).read_text())) for lab, p in zip(labels, eval_paths, strict=False)]
    charts = render(runs)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, svg in charts.items():
        (out / name).write_text(svg)
        written.append(out / name)
    (out / "README.md").write_text(index_md(list(zip(labels, eval_paths, strict=False)), charts))
    return written + [out / "README.md"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("eval_json", nargs="+", help="eval JSON files, oldest first; the last is charted in detail")
    ap.add_argument("--label", action="append", default=[], help="display label per eval JSON (default: file stem)")
    ap.add_argument("--out", required=True, type=Path, help="output directory for SVGs + README.md")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    labels = args.label or [Path(p).stem.removeprefix("eval_") for p in args.eval_json]
    if len(labels) != len(args.eval_json):
        raise SystemExit("--label must be given once per eval JSON")
    for p in write(args.eval_json, labels, args.out):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
