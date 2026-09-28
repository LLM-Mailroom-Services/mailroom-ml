"""Artifact-driven markdown reports for ModernBERT eval (no torch)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from mailroom_ml.config import CLASSIFIER_MODEL_REPO
from mailroom_ml.m9a_gates import gate_status, gates_all_met, m9a_gate_rows

PER_DOC_FULL_MAX = 500
ECE_EXCLUDE = 0.05


@dataclass
class ReportContext:
    run_tag: str
    arm: str = ""
    run_id: str = ""
    checkpoint: str = ""
    training_log: str = ""
    gates_check_file: str = ""
    baseline_doc_type: dict[str, float] = field(default_factory=dict)


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _dt_counts(report: dict) -> tuple[int, int]:
    per_doc = report.get("per_doc") or []
    n = len(per_doc) or int(report.get("n_docs") or 0)
    ok = sum(1 for r in per_doc if r.get("dt_correct"))
    return ok, n


def _sc_counts(report: dict) -> tuple[int, int]:
    per_doc = report.get("per_doc") or []
    scorable = [r for r in per_doc if r.get("dt_correct")]
    ok = sum(1 for r in scorable if r.get("sc_correct"))
    return ok, len(scorable)


def _training_config_lines(summary: dict | None, arm: str) -> list[str]:
    if not summary:
        return []
    cfg = summary.get("config") or summary.get("train_config") or {}
    if not cfg and summary.get("hyperparameters"):
        cfg = summary["hyperparameters"]
    lam = cfg.get("loss_lambda_dt", cfg.get("loss-lambda-dt"))
    arm_note = f" (Arm {arm})" if arm else ""
    rows = [
        ("`loss_lambda_dt`", f"**{lam}**{arm_note}" if lam is not None else "—"),
        ("epochs", cfg.get("epochs", summary.get("epochs"))),
        ("batch × grad_accum", f"{cfg.get('batch_size', cfg.get('batch', '?'))} × "
         f"{cfg.get('grad_accum', cfg.get('gradient_accumulation_steps', '?'))}"),
        ("lr", cfg.get("lr", cfg.get("learning_rate"))),
        ("subclass_head_lr", cfg.get("subclass_head_lr") or "= lr"),
        ("seed", cfg.get("seed")),
        ("max_length", cfg.get("max_length")),
        ("label_smoothing", cfg.get("label_smoothing", cfg.get("subclass_label_smoothing"))),
        ("weight_mode / cap", f"{cfg.get('weight_mode', '?')} / {cfg.get('weight_cap', '?')}"),
        ("subclass_loss_norm", cfg.get("subclass_loss_norm", "weighted-mean")),
        ("subclass_logit_adjust", cfg.get("subclass_logit_adjust", 0.0)),
        ("warmup_frac", cfg.get("warmup_frac")),
        ("mlp_heads", cfg.get("mlp_heads")),
        ("freeze_backbone_epochs", cfg.get("freeze_backbone_epochs")),
        ("model", cfg.get("model_name", cfg.get("model"))),
        ("data", cfg.get("data_path", cfg.get("stage"))),
    ]
    wall = summary.get("wall_seconds") or summary.get("training_wall_seconds")
    if wall:
        rows.append(("training wall time", f"~{wall / 3600:.1f} h ({wall:.0f} s)"))
    lines = ["## Training config (context for test readout)", "", "| Setting | Value |", "| --- | --- |"]
    for k, v in rows:
        if v is not None and v != "?":
            lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def _trainer_vs_harness(summary: dict | None, report: dict) -> list[str]:
    if not summary:
        return []
    sel = summary.get("checkpoint_selection") or {}
    epochs = summary.get("epoch_metrics") or summary.get("validation_epochs") or []
    ep = sel.get("epoch")
    val_acc = sel.get("doc_acc") or sel.get("doc_type_acc")
    val_mf1 = sel.get("macro_f1")
    val_sc = sel.get("subclass_objective")
    val_ece = sel.get("ece")
    if not ep and epochs:
        last = epochs[-1] if isinstance(epochs, list) else {}
        ep = last.get("epoch", len(epochs))
        val_acc = val_acc or last.get("doc_type_acc") or last.get("doc_acc")
        val_mf1 = val_mf1 or last.get("doc_type_macro_f1")
        val_sc = val_sc or last.get("subclass_objective")
        val_ece = val_ece or last.get("ece_calibrated")
    wc = report.get("window_calibration") or {}
    lines = [
        "### Trainer vs eval harness",
        "",
        "For **held-out test**, use the eval JSON — not inline trainer test fields alone — "
        "when judging #112 or updating the Hub model card.",
        "",
        "| Surface | doc_type | subclass (cond.) | window ECE |",
        "| --- | ---: | ---: | ---: |",
    ]
    if ep is not None:
        val_line = f"acc {_fmt(val_acc)}; macro-F1 {_fmt(val_mf1)}"
        lines.append(
            f"| Val epoch {ep} (trainer) | {val_line} | objective {_fmt(val_sc)} | "
            f"ECE {_fmt(val_ece)} (cal.) |"
        )
    dt_ok, n = _dt_counts(report)
    sc_ok, sc_n = _sc_counts(report)
    lines.append(
        f"| **Held-out test (harness)** | **{_fmt(report.get('doc_type_accuracy'))}** "
        f"({dt_ok}/{n}) | **{_fmt(report.get('subclass_accuracy_conditional'))}** "
        f"({sc_ok}/{sc_n} scorable) | **{_fmt(wc.get('ece'))}** |"
    )
    lines.append("")
    return lines


def _cohorts_section(report: dict) -> list[str]:
    cohorts = report.get("cohorts") or {}
    if not cohorts:
        return []
    lines = [
        "## Cohorts (#104 single vs multi-window)",
        "",
        "| cohort | n_docs | doc_type_accuracy | mean_agreement | window ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("single-window", "multi-window"):
        c = cohorts.get(name) or {}
        if c:
            lines.append(
                f"| {name} | {c.get('n_docs')} | {_fmt(c.get('doc_type_accuracy'))} | "
                f"{_fmt(c.get('mean_agreement'))} | {_fmt(c.get('window_ece'))} |"
            )
    lines.append("")
    return lines


def _selective_risk_section(report: dict) -> list[str]:
    sr = report.get("selective_risk") or {}
    if not sr:
        return []
    lines = [
        "## Selective-risk sweep (global)",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| budget_met | **{sr.get('budget_met')}** |",
        f"| recommended deployment threshold | **{sr.get('recommended_threshold')}** |",
        f"| error_budget | {sr.get('error_budget')} |",
        f"| coverage at pick | ~{_fmt(sr.get('coverage'), 3)} |",
        "",
    ]
    cohorts = report.get("cohorts") or {}
    mw = cohorts.get("multi-window") or {}
    mwsr = mw.get("selective_risk") or {}
    if mwsr:
        lines.append(
            "Per-cohort selective-risk on multi-window reports "
            f"`budget_met: {mwsr.get('budget_met')}` with "
            f"`recommended_threshold: {mwsr.get('recommended_threshold')}`."
        )
        lines.append("")
    return lines


def _per_head_section(report: dict) -> list[str]:
    per_head = report.get("per_head") or {}
    if not per_head:
        return []
    lines = [
        "## Per-head test macro-F1 (observed)",
        "",
        "Support counts are per-class **test** support from `per_head.<head>.support`.",
        "",
        "| head | macro-F1 | notes |",
        "| --- | ---: | --- |",
    ]
    for head in sorted(per_head):
        info = per_head[head]
        if not isinstance(info, dict):
            continue
        mf1 = info.get("macro_f1")
        note = ""
        if head == "contract" and mf1 is not None and mf1 < 0.05:
            note = "majority-prior collapse persists"
        elif head == "correspondence" and mf1 is not None and mf1 < 0.25:
            note = "below #112 floor"
        elif head == "insurance_claim" and mf1 is not None and mf1 > 0.9:
            note = "near-balanced control head"
        elif mf1 is not None and mf1 >= 0.2:
            note = "partial learning"
        lines.append(f"| {head} | {_fmt(mf1)} | {note} |")
    lines.append("")
    return lines


def _head_ece_policy(summary: dict | None, report: dict) -> list[str]:
    head_ece = report.get("head_ece") or {}
    excl = {}
    if summary:
        excl = ((summary.get("checkpoint_selection") or {})
                .get("head_exclusion_policy") or {}).get("excluded") or {}
    if not head_ece and not excl:
        return []
    lines = [
        "### Calibrated head ECE (validation / exclusion policy)",
        "",
        "From trainer checkpoint selection (validation, not test) when summary is present; "
        "else from eval `head_ece` sidecar.",
        "",
        "| head | ECE (calibrated) | fast-path excluded |",
        "| --- | ---: | --- |",
    ]
    heads = sorted(set(head_ece) | set(excl))
    for h in heads:
        ece = head_ece.get(h)
        if ece is None and h in excl:
            ece = (excl[h] or {}).get("ece")
        excluded = h in excl or (ece is not None and float(ece) > ECE_EXCLUDE)
        lines.append(
            f"| {h} | {_fmt(ece)} | {'**yes**' if excluded else 'no'} |"
        )
    lines.append("")
    return lines


def _gates_section(report: dict, gates_file: str) -> list[str]:
    lines = ["## M9a #112 gates (held-out test)", ""]
    if gates_file:
        lines.append(f"Verified with `training/check_m9a_gates.py` → `{gates_file}`.")
        lines.append("")
    lines += [
        "| Gate | Actual | Threshold | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, actual, thr, how in m9a_gate_rows(report):
        op = "≥" if how == "ge" else "≤"
        st = gate_status(actual, thr, how)
        bold = "**" if st == "NOT MET" else ""
        lines.append(
            f"| {name} | {_fmt(actual)} | {op} {thr} | {bold}{st}{bold} |"
        )
    verdict = "PASS" if gates_all_met(report) else "FAIL"
    n_fail = sum(
        1 for _, a, t, h in m9a_gate_rows(report) if gate_status(a, t, h) != "MET"
    )
    lines += [
        "",
        f"**Overall #112 gate verdict:** **{verdict}** "
        f"({n_fail}/4 gates not met)." if verdict == "FAIL" else
        f"**Overall #112 gate verdict:** **{verdict}**.",
        "",
    ]
    rg = report.get("recorded_gates") or {}
    if rg:
        lines += [
            "### Report-only P0 (eval harness)",
            "",
            "| Gate | Actual | Threshold | met |",
            "| --- | ---: | ---: | --- |",
        ]
        for key, g in sorted(rg.items()):
            if isinstance(g, dict):
                lines.append(
                    f"| {key.replace('_', ' ')} | {_fmt(g.get('actual'))} | "
                    f"{g.get('threshold')} | {g.get('met')} |"
                )
        lines.append("")
    return lines


def _baseline_section(report: dict, baselines: dict[str, float]) -> list[str]:
    if not baselines:
        return []
    dt = report.get("doc_type_accuracy")
    lines = [
        "## Baseline comparison (doc_type test accuracy)",
        "",
        "| source | doc_type_accuracy |",
        "| ---: | ---: |",
    ]
    for label, acc in baselines.items():
        mark = "**" if label.startswith("this") else ""
        lines.append(f"| {label} | {mark}{_fmt(acc)}{mark} |")
    if dt is not None and "run-3" in str(baselines):
        r3 = baselines.get("run-3 (`eval_run3_20260921.json`)") or baselines.get("run-3")
        if r3 is not None:
            pp = (float(dt) - float(r3)) * 100
            sc = report.get("subclass_accuracy_conditional")
            lines.append("")
            lines.append(
                f"Δ vs run-3: **{pp:+.2f} pp** doc_type; conditional subclass "
                f"{_fmt(sc)} on this checkpoint."
            )
    lines.append("")
    return lines


def _telemetry_section(report: dict) -> list[str]:
    tel = report.get("run_telemetry") or {}
    cm = report.get("comparable_metrics") or {}
    lat = tel.get("latency_seconds_per_document") or cm.get("latency_seconds_per_document")
    if lat is None and not tel:
        return []
    lines = ["## Serving / telemetry", "", "| metric | value |", "| --- | --- |"]
    if lat is not None:
        lines.append(f"| latency_s_per_doc | {_fmt(lat, 6)} |")
    for k in ("wall_seconds", "gpu", "eval_subset"):
        if tel.get(k) is not None:
            lines.append(f"| {k} | {tel[k]} |")
    ood = report.get("ood") or {}
    if ood.get("rate") is not None:
        lines.append(f"| ood_rate | {_fmt(ood.get('rate'))} |")
    lines.append("")
    return lines


def _analyst_insights(report: dict, summary: dict | None) -> list[str]:
    per_doc = report.get("per_doc") or []
    lines = ["## Analyst insights & findings", ""]
    if not per_doc:
        lines.append("- No `per_doc` rows in eval JSON; insights limited to headline scalars.")
        lines.append("")
        return lines
    by_dt: dict[str, list[bool]] = defaultdict(list)
    for r in per_doc:
        by_dt[str(r.get("gt_doc_type"))].append(bool(r.get("dt_correct")))
    if by_dt:
        worst = min(by_dt, key=lambda k: sum(by_dt[k]) / len(by_dt[k]))
        best = max(by_dt, key=lambda k: sum(by_dt[k]) / len(by_dt[k]))
        lines.append(
            f"- **Doc_type routing:** best `{best}` "
            f"{sum(by_dt[best]) / len(by_dt[best]):.3f} acc (n={len(by_dt[best])}); "
            f"weakest `{worst}` "
            f"{sum(by_dt[worst]) / len(by_dt[worst]):.3f} acc (n={len(by_dt[worst])})."
        )
    per_head = report.get("per_head") or {}
    sparse = [
        (h, (info or {}).get("macro_f1"))
        for h, info in per_head.items()
        if isinstance(info, dict) and h not in ("doc_type",)
    ]
    sparse.sort(key=lambda x: (x[1] is None, x[1] or 0))
    if sparse:
        h, mf1 = sparse[0]
        lines.append(
            f"- **Subclass bottleneck:** lowest test macro-F1 head `{h}` at {_fmt(mf1)} "
            f"(#112 floors: contract ≥0.20, correspondence ≥0.25)."
        )
    cohorts = report.get("cohorts") or {}
    sw = cohorts.get("single-window") or {}
    mw = cohorts.get("multi-window") or {}
    if sw and mw:
        dsw = sw.get("doc_type_accuracy")
        dmw = mw.get("doc_type_accuracy")
        if dsw is not None and dmw is not None:
            lines.append(
                f"- **Window cohort (#104):** multi-window doc_type acc {_fmt(dmw)} vs "
                f"single-window {_fmt(dsw)} (n={mw.get('n_docs')}/{sw.get('n_docs')})."
            )
    sr = report.get("selective_risk") or {}
    if sr.get("budget_met") is True:
        lines.append(
            f"- **Selective-risk:** global budget_met with deployment threshold "
            f"{sr.get('recommended_threshold')} at coverage ~{_fmt(sr.get('coverage'), 3)}."
        )
    elif sr.get("budget_met") is False:
        lines.append("- **Selective-risk:** global budget_met **false** — do not deploy on threshold sweep alone.")
    fp = report.get("fast_path_rate")
    if fp is not None:
        lines.append(f"- **Fast-path rate:** {_fmt(fp)} of docs would route on classifier gate.")
    lines.append("")
    return lines


def _per_doc_table(report: dict) -> list[str]:
    per_doc = report.get("per_doc") or []
    n = len(per_doc)
    if not per_doc:
        return []
    if n > PER_DOC_FULL_MAX:
        rows = [r for r in per_doc if not r.get("dt_correct") or not r.get("sc_correct", True)]
        title = f"## Per-document errors ({len(rows)} of {n} docs)"
        note = (
            f"_Full per-doc table omitted (n={n} > {PER_DOC_FULL_MAX}); "
            "see `per_doc` in eval JSON._"
        )
    else:
        rows = per_doc
        title = "## Per-document scores"
        note = ""
    lines = [title, "", note, ""] if note else [title, ""]
    lines += [
        "| filename | gt doc_type | pred | dt ok | sc ok | n_win | fast_path |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for r in rows[:500]:
        lines.append(
            f"| `{r.get('filename')}` | {r.get('gt_doc_type')} | {r.get('pred_doc_type')} | "
            f"{r.get('dt_correct')} | {r.get('sc_correct')} | {r.get('n_windows')} | "
            f"{r.get('fast_path')} |"
        )
    if len(rows) > 500:
        lines.append(f"| … | | | | | | | ({len(rows) - 500} more) |")
    lines.append("")
    return lines


def _reproduce_test(ctx: ReportContext, eval_path: str) -> list[str]:
    ckpt = ctx.checkpoint or f"data/modernbert_training/runs/{ctx.run_tag}/latest"
    return [
        "## Reproduce",
        "",
        "```bash",
        f"uv run python training/eval_modernbert.py --checkpoint {ckpt} \\",
        "  --stage data/modernbert_training/stage --sample 0 --selective-risk \\",
        "  --write-routing-thresholds \"${CKPT}/routing_thresholds.json\" --json \\",
        f"  > reports/eval_{ctx.run_tag}.json",
        f"uv run python training/check_m9a_gates.py {eval_path}",
        f"uv run python training/write_eval_report.py test --run-tag {ctx.run_tag} \\",
        f"  --eval-json {eval_path}",
        "```",
        "",
    ]


def _artifacts_table(ctx: ReportContext, eval_path: str, test_eval_copy: str) -> list[str]:
    return [
        "## Artifacts",
        "",
        "| path | role |",
        "| --- | --- |",
        f"| `{eval_path}` | canonical eval JSON |",
        f"| `{test_eval_copy}` | TEST-EVAL copy |",
        f"| `reports/M9a-REPORT-{ctx.run_tag}.md` | training-focused report |",
        f"| `reports/TEST-EVAL/TEST-EVAL-REPORT-{ctx.run_tag}.md` | held-out test harness report |",
        "| `reports/TEST-EVAL/MANIFEST.json` | run index + gate snapshot |",
        "",
    ]


def render_test_eval_report(
    report: dict,
    *,
    ctx: ReportContext,
    summary: dict | None = None,
    eval_json_canonical: str = "",
    eval_json_test_eval: str = "",
) -> str:
    """323-doc TEST-EVAL markdown from eval JSON."""
    run_tag = ctx.run_tag
    canonical = eval_json_canonical or f"reports/eval_{run_tag}.json"
    te_copy = eval_json_test_eval or f"reports/TEST-EVAL/eval_{run_tag}.json"
    wc = report.get("window_calibration") or {}
    dt_ok, n = _dt_counts(report)
    sc_ok, sc_n = _sc_counts(report)
    arm_label = f"Arm {ctx.arm} " if ctx.arm else ""
    lines = [
        f"# TEST-EVAL — {arm_label}(`{run_tag}`)".replace("  ", " "),
        "",
        "**Eval role:** authoritative held-out test for M9a **#112** gates and Hub release  ",
        f"**Run tag:** `{run_tag}`  ",
    ]
    if ctx.run_id:
        lines.append(f"**Run ID:** `{ctx.run_id}`  ")
    if ctx.arm:
        lines.append(f"**Arm:** {ctx.arm}  ")
    lines += [
        f"**Checkpoint:** `{report.get('checkpoint', ctx.checkpoint)}`  ",
        f"**Eval JSON (this dir):** `{te_copy}`  ",
        f"**Eval JSON (canonical):** `{canonical}`  ",
        f"**Artifact SHA:** `{report.get('artifact_sha', '')}`  ",
    ]
    if ctx.training_log:
        lines.append(f"**Training log:** `{ctx.training_log}`  ")
    lines.append(f"**Training report:** `reports/M9a-REPORT-{run_tag}.md`")
    lines.append("")
    lines.extend(_training_config_lines(summary, ctx.arm))
    lines += [
        "## Test harness protocol",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| Command surface | `training/eval_modernbert.py` |",
        f"| `--subset` | `{report.get('eval_subset', 'test')}` |",
        f"| Documents | **{report.get('n_docs', n)}** (`--sample 0`) |",
        "| `--max-length` | 8192 |",
        f"| `--seed` | {report.get('seed', 42)} |",
        f"| Windows evaluated | {wc.get('n_windows')} (`window_calibration.n_windows`) |",
        "",
        "The test split is isolated from training and validation checkpoint selection. "
        "Report-only **P0** thresholds in `recorded_gates` are diagnostic; **#112** gates "
        "below are the release bar.",
        "",
        "## Headline test metrics (eval harness)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| doc_type_accuracy | **{_fmt(report.get('doc_type_accuracy'))}** ({dt_ok}/{n}) |",
        f"| subclass_accuracy_conditional | **{_fmt(report.get('subclass_accuracy_conditional'))}** "
        f"({sc_ok}/{sc_n} scorable) |",
        f"| window_calibration ECE | **{_fmt(wc.get('ece'))}** |",
        f"| window_calibration band ECE | {_fmt(wc.get('band_ece'))} |",
        f"| fast_path_rate | {_fmt(report.get('fast_path_rate'))} |",
        "",
    ]
    lines.extend(_trainer_vs_harness(summary, report))
    lines.extend(_telemetry_section(report))
    lines.extend(_cohorts_section(report))
    lines.extend(_selective_risk_section(report))
    lines.extend(_per_head_section(report))
    lines.extend(_head_ece_policy(summary, report))
    lines.extend(_gates_section(report, ctx.gates_check_file))
    lines.extend(_baseline_section(report, ctx.baseline_doc_type))
    lines += [
        "## Hub release",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model repo | `{CLASSIFIER_MODEL_REPO}` |",
        f"| Release tag | `{run_tag}` |",
        f"| Eval on Hub | `eval_report_{run_tag}.json` |",
        "",
        f"Publish via `./training/complete_run.sh --run-tag {run_tag} --publish` "
        "(requires all #112 gates MET, or `--force-publish`).",
        "",
    ]
    lines.extend(_analyst_insights(report, summary))
    verdict = "PASS" if gates_all_met(report) else "FAIL"
    lines += [
        "## TEST-EVAL verdict",
        "",
        f"- **#112 gates:** **{verdict}** on held-out test.",
        f"- **doc_type_accuracy:** {_fmt(report.get('doc_type_accuracy'))}; "
        f"**window ECE:** {_fmt(wc.get('ece'))}.",
        f"- **Subclass conditional:** {_fmt(report.get('subclass_accuracy_conditional'))}.",
        "",
    ]
    lines.extend(_per_doc_table(report))
    lines.extend(_reproduce_test(ctx, canonical))
    lines.extend(_artifacts_table(ctx, canonical, te_copy))
    return "\n".join(lines)


def _slice_metrics(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n_docs": 0}
    dt_ok = sum(1 for r in rows if r.get("dt_correct"))
    sc_rows = [r for r in rows if r.get("dt_correct")]
    sc_ok = sum(1 for r in sc_rows if r.get("sc_correct"))
    by_dt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dt[str(r.get("gt_doc_type"))].append(r)
    per_type = {}
    for dt, rs in sorted(by_dt.items()):
        d_ok = sum(1 for r in rs if r.get("dt_correct"))
        per_type[dt] = {
            "n": len(rs),
            "doc_type_accuracy": round(d_ok / len(rs), 4) if rs else None,
        }
    return {
        "n_docs": n,
        "doc_type_accuracy": round(dt_ok / n, 4),
        "subclass_accuracy_conditional": round(sc_ok / len(sc_rows), 4) if sc_rows else None,
        "per_doc_type": per_type,
    }


def render_heldout_plus_report(
    report: dict,
    *,
    ctx: ReportContext,
    plus_filenames: set[str],
    baseline: dict | None = None,
    eval_json_name: str = "",
) -> str:
    run_tag = ctx.run_tag
    per_doc = report.get("per_doc") or []
    canon_rows = [r for r in per_doc if str(r.get("filename")) not in plus_filenames]
    plus_rows = [r for r in per_doc if str(r.get("filename")) in plus_filenames]
    canon_m = _slice_metrics(canon_rows)
    plus_m = _slice_metrics(plus_rows)
    wc = report.get("window_calibration") or {}
    ej = eval_json_name or f"eval_{run_tag}-heldout-plus.json"
    lines = [
        f"# TEST-EVAL — held-out-plus (`{run_tag}`)",
        "",
        "**Eval role:** extended monitoring pool (1,323 docs); **#112 gates remain "
        "on canonical 323 only** (`--subset test`).",
        f"**Run tag:** `{run_tag}`",
        f"**Eval JSON:** `{ej}`",
        f"**Checkpoint:** `{report.get('checkpoint', ctx.checkpoint)}`",
        f"**Artifact SHA:** `{report.get('artifact_sha', '')}`",
        "",
        "## Harness protocol",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| CLI | `training/eval_modernbert.py` |",
        "| `--subset` | `heldout-plus` |",
        f"| Documents | **{report.get('n_docs')}** (`--sample 0`) |",
        f"| Windows | **{wc.get('n_windows')}** |",
        "| `--max-length` | 8192 |",
        f"| seed | {report.get('seed', 42)} |",
        "",
        "## Headline metrics (full 1,323)",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| doc_type_accuracy | **{_fmt(report.get('doc_type_accuracy'))}** |",
        f"| subclass_accuracy_conditional | **{_fmt(report.get('subclass_accuracy_conditional'))}** |",
        f"| window ECE | {_fmt(wc.get('ece'))} |",
        f"| fast_path_rate | {_fmt(report.get('fast_path_rate'))} |",
        "",
        "## Slices (canonical 323 vs Enron plus 1,000)",
        "",
        "| slice | n | doc_type_acc | subclass (cond.) |",
        "| --- | ---: | ---: | ---: |",
        f"| canonical test | {canon_m.get('n_docs')} | {canon_m.get('doc_type_accuracy')} "
        f"| {canon_m.get('subclass_accuracy_conditional')} |",
        f"| plus v1 (Enron) | {plus_m.get('n_docs')} | {plus_m.get('doc_type_accuracy')} "
        f"| {plus_m.get('subclass_accuracy_conditional')} |",
        "",
    ]
    if plus_m.get("per_doc_type"):
        lines += ["### Per doc_type (plus slice)", "", "| doc_type | n | doc_type_acc |", "| --- | ---: | ---: |"]
        for dt, info in plus_m["per_doc_type"].items():
            lines.append(f"| {dt} | {info['n']} | {info['doc_type_accuracy']} |")
        lines.append("")
    lines.extend(_per_head_section(report))
    cm = report.get("comparable_metrics") or {}
    if cm:
        lines += [
            "## LLM sorter comparable metrics",
            "",
            "Schema `mailroom-ml/comparable-metrics/v1` in eval JSON.",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        for key in (
            "doc_type_accuracy", "subclass_accuracy_conditional", "window_ece",
            "fast_path_rate", "latency_seconds_per_document",
        ):
            if cm.get(key) is not None:
                lines.append(f"| {key} | {cm[key]} |")
        lines.append("")
    lines.extend(_analyst_insights(report, None))
    lines.extend(_per_doc_table(report))
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "uv run python training/eval_modernbert.py --subset heldout-plus \\",
        f"  --checkpoint {ctx.checkpoint or '…'} --sample 0 --json \\",
        f"  > reports/eval_{run_tag}-heldout-plus.json",
        f"uv run python training/write_eval_report.py heldout-plus --run-tag {run_tag} \\",
        f"  --eval-json reports/eval_{run_tag}-heldout-plus.json \\",
        f"  --baseline-json reports/eval_{run_tag}.json",
        "```",
        "",
        "## Artifacts",
        "",
        f"| `reports/eval_{run_tag}-heldout-plus.json` | canonical heldout-plus eval |",
        f"| `reports/TEST-EVAL/TEST-EVAL-REPORT-heldout-plus-{run_tag}.md` | this report |",
        "",
    ]
    return "\n".join(lines)


def render_training_report(
    report: dict,
    *,
    ctx: ReportContext,
    summary: dict | None,
) -> str:
    run_tag = ctx.run_tag
    lines = [
        f"# M9a local {ctx.arm + ' — ' if ctx.arm else ''}full report",
        "",
        f"**Run tag:** `{run_tag}`  ",
    ]
    if ctx.run_id:
        lines.append(f"**Run ID:** `{ctx.run_id}`  ")
    if ctx.arm:
        lines.append(f"**Arm:** {ctx.arm}  ")
    lines += [
        f"**Checkpoint:** `{report.get('checkpoint', ctx.checkpoint)}`  ",
        f"**Held-out eval JSON:** `reports/eval_{run_tag}.json`",
        "",
    ]
    lines.extend(_training_config_lines(summary, ctx.arm))
    if summary:
        epochs = summary.get("epoch_metrics") or summary.get("validation_epochs") or []
        if epochs:
            lines += ["## Validation (per epoch)", "", "| Epoch | doc_type macro-F1 | doc acc | selected |",
                      "| ---: | ---: | ---: | --- |"]
            for em in epochs:
                if isinstance(em, dict):
                    lines.append(
                        f"| {em.get('epoch')} | {_fmt(em.get('doc_type_macro_f1'))} | "
                        f"{_fmt(em.get('doc_type_acc'))} | {em.get('selected', '')} |"
                    )
            lines.append("")
        sel = summary.get("checkpoint_selection") or {}
        if sel:
            lines += [
                "### Trainer checkpoint selection (validation)",
                "",
                f"- **Selected epoch:** {sel.get('epoch')}  ",
                f"- **Selection `gate_met`:** **{sel.get('gate_met')}**  ",
                f"- **Val doc_type macro-F1 (observed):** {_fmt(sel.get('macro_f1'))}  ",
                f"- **Val doc_type ECE (calibrated):** {_fmt(sel.get('ece'))}  ",
                f"- **Val subclass objective:** {_fmt(sel.get('subclass_objective'))}  ",
                "",
            ]
    lines.extend(_head_ece_policy(summary, report))
    wc = report.get("window_calibration") or {}
    dt_ok, n = _dt_counts(report)
    sc_ok, sc_n = _sc_counts(report)
    lines += [
        "## Held-out test (323 docs @ 8,192 tokens)",
        "",
        f"Source: `reports/eval_{run_tag}.json`.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| doc_type_accuracy | **{_fmt(report.get('doc_type_accuracy'))}** ({dt_ok}/{n}) |",
        f"| subclass_accuracy_conditional | **{_fmt(report.get('subclass_accuracy_conditional'))}** "
        f"({sc_ok}/{sc_n} scorable) |",
        f"| window_calibration ECE | **{_fmt(wc.get('ece'))}** |",
        f"| window_calibration band ECE | {_fmt(wc.get('band_ece'))} |",
        "",
    ]
    lines.extend(_cohorts_section(report))
    lines.extend(_selective_risk_section(report))
    per_head = report.get("per_head") or {}
    if per_head:
        lines += ["### Per-head test macro-F1 (observed)", "", "| head | macro-F1 |", "| --- | ---: |"]
        for head in sorted(per_head):
            info = per_head[head]
            if isinstance(info, dict):
                lines.append(f"| {head} | {_fmt(info.get('macro_f1'))} |")
        lines.append("")
    lines.extend(_gates_section(report, ctx.gates_check_file))
    lines += [
        "## Hub release",
        "",
        f"- **Model repo:** `{CLASSIFIER_MODEL_REPO}`  ",
        f"- **Release tag:** `{run_tag}`  ",
        f"- **Eval artifact on Hub:** `eval_report_{run_tag}.json`  ",
        "",
        "## Verdict",
        "",
        f"- **Doc-type routing:** test acc {_fmt(report.get('doc_type_accuracy'))}, "
        f"ECE {_fmt(wc.get('ece'))}.",
        f"- **#112 gates:** {'PASS' if gates_all_met(report) else 'FAIL'}.",
        "",
    ]
    return "\n".join(lines)


def render_compare_report(
    cmp: dict,
    *,
    label_a: str,
    label_b: str,
    path_a: str,
    path_b: str,
    title: str = "",
) -> str:
    lines = [
        title or f"# compare_runs: {label_a} vs {label_b}",
        "",
        f"**A:** `{path_a}`  ",
        f"**B:** `{path_b}`  ",
        "",
        f"n_docs {label_a}={cmp.get('n_docs_a')}  {label_b}={cmp.get('n_docs_b')}  "
        f"paired={cmp.get('n_paired')}  pairing={cmp.get('pairing')}",
        f"bootstrap resamples={cmp.get('resamples')} seed={cmp.get('seed')}",
        "",
        f"| metric | {label_a} | {label_b} | delta | 95% CI (A-B) |",
        "|---|---:|---:|---:|---|",
    ]
    for name, row in (cmp.get("metrics") or {}).items():
        ci = row.get("paired") or {}
        ci_s = (
            f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
            if ci.get("ci_low") is not None else "—"
        )
        lines.append(
            f"| {name} | {_fmt(row.get('a'))} | {_fmt(row.get('b'))} | "
            f"{_fmt(row.get('delta'))} | {ci_s} |"
        )
    lines += ["", "per-head macro-F1 (observed):", f"| head | {label_a} | {label_b} | delta |",
              "|---|---:|---:|---:|"]
    for head, row in (cmp.get("per_head") or {}).items():
        lines.append(
            f"| {head} | {_fmt(row.get('a'))} | {_fmt(row.get('b'))} | {_fmt(row.get('delta'))} |"
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        f"uv run python training/compare_runs.py --a {path_a} --b {path_b}",
        "```",
        "",
        "## Artifacts",
        "",
        f"| `{path_a}` | eval A |",
        f"| `{path_b}` | eval B |",
        "",
    ]
    return "\n".join(lines)
