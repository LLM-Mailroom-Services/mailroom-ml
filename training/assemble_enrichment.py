#!/usr/bin/env python3
"""CLI: assemble the enrichment tiers into the committed stage layout.

Sits on ``mailroom_ml.enrichment`` (the pure assemblers) + the committed
stage discipline (``dataset.stage`` layout, byte-deterministic manifest,
#52 revision pins, #57 single-labels canon, #75 surface guardrail).

Usage:
    .venv/bin/python training/assemble_enrichment.py --stage data/modernbert_training/stage --tiers 1
    .venv/bin/python training/assemble_enrichment.py --stage DIR --tiers 1,2 --dry-run
    .venv/bin/python training/assemble_enrichment.py --stage DIR --tiers 1 --no-windows

Tiers (plan §6.1 ladder — nothing is "adopted" without the ladder A/B):
  1  source-matched enrichment (Enron GT expansion + insurance pools)
  2  Enron pseudo-label distillation scaffold (confidence-gated) — the blind
     pool must carry doc_type_conf/subclass_conf/agreement columns from the
     Tier-1 classifier snapshot (no live labeling here)
  3  constrained-synthesis machinery (eligibility + mixture + 7 gates) —
     needs --tier3-cards; optional --tier3-candidates for gate runs

Pools: local parquet/CSV paths, or Hub repo ids @ pinned revisions (the
config pins are the default — issue #52 discipline, never live tips).
Local overrides record the pool file's content-sha256 as the revision.

Writes (married to the committed stage layout — the trainer's globs):
  parquet/documents/train/enrichment-00000-of-00001.parquet   documents schema
  parquet/windows/train/enrichment-00000-of-00001.parquet     window schema (--no-windows skips)
  enrichment_provenance.parquet   per-row §5 provenance (filename-joined)
  enrichment_audit.jsonl          every rejected/cut row, with reasons
  labels.json                     REGENERATED from canonical + enrichment docs
  manifest.txt                    deterministic enrichment block appended

Ladder rule (plan §6.1): enrichment rows land in TRAIN ONLY — validation and
the held-out test split are never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# training/ scripts sit outside the package — anchor on the repo root (.git)
# and put src/ on sys.path (mirrors the committed build_dataset.py bootstrap).
_b = Path(__file__).resolve()
while not (_b / ".git").is_dir():
    _b = _b.parent
ROOT = _b
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mailroom_ml import config as cfg  # noqa: E402
from mailroom_ml.dataset import verify_stage  # noqa: E402
from mailroom_ml.enrichment import (  # noqa: E402
    DOCS_SCHEMA_COLUMNS,
    AuditStore,
    LabelCard,
    PoolResult,
    apply_insurance_cap,
    apply_mixture_caps,
    assemble_bdr_pool,
    assemble_cms_pool,
    assemble_enron_gt,
    assemble_gnotheia_pool,
    assemble_insurbias_pool,
    assemble_pseudo_labels,
    build_enrichment_windows,
    combine_tier1,
    evaluate_eligibility,
    human_spot_audit,
    mixture_caps,
    rows_frame,
    run_seven_gates,
)
from mailroom_ml.labels import label_maps  # noqa: E402

MANIFEST_START = "# ---- enrichment ----"
MANIFEST_END = "# ---- end enrichment ----"

# ---------------------------------------------------------------------------
# Caps: CLI knobs with committed defaults (config.py is read, never edited).
# ---------------------------------------------------------------------------
DEFAULT_CAPS = {
    "enron_cap_mult": 2.0,          # plan §6.2: <= 2x correspondence train rows
    "insurance_cap_mult": 2.0,      # plan §6.3: total insurance train rows <= 2x class size
    "pseudo_max_fraction": 0.30,    # plan §6.2: <= 30% of correspondence train rows
    "global_share": cfg.SYNTHETIC_MAX_GLOBAL_SHARE,            # §6.5 <= 40%
    "per_subclass_share": cfg.SYNTHETIC_MAX_PER_SUBCLASS_SHARE,  # §6.5 <= 50%
    "synthetic_weight": cfg.SYNTHETIC_EXAMPLE_WEIGHT,          # §6.5 0.6
}

# (flag attribute, pool name, default repo, default revision)
POOL_FLAGS = (
    ("enron_pool", "enron", cfg.ENRON_DEDUP_REPO, cfg.ENRON_DEDUP_REVISION),
    ("cms_pool", "cms", cfg.CMS_POOL_REPO, cfg.CMS_POOL_REVISION),
    ("gnotheia_pool", "gnotheia", cfg.GNOTHEIA_REPO, cfg.GNOTHEIA_REVISION),
    ("bdr_pool", "bdr", cfg.BDR_REPO, cfg.BDR_REVISION),
    ("insurbias_pool", "insurbias", cfg.INSURBIAS_REPO, cfg.INSURBIAS_REVISION),
)


def _parse_caps(specs: list[str] | None) -> dict[str, float]:
    caps = dict(DEFAULT_CAPS)
    for spec in specs or []:
        for kv in spec.split(","):
            if not kv.strip():
                continue
            key, _, value = kv.partition("=")
            key, value = key.strip(), value.strip()
            if key not in caps:
                raise SystemExit(
                    f"unknown cap {key!r}; known: {sorted(caps)}")
            caps[key] = float(value)
    return caps


def _load_stage_docs(stage_dir: Path) -> pd.DataFrame:
    """Canonical documents = the existing staged tree (train+val+test),
    EXCLUDING any prior ``enrichment-*`` files (the enrichment parquet is
    regenerated wholesale every run — a rerun must never double-count).

    The stage is the source of truth for dedup, observed surfaces, caps and
    the leak audit: enrichment is measured against exactly what the trainer
    consumes.  Missing tree -> loud error naming the build CLI.
    """
    frames = []
    for split in ("train", "validation", "test"):
        files = sorted((stage_dir / "parquet" / "documents" / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"no staged documents under {stage_dir / 'parquet' / 'documents' / split} "
                f"— run training/build_dataset.py --stage-only first")
        for f in files:
            if f.name.startswith("enrichment-"):
                continue  # replaced by this run; never double-counted
            frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True)


def _replace_enrichment_files(stage_dir: Path) -> None:
    """Remove stale enrichment parquet files before writing fresh ones."""
    for pattern in ("parquet/documents/train/enrichment-*",
                    "parquet/windows/train/enrichment-*"):
        for f in stage_dir.glob(pattern):
            f.unlink()


def _read_pool_dir(local: Path, pool_name: str) -> pd.DataFrame:
    parquet = sorted(local.rglob("*.parquet"))
    csvs = sorted(local.rglob("*.csv"))
    candidates = parquet or csvs
    if not candidates:
        raise FileNotFoundError(
            f"pool {pool_name}: no parquet/csv files under {local}")
    return pd.read_parquet(candidates[0]) if parquet \
        else pd.read_csv(candidates[0])


def _load_pool(spec: str | None, pool_name: str, default_repo: str,
               default_revision: str) -> tuple[pd.DataFrame, str, str]:
    """Load one pool: local parquet/CSV path, or Hub repo id[@revision]
    (config pin default — #52 discipline).  Returns (df, source_corpus,
    source_revision); local files record their content-sha256 as the
    revision so provenance stays deterministic and self-verifying."""
    if spec is None or (spec.count("@") == 1 and "://" not in spec
                        and not Path(spec.split("@", 1)[0]).exists()):
        repo_id = default_repo if spec is None else spec.partition("@")[0]
        revision = default_revision if spec is None else spec.partition("@")[2]
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        local = Path(snapshot_download(repo_id=repo_id,
                                       repo_type="dataset", revision=revision))
        return _read_pool_dir(local, pool_name), repo_id, revision
    path = Path(spec).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"pool {pool_name}: {spec!r} is neither an existing path nor a "
            f"repo_id@revision")
    if path.is_dir():
        df = _read_pool_dir(path, pool_name)
        revision = _dir_revision(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        revision = hashlib.sha256(path.read_bytes()).hexdigest()
    elif path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        revision = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        raise ValueError(f"pool {pool_name}: unsupported file {path.suffix}")
    return df, f"local:{pool_name}", revision


def _dir_revision(local: Path) -> str:
    """Deterministic content revision for a local pool DIRECTORY: sorted
    relative paths + per-file content sha256s, hashed again."""
    h = hashlib.sha256()
    for rel in sorted(p.relative_to(local).as_posix()
                  for p in local.rglob("*") if p.is_file()):
        h.update(str(rel).encode("utf-8"))
        h.update((local / rel).read_bytes())
    return h.hexdigest()


def _subset_docs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop enrichment rows to EXACTLY the trainer documents schema."""
    return df[list(DOCS_SCHEMA_COLUMNS)].sort_values("filename") \
        .reset_index(drop=True)


def _subclass_class_map(canonical: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for cls, grp in canonical.groupby("doc_type"):
        for sc in grp["subclass"].unique():
            out[str(sc)] = str(cls)
    return out


def _authentic_counts(canonical: pd.DataFrame) -> dict[str, int]:
    sub = canonical[canonical["split"] != "test"]
    return sub["subclass"].value_counts().sort_index().to_dict()


# ---------------------------------------------------------------------------
# Tier orchestration (pure — pool frames in, rows + stats + audit out)
# ---------------------------------------------------------------------------

def _run_pool(name: str, fn, pool_df: pd.DataFrame, canonical: pd.DataFrame,
              kwargs: dict, audit: AuditStore) -> tuple[PoolResult, dict]:
    res = fn(pool_df, canonical, **kwargs)
    audit.add_rejected_frame(f"tier1:{name}", res.rejected)
    if not res.needs_render.empty:
        audit.add_rejected_frame(f"tier1:{name}", res.needs_render)
    return res, {"kept": int(len(res.rows)),
                 "rejected": int(len(res.rejected)),
                 "needs_render": int(len(res.needs_render))}


def assemble_tier1(canonical: pd.DataFrame, caps: dict[str, float],
                   pools: dict[str, pd.DataFrame],
                   audit: AuditStore) -> tuple[pd.DataFrame, dict]:
    """Enron GT + insurance pools -> combined rows (deduped, capped) + stats."""
    results: list[tuple[str, PoolResult]] = []
    stats: dict[str, dict] = {}
    spec: dict[str, tuple] = {
        "enron": (assemble_enron_gt, {"cap_mult": caps["enron_cap_mult"]}),
        "cms": (assemble_cms_pool, {}),
        "gnotheia": (assemble_gnotheia_pool, {}),
        "bdr": (assemble_bdr_pool, {}),
        "insurbias": (assemble_insurbias_pool, {}),
    }
    for name in sorted(pools):
        fn, kwargs = spec[name]
        res, stat = _run_pool(name, fn, pools[name], canonical, kwargs, audit)
        results.append((name, res))
        stats[name] = stat
    combined, cross_rejects = combine_tier1(results)
    audit.add_rejected_frame("tier1:cross_pool", cross_rejects)
    insurance, insurance_cuts = apply_insurance_cap(
        combined[combined["doc_type"] == "insurance_claim"],
        canonical, class_cap_mult=caps["insurance_cap_mult"])
    audit.add_rejected_frame("tier1:insurance_cap", insurance_cuts)
    rest = combined[combined["doc_type"] != "insurance_claim"]
    rows = pd.concat([rest, insurance], ignore_index=True) \
        .sort_values("filename").reset_index(drop=True) \
        if not combined.empty else combined
    stats["insurance_cap"] = {"kept": int(len(insurance)),
                              "cut": int(len(insurance_cuts))}
    return rows, stats


def assemble_tier2(canonical: pd.DataFrame, caps: dict[str, float],
                   blind: pd.DataFrame, audit: AuditStore,
                   family_col: str | None) -> tuple[pd.DataFrame, dict]:
    """Pseudo-label scaffold: gates -> seam -> cap -> balance."""
    res = assemble_pseudo_labels(
        blind, canonical, max_fraction=caps["pseudo_max_fraction"],
        family_col=family_col)
    audit.add_rejected_frame("tier2:pseudo", res.rejected)
    return res.rows, {"kept": int(len(res.rows)),
                      "rejected": int(len(res.rejected))}


def assemble_tier3(canonical: pd.DataFrame, caps: dict[str, float],
                   cards: list[dict] | None, candidates: pd.DataFrame,
                   audit: AuditStore) -> tuple[pd.DataFrame, dict]:
    """Synthesis machinery: eligibility + mixture caps (+ gate runs when a
    candidates frame is given)."""
    authentic = _authentic_counts(canonical)
    elig = evaluate_eligibility(
        authentic, subclass_class=_subclass_class_map(canonical))
    stat = {
        "eligible": [sc for sc, e in sorted(elig.items())
                     if e.cap > 0 and e.eligible],
        "caps": {sc: e.cap for sc, e in sorted(elig.items())},
    }
    mix = mixture_caps(authentic, global_share=caps["global_share"],
                       per_subclass_share=caps["per_subclass_share"])
    stat["mixture_global_cap"] = mix["__global_cap__"]
    stat["mixture_authentic_total"] = mix["__authentic_total__"]
    if cards is None or candidates.empty:
        return pd.DataFrame(), stat
    by_key = {(str(c["parent_class"]), str(c["subclass"])): LabelCard(**c)
              for c in cards}
    corpus_texts = list(canonical[canonical["split"] != "test"]["doc_text"])
    accepted_records: list[dict] = []
    n_passed = 0
    for r in candidates.to_dict("records"):
        key = (str(r.get("parent_class")), str(r.get("subclass")))
        card = by_key.get(key)
        if card is None:
            audit.add({"candidate_id": str(r.get("id") or r.get("filename")),
                       "card": "/".join(key), "passed": False,
                       "first_failure": "schema",
                       "reason": f"no label card for {key}"})
            continue
        report = run_seven_gates(r, card, corpus_texts, audit=audit)
        if not report.passed:
            continue
        n_passed += 1
        text = str(r.get("doc_text") or "")
        accepted_records.append({
            "filename": str(r.get("filename") or r.get("id")),
            "doc_text": text, "doc_type": card.parent_class,
            "subclass": card.subclass,
            "title": str(r.get("title") or r.get("filename") or ""),
            "source_corpus": f"synthetic:{card.parent_class}/{card.subclass}",
            "source_revision": "card",
            "label_source": "synthetic_card",
            "label_confidence": float(r.get("confidence", 1.0)),
            "example_weight": caps["synthetic_weight"],
            "lineage": "label_card+7gates", "tier": 3,
        })
    rows = rows_frame(accepted_records) if accepted_records \
        else pd.DataFrame()
    capped, mix_cuts = apply_mixture_caps(
        rows, authentic, global_share=caps["global_share"],
        per_subclass_share=caps["per_subclass_share"])
    audit.add_rejected_frame("tier3:mixture", mix_cuts)
    stat["adopted"] = int(len(capped))
    stat["gate_rejected"] = int(max(0, len(candidates) - n_passed))
    stat["human_audit_pending"] = human_spot_audit(capped, authentic)
    return capped, stat


# ---------------------------------------------------------------------------
# Stage writes (byte-deterministic — no timestamps, ever)
# ---------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def append_manifest_block(manifest_path: Path, block: str) -> None:
    """Append (or replace) the deterministic enrichment block in manifest.txt.
    Re-running with identical inputs reproduces identical bytes (the block
    is delimited and replaced in place; no timestamps)."""
    text = manifest_path.read_text(encoding="utf-8")
    block = f"\n{MANIFEST_START}\n{block.strip()}\n{MANIFEST_END}\n"
    if MANIFEST_START in text:
        head = text.split(MANIFEST_START, 1)[0].rstrip() + "\n"
        manifest_path.write_text(head + block, encoding="utf-8")
    else:
        manifest_path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def render_report(canonical: pd.DataFrame, tier1_rows: pd.DataFrame | None,
                  tier2_rows: pd.DataFrame | None,
                  tier3_rows: pd.DataFrame | None) -> str:
    lines: list[str] = []
    docs = canonical["split"].value_counts().to_dict()
    lines.append(f"staged-docs            : documents train={docs.get('train', 0)} "
                 f"validation={docs.get('validation', 0)} test={docs.get('test', 0)}")
    for label, df in (("tier1", tier1_rows), ("tier2", tier2_rows),
                      ("tier3", tier3_rows)):
        if df is not None and not df.empty:
            lines.append(f"{label:<22s}: kept={len(df)} "
                         f"subclasses={sorted(df['subclass'].unique())}")
    lines.append("enrichment-note        : rows land in documents/train ONLY — "
                 "validation/test never touched (ladder rule)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=Path, default=cfg.STAGE_DIR)
    ap.add_argument("--tiers", default="1",
                    help="comma list of tiers to assemble (1|2|3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print counts/caps/leak checks; write nothing")
    ap.add_argument("--caps", action="append", default=None,
                    help="KEY=VALUE[,KEY=VALUE...] overrides, e.g. "
                         "enron_cap_mult=3,insurance_cap_mult=2.5")
    ap.add_argument("--no-windows", action="store_true",
                    help="skip writing parquet/windows/train enrichment windows")
    ap.add_argument("--enron-pool", default=None, help="path or repo_id@rev")
    ap.add_argument("--cms-pool", default=None)
    ap.add_argument("--gnotheia-pool", default=None)
    ap.add_argument("--bdr-pool", default=None)
    ap.add_argument("--insurbias-pool", default=None)
    ap.add_argument("--blind-pool", default=None,
                    help="tier-2 candidates (must carry confidence columns)")
    ap.add_argument("--family-col", default="thread_id",
                    help="family column for the grouped-split seam (tier 2)")
    ap.add_argument("--tier3-cards", type=Path, default=None,
                    help="JSON list of LabelCard dicts (tier 3)")
    ap.add_argument("--tier3-candidates", type=Path, default=None,
                    help="JSON-lines candidates to gate (tier 3)")
    args = ap.parse_args(argv)
    caps = _parse_caps(args.caps)
    tiers = {int(t) for t in args.tiers.split(",") if t.strip()}
    if not tiers or not tiers <= {1, 2, 3}:
        print(f"ERROR: --tiers {args.tiers!r} invalid (components 1|2|3)",
              file=sys.stderr)
        return 2

    try:
        canonical = _load_stage_docs(args.stage)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    audit = AuditStore()
    tier1_rows: pd.DataFrame | None = None
    tier2_rows: pd.DataFrame | None = None
    tier3_rows: pd.DataFrame | None = None
    stats: dict[str, dict] = {}

    if 1 in tiers:
        pools: dict[str, pd.DataFrame] = {}
        for flag_attr, name, repo, rev in POOL_FLAGS:
            spec = getattr(args, flag_attr)
            pool_df, src, rev_used = _load_pool(spec, name, repo, rev)
            if not pool_df.empty:
                pools[name] = pool_df
                stats.setdefault("pools", {})[name] = {
                    "source": src, "revision": rev_used,
                    "rows_in": int(len(pool_df))}
        tier1_rows, tier1_stats = assemble_tier1(canonical, caps, pools, audit)
        stats["tier1"] = tier1_stats

    if 2 in tiers:
        if args.blind_pool is None:
            print("ERROR: tier 2 needs --blind-pool (candidates with "
                  "doc_type_conf/subclass_conf/agreement columns)",
                  file=sys.stderr)
            return 2
        blind, src, rev = _load_pool(args.blind_pool, "blind",
                                     cfg.ENRON_DEDUP_REPO, cfg.ENRON_DEDUP_REVISION)
        stats.setdefault("pools", {})["blind"] = {"source": src,
                                                  "revision": rev,
                                                  "rows_in": int(len(blind))}
        tier2_rows, tier2_stats = assemble_tier2(
                canonical, caps, blind, audit, family_col=args.family_col)
        stats["tier2"] = tier2_stats
        if args.family_col and args.family_col not in canonical.columns:
            print("family seam inactive: canonical stage docs lack "
                  f"{args.family_col!r} — thread-families of the canonical "
                  "corpus cannot be blocked (enrichment rows still land in "
                  "train only)")

    if 3 in tiers:
        cards = None
        if args.tier3_cards is not None:
            cards = json.loads(args.tier3_cards.read_text(encoding="utf-8"))
        candidates = pd.DataFrame()
        if args.tier3_candidates is not None:
            cand_lines = [json.loads(line) for line in
                          args.tier3_candidates.read_text(encoding="utf-8")
                          .splitlines()
                          if line.strip() and not line.startswith("#")]
            candidates = pd.DataFrame(cand_lines)
        tier3_rows, tier3_stats = assemble_tier3(
            canonical, caps, cards, candidates, audit)
        stats["tier3"] = tier3_stats

    # ---- deterministic report (dry-run and real run share it) -------------
    out = [render_report(canonical, tier1_rows, tier2_rows, tier3_rows)]
    for name, s in sorted(stats.get("pools", {}).items()):
        out.append(f"pool {name:<10s}: source={s['source']} "
                   f"revision={s['revision'][:16]} rows_in={s['rows_in']}")
    if "tier1" in stats:
        for name, s in sorted(stats["tier1"].items()):
            out.append(f"tier1 {name:<16s}: {json.dumps(s, sort_keys=True)}")
    if "tier2" in stats:
        out.append(f"tier2 pseudo          : {json.dumps(stats['tier2'], sort_keys=True)}")
    if "tier3" in stats:
        t3 = stats["tier3"]
        out.append("tier3 eligibility     : "
                   f"eligible={t3.get('eligible', [])} "
                   f"caps={t3.get('caps', {})}")
        out.append("tier3 mixture         : "
                   f"global_cap={t3['mixture_global_cap']} "
                   f"authentic_total={t3['mixture_authentic_total']}")
        if "adopted" in t3:
            out.append(f"tier3 adopted         : {t3['adopted']} rows "
                       f"(gate_rejected={t3['gate_rejected']}, "
                       f"human_audit_pending={t3['human_audit_pending']})")
    out.append(f"audit store           : {len(audit)} rejected/cut rows "
               "(never fitted)")
    leak: dict[str, int] = {}
    for r in audit.records:
        leak[str(r["reason"])] = leak.get(str(r["reason"]), 0) + 1
    out.append(f"leak summary          : {json.dumps(leak, sort_keys=True)}")
    for r in audit.records[:20]:
        out.append(f"  audit {str(r.get('pool', 'gates')):<20s} "
                   f"{r['filename' if 'filename' in r else 'candidate_id']:<28s} "
                   f"{r['reason']}")
    if len(audit) > 20:
        out.append(f"  ... {len(audit) - 20} more")
    print("\n".join(out))
    if args.dry_run:
        print("dry-run: no files written")
        return 0

    # ---- writes -----------------------------------------------------------
    all_rows = pd.concat(
        [r for r in (tier1_rows, tier2_rows, tier3_rows)
         if r is not None and not r.empty], ignore_index=True) \
        .sort_values("filename").reset_index(drop=True) \
        if any(r is not None and not r.empty
               for r in (tier1_rows, tier2_rows, tier3_rows)) else pd.DataFrame()

    merged = canonical.copy()
    if not all_rows.empty:
        _replace_enrichment_files(args.stage)
        merged = pd.concat([merged, _subset_docs(all_rows)],
                           ignore_index=True)
        _write_parquet(_subset_docs(all_rows),
                       args.stage / "parquet" / "documents" / "train"
                       / "enrichment-00000-of-00001.parquet")
        prov_cols = ("filename", "source_corpus", "source_revision",
                     "purpose", "label_source", "label_confidence",
                     "example_weight", "lineage", "tier")
        _write_parquet(all_rows[list(prov_cols)]
                       .sort_values("filename").reset_index(drop=True),
                       args.stage / "enrichment_provenance.parquet")
        if not args.no_windows:
            from mailroom_ml.windows import window_document  # noqa: PLC0415
            recs = [dict(r, windows=window_document(r["title"], r["doc_text"]))
                    for r in all_rows.to_dict("records")]
            wins = build_enrichment_windows(recs)
            if not wins.empty:
                _write_parquet(wins, args.stage / "parquet" / "windows" / "train"
                               / "enrichment-00000-of-00001.parquet")
    else:
        empty = pd.DataFrame(columns=("filename", "source_corpus",
                                      "source_revision", "purpose",
                                      "label_source", "label_confidence",
                                      "example_weight", "lineage", "tier"),
                             dtype=str)
        _write_parquet(empty, args.stage / "enrichment_provenance.parquet")

    audit_path = args.stage / "enrichment_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as fh:
        for rec in audit.records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    maps = label_maps(merged)
    (args.stage / "labels.json").write_text(
        json.dumps(maps, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    block = "\n".join([
        f"tiers        : {sorted(tiers)}",
        f"pools        : {json.dumps(stats.get('pools', {}), sort_keys=True)}",
        f"tier1        : {json.dumps(stats.get('tier1', {}), sort_keys=True)}",
        f"tier2        : {json.dumps(stats.get('tier2', {}), sort_keys=True)}",
        f"tier3        : {json.dumps(stats.get('tier3', {}), sort_keys=True)}",
        f"audit_rows   : {len(audit)}",
        f"labels       : regenerated over {len(merged)} documents",
    ])
    append_manifest_block(args.stage / "manifest.txt", block)

    check = verify_stage(args.stage)
    if not check["ok"]:
        print("VERIFY FAILED:")
        for p in check["problems"]:
            print(f"  - {p}")
        return 1
    print(f"verify ok: {check['rows']} rows, splits {check['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
