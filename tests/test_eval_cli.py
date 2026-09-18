"""Eval CLI tests (plan §11, issues #92 M7): CLI surface + sampling math.

Hermetic: argparse surface and the stratified sampler only — no model, no
network, no stage tree.
"""
from __future__ import annotations

from pathlib import Path

from training.eval_modernbert import build_parser, stratified_sample


def test_cli_surface_accepts_documented_flags():
    ns = build_parser().parse_args([
        "--checkpoint", "artifacts/onnx/model", "--stage", "/tmp/stage",
        "--subset", "test", "--sample", "50", "--seed", "42",
        "--max-length", "4096", "--selective-risk", "--json",
    ])
    assert ns.checkpoint == "artifacts/onnx/model"
    assert ns.stage == Path("/tmp/stage")
    assert ns.subset == "test"
    assert ns.sample == 50
    assert ns.seed == 42
    assert ns.max_length == 4096
    assert ns.selective_risk is True
    assert ns.as_json is True


def test_cli_defaults_match_plan_surface():
    ns = build_parser().parse_args([])
    assert ns.subset == "test"
    assert ns.sample == 50
    assert ns.seed == 42
    assert ns.checkpoint == ""
    assert ns.selective_risk is False
    assert ns.as_json is False


def test_stratified_sample_deterministic_and_bounded():
    filenames = [f"f{i:02d}" for i in range(30)]
    strata = ["contract"] * 10 + ["correspondence"] * 10 + ["insurance_claim"] * 10
    a = stratified_sample(filenames, strata, 4, seed=42)
    b = stratified_sample(filenames, strata, 4, seed=42)
    assert a == b  # deterministic
    assert len(a) == 12  # min(4, 10) per stratum -> 12 total
    from collections import Counter

    picked = Counter(fn[:1] for fn in a)  # f<digit> -> stratum group
    assert picked["f0"] <= 4 and picked["f1"] <= 4 and picked["f2"] <= 4
    # order preserved relative to the input
    pos = {fn: i for i, fn in enumerate(filenames)}
    assert [pos[fn] for fn in a] == sorted(pos[fn] for fn in a)


def test_stratified_sample_seed_changes_pick():
    filenames = [f"f{i:02d}" for i in range(10)]
    strata = ["contract"] * 10
    assert stratified_sample(filenames, strata, 3, seed=1) != \
        stratified_sample(filenames, strata, 3, seed=2)


def test_stratified_sample_above_stratum_size_keeps_all():
    filenames = ["a", "b", "c"]
    strata = ["x"] * 3
    assert stratified_sample(filenames, strata, 100, seed=0) == filenames
