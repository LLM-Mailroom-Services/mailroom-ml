"""CLI tests for training/build_dataset.py (stage-only default, publish guard).

Hermetic: no network, no torch.  The publish path is exercised against a
fake ``huggingface_hub`` injected into ``sys.modules`` (spies record the
HfApi calls; byte-verification compares sidecar bytes), and the stage-only
base path never touches the Hub at all.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import ROOT, fixture_rows, requires_transformers

CLI_PATH = ROOT / "training" / "build_dataset.py"

spec = importlib.util.spec_from_file_location("build_dataset_cli", CLI_PATH)
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)  # type: ignore[union-attr]


class _CommitInfo:
    """Mirror the installed huggingface_hub CommitInfo surface (commit_url,
    not the pre-1.0 commit_hash attribute)."""

    def __init__(self, commit_hash: str = "abc" + "0" * 37):
        self.commit_url = f"https://huggingface.co/datasets/x/commit/{commit_hash}"


class _FakeHfApi:
    """Minimal HfApi double: records calls; the 'hub' mirrors local bytes."""

    def __init__(self, stage_dir: Path):
        self.stage_dir = Path(stage_dir)
        self.created: list[dict] = []
        self.uploaded: list[dict] = []
        self.downloaded: list[dict] = []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        self.created.append({"repo_id": repo_id, "repo_type": repo_type,
                             "private": private, "exist_ok": exist_ok})

    def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
        self.uploaded.append({
            "folder_path": folder_path, "repo_id": repo_id,
            "repo_type": repo_type, "commit_message": commit_message})
        return _CommitInfo()

    def fake_hf_hub_download(self, repo_id, filename, repo_type, revision):
        """Hub double: sidecar bytes identical to the local stage copy."""
        self.downloaded.append({"repo_id": repo_id, "filename": filename,
                                "repo_type": repo_type, "revision": revision})
        p = self.stage_dir / f"hub_{filename}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((self.stage_dir / filename).read_bytes())
        return str(p)


def _install_fake_hub(monkeypatch, stage_dir: Path) -> _FakeHfApi:
    """Patch HfApi/hf_hub_download ON the real huggingface_hub module.

    Replacing sys.modules["huggingface_hub"] wholesale breaks transformers'
    lazy imports (AutoTokenizer.from_pretrained imports huggingface_hub.utils
    on a cold tokenizer cache) — attribute patching keeps the package intact.
    """
    import huggingface_hub as hf

    api = _FakeHfApi(stage_dir)
    monkeypatch.setattr(hf, "HfApi", lambda: api)
    monkeypatch.setattr(hf, "hf_hub_download", api.fake_hf_hub_download)
    return api


def _patch_stage(monkeypatch, tmp_path: Path) -> Path:
    """Point the CLI at a tmp stage dir + the committed fixture rows."""
    import mailroom_ml.config as cfg_mod
    import mailroom_ml.dataset as dataset_mod

    stage_dir = tmp_path / "stage"
    monkeypatch.setattr(cfg_mod, "STAGE_DIR", stage_dir)
    monkeypatch.setattr(dataset_mod, "load_corpus_rows", fixture_rows)
    return stage_dir


def _documents_split(stage_dir: Path, split: str) -> pd.DataFrame:
    f = stage_dir / "data" / "documents" / split / f"{split}-00000-of-00001.parquet"
    return pd.read_parquet(f)


@requires_transformers
@pytest.mark.train
def test_stage_only_is_the_default(tmp_path, monkeypatch):
    stage_dir = _patch_stage(monkeypatch, tmp_path)
    exit_code = build_dataset.main(["--stage-only"])
    assert exit_code == 0
    # full staged tree, verified
    assert (stage_dir / "manifest.txt").exists()
    assert (stage_dir / "labels.json").exists()
    assert (stage_dir / "vocabularies.json").exists()
    assert (stage_dir / "data" / "windows" / "train").exists()
    # README.md is publish-only, never staged
    assert not (stage_dir / "README.md").exists()
    # fixture: exactly one held-out test row
    assert len(_documents_split(stage_dir, "test")) == 1
    assert len(_documents_split(stage_dir, "train")) == len(fixture_rows()) - 1


def test_stage_only_never_touches_the_hub(tmp_path, monkeypatch):
    """Publishing is operator-only: no --publish flag -> no HfApi (ever)."""
    _patch_stage(monkeypatch, tmp_path)
    import huggingface_hub as hf

    def _boom(*_a, **_k):
        raise AssertionError("HfApi must not be constructed on --stage-only")

    monkeypatch.setattr(hf, "HfApi", _boom)
    # --no-windows keeps this hermetic without the tokenizer (train extra)
    assert build_dataset.main(["--stage-only", "--no-windows"]) == 0


def test_no_windows_flag(tmp_path, monkeypatch):
    stage_dir = _patch_stage(monkeypatch, tmp_path)
    assert build_dataset.main(["--no-windows"]) == 0
    assert not (stage_dir / "data" / "windows").exists()
    assert len(_documents_split(stage_dir, "test")) == 1


@requires_transformers
@pytest.mark.train
def test_publish_uploads_and_byte_verifies(tmp_path, monkeypatch, capsys):
    stage_dir = _patch_stage(monkeypatch, tmp_path)
    api = _install_fake_hub(monkeypatch, stage_dir)
    repo_id = "fake/mailroom-modernbert-training"
    exit_code = build_dataset.main(["--publish", "--repo-id", repo_id])
    assert exit_code == 0
    # repo created public, tree uploaded with a clear commit message
    assert api.created == [{
        "repo_id": repo_id, "repo_type": "dataset", "private": False,
        "exist_ok": True}]
    assert len(api.uploaded) == 1
    assert api.uploaded[0]["folder_path"] == str(stage_dir)
    assert api.uploaded[0]["repo_type"] == "dataset"
    assert "mailroom-finetune" in api.uploaded[0]["commit_message"]
    # publish-only README card written + sidecars byte-verified vs the "hub"
    assert (stage_dir / "README.md").exists()
    n_parquets = len(list(stage_dir.glob("data/**/*.parquet")))
    assert len(api.downloaded) == 5 + n_parquets  # sidecars + every parquet
    out = capsys.readouterr().out
    assert all(f"  {rel}: OK" in out for rel in
               ("manifest.txt", "labels.json", "vocabularies.json", "README.md"))
    assert "WARNING" not in out


def test_publish_detects_hub_mismatch(tmp_path, monkeypatch, capsys):
    """A hub whose sidecar bytes differ must NOT report byte-verified."""
    _patch_stage(monkeypatch, tmp_path)
    import huggingface_hub as hf

    monkeypatch.setattr(hf, "HfApi", lambda: _FakeHfApi(tmp_path / "stage"))
    monkeypatch.setattr(
        hf, "hf_hub_download",
        lambda repo_id, filename, repo_type, revision: _corrupt(
            tmp_path, filename))
    # --no-windows: the mismatch check concerns sidecars, not tokenization
    exit_code = build_dataset.main(["--publish", "--no-windows", "--repo-id", "fake/mismatch"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "MISMATCH" in captured.out
    assert "byte-verify failed" in captured.err


def _corrupt(tmp_path: Path, filename: str) -> str:
    p = tmp_path / f"hub_{filename}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("corrupted-bytes-0000", encoding="utf-8")
    return str(p)


def _write_enrichment(stage_dir: Path, n: int = 2) -> None:
    """Drop a schema-valid enrichment parquet beside the canonical train file
    (the publish-order hazard's object of interest)."""
    d = stage_dir / "data" / "documents" / "train"
    extra = pd.read_parquet(d / "train-00000-of-00001.parquet").head(n).copy()
    extra["filename"] = [f"zzzz-enrich-{i}" for i in range(len(extra))]
    extra["split"] = "train"
    extra.to_parquet(d / "enrichment-00000-of-00001.parquet")


def test_no_stage_without_publish_errors(tmp_path, monkeypatch, capsys):
    _patch_stage(monkeypatch, tmp_path)
    assert build_dataset.main(["--no-stage"]) == 2
    assert "only affects the publish path" in capsys.readouterr().err


def test_no_stage_requires_an_existing_tree(tmp_path, monkeypatch, capsys):
    _patch_stage(monkeypatch, tmp_path)  # stage dir created but empty
    _install_fake_hub(monkeypatch, tmp_path / "stage")
    assert build_dataset.main(["--publish", "--no-stage"]) == 2
    assert "no staged tree" in capsys.readouterr().err


def test_publish_refuses_to_restage_an_enriched_tree(
        tmp_path, monkeypatch, capsys):
    """--publish without --no-stage over a tree carrying enrichment would
    rewrite labels.json/dataset_info.json canonical-only: refuse loudly."""
    stage_dir = _patch_stage(monkeypatch, tmp_path)
    assert build_dataset.main(["--stage-only", "--no-windows"]) == 0
    _write_enrichment(stage_dir, n=1)
    _install_fake_hub(monkeypatch, stage_dir)
    assert build_dataset.main(["--publish", "--repo-id", "fake/x"]) == 2
    assert "inconsistent publish" in capsys.readouterr().err


def test_no_stage_publishes_enriched_tree_verbatim(
        tmp_path, monkeypatch, capsys):
    stage_dir = _patch_stage(monkeypatch, tmp_path)
    assert build_dataset.main(["--stage-only", "--no-windows"]) == 0
    _write_enrichment(stage_dir, n=2)
    api = _install_fake_hub(monkeypatch, stage_dir)
    exit_code = build_dataset.main(
        ["--publish", "--no-stage", "--repo-id", "fake/enriched"])
    assert exit_code == 0
    # dataset_info.json advertises the enrichment-inclusive total, not the
    # canonical-only count stage() wrote.
    info = json.loads((stage_dir / "dataset_info.json").read_text())
    canonical_train = len(_documents_split(stage_dir, "train"))
    assert canonical_train == len(fixture_rows()) - 1
    assert info["documents"]["splits"]["train"]["num_examples"] == canonical_train + 2
    # the card tells the truth about adoption (doc-currency law)
    readme = (stage_dir / "README.md").read_text()
    assert "ADOPTS tier-1" in readme
    assert "pure-canonical baseline" not in readme
    # uploaded once, sidecars byte-verified
    assert len(api.uploaded) == 1
    assert "WARNING" not in capsys.readouterr().out


def test_repo_root_walk_errors_without_git(tmp_path):
    start = tmp_path / "deep" / "nested"
    start.mkdir(parents=True)
    _b = start.resolve()
    with pytest.raises(SystemExit, match="no .git directory"):
        while not (_b / ".git").is_dir():
            _parent = _b.parent
            if _parent == _b:
                raise SystemExit(
                    f"repo root not found: no .git directory above {start}")
            _b = _parent
