"""CLI tests for training/build_dataset.py (stage-only default, publish guard).

Hermetic: no network, no torch.  The publish path is exercised against a
fake ``huggingface_hub`` injected into ``sys.modules`` (spies record the
HfApi calls; byte-verification compares sidecar bytes), and the stage-only
base path never touches the Hub at all.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

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
    assert len(api.downloaded) == 5  # 4 sidecars + root dataset_info.json
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
    assert exit_code == 0  # warning, not failure (mirrors committed publish)
    out = capsys.readouterr().out
    assert "MISMATCH" in out
    assert "WARNING" in out


def _corrupt(tmp_path: Path, filename: str) -> str:
    p = tmp_path / f"hub_{filename}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("corrupted-bytes-0000", encoding="utf-8")
    return str(p)
