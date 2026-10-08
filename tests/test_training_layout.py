"""Every driver under training/ resolves the repo root by a FIXED depth
(``Path(__file__).resolve().parents[N]`` / ``dirname/..``).  Commit 134c179
moved the drivers one level deeper without updating that depth, so reports,
data and the gate script resolved under ``training/training/...``.  This
guard fails if a driver is moved (or its depth edited) inconsistently, and
checks that every ``training/...`` script path referenced from a driver
exists.  Static: nothing is imported or executed."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "training"

_PY_DEPTH = re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[(\d)\]")
_SH_DEPTH = re.compile(r'dirname "\$\{BASH_SOURCE\[0\]\}"\)((?:/\.\.)+)"')
_SCRIPT_REF = re.compile(r"(?<![\w/.-])\.?/?(training/[\w./-]+\.(?:py|sh))")


def _drivers(suffix: str) -> list[Path]:
    return sorted(p for p in TRAINING.rglob(f"*{suffix}")
                  if "__pycache__" not in p.parts)


def test_python_drivers_resolve_repo_root():
    checked = 0
    for path in _drivers(".py"):
        for m in _PY_DEPTH.finditer(path.read_text(encoding="utf-8")):
            target = path.resolve().parents[int(m.group(1))]
            # pretty_log / publish_run_to_hub resolve training/ itself for
            # check_m9a_gates.py; everything else wants the repo root
            assert target in (REPO, TRAINING), (path, target)
            checked += 1
    assert checked >= 8


def test_python_root_constants_are_repo_root():
    for path in _drivers(".py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROOT = Path(__file__)"):
                n = int(_PY_DEPTH.search(line).group(1))
                assert path.resolve().parents[n] == REPO, path


def test_shell_drivers_resolve_repo_root():
    checked = 0
    for path in _drivers(".sh"):
        m = _SH_DEPTH.search(path.read_text(encoding="utf-8"))
        if not m:
            continue
        ups = m.group(1).count("..")
        assert path.resolve().parents[ups] == REPO, path
        checked += 1
    assert checked >= 8


def test_referenced_training_scripts_exist():
    missing = []
    for path in _drivers(".py") + _drivers(".sh"):
        for ref in _SCRIPT_REF.findall(path.read_text(encoding="utf-8")):
            if not (REPO / ref).is_file():
                missing.append(f"{path.relative_to(REPO)} -> {ref}")
    assert not missing, "\n".join(missing)
