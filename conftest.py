"""Test-suite bootstrap.

Adds the repo root and ``src/`` to ``sys.path`` so tests can import the
``deploy`` package (root) and ``mailroom_ml`` (src) without installing the
project — keeps the core suite runnable before/without a ``uv sync``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)