"""The Dojo pin and the vendored Dojo ML contract.

``llm-dojo-scoring`` (the Dojo) owns the constellation's taxonomy and metric
definitions. mailroom-ml uses it two ways:

- **Eval and gate reports** call the Dojo directly (``mailroom_ml.dojo_metrics``).
  It is installed from the ``dojo`` extra, pinned to :data:`DOJO_COMMIT`.
- **Training and serving** do not install the Dojo. They read the label tables and
  the intake clerk from ``labels.py``, ``normalize.py`` and ``config.py``. Those
  copies are held to the vendored contract (``contracts/dojo_ml_contract.json``, exported
  by the Dojo's ``scripts/export_ml_contract.py`` at the same commit) by
  ``tests/test_dojo_contract.py``.

To move the pin: change :data:`DOJO_COMMIT` and the ``dojo`` extra in
``pyproject.toml`` together, copy ``llm_dojo_scoring/ml/data/ml_contract.json`` from
that commit over ``contracts/dojo_ml_contract.json``, run ``uv lock``, then the tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: The Dojo commit mailroom-ml is aligned to (40-hex).
DOJO_COMMIT = "ab7d9bc9b5e7f9c1c14d6a9887c1b446c3ad3729"

#: The requirement string the ``dojo`` extra and the Modal eval image install.
DOJO_REQUIREMENT = (
    "llm-dojo-scoring @ git+https://github.com/Exios66/llm-dojo-scoring@" + DOJO_COMMIT
)

CONTRACT_PATH = Path(__file__).with_name("contracts") / "dojo_ml_contract.json"


def load_contract() -> dict[str, Any]:
    """Read the vendored Dojo ML contract."""
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
