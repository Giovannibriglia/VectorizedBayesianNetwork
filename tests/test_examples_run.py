from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = [
    "examples.01_basic_fit",
    "examples.05_save_load",
]


@pytest.mark.parametrize("module", EXAMPLES)
def test_examples_run(module: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VBN_SKIP_PLOTS"] = "1"
    env["VBN_OUT_DIR"] = str(tmp_path / "out")
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [sys.executable, "-m", module],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        timeout=120,
    )
