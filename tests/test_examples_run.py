from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = [
    "examples.01_basic_fit",
    "examples.02_infer_posterior",
    "examples.03_sampling",
    "examples.04_update_online",
    "examples.05_save_load",
]


@pytest.mark.parametrize("module", EXAMPLES)
def test_examples_run(module: str, tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VBN_SKIP_PLOTS"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [sys.executable, "-m", module],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
        timeout=600,
    )
