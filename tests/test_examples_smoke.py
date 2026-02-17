from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
EXAMPLES = sorted(
    f"examples.{path.stem}"
    for path in EXAMPLES_DIR.glob("*.py")
    if path.name != "__init__.py"
)


def test_examples_smoke(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["VBN_SKIP_PLOTS"] = "1"
    env["VBN_OUT_DIR"] = str(tmp_path / "out")
    env.setdefault("MPLBACKEND", "Agg")

    for module in EXAMPLES:
        subprocess.run(
            [sys.executable, "-m", module],
            check=True,
            env=env,
            cwd=ROOT,
        )
