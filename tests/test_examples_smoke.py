import os
import subprocess
import sys
from pathlib import Path


def test_examples_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    example_files = sorted(
        p for p in examples_dir.glob("*.py") if not p.name.startswith("_")
    )
    assert example_files, "No example scripts found."

    env = dict(os.environ)
    env.setdefault("VBN_SKIP_PLOTS", "1")

    for path in example_files:
        result = subprocess.run(
            [sys.executable, str(path), "--seed", "0"],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"Example failed: {path.name}")
            print("stdout:")
            print(result.stdout)
            print("stderr:")
            print(result.stderr)
        assert result.returncode == 0
