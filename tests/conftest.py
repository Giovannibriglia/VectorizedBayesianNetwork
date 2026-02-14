import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="PyTorch is required to run VBN tests")

ROOT = Path(__file__).resolve().parent.parent
for path in [str(ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)
