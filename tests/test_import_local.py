from __future__ import annotations

from pathlib import Path

import vbn


def test_imports_resolve_to_local_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vbn_path = Path(vbn.__file__).resolve()
    assert vbn_path.is_relative_to(
        repo_root
    ), f"vbn imported from unexpected location: {vbn_path}"
