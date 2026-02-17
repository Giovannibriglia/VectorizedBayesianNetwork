from __future__ import annotations

from pathlib import Path

import vbn
from vbn import VBN
from vbn.cpds import KDECPD
from vbn.cpds.kde import KDECPD as KDECPD_Canonical


def test_imports_are_lightweight() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    vbn_path = Path(vbn.__file__).resolve()
    assert vbn_path.is_relative_to(repo_root)

    # Eager API from vbn.cpds and canonical import path should both work.
    assert KDECPD is KDECPD_Canonical
    assert VBN is not None
