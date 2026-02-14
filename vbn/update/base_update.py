from __future__ import annotations

from typing import Dict

import torch


class BaseUpdatePolicy:
    def update(self, vbn, data: Dict[str, torch.Tensor], **kwargs):
        raise NotImplementedError

    def get_state(self) -> Dict[str, object]:
        return {}

    def set_state(self, state: Dict[str, object]) -> None:
        return None
