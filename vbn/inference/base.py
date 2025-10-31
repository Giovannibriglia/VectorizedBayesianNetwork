from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import torch

Tensor = torch.Tensor


class InferenceBackend(ABC):
    """Minimal sampling-based backend interface.
    Supports evidence and interventions via node clamping/override during sampling.
    """

    def __init__(self, device: str | torch.device = "cpu", **kwargs):
        self.device = torch.device(device)

    @abstractmethod
    def posterior(
        self,
        bn,
        query: Sequence[str],
        evidence: Optional[Dict[str, Tensor]] = None,
        do: Optional[Dict[str, Tensor]] = None,
        **kw,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError
