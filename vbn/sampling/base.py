# vbn/sampling/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

Tensor = torch.Tensor


class BaseSampler(ABC):
    def __init__(
        self,
        device: str | torch.device = "cpu",
        qmc: bool = False,
        qmc_seed: int = 42,
    ):
        self.device = torch.device(device)
        self.qmc = qmc
        self.qmc_seed = int(qmc_seed)

    # --- QMC utilities (Sobol → Normal) ---
    def _qmc_normal(self, n: int, d: int) -> Tensor:
        # Sobol in (0,1) then inverse-CDF to N(0,1)
        eng = torch.quasirandom.SobolEngine(
            dimension=d, scramble=True, seed=self.qmc_seed
        )
        u = eng.draw(n).to(self.device).clamp_(1e-7, 1 - 1e-7)
        return torch.sqrt(torch.tensor(2.0, device=self.device)) * torch.erfinv(
            2.0 * u - 1.0
        )

    def _standard_normal(self, n: int, d: int) -> Tensor:
        if self.qmc:
            return self._qmc_normal(n, d)
        return torch.randn(n, d, device=self.device)

    # --- generic helpers ---
    @staticmethod
    def _is_discrete_cpd(cpd) -> bool:
        return hasattr(cpd, "K") and hasattr(cpd, "table")

    @staticmethod
    def _is_linear_gaussian_cpd(cpd) -> bool:
        return all(hasattr(cpd, k) for k in ("W", "b", "sigma2"))

    @abstractmethod
    def sample(
        self, bn, n: int, do: Optional[Dict[str, Tensor]] = None, **kw
    ) -> Dict[str, Tensor]:
        raise NotImplementedError
