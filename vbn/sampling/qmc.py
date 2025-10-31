# vbn/sampling/qmc.py
from __future__ import annotations

import torch


def normal_icdf(u: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.tensor(2.0, device=u.device)) * torch.erfinv(2.0 * u - 1.0)


def sobol_normals(n: int, d: int, device="cpu", seed: int = 12345) -> torch.Tensor:
    eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seed))
    u = eng.draw(n).to(device).clamp_(1e-7, 1 - 1e-7)
    return normal_icdf(u)
