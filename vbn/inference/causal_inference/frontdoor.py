# vbn/inference/causal_inference/frontdoor.py
from __future__ import annotations

from typing import Sequence

import torch

from .base import CausalQuery, Tensor


class FrontdoorAdjuster(CausalQuery):
    r"""
    Front-door adjustment (discrete mediator M):
      P(y | do(x)) = Σ_m P(m | x) Σ_{x'} P(y | m, x') P(x')
    """

    @torch.no_grad()
    def effect(
        self,
        x_var: str,
        m_var: str,
        y_var: str,
        x_support: Sequence[Tensor],
        m_support: Sequence[Tensor],
        x_value: Tensor,
    ) -> Tensor:
        device = self.device
        x_value = x_value.to(device)
        total = 0.0
        # sum over mediator
        for m in m_support:
            pm_x = self._p([m_var], evidence={x_var: x_value})[m_var].view(-1)
            pm = pm_x[int(m.item())]
            inner = 0.0
            for xp in x_support:
                val = float(self._expect(y_var, evidence={m_var: m, x_var: xp}).item())
                pxp = self._p([x_var], evidence={})[x_var].view(-1)[int(xp.item())]
                inner += val * float(pxp.item())
            total += float(pm.item()) * inner
        return torch.as_tensor(total, device=device).float()


def _to_scalar(obj) -> float:
    if isinstance(obj, dict) and "mean" in obj:
        return float(obj["mean"].squeeze().item())
    if torch.is_tensor(obj):
        t = obj.squeeze()
        return float(t.item()) if t.ndim == 0 else float(t.mean().item())
    import numpy as np

    arr = obj.detach().cpu().numpy()
    return float((arr * np.arange(arr.shape[-1])).sum())
