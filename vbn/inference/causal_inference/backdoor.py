# vbn/inference/causal_inference/backdoor.py
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .base import CausalQuery, Tensor


class BackdoorAdjuster(CausalQuery):
    r"""
    Back-door adjustment:
      P(y | do(x)) = Σ_z P(y | x,z) P(z)
    Discrete-Z path is exact by summation; continuous-Z uses MC over P(z).
    """

    @torch.no_grad()
    def effect(
        self,
        x_var: str,
        y_var: str,
        z_vars: Sequence[str],
        x_value: Tensor,
        z_support: Optional[Dict[str, Sequence[Tensor]]] = None,
        mc_samples_z: int = 0,
    ) -> Tensor:
        device = self.device
        x_value = x_value.to(device)
        # --- discrete Z: exact sum over supports
        if z_support:
            total = 0.0
            for assign in _cartesian(z_support):
                ev = {**assign}
                # assume discrete y => categorical vector; gaussian => mean
                val = float(self._expect(y_var, evidence={**ev, x_var: x_value}).item())
                # pz = self._p(list(assign.keys()), evidence=assign)[list(assign.keys())[-1]]  # any factor ok
                pz_val = _prob_of_assign(self.bn, assign)
                total = total + val * pz_val
            return torch.as_tensor(total, device=device).float()

        # --- continuous or unspecified Z: MC over P(Z)
        if mc_samples_z <= 0:
            mc_samples_z = 4096
        draws = self.bn.sample(n_samples=mc_samples_z, method="ancestral")
        total = 0.0
        for i in range(mc_samples_z):
            ev = {z: draws[z][i : i + 1] for z in z_vars}
            py = self._p([y_var], evidence={**ev, x_var: x_value})
            total = total + _to_scalar(py[y_var])
        return torch.as_tensor(total / mc_samples_z, device=device)

    @torch.no_grad()
    def ate(
        self,
        x_var: str,
        y_var: str,
        z_vars: Sequence[str],
        x0: Tensor,
        x1: Tensor,
        **kw,
    ) -> Tensor:
        mu0 = self.effect(x_var, y_var, z_vars, x0, **kw)
        mu1 = self.effect(x_var, y_var, z_vars, x1, **kw)
        return mu1 - mu0


# ---- small utilities ----


def _to_scalar(obj) -> float:
    if isinstance(obj, dict) and "mean" in obj:  # gaussian exact
        return float(obj["mean"].squeeze().item())
    if torch.is_tensor(obj):
        t = obj
        if t.ndim > 0:
            t = t.squeeze()
        return float(t.item())
    # categorical vector case → return E[Y] assuming {0,1,...}
    import numpy as np

    arr = obj.detach().cpu().numpy()
    return float((arr * np.arange(arr.shape[-1])).sum())


def _prob_of_assign(bn, assign: Dict[str, Tensor]) -> Tensor:
    """Compute P(Z=assign) via VE on their joint (works for discrete small Z)."""
    # out = bn.posterior(list(assign.keys()), evidence=assign, method="ve")
    # VE returns factors normalized; probability of that exact assignment is 1.0 in posterior.
    # Use prior P(Z) instead: compute with evidence={}, then pick indices.
    prior = bn.posterior(list(assign.keys()), evidence={}, method="ve")
    p_val = 1.0
    for z, v in assign.items():
        vec = prior[z].view(-1)
        idx = int(v.view(-1)[0].item())
        p_val = p_val * vec[idx].item()
    return torch.as_tensor(p_val, device=bn.device)


def _cartesian(grid: Dict[str, Sequence[Tensor]]) -> List[Dict[str, Tensor]]:
    keys = list(grid.keys())
    pools = [list(vals) for vals in grid.values()]
    out = []

    def rec(i, cur):
        if i == len(keys):
            out.append(cur.copy())
            return
        k = keys[i]
        for v in pools[i]:
            cur[k] = v
            rec(i + 1, cur)

    rec(0, {})
    return out
