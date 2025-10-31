# vbn/inference/causal_inference/counterfactual_lg.py
from __future__ import annotations

from typing import Dict

import torch

from .base import CausalQuery, Tensor


def _as_col(t: torch.Tensor) -> torch.Tensor:
    t = t.to(dtype=torch.get_default_dtype())
    if t.ndim == 0:
        return t.view(1, 1)
    if t.ndim == 1:
        return t.view(-1, 1)
    return t  # assume already [B, d]


class CounterfactualLG(CausalQuery):
    r"""
    Linear-Gaussian counterfactuals via Abduction–Action–Prediction.
    Works with LinearGaussianCPD that expects `forward(parents_dict)`.
    """

    @torch.no_grad()
    def y_cf(
        self,
        y: str,
        factual_parents: Dict[str, Tensor],
        factual_y: Tensor,
        intervened_parents: Dict[str, Tensor],
    ) -> Tensor:
        device = self.device
        cpd = self.bn.cpd[y]  # LinearGaussianCPD

        # Parent order used by the CPD/trainer
        par_names = list(self.bn.parents.get(y, []))

        # Normalize parent dicts -> ensure tensors are on device and 2D [B, d]
        def norm_parents(pdict: Dict[str, Tensor]) -> Dict[str, torch.Tensor]:
            out = {}
            for p in par_names:
                v = pdict[p].to(device)
                out[p] = _as_col(v).float()
            # broadcast batch B if needed
            B = max(out[p].shape[0] for p in par_names) if par_names else 1
            for p in par_names:
                if out[p].shape[0] == 1 and B > 1:
                    out[p] = out[p].expand(B, out[p].shape[1]).clone()
            return out

        par_f = norm_parents(factual_parents)
        par_cf = norm_parents(intervened_parents)

        # y_f -> [B,1]
        y_f = factual_y.to(device)
        if y_f.ndim == 0:
            y_f = y_f.view(1, 1)
        elif y_f.ndim == 1:
            y_f = y_f.view(-1, 1)
        else:
            y_f = y_f.view(y_f.shape[0], -1)
        if y_f.shape[1] != 1:
            raise ValueError("CounterfactualLG expects scalar Y (dim==1).")

        # Abduction: eps_hat = y_f - E[Y | factual parents]
        mu_f = cpd.forward(par_f).view(-1, 1)  # calls CPD with parents dict
        eps_hat = y_f - mu_f

        # Prediction under intervention: Y_cf = E[Y | intervened parents] + eps_hat
        mu_cf = cpd.forward(par_cf).view(-1, 1)
        y_cf = mu_cf + eps_hat
        return y_cf.squeeze()
