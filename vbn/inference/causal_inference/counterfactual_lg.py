# vbn/inference/causal_inference/counterfactual_lg.py
from __future__ import annotations

from typing import Dict

import torch

from .base import CausalQuery, Tensor


class CounterfactualLG(CausalQuery):
    r"""
    Linear-Gaussian counterfactuals via Abduction–Action–Prediction.
    For a node Y with SE: Y = W·Pa(Y) + b + ε, ε ~ N(0, σ²)
    Given factual (observed) assignments for parents and Y, abduct ε̂,
    then predict Y^{do(X=x')} by reusing ε̂ under the intervention.
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
        cpd = self.bn.cpd[y]
        assert (
            hasattr(cpd, "W") and hasattr(cpd, "b") and hasattr(cpd, "sigma2")
        ), "CounterfactualLG requires LinearGaussianCPD."
        # build X_f and X_cf from factual/intervened parent tensors (concat in CPD order)
        par_names = list(cpd.parents.keys())  # matches W’s column order

        def build_X(vals: Dict[str, Tensor]):
            cols = [vals[p].to(device).view(1, -1).float() for p in par_names]
            return (
                torch.cat(cols, dim=1) if cols else torch.zeros((1, 0), device=device)
            )

        X_f = build_X(factual_parents)
        X_cf = build_X(intervened_parents)

        mean_f = (X_f @ cpd.W.to(device) + cpd.b.to(device)).view(1, -1)
        eps_hat = factual_y.to(device).view(1, -1) - mean_f  # abduct noise
        mean_cf = (X_cf @ cpd.W.to(device) + cpd.b.to(device)).view(1, -1)
        y_cf = mean_cf + eps_hat  # reuse noise
        return y_cf.squeeze()
