# vbn/inference/causal_inference/base.py
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

Tensor = torch.Tensor


class CausalQuery:
    """
    Thin façade around a CausalBayesNet providing common utilities
    for causal effect and counterfactual queries.
    """

    def __init__(self, bn, device: Optional[str] = None, exact_method: str = "ve"):
        self.bn = bn
        self.device = torch.device(device or getattr(bn, "device", "cpu"))
        self.exact_method = exact_method  # "ve" or "gaussian_exact" depending on model

    # ---- helpers ----
    def _p(
        self,
        query: Sequence[str],
        evidence: Optional[Dict[str, Tensor]] = None,
        do: Optional[Dict[str, Tensor]] = None,
    ):
        """Return posterior factors for query (delegates to bn.posterior)."""
        return self.bn.posterior(
            query, evidence=evidence, do=do, method=self.exact_method
        )

    def _expect(self, var, evidence=None, do=None) -> Tensor:
        spec = self.bn.nodes[var]
        t = spec.get("type")
        try:
            if t == "discrete":
                # E[Y] from categorical: sum k * P(Y=k | ...)
                out = self.bn.posterior([var], evidence=evidence, do=do, method="ve")
                p = out[var].view(-1)
                vals = torch.arange(p.numel(), device=self.device, dtype=p.dtype)
                return (p * vals).sum()
            if t == "gaussian" and spec.get("dim", 1) == 1:
                out = self.bn.posterior(
                    [var], evidence=evidence, do=do, method="gaussian"
                )
                return out[var]["mean"].squeeze()
        except Exception:
            pass
        # fallback MC
        draws = self.bn.sample_conditional(
            evidence=evidence or {}, do=do, n_samples=4096
        )
        return draws[var].float().mean(0).squeeze()
