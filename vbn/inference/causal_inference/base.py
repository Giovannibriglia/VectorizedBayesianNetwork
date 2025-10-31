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

    def _expect(
        self,
        var: str,
        evidence: Optional[Dict[str, Tensor]] = None,
        do: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """E[var | evidence, do].
        - Gaussian nodes: try exact Gaussian backend; fallback to MC.
        - Discrete or other: MC fallback.
        """
        spec = self.bn.nodes[var]
        if spec["type"] == "gaussian":
            try:
                out = self.bn.posterior(
                    [var], evidence=evidence, do=do, method="gaussian_exact"
                )
                m = out[var]["mean"]  # dict with "mean"/"var"
                return m.squeeze()
            except Exception:
                pass  # fallback to MC below

        # MC fallback (works for any type)
        draws = self.bn.sample_conditional(
            evidence=evidence or {}, do=do, n_samples=4096
        )
        return draws[var].float().mean(0).squeeze()
