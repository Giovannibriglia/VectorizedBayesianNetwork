# vbn/sampling/smc_conditional.py
from __future__ import annotations

from typing import Dict, Optional

import torch

from .ancestral import _flat_index_select_cpt
from .base import BaseSampler, Tensor


class ConditionalSMCSampler(BaseSampler):
    """
    Posterior sampling via SMC with systematic resampling.
    - Supports evidence & do(·)
    - Returns approximately i.i.d. samples from p(· | evidence, do)
    """

    def __init__(self, ess_threshold: float = 0.5, **kw):
        super().__init__(**kw)
        self.ess_threshold = ess_threshold

    @staticmethod
    def _ess(w: Tensor) -> Tensor:
        w = w / w.sum().clamp_min(1e-12)
        return 1.0 / (w * w).sum()

    def _resample(self, xdict: Dict[str, Tensor], w: Tensor):
        N = w.shape[0]
        w = (w / w.sum().clamp_min(1e-12)).detach()
        u0 = torch.rand(1, device=self.device) / N
        cum = torch.cumsum(w, dim=0)
        idx = torch.searchsorted(cum, (u0 + torch.arange(N, device=self.device) / N))
        return {k: v[idx] for k, v in xdict.items()}, torch.full_like(w, 1.0 / N)

    def sample(
        self,
        bn,
        n: int = 1024,
        evidence: Optional[Dict[str, Tensor]] = None,
        do: Optional[Dict[str, Tensor]] = None,
        **kw,
    ) -> Dict[str, Tensor]:
        device = self.device
        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}

        N = n
        w = torch.ones(N, device=device)
        samples: Dict[str, Tensor] = {}

        for node in bn.topo_order:
            cpd = bn.cpd[node]
            parents = bn.parents.get(node, [])

            if node in do:
                v = do[node]
                if self._is_discrete_cpd(cpd):
                    samples[node] = v.view(1, 1).long().expand(N, 1)
                else:
                    samples[node] = v.view(1, -1).expand(N, -1).float()
            elif self._is_linear_gaussian_cpd(cpd) and parents:
                cols = [samples[p].to(device).view(N, -1).float() for p in parents]
                X = (
                    torch.cat(cols, dim=1)
                    if cols
                    else torch.zeros((N, 0), device=device)
                )
                mean = (X @ cpd.W.to(device) + cpd.b.to(device)).view(N, -1)
                D = mean.shape[1]
                z = self._standard_normal(N, D) * cpd.sigma2.to(device).sqrt()
                samples[node] = mean + z
            else:
                if self._is_discrete_cpd(cpd):
                    parent_cards = [bn.nodes[p]["card"] for p in parents]
                    K = int(cpd.K)
                    idxs = (
                        [samples[p].view(-1).long().to(device) for p in parents]
                        if parents
                        else []
                    )
                    probs = (
                        _flat_index_select_cpt(cpd.table, parent_cards, idxs, K, device)
                        if parents
                        else cpd.table.to(device).view(1, K).expand(N, K)
                    )
                    cat = torch.distributions.Categorical(probs=probs)
                    samples[node] = cat.sample().view(N, 1)
                else:
                    # existing continuous path
                    par_vals = {p: samples[p] for p in parents}
                    samp = cpd.sample(
                        par_vals if parents else {}, n_samples=1 if parents else N
                    )
                    samples[node] = samp.squeeze(1) if parents else samp.squeeze(0)

            if evidence and node in evidence:
                y = evidence[node]
                lp = bn.cpd[node].log_prob(y, {p: samples[p] for p in parents})
                w *= lp.exp().clamp_min(1e-12)
                # clamp to evidence
                if self._is_discrete_cpd(cpd):
                    samples[node] = y.view(1, 1).long().expand(N, 1)
                else:
                    samples[node] = y.view(1, -1).expand(N, -1).float()

            if self._ess(w) < self.ess_threshold * N:
                samples, w = self._resample(samples, w)

        # final resample to return i.i.d. posterior draws
        samples, w = self._resample(samples, w)
        return samples
