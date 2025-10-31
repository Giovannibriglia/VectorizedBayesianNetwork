# vbn/sampling/rao_blackwellized.py
from __future__ import annotations

from typing import Dict, Optional

import torch

from .ancestral import _flat_index_select_cpt
from .base import BaseSampler, Tensor


class RaoBlackwellizedSampler(BaseSampler):
    """
    Mixed sampler:
      - Sample all DISCRETE nodes ancestrally (supports do/evidence via kw)
      - For Linear-Gaussian nodes:
          * if return_gaussian_params=True: return {"mean": [n,D], "var": [n,D]}
          * else: sample from N(mean, var) (QMC if enabled)
    """

    def sample(
        self,
        bn,
        n: int = 1024,
        do: Optional[Dict[str, Tensor]] = None,
        return_gaussian_params: bool = False,
        **kw,
    ) -> Dict[str, Tensor]:
        device = self.device
        do = {k: v.to(device) for k, v in (do or {}).items()}
        out: Dict[str, Tensor] = {}
        # 1) sample discretes ancestrally
        for node in bn.topo_order:
            cpd = bn.cpd[node]
            parents = bn.parents.get(node, [])
            # robust discrete: canonical CPT + flat index
            if self._is_discrete_cpd(cpd):
                if node in do:
                    out[node] = do[node].view(1, 1).long().expand(n, 1)
                else:
                    parent_cards = [bn.nodes[p]["card"] for p in parents]
                    K = int(cpd.K)
                    if not parents:
                        probs = cpd.table.to(device).view(1, K).expand(n, K)
                    else:
                        idxs = [out[p].view(-1).long().to(device) for p in parents]
                        probs = _flat_index_select_cpt(
                            cpd.table, parent_cards, idxs, K, device
                        )
                    cat = torch.distributions.Categorical(probs=probs)
                    out[node] = cat.sample().view(n, 1)
                continue  # move to next node

        # 2) handle linear-gaussian nodes conditioned on sampled parents
        for node in bn.topo_order:
            cpd = bn.cpd[node]
            if not self._is_linear_gaussian_cpd(cpd):
                continue
            parents = bn.parents.get(node, [])
            if node in do:
                if return_gaussian_params:
                    out[node] = {
                        "mean": do[node].view(1, -1).expand(n, -1).float(),
                        "var": torch.zeros(n, 1, device=device),
                    }
                else:
                    out[node] = do[node].view(1, -1).expand(n, -1).float()
                continue
            if parents:
                cols = [out[p].to(device).view(n, -1).float() for p in parents]
                X = (
                    torch.cat(cols, dim=1)
                    if cols
                    else torch.zeros((n, 0), device=device)
                )
            else:
                X = torch.zeros((n, 0), device=device)
            mean = (X @ cpd.W.to(device) + cpd.b.to(device)).view(n, -1)
            var = cpd.sigma2.to(device).view(1).expand_as(mean)
            if return_gaussian_params:
                out[node] = {"mean": mean, "var": var}
            else:
                D = mean.shape[1]
                z = self._standard_normal(n, D) * var.sqrt()
                out[node] = mean + z
        return out
