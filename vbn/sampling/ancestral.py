# vbn/sampling/ancestral.py
from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import BaseSampler, Tensor


def _flat_index_select_cpt(
    table: torch.Tensor,
    parent_cards: list[int],
    parent_vals: list[torch.Tensor],
    K: int,
    device,
) -> torch.Tensor:
    """
    table: CPT as Tensor with shape [prod(parent_cards), K] or canonical [C1,...,Cm,K]
    parent_cards: [C1,...,Cm]
    parent_vals: list of length m, each LongTensor shape [N] with values in [0, Ci-1]
    returns probs: [N, K]
    """
    m = len(parent_cards)
    if m == 0:
        # root: broadcast prior
        t2d = table.view(1, K)
        return t2d.expand(parent_vals[0].shape[0] if parent_vals else 1, K)

    # make sure table is 2D [prodC, K]
    prodC = 1
    for c in parent_cards:
        prodC *= c
    t2d = table.to(device).contiguous().view(prodC, K)

    # strides for row-major flattening of [C1,...,Cm]
    strides = []
    acc = 1
    for c in reversed(parent_cards):
        strides.append(acc)
        acc *= c
    strides = list(reversed(strides))  # length m

    # no in-place ops on possibly 0-stride tensors
    vals = [v.view(-1).to(device).long() for v in parent_vals]
    terms = [val.clamp(0, c - 1) * s for val, c, s in zip(vals, parent_cards, strides)]
    flat = terms[0]
    for t in terms[1:]:
        flat = flat + t

    return t2d.index_select(0, flat)


class AncestralSampler(BaseSampler):
    """
    Unconditional / interventional ancestral sampling.
    - Supports do(·)
    - If CPD is LinearGaussian, uses samplers own Normal (QMC if enabled) to inject noise.
    """

    def sample(
        self, bn, n: int = 1024, do: Optional[Dict[str, Tensor]] = None, **kw
    ) -> Dict[str, Tensor]:
        device = self.device
        do = {k: v.to(device) for k, v in (do or {}).items()}
        out: Dict[str, Tensor] = {}

        for node in bn.topo_order:
            cpd = bn.cpd[node]
            parents = bn.parents.get(node, [])
            if node in do:
                v = do[node]
                # Make [n, 1] for discrete / [n, D] for continuous
                if self._is_discrete_cpd(cpd):
                    out[node] = v.view(1, 1).long().expand(n, 1)
                else:
                    out[node] = v.view(1, -1).expand(n, -1).float()
                continue

            if self._is_linear_gaussian_cpd(cpd):
                # Compute mean = X W + b, then add our own noise (QMC if enabled)
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
                D = mean.shape[1]
                z = self._standard_normal(n, D) * cpd.sigma2.to(device).sqrt()
                out[node] = mean + z
            else:
                if self._is_discrete_cpd(cpd):
                    # Robust discrete conditional sampling via canonical reshape + flat index
                    parent_cards = [bn.nodes[p]["card"] for p in parents]
                    K = int(cpd.K)
                    # ensure parent indices are [N]
                    idxs = (
                        [out[p].view(-1).long().to(device) for p in parents]
                        if parents
                        else []
                    )
                    probs = (
                        _flat_index_select_cpt(cpd.table, parent_cards, idxs, K, device)
                        if parents
                        else cpd.table.to(device).view(1, K).expand(n, K)
                    )
                    cat = torch.distributions.Categorical(probs=probs)
                    out[node] = cat.sample().view(n, 1)
                else:
                    # continuous / other CPDs → delegate
                    if not parents:
                        samp = cpd.sample({}, n_samples=n)
                        out[node] = samp.squeeze(0)
                    else:
                        par_vals = {p: out[p] for p in parents}
                        samp = cpd.sample(par_vals, n_samples=1)
                        out[node] = samp.squeeze(1)

        return out
