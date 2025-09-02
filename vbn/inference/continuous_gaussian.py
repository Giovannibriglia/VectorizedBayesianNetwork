from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..core import BNMeta, LearnParams


class ContinuousLGInference:
    def __init__(self, meta: BNMeta, device=None, dtype=torch.float32):
        self.meta = meta
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

    @torch.no_grad()
    def posterior(
        self,
        lp: LearnParams,
        evidence: Dict[str, torch.Tensor],
        query: List[str],
        do: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        assert lp.lg is not None, "Need LGParams."
        do = do or {}

        # Apply interventions: cut incoming edges, set mean=b=c, variance tiny
        W = lp.lg.W.clone()
        b = lp.lg.b.clone()
        sigma2 = lp.lg.sigma2.clone()
        for nm, val in do.items():
            if nm in lp.lg.name2idx:
                i = lp.lg.name2idx[nm]
                W[i, :] = 0.0
                b[i] = val.to(device=self.device, dtype=self.dtype).reshape(())
                sigma2[i] = 1e-12  # huge precision

        order = lp.lg.order
        name2idx = lp.lg.name2idx
        n = len(order)
        Identity = torch.eye(n, device=self.device, dtype=self.dtype)
        M = Identity - W
        invD = torch.diag_embed(1.0 / (sigma2 + 1e-20))
        J = M.transpose(-1, -2) @ invD @ M
        h = (M.transpose(-1, -2) @ (invD @ b.unsqueeze(-1))).squeeze(-1)

        # evidence may include non-continuous vars — keep only those present in LGParams
        e_idx = torch.tensor(
            [name2idx[nm] for nm in evidence.keys() if nm in name2idx],
            device=self.device,
            dtype=torch.long,
        )
        e_val = (
            torch.stack(
                [
                    evidence[nm].to(device=self.device, dtype=self.dtype).reshape(())
                    for nm in evidence.keys()
                    if nm in name2idx
                ]
            )
            if e_idx.numel() > 0
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )
        q_names = [nm for nm in query if nm in name2idx]
        q_idx = torch.tensor(
            [name2idx[nm] for nm in q_names], device=self.device, dtype=torch.long
        )

        perm = torch.cat([q_idx, e_idx], dim=0)
        Jp = J.index_select(0, perm).index_select(1, perm)
        hp = h.index_select(0, perm)

        if q_idx.numel() == 0:
            return {}, torch.empty(0, 0, device=self.device, dtype=self.dtype)

        k = q_idx.numel()
        Juu = Jp[:k, :k] + 1e-8 * torch.eye(k, device=self.device, dtype=self.dtype)
        Jue = Jp[:k, k:]
        hu = hp[:k]

        L = torch.linalg.cholesky(Juu)
        eta = hu - (Jue @ e_val if e_val.numel() else 0.0)
        mu = torch.cholesky_solve(eta.unsqueeze(-1), L).squeeze(-1)
        cov = torch.cholesky_inverse(L)

        mu_dict = {nm: mu[i] for i, nm in enumerate(q_names)}
        return mu_dict, cov
