from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..core import BNMeta, LearnParams

from .base import BaseInference


class ContinuousLGInference(BaseInference):
    def __init__(self, meta: BNMeta, device=None, dtype=torch.float32, **kwargs):
        super().__init__(meta, device, dtype)
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
        return_samples: bool = False,
        **kwargs,
    ) -> (
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
        | Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]
    ):
        """
        Returns:
            mu_dict: {q_i: posterior mean (scalar tensor on device)}
            cov:     posterior covariance over queries, shape [k, k] (k=len(query_in_graph))

            If return_samples=True:
              (mu_dict, cov, samples_dict)
              where samples_dict[q_i] has shape [n_samples].
        """

        num_samples = kwargs.get("num_samples", 512)

        assert lp.lg is not None, "Need LGParams."
        do = do or {}

        # Apply interventions: cut incoming edges; set mean=b=c; tiny variance
        W = lp.lg.W.clone()
        b = lp.lg.b.clone()
        sigma2 = lp.lg.sigma2.clone().to(self.device, self.dtype)
        for nm, val in do.items():
            if nm in lp.lg.name2idx:
                i = lp.lg.name2idx[nm]
                W[i, :] = 0.0
                b[i] = val.to(device=self.device, dtype=self.dtype).reshape(())
                sigma2[i] = torch.tensor(
                    1e-12, device=self.device, dtype=self.dtype
                )  # huge precision

        order = lp.lg.order
        name2idx = lp.lg.name2idx
        n = len(order)
        identity = torch.eye(n, device=self.device, dtype=self.dtype)
        M = identity - W.to(self.device, self.dtype)
        invD = torch.diag_embed(1.0 / (sigma2 + 1e-20))
        J = M.transpose(-1, -2) @ invD @ M  # precision of joint
        h = (
            M.transpose(-1, -2) @ (invD @ b.to(self.device, self.dtype).unsqueeze(-1))
        ).squeeze(-1)

        # Keep only evidence variables present in LGParams
        e_names = [nm for nm in evidence.keys() if nm in name2idx]
        e_idx = torch.tensor(
            [name2idx[nm] for nm in e_names], device=self.device, dtype=torch.long
        )
        e_val = (
            torch.stack(
                [evidence[nm].to(self.device, self.dtype).reshape(()) for nm in e_names]
            )
            if e_idx.numel() > 0
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

        # Queries present in the LG
        q_names = [nm for nm in query if nm in name2idx]
        q_idx = torch.tensor(
            [name2idx[nm] for nm in q_names], device=self.device, dtype=torch.long
        )

        if q_idx.numel() == 0:
            # nothing to infer within this LG
            empty_cov = torch.empty(0, 0, device=self.device, dtype=self.dtype)
            return ({}, empty_cov) if not return_samples else ({}, empty_cov, {})

        # Reorder as [q, e] to compute conditional
        perm = torch.cat([q_idx, e_idx], dim=0)
        Jp = J.index_select(0, perm).index_select(1, perm)
        hp = h.index_select(0, perm)

        k = q_idx.numel()
        Juu = Jp[:k, :k] + 1e-8 * torch.eye(
            k, device=self.device, dtype=self.dtype
        )  # stabilizer
        Jue = Jp[:k, k:]
        hu = hp[:k]

        # Posterior natural params for q | e:  J_post = Juu,   h_post = hu - Jue * e
        eta = hu - (Jue @ e_val if e_val.numel() else 0.0)

        # Solve Juu * mu = eta  via Cholesky
        L = torch.linalg.cholesky(Juu)
        mu = torch.cholesky_solve(eta.unsqueeze(-1), L).squeeze(-1)  # [k]
        cov = torch.cholesky_inverse(L)  # [k, k]

        mu_dict = {nm: mu[i] for i, nm in enumerate(q_names)}

        if not return_samples:
            return mu_dict, cov

        # ---- Sample from N(mu, cov) for the queried block ----
        # Use MultivariateNormal with a tiny jitter for extra safety
        jitter = 1e-10
        cov_safe = cov + jitter * torch.eye(k, device=self.device, dtype=self.dtype)
        dist = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=cov_safe
        )
        S = max(int(num_samples), 1)
        draws = dist.sample((S,))  # [S, k]

        samples_dict: Dict[str, torch.Tensor] = {
            nm: draws[:, i] for i, nm in enumerate(q_names)
        }
        return mu_dict, cov, samples_dict
