# vbn/inference/gaussian_exact.py
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from vbn.inference.base import InferenceBackend

Tensor = torch.Tensor


class GaussianExact(InferenceBackend):
    """Exact inference for linear-Gaussian scalar nodes.
    X = B X + b + ε,  ε~N(0, diag(σ²))
    - Uses current LinearGaussianCPD with out_dim=1: .W (in_dim,), .b (1,), .sigma2 (scalar)
    - do(X=c): cut incoming edges, set b_i=c, set σ²_i≈0
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    def _collect_nodes(self, bn):
        cont = [
            n
            for n, sp in bn.nodes.items()
            if sp.get("type") == "gaussian" and sp.get("dim", 1) == 1
        ]
        idx = {n: i for i, n in enumerate(cont)}
        return cont, idx

    def _build_B_b_S(self, bn, cont, idx):
        k = len(cont)
        B = torch.zeros((k, k), device=self.device)
        b = torch.zeros((k,), device=self.device)
        S = torch.zeros((k,), device=self.device)
        for n in cont:
            i = idx[n]
            cpd = bn.cpd[n]
            b[i] = cpd.b.view(-1)[0]
            S[i] = float(cpd.sigma2)
            parents = bn.parents.get(n, [])
            if parents:
                off = 0
                for p in parents:
                    j = idx.get(p, None)
                    if j is not None:
                        # assume each parent contributes one scalar weight
                        w = cpd.W[off : off + 1].view(-1)[0]
                        B[i, j] = w
                        off += 1
        return B, b, S

    @staticmethod
    def _joint_from_linear_gaussian(B: Tensor, b: Tensor, S_diag: Tensor):
        Id = torch.eye(B.shape[0], device=B.device)
        M = torch.linalg.inv(Id - B)  # (I-B)^{-1}
        mu = M @ b
        Cov_eps = torch.diag(S_diag)
        Sigma = M @ Cov_eps @ M.T
        return mu, Sigma

    def posterior(
        self,
        bn,
        query: Sequence[str],
        evidence: Optional[Dict[str, Tensor]] = None,
        do: Optional[Dict[str, Tensor]] = None,
        **kw,
    ):
        evidence = {
            k: v.to(self.device).view(-1)[0] for k, v in (evidence or {}).items()
        }
        do = {k: v.to(self.device).view(-1)[0] for k, v in (do or {}).items()}

        cont, idx = self._collect_nodes(bn)
        if not cont:
            return {}  # nothing Gaussian to do

        B, b, S = self._build_B_b_S(bn, cont, idx)

        # do-surgery
        for x, val in do.items():
            if x in idx:
                i = idx[x]
                B[i, :].zero_()
                b[i] = val
                S[i] = 1e-10

        mu, Sigma = self._joint_from_linear_gaussian(B, b, S)

        # If no evidence: return marginals (mean,var) for Gaussian queries
        if not evidence:
            out = {}
            for q in query:
                if q in idx:
                    i = idx[q]
                    out[q] = {"mean": mu[i], "var": Sigma[i, i]}
            return out

        # Condition on Gaussian evidence only
        E_idx = [idx[k] for k in evidence.keys() if k in idx]
        Q_idx = [idx[k] for k in query if k in idx]
        if not Q_idx:
            return {}

        E = torch.tensor(E_idx, device=self.device, dtype=torch.long)
        Q = torch.tensor(Q_idx, device=self.device, dtype=torch.long)

        mu_E = mu[E]
        mu_Q = mu[Q]
        Sigma_EE = Sigma.index_select(0, E).index_select(1, E)
        Sigma_QE = Sigma.index_select(0, Q).index_select(1, E)
        Sigma_QQ = Sigma.index_select(0, Q).index_select(1, Q)

        y_E = torch.stack([evidence[cont[i]] for i in E_idx]).to(self.device)

        # Stable conditioning via Cholesky
        L = torch.linalg.cholesky(
            Sigma_EE + 1e-8 * torch.eye(len(E_idx), device=self.device)
        )
        delta = torch.cholesky_solve((y_E - mu_E).unsqueeze(1), L).squeeze(1)
        post_mean = mu_Q + Sigma_QE @ delta
        inv_Sigma_EE = torch.cholesky_inverse(L)
        post_cov = Sigma_QQ - Sigma_QE @ inv_Sigma_EE @ Sigma_QE.T

        out = {}
        for k, pos in zip([q for q in query if q in idx], range(len(Q_idx))):
            out[k] = {"mean": post_mean[pos], "var": post_cov[pos, pos]}
        return out
