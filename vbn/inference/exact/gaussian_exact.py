from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from vbn.inference.base import InferenceBackend


class GaussianExact(InferenceBackend):
    """Exact inference for linear-Gaussian BNs with scalar nodes.
    Assumptions:
      - Each continuous node X has CPD: X = sum_j w_j * parent_j + b + ε, ε~N(0, σ²)
      - Implemented via current LinearGaussianCPD (out_dim=1).
    Supports evidence and do(·) (incoming edges cut; X fixed with tiny variance).
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    def _build_structures(self, bn):
        idx = {}
        cont_nodes = [
            n
            for n, spec in bn.nodes.items()
            if spec.get("type") == "gaussian" and spec.get("dim", 1) == 1
        ]
        for i, n in enumerate(cont_nodes):
            idx[n] = i
        k = len(cont_nodes)
        B = torch.zeros((k, k), device=self.device)
        b = torch.zeros((k,), device=self.device)
        Sigma_eps = torch.zeros((k,), device=self.device)
        for n in cont_nodes:
            i = idx[n]
            cpd = bn.cpd[n]
            parents = bn.parents.get(n, [])
            b[i] = cpd.b.view(-1)[0]
            Sigma_eps[i] = cpd.sigma2
            if parents:
                # cpd.W maps concat(parents) -> out. We assume scalar parents each with dim=1
                offset = 0
                for p in parents:
                    j = idx.get(p, None)
                    if j is not None:
                        w = cpd.W[offset : offset + 1].view(-1)[0]
                        B[i, j] = w
                        offset += 1
        return cont_nodes, idx, B, b, Sigma_eps

    def _joint_mean_cov(self, B, b, Sigma_eps):
        # X = B X + b + ε ⇒ (I - B) X = b + ε ⇒ X = M(b + ε), M = (I - B)^{-1}
        Id = torch.eye(B.shape[0], device=B.device)
        M = torch.linalg.inv(Id - B)
        mu = M @ b
        Cov_eps = torch.diag(Sigma_eps)
        Sigma = M @ Cov_eps @ M.T
        return mu, Sigma

    def posterior(
        self,
        bn,
        query: Sequence[str],
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        evidence = {
            k: v.to(self.device).view(-1)[0] for k, v in (evidence or {}).items()
        }
        do = {k: v.to(self.device).view(-1)[0] for k, v in (do or {}).items()}
        cont_nodes, idx, B, b, Sigma_eps = self._build_structures(bn)
        # do-surgery: cut incoming edges to intervened nodes, set their mean to value and near-zero noise
        for x, val in do.items():
            if x in idx:
                i = idx[x]
                B[i, :] = 0.0
                b[i] = val
                Sigma_eps[i] = 1e-8
        mu, Sigma = self._joint_mean_cov(B, b, Sigma_eps)
        if not evidence:
            # return marginals for query
            out = {}
            for q in query:
                i = idx[q]
                out[q] = {"mean": mu[i], "var": Sigma[i, i]}
            return out
        # Condition joint Gaussian on evidence
        E = [idx[k] for k in evidence if k in idx]
        Q = [idx[k] for k in query if k in idx]
        E = torch.tensor(E, device=self.device, dtype=torch.long)
        Q = torch.tensor(Q, device=self.device, dtype=torch.long)
        mu_E = mu[E]
        mu_Q = mu[Q]
        Sigma_EE = Sigma[E][:, E]
        Sigma_QE = Sigma[Q][:, E]
        Sigma_QQ = Sigma[Q][:, Q]
        y_E = torch.stack([evidence[cont_nodes[i]] for i in E.cpu().tolist()]).to(
            self.device
        )
        L = torch.linalg.cholesky(
            Sigma_EE + 1e-8 * torch.eye(len(E), device=self.device)
        )
        delta = torch.cholesky_solve((y_E - mu_E).unsqueeze(1), L).squeeze(1)
        post_mean = mu_Q + Sigma_QE @ delta
        inv_Sigma_EE = torch.cholesky_inverse(L)
        post_cov = Sigma_QQ - Sigma_QE @ inv_Sigma_EE @ Sigma_QE.T
        out = {}
        for k, i in zip(query, Q.tolist()):
            out[k] = {
                "mean": post_mean[Q.tolist().index(i)],
                "var": post_cov[Q.tolist().index(i), Q.tolist().index(i)],
            }
        return out
