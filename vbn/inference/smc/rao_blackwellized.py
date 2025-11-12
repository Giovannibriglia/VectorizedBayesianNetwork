# vbn/inference/smc/rao_blackwellized.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx
import torch
import torch.nn.functional as F

from ...learning.parametric_cpds.linear_gaussian import ParametricLinearGaussianCPD

from ..base import BaseInferencer


class RaoBlackwellizedSMCInferencer(BaseInferencer):
    """
    Rao–Blackwellized SMC (particle filter over BN) with:
      • Discrete nodes: categorical proposals (prior)
      • Linear-Gaussian nodes: analytically integrated (RB; no sampling)
      • Generic continuous nodes: fallback to CPD.sample
      • evidence & do(·) supported
      • systematic resampling when ESS < alpha * N (per batch row)
      • optional top-K pruning at the end

    API:
      infer_posterior(query, evidence={name: [B,1]}, do={name: [B,1]}, ...)
      -> pdf={"weights":[B,K], "logZ":[B]}, samples={query:[B,K]}
         where K = keep_top_k if set, else N.
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        nodes: Dict[str, object],
        device: torch.device,
        *,
        num_samples: int = 4096,
        ess_alpha: float = 0.5,
        proposal: str = "prior",
        resample: str = "systematic",
        # tolerate extra factory kwargs
        parents: Optional[Dict[str, list]] = None,
        keep_top_k: Optional[int] = 256,
        **kwargs,
    ):
        super().__init__(dag=dag, nodes=nodes, device=device, parents=parents)

        self.N = int(num_samples)
        self.ess_alpha = float(ess_alpha)
        self.proposal = proposal
        self.resample = resample
        self.keep_top_k = None if keep_top_k is None else int(keep_top_k)
        self.topo = list(nx.topological_sort(self.dag))

    # ------------------- helpers -------------------
    @staticmethod
    def _is_linear_gaussian(node) -> bool:
        return isinstance(node, ParametricLinearGaussianCPD)

    @staticmethod
    def _is_discrete(node) -> bool:
        # Softmax-like CPDs typically expose .linear or .net
        return hasattr(node, "linear") or hasattr(node, "net")

    @staticmethod
    def _first_tensor(d: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        for v in d.values():
            return v
        return None

    def _with_features(self, node, X: torch.Tensor) -> torch.Tensor:
        # Adds bias for roots in MLESoftmaxCPD; otherwise passthrough
        if hasattr(node, "_features"):
            return node._features(X)
        if X.shape[1] == 0:
            return torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        return X

    @torch.no_grad()
    def _systematic_resample(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B,N] normalized weights -> indices [B,N] via systematic resampling
        """
        B, N = w.shape
        cdf = torch.cumsum(w, dim=1)  # [B,N]
        # u in [0,1) stratified per particle
        u0 = (torch.rand(B, 1, device=w.device) + torch.arange(N, device=w.device)) / N
        u = (u0 % 1.0).expand_as(w)
        idx = torch.searchsorted(cdf, u, right=True).clamp(max=N - 1)
        return idx  # [B,N]

    @torch.no_grad()
    def _maybe_resample(
        self, logw: torch.Tensor, state: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        """
        Per-row ESS trigger. Only rows below threshold are resampled.
        logw: [B,N]; state[k]: [B,N] or None
        """
        B, N = logw.shape
        # normalize
        m = torch.amax(logw, dim=1, keepdim=True)
        w = torch.exp(logw - m)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)

        ess = 1.0 / (torch.sum(w * w, dim=1) + 1e-12)  # [B]
        need = ess < (self.ess_alpha * N)
        if not torch.any(need):
            return logw, state

        idx_all = self._systematic_resample(w)  # [B,N]
        new_state = {}
        for k, v in state.items():
            if v is None:
                new_state[k] = None
                continue
            # v: [B,N]
            v_new = v.clone()
            if torch.any(need):
                rows = need.nonzero(as_tuple=True)[0]
                if rows.numel() > 0:
                    v_new[rows] = torch.gather(v[rows], dim=1, index=idx_all[rows])
            new_state[k] = v_new

        # reset resampled rows to uniform log-weights; keep others untouched
        logw_new = logw.clone()
        if torch.any(need):
            rows = need.nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                logw_new[rows] = -torch.log(
                    torch.tensor(N, device=logw.device, dtype=logw.dtype)
                )
        return logw_new, new_state

    # ------------------- main -------------------
    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ):
        evidence = evidence or {}
        do = do or {}

        # Determine batch size B robustly (no ambiguous tensor truthiness)
        t0 = self._first_tensor(evidence)
        if t0 is None:
            t0 = self._first_tensor(do)
        B = int(t0.shape[0]) if t0 is not None else 1

        device = self.device
        N = self.N

        state: Dict[str, Optional[torch.Tensor]] = {n: None for n in self.topo}
        logw = torch.zeros(B, N, device=device)

        def expand_to_particles(x: torch.Tensor, as_long: bool = False) -> torch.Tensor:
            """
            x: [B,1] or [B] -> [B,N] (broadcast)
            """
            x = x.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(1)
            xBN = x.expand(B, N)
            return xBN.long() if as_long else xBN.float()

        # Forward pass in topological order
        for n in self.topo:
            node = self.nodes[n]
            parents = list(self.dag.predecessors(n))

            # Build parent matrix X for all particles
            if len(parents) == 0:
                X = torch.zeros(B * N, 0, device=device)
            else:
                pa_vals = [state[p] for p in parents]
                if any(v is None for v in pa_vals):
                    raise RuntimeError(f"Parent missing for node '{n}' in SMC.")
                X_bn_d = torch.stack(pa_vals, dim=-1)  # [B,N,|Pa|]
                X = X_bn_d.reshape(B * N, -1).float()  # [B*N, |Pa|]

            # Interventions have priority: set value, no weight
            if n in do:
                as_long = self._is_discrete(node)
                v = expand_to_particles(do[n], as_long=as_long)  # [B,N]
                state[n] = v.float()
                # ignore evidence if both set; or you may penalize mismatch
                logw, state = self._maybe_resample(logw, state)
                continue

            # Evidence: add likelihood weight and set state
            if n in evidence:
                as_long = self._is_discrete(node)
                y_obs = expand_to_particles(evidence[n], as_long=as_long)  # [B,N]
                if self._is_discrete(node):
                    y_flat = y_obs.reshape(B * N, 1).long()
                    lp = node.log_prob(X, y_flat)  # [B*N]
                    logw = logw + lp.reshape(B, N)
                    state[n] = y_obs.float()
                elif self._is_linear_gaussian(node):
                    # Analytic Gaussian likelihood
                    Xb = node._with_bias(X)  # [B*N, p]
                    mu = (Xb @ node.beta).reshape(B, N)  # [B,N]
                    var = node.sigma2.clamp_min(1e-8)
                    # out_dim==1 expected; if more, sum diagonal terms
                    if var.numel() == 1:
                        var_s = var[0]
                        logZ = 0.5 * torch.log(2 * torch.pi * var_s)
                        lp = -0.5 * ((y_obs - mu) ** 2) / var_s - logZ  # [B,N]
                    else:
                        # multi-dim (rare in examples): assume diagonal
                        var_s = var.view(1, 1, -1)  # [1,1,D]
                        mu3 = mu.view(B, N, -1)
                        y3 = y_obs.view(B, N, -1)
                        logZ = 0.5 * torch.sum(torch.log(2 * torch.pi * var_s), dim=-1)
                        lp = -0.5 * torch.sum((y3 - mu3) ** 2 / var_s, dim=-1) - logZ
                    logw = logw + lp
                    state[n] = y_obs.float()
                else:
                    # generic continuous with log_prob
                    if getattr(node.capabilities, "has_log_prob", False):
                        lp = node.log_prob(X, y_obs.reshape(B * N, 1).float())
                        logw = logw + lp.reshape(B, N)
                    state[n] = y_obs.float()
                logw, state = self._maybe_resample(logw, state)
                continue

            # Non-evidence, non-do: propose state
            if self._is_discrete(node):
                Xf = self._with_features(node, X)
                if hasattr(node, "linear"):
                    logits = node.linear(Xf)
                elif hasattr(node, "net"):
                    logits = node.net(Xf)
                else:
                    raise RuntimeError(f"Discrete node '{n}' has no logits method.")
                probs = F.softmax(logits, dim=-1)
                cat = torch.distributions.Categorical(probs=probs)
                y = cat.sample().reshape(B, N).float()
                state[n] = y
            elif self._is_linear_gaussian(node):
                # RB: propagate mean; weight adjusts only when evidence constrains
                Xb = node._with_bias(X)
                mu = (Xb @ node.beta).reshape(B, N)
                state[n] = (
                    mu  # store mean for children; sample later for query if needed
                )
            else:
                # fallback: sample from CPD
                ys = node.sample(X, n=1)  # [1, B*N, Dy]
                y = ys.reshape(1, B, N, -1).squeeze(0).squeeze(-1).float()  # [B,N]
                state[n] = y

            logw, state = self._maybe_resample(logw, state)

        # Normalize weights & compute logZ
        m = torch.amax(logw, dim=1, keepdim=True)  # [B,1]
        w = torch.exp(logw - m)  # [B,N]
        Z = w.sum(dim=1) + 1e-12  # [B]
        w = w / Z.unsqueeze(1)  # [B,N]
        logZ = m.squeeze(1) + torch.log(Z)  # [B]

        # Build query samples
        node_q = self.nodes[query]
        if self._is_discrete(node_q):
            q_samples = state[query]  # [B,N]
        elif self._is_linear_gaussian(node_q):
            mu = state[query]  # [B,N]
            var = node_q.sigma2.clamp_min(1e-8)
            if var.numel() == 1:
                eps = torch.randn_like(mu)
                q_samples = mu + eps * torch.sqrt(var[0])
            else:
                # simple diag sampling: take first dim (or sum) to scalar — adapt if needed
                eps = torch.randn_like(mu)
                q_samples = mu + eps * torch.sqrt(var.mean())  # coarse fallback
        else:
            q_samples = state[query]  # [B,N]

        # Optional per-row top-K pruning
        if self.keep_top_k is not None and 0 < self.keep_top_k < N:
            K = self.keep_top_k
            topw, idx = torch.topk(w, k=K, dim=1, largest=True, sorted=False)  # [B,K]
            # gather query samples
            q_samples = torch.gather(q_samples, 1, idx)  # [B,K]
            # renormalize weights in top-K
            w = topw / (topw.sum(dim=1, keepdim=True) + 1e-12)

        pdf = {"weights": w, "logZ": logZ}
        samples = {query: q_samples}
        return pdf, samples
