from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vbn.inference.base import _as_B1, BaseInferencer

"""
Mean-field VI with amortized factors per batch query.
- Discrete nodes: categorical q_i with logits φ_i ∈ R^{K_i}; sampled via Gumbel-Softmax (relaxed)
- Continuous reparameterized nodes: diagonal Gaussian q_i(y)=N(μ_i, σ_i^2)
Optimizes ELBO via SGD for T steps (inner loop) and S samples for Monte Carlo expectations.
Returns posterior samples for 'query' by drawing from the learned q(query).
"""


def _is_discrete(node) -> bool:
    cfg = getattr(node, "cfg", {})
    k = int(cfg.get("num_classes", cfg.get("out_dim", 1)))
    return k >= 2 and not getattr(node.capabilities, "is_reparameterized", False)


def _num_classes(node) -> int:
    cfg = getattr(node, "cfg", {})
    return int(cfg.get("num_classes", cfg.get("out_dim", 1)))


class MeanFieldFullVIInferencer(BaseInferencer):
    @torch.no_grad()
    def _parents_X(
        self, assign: Dict[str, torch.Tensor], ps: List[str]
    ) -> torch.Tensor:
        # assign: node -> [B,S] (samples); if empty parents -> [B*S,0]
        if not ps:
            return torch.empty(
                (next(iter(assign.values())).numel(), 0), device=self.device
            )
        xs = [assign[p].float().unsqueeze(-1) for p in ps]  # each [B,S,1]
        X = torch.cat(xs, dim=-1)  # [B,S,|Pa|]
        B, S, D = X.shape
        return X.reshape(B * S, D)

    def _build_vi_params(
        self, free_nodes: List[str], K: Dict[str, int], B: int, device: torch.device
    ):
        vi = {}
        for n in free_nodes:
            if K.get(n, 1) >= 2:  # discrete
                # logits φ: [B, K]
                # estimate marginal class frequencies from CPD by sampling roots (cheap) or set small bias
                phi = torch.zeros(B, K[n], device=device, requires_grad=True)
                phi.data += 0.01 * torch.randn_like(phi)  # small noise break symmetry
                vi[n] = {"type": "cat", "phi": phi}
            else:
                # Gaussian reparam (μ, logσ): [B,1]
                mu = torch.zeros(B, 1, device=device, requires_grad=True)
                log_sigma = torch.zeros(B, 1, device=device, requires_grad=True)
                vi[n] = {"type": "gauss", "mu": mu, "log_sigma": log_sigma}
        return vi

    def _sample_from_vi(
        self,
        vi: dict,
        num_samples: int,
        tau: float,
        *,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for n, prm in vi.items():
            if prm["type"] == "cat":
                phi = prm["phi"]  # [B,K]
                B, K = phi.shape
                logits = phi.unsqueeze(1).expand(B, num_samples, K)  # [B,S,K]
                g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
                probs = torch.softmax((logits + g) / max(1e-4, tau), dim=-1)  # [B,S,K]
                if hard:
                    idx = probs.argmax(dim=-1)  # [B,S]
                else:
                    cat = torch.distributions.Categorical(probs=probs.reshape(-1, K))
                    idx = cat.sample().view(B, num_samples)  # [B,S]
                out[n] = idx
            else:
                mu = prm["mu"]  # [B,1]
                log_sigma = prm["log_sigma"]  # [B,1]
                B = mu.shape[0]
                eps = torch.randn(B, num_samples, 1, device=self.device)  # [B,S,1]
                y = mu.unsqueeze(1) + eps * log_sigma.exp().unsqueeze(1)  # [B,S,1]
                out[n] = y.squeeze(-1)  # [B,S]
        return out

    def _elbo(
        self,
        vi: dict,
        samples: Dict[str, torch.Tensor],  # {node: [B,S]}
        evidence_B1: Dict[str, torch.Tensor],  # {node: [B,1]}
        do_B1: Dict[str, torch.Tensor],  # {node: [B,1]}
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        ELBO = E_q[ sum_n log p(n | Pa_n) ]  -  E_q[ sum_free log q(n) ]
        • Evidence contributes to likelihood (clamped), no entropy term.
        • Do-nodes are clamped and excluded from both sums.
        • Uses analytic E_q[log q] for Cat/Gauss (stable & faster).
        """
        # infer B,S
        some_key = next(iter(samples))
        B, S = samples[some_key].shape

        # ----- unified assignment: assign[n]: [B,S] -----
        assign: Dict[str, torch.Tensor] = {}

        # free nodes (sampled)
        for n, y in samples.items():
            assign[n] = y  # [B,S]

        # evidence (repeat across S)
        for n, v in evidence_B1.items():
            assign[n] = v.expand(B, S)

        # interventions (repeat across S)
        for n, v in do_B1.items():
            assign[n] = v.expand(B, S)

        # ----- E_q[ sum log p(n | Pa_n) ] -----
        logp = 0.0
        for n in self.topo:
            if n in do_B1:
                continue
            node = self.nodes[n]
            ps = self.parents[n]

            if ps:
                # stack parents and CAST TO FLOAT for CPDs with Linear layers
                X_b_s_p = torch.stack([assign[p] for p in ps], dim=-1).to(
                    self.device
                )  # [B,S,|Pa|]
                X_flat = X_b_s_p.reshape(B * S, -1).float()  # [B*S,|Pa|]
            else:
                X_flat = torch.empty(
                    (B * S, 0), device=self.device, dtype=torch.float32
                )

            y_flat = assign[n].reshape(B * S, 1).to(self.device)

            lp = node.log_prob(
                X_flat, y_flat
            )  # expects float X, y index/type handled internally
            logp = logp + beta * lp.mean()

        # ----- Analytic E_q[ sum_free log q(n) ]  (exclude evidence & do) -----
        logq = 0.0
        LOG_2PI_E = math.log(2.0 * math.pi * math.e)  # for 1-D Gaussian entropy

        for n, prm in vi.items():
            if (n in evidence_B1) or (n in do_B1):
                continue

            if prm["type"] == "cat":
                # q(y) = Categorical(softmax(phi))
                phi = prm["phi"]  # [B,K]
                lsm = F.log_softmax(phi, dim=-1)  # [B,K]
                probs = lsm.exp()  # [B,K]
                # E_q[log q] = sum_k q_k log q_k  (=-H(q))
                lq_b = (probs * lsm).sum(dim=-1)  # [B]
                logq = logq + lq_b.mean()

            else:
                # q(y) = N(mu, sigma^2), 1-D
                log_sigma = prm["log_sigma"]  # [B,1], sigma^2 = exp(2 log_sigma)
                # E_q[log q] = -H = -0.5 * log(2π e σ^2)
                lq_b = -0.5 * (LOG_2PI_E + 2.0 * log_sigma.squeeze(-1))  # [B]
                logq = logq + lq_b.mean()

        return logp - logq

    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 1024,
        steps: int = 300,
        lr: float = 1e-2,
        tau: float = 0.5,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        ev = {k: _as_B1(v, self.device) for k, v in (evidence or {}).items()}
        do_ = {k: _as_B1(v, self.device) for k, v in (do or {}).items()}
        # capabilities: need log_prob; for continuous VI also fine; sampling not required from CPD
        self._check_caps({n: ("has_log_prob",) for n in self.topo if n not in do_})

        # cardinalities (for discrete vs continuous)
        K = {n: _num_classes(self.nodes[n]) for n in self.topo}
        # batch size B from evidence/do or default 1
        B = 1
        for t in list(ev.values()) + list(do_.values()):
            B = max(B, t.shape[0])

        # free nodes (to be inferred)
        free_nodes = [n for n in self.topo if n not in ev and n not in do_]
        if len(free_nodes) == 0:
            # trivial: just return query fixed (ev/do must contain it)
            q = ev.get(query, do_.get(query))
            qS = q.expand(B, num_samples)
            return {
                "weights": torch.ones(B, num_samples, device=self.device) / num_samples
            }, {query: qS}

        # Initialize VI parameters
        vi = self._build_vi_params(free_nodes, K, B, self.device)

        # Optimize ELBO
        params = []
        for prm in vi.values():
            if prm["type"] == "cat":
                params.append(prm["phi"])
            else:
                params.extend([prm["mu"], prm["log_sigma"]])
        opt = torch.optim.Adam(params, lr=lr)

        # VI optimization loop
        for step in range(steps):
            opt.zero_grad(set_to_none=True)

            tau_t = max(0.2, tau * (0.5 ** (step / max(1, steps // 3))))
            beta_t = min(1.0, 0.1 + 0.9 * (step / steps))
            hard = step >= int(0.6 * steps)

            S_elbo = max(16, num_samples // 8)
            samples_q = self._sample_from_vi(
                vi, num_samples=S_elbo, tau=tau_t, hard=hard
            )

            elbo = self._elbo(
                vi, samples_q, ev, do_, beta=beta_t
            )  # ← no num_samples kwarg
            loss = -elbo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        self.last_elbo = elbo.detach()

        # Final posterior samples from q(query)
        final_samples = self._sample_from_vi(vi, num_samples=num_samples, tau=tau)
        if query in final_samples:
            q_BN = final_samples[query]  # [B,N]
        elif query in ev:
            q_BN = ev[query].expand(B, num_samples)
        elif query in do_:
            q_BN = do_[query].expand(B, num_samples)
        else:
            raise RuntimeError(
                f"Query '{query}' not found among VI variables or fixed sets."
            )

        pdf = {
            "weights": torch.full(
                (B, num_samples), 1.0 / num_samples, device=self.device
            )
        }
        samples = {query: q_BN.detach()}
        return pdf, samples
