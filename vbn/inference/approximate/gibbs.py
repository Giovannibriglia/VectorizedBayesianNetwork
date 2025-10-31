from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from vbn.inference.base import InferenceBackend


class ParallelGibbs(InferenceBackend):
    """Blocked/parallel Gibbs sampling for discrete BNs.
    - Multiple chains in parallel; evidence and do handle via clamping.
    - Uses CPDs to sample each variable given current Markov blanket.
    """

    def __init__(
        self,
        burn_in: int = 200,
        steps: int = 1000,
        chains: int = 512,
        device="cpu",
        **kwargs,
    ):
        super().__init__(device, **kwargs)
        self.burn_in = burn_in
        self.steps = steps
        self.chains = chains

    def _init_state(self, bn, evidence, do):
        state = {}
        for X in bn.topo_order:
            if bn.nodes[X]["type"] != "discrete":
                continue
            K = bn.nodes[X]["card"]
            if X in do:
                state[X] = do[X].to(self.device).view(1, 1).repeat(self.chains, 1)
            elif X in evidence:
                state[X] = evidence[X].to(self.device).view(1, 1).repeat(self.chains, 1)
            else:
                state[X] = torch.randint(0, K, (self.chains, 1), device=self.device)
        return state

    def _sample_var(self, bn, X, state):
        """Sample X ~ P(X | MB(X)) using either CPD.table (fast path) or CPD.log_prob (fallback)."""
        K = int(bn.nodes[X]["card"])
        parents = bn.parents.get(X, [])
        chains = self.chains

        # fast path: use explicit probability table if present
        cpd = bn.cpd[X]
        if hasattr(cpd, "table"):
            # Build parent index per chain and gather probabilities
            probs = cpd.table.to(
                self.device
            )  # assumed non-log, shape [Π card(parents), K] or reshaped
            if parents:
                # canonical shape [card[p1], ..., card[p_m], K]
                cards = [bn.nodes[p]["card"] for p in parents] + [K]
                probs = probs.view(*cards)
                # index-select along each parent axis
                out = probs
                for dim, p in enumerate(parents):
                    idx = state[p].view(-1)  # (chains,)
                    out = out.index_select(dim, idx)
                # now out shape is [chains, K]
                probs = out
            else:
                probs = probs.view(1, K).expand(chains, K)
        else:
            # fallback: compute probs via log_prob for each class k
            # build parents dict for this step (per-chain tensors)
            pars = {p: state[p] for p in parents}
            lps = []
            for k in range(K):
                yk = torch.full((chains, 1), k, device=self.device, dtype=torch.long)
                lp = cpd.log_prob(yk, pars)  # (chains,)
                lps.append(lp.view(-1, 1))
            lps = torch.cat(lps, dim=1)  # (chains, K)
            probs = F.softmax(lps, dim=1)  # normalize across classes

        cat = torch.distributions.Categorical(probs=probs)
        return cat.sample().view(chains, 1)

    def posterior(
        self,
        bn,
        query,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        evidence = {k: v.to(self.device) for k, v in (evidence or {}).items()}
        do = {k: v.to(self.device) for k, v in (do or {}).items()}
        state = self._init_state(bn, evidence, do)
        order = [X for X in bn.topo_order if bn.nodes[X]["type"] == "discrete"]
        samples_acc = {
            q: torch.zeros(bn.nodes[q]["card"], device=self.device) for q in query
        }
        kept = 0
        for t in range(self.burn_in + self.steps):
            for X in order:
                if X in do or X in evidence:
                    continue
                state[X] = self._sample_var(bn, X, state)
            if t >= self.burn_in:
                kept += 1
                for q in query:
                    hist = torch.bincount(
                        state[q].view(-1), minlength=bn.nodes[q]["card"]
                    ).float()
                    samples_acc[q] += hist
        out = {q: (samples_acc[q] / samples_acc[q].sum()) for q in query}
        return out
