from __future__ import annotations

from typing import Dict, Optional

import torch

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
        parents = bn.parents.get(X, [])
        par_vals = {p: state[p] for p in parents if p in state}
        # build per-chain probs from CPD table indexing parents
        cpd = bn.cpd[X]
        table = cpd.table.to(self.device)  # shape [card[p1],..., card[X]]
        if parents:
            # gather along each parent dim
            idx = [par_vals[p].view(-1) for p in parents]
            probs = table
            for dim, idv in enumerate(idx):
                probs = probs.index_select(dim, idv)
            # now probs has shape [chains, card[X]] (because each select reduces dim)
        else:
            probs = table.view(1, -1).expand(self.chains, -1)
        cat = torch.distributions.Categorical(probs=probs)
        return cat.sample().view(self.chains, 1)

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
