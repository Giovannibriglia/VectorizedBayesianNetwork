from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from vbn.inference.base import InferenceBackend


def _cpd_table_from_logprob(bn, X, device):
    """Return probs table with canonical shape [card[p1], ..., card[p_m], card[X]]
    by querying cpd.log_prob over all parent assignments and all y∈{0..K-1}."""
    parents = bn.parents.get(X, [])
    K = int(bn.nodes[X]["card"])
    if not parents:
        # just P(X)
        cpd = bn.cpd[X]
        lps = []
        for k in range(K):
            yk = torch.tensor([k], device=device).view(1, 1).long()
            lp = cpd.log_prob(yk, {})  # (1,)
            lps.append(lp)
        lps = torch.stack(lps, dim=0).view(1, K)  # [1,K]
        probs = F.softmax(lps, dim=-1).squeeze(0)  # [K]
        return probs  # shape [K]

    # parent cards and grids
    cards = [bn.nodes[p]["card"] for p in parents]
    grids = [torch.arange(c, device=device, dtype=torch.long) for c in cards]
    mesh = torch.cartesian_prod(*grids)  # [M, len(parents)]
    M = mesh.shape[0]

    # evaluate log_prob for all classes
    cpd = bn.cpd[X]
    lps = []
    for k in range(K):
        yk = torch.full((M, 1), k, device=device, dtype=torch.long)
        pars = {p: mesh[:, i].view(M, 1) for i, p in enumerate(parents)}
        lp = cpd.log_prob(yk, pars).view(M, 1)  # [M,1]
        lps.append(lp)
    lps = torch.cat(lps, dim=1)  # [M, K]
    probs = F.softmax(lps, dim=1)  # normalize over classes

    # reshape to [card[p1],...,card[p_m], K]
    return probs.view(*cards, K)


class LoopyBP(InferenceBackend):
    """Sum-product Loopy Belief Propagation on discrete factor graphs.
    - Works with discrete CPDs (uses .table)
    - Evidence via clamping, do via delta factor on intervened vars.
    """

    def __init__(self, iters: int = 20, damping: float = 0.0, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        self.iters = iters
        self.damping = damping

    def _build_factors(self, bn, evidence, do):
        cards = {
            n: bn.nodes[n]["card"]
            for n, s in bn.nodes.items()
            if s["type"] == "discrete"
        }
        factors = []
        for X in bn.topo_order:
            if bn.nodes[X]["type"] != "discrete":
                continue

            if X in do:
                K = cards[X]
                logt = torch.full((K,), -1e9, device=self.device)
                logt[int(do[X].view(-1)[0].item())] = 0.0
                factors.append(((X,), logt))
                continue

            parents = bn.parents.get(X, [])
            # get (probability) table
            if hasattr(bn.cpd[X], "table"):
                raw = bn.cpd[X].table.to(self.device).clamp_min(1e-12)
                shape = [cards[p] for p in parents] + [cards[X]]
                tab = raw.view(*shape)
            else:
                tab = _cpd_table_from_logprob(
                    bn, X, self.device
                )  # [cards(parents)..., K]

            logt = torch.log(tab.clamp_min(1e-12))
            scope = tuple(parents + [X])

            # slice by evidence
            for v, val in (evidence or {}).items():
                if v in scope:
                    axis = scope.index(v)
                    idx = int(val.view(-1)[0].item())
                    logt = logt.select(dim=axis, index=idx)
                    scope = tuple([s for s in scope if s != v])

            factors.append((scope, logt))
        return factors

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
        factors = self._build_factors(bn, evidence, do)
        # collect variable domains
        cards = {
            n: bn.nodes[n]["card"]
            for n, spec in bn.nodes.items()
            if spec["type"] == "discrete"
        }
        vars_all = list(cards.keys())
        # init messages var->factor and factor->var as zeros (log-domain uniform)
        # Build adjacency
        var_to_f = {v: [] for v in vars_all}
        for fi, (scope, _) in enumerate(factors):
            for v in scope:
                var_to_f[v].append(fi)
        # messages dict: (fi, v) factor->var ; (v, fi) var->factor
        m_fv = {}
        m_vf = {}
        for v in vars_all:
            for fi in var_to_f[v]:
                m_vf[(v, fi)] = torch.zeros(cards[v], device=self.device)
                m_fv[(fi, v)] = torch.zeros(cards[v], device=self.device)
        # iterate
        for _ in range(self.iters):
            # factor -> var (safe broadcasting & unary factors)
            for fi, (scope, logt) in enumerate(factors):
                # scope: tuple of variable names; logt: tensor shaped [cards[scope[0]], ..., cards[scope[-1]]]
                for v in scope:
                    pot = logt
                    # add incoming var->factor messages from all u≠v
                    for u in scope:
                        if u == v:
                            continue
                        mu = m_vf[(u, fi)]  # shape [cards[u]]
                        # reshape to broadcast on axis of u within this factor
                        shape = [1] * len(scope)
                        shape[scope.index(u)] = mu.shape[0]
                        pot = pot + mu.view(*shape)

                    axis_v = scope.index(v)
                    if pot.dim() == 1:
                        # unary factor: no marginalization needed
                        msg = pot
                    else:
                        # move v to last dim, sum out all others
                        pot = pot.movedim(axis_v, -1)
                        msg = torch.logsumexp(pot, dim=tuple(range(pot.dim() - 1)))

                    # normalize in log-domain
                    msg = msg - torch.logsumexp(msg, dim=0)

                    # damping
                    if self.damping > 0:
                        msg = (1 - self.damping) * msg + self.damping * m_fv[(fi, v)]

                    m_fv[(fi, v)] = msg

            # var -> factor
            for v in vars_all:
                for fi in var_to_f[v]:
                    msg = torch.zeros(cards[v], device=self.device)
                    for fj in var_to_f[v]:
                        if fj == fi:
                            continue
                        msg = msg + m_fv[(fj, v)]
                    msg = msg - torch.logsumexp(msg, dim=0)
                    if self.damping > 0:
                        msg = (1 - self.damping) * msg + self.damping * m_vf[(v, fi)]
                    m_vf[(v, fi)] = msg

        # beliefs
        out = {}
        for q in query:
            logb = torch.zeros(cards[q], device=self.device)
            for fi in var_to_f[q]:
                logb = logb + m_fv[(fi, q)]
            logb = logb - torch.logsumexp(logb, dim=0)
            out[q] = torch.exp(logb)
        return out
