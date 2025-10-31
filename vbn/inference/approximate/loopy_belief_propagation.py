from __future__ import annotations

from typing import Dict, Optional

import torch

from vbn.inference.base import InferenceBackend


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
        # cards once here
        cards = {
            n: bn.nodes[n]["card"]
            for n, spec in bn.nodes.items()
            if spec["type"] == "discrete"
        }
        factors = []
        for X in bn.topo_order:
            if bn.nodes[X]["type"] != "discrete":
                continue

            # handle do(X=·) as a delta (log 1 at the clamped index, -inf elsewhere)
            if X in do:
                K = cards[X]
                logt = torch.full((K,), -1e9, device=self.device)
                logt[int(do[X].view(-1)[0].item())] = 0.0
                factors.append(((X,), logt))
                continue

            parents = bn.parents.get(X, [])
            # --- CANONICAL RESHAPE ---
            # CPD table may be stored as [prod(parent_cards), K]; view it as
            # [card[p1], ..., card[p_m], card[X]] so evidence slicing works.
            raw = bn.cpd[X].table.to(self.device).clamp_min(1e-12)
            shape = [cards[p] for p in parents] + [cards[X]]
            logt = torch.log(raw.view(*shape))  # <-- key fix
            scope = tuple(parents + [X])

            # apply evidence by slicing (reduce scope on matched vars)
            for v, val in evidence.items():
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
