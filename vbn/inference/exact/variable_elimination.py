from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from vbn.inference.base import InferenceBackend

Tensor = torch.Tensor


@dataclass
class Factor:
    table: torch.Tensor  # flat or shaped
    scope: List[str]  # variable names in this factor
    cards: Dict[str, int]  # cardinality per variable in scope

    def to(self, device):
        # keep scope/cards, move table
        return Factor(
            table=self.table.to(device), scope=list(self.scope), cards=dict(self.cards)
        )

    def shaped(self) -> torch.Tensor:
        """Return table reshaped to the canonical shape [cards[v] for v in scope].
        If scope is empty, return a scalar tensor with shape [1] to allow broadcasting.
        """
        if len(self.scope) == 0:
            return self.table.view(1)
        shape = [int(self.cards[v]) for v in self.scope]
        return self.table.view(*shape)

    def normalize(self) -> "Factor":
        t = self.shaped()
        s = t.sum()
        if s.item() == 0:
            return self
        self.table = (t / s).reshape(-1)
        return self


def factor_from_cpd(
    node: str, parents: List[str], card: Dict[str, int], table: Tensor
) -> Factor:
    """
    CPD P(node | parents) -> factor over (parents + node).
    table can be [prod_parent_cards, K] or already shaped; we canonicalize to flat.
    """
    shape = [int(card[p]) for p in parents] + [int(card[node])]
    t = table.view(*shape)  # safe if numel matches
    return Factor(
        table=t.reshape(-1),
        scope=list(parents) + [node],
        cards={**{p: int(card[p]) for p in parents}, node: int(card[node])},
    )


def multiply(a: Factor, b: Factor) -> Factor:
    # union scope with stable order: vars in a first, then b\ a
    scope = list(a.scope) + [v for v in b.scope if v not in a.scope]
    # merged cards
    cards = dict(a.cards)
    for v in b.cards:
        cards[v] = int(b.cards[v])

    def align(f: Factor) -> torch.Tensor:
        # reshape to its own canonical shape
        t = f.shaped()

        # If f has empty scope, build a rank==len(scope) view so expand works
        if len(f.scope) == 0:
            if len(scope) == 0:
                return t.view(1)  # scalar stays scalar
            target = [int(cards[v]) for v in scope]
            # start as all-ones so broadcast/expand is legal
            t = t.view(*([1] * len(scope)))
            return t.expand(*target)

        if len(scope) == 0:
            return t.view(1)  # scalar factor

        # permute existing axes to match 'scope' order for variables present in f
        present = [v for v in scope if v in f.scope]
        if present:
            perm = [f.scope.index(v) for v in present]
            if perm != list(range(len(perm))):
                t = t.permute(perm)

        # insert singleton dims for variables not in f
        full_shape = []
        cur_axes = 0
        for v in scope:
            if v in f.scope:
                full_shape.append(int(f.cards[v]))
                cur_axes += 1
            else:
                t = t.unsqueeze(cur_axes)
                full_shape.append(int(cards[v]))
                cur_axes += 1

        return t.expand(*full_shape)

    ta = align(a)
    tb = align(b)
    prod = ta * tb
    return Factor(table=prod.reshape(-1), scope=scope, cards=cards)


def sum_out(f: Factor, var: str) -> Factor:
    if var not in f.scope:
        return f
    t = f.shaped()
    axis = f.scope.index(var)
    scope = list(f.scope)
    cards = dict(f.cards)
    # sum over the axis
    t = t.sum(dim=axis)
    scope.pop(axis)
    cards.pop(var, None)
    return Factor(table=t.reshape(-1), scope=scope, cards=cards)


def reduce_evidence(f: Factor, evidence: Dict[str, torch.Tensor]) -> Factor:
    if not evidence:
        return f
    t = f.shaped()
    scope = list(f.scope)
    cards = dict(f.cards)
    for var, val in evidence.items():
        if var not in scope:
            continue
        axis = scope.index(var)
        idx = int(val.view(-1)[0].item())
        # select along the explicit axis
        t = t.select(dim=axis, index=idx)
        # drop var from scope/cards
        scope.pop(axis)
        cards.pop(var, None)
    # keep as flat for consistency; Factor.shaped() will re-shape when needed
    return Factor(table=t.reshape(-1), scope=scope, cards=cards)


def delta_factor(var: str, val: int, card: Dict[str, int], device) -> Factor:
    t = torch.zeros(int(card[var]), device=device)
    t[val] = 1.0
    return Factor(table=t.reshape(-1), scope=[var], cards={var: int(card[var])})


class VariableElimination(InferenceBackend):
    """Exact discrete inference with VE (GPU-friendly)."""

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    def posterior(
        self,
        bn,
        query: Sequence[str],
        evidence: Optional[Dict[str, Tensor]] = None,
        do: Optional[Dict[str, Tensor]] = None,
        **kw,
    ) -> Dict[str, Tensor]:
        device = self.device
        # cards
        card = {
            n: bn.nodes[n]["card"]
            for n, spec in bn.nodes.items()
            if spec["type"] == "discrete"
        }
        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}

        factors: List[Factor] = []
        for X in bn.topo_order:
            if bn.nodes[X]["type"] != "discrete":
                continue  # VE here handles only discrete nodes
            parents = bn.parents.get(X, [])
            if X in do:
                factors.append(
                    delta_factor(X, int(do[X].view(-1)[0].item()), card, device)
                )
                continue
            cpd = bn.cpd[X]
            assert hasattr(
                cpd, "table"
            ), f"VE requires discrete CPD with .table for {X}"
            table = cpd.table.to(device)
            factors.append(factor_from_cpd(X, parents, card, table))

        if evidence:
            factors = [reduce_evidence(f, evidence) for f in factors]

        keep = set(query) | set(evidence or {}) | set(do or {})
        elim_vars = [v for v in bn.topo_order if v in card and v not in keep]

        for Z in elim_vars:
            bucket = [f for f in factors if Z in f.scope]
            if not bucket:
                continue
            prod = bucket[0]
            for f in bucket[1:]:
                prod = multiply(prod, f)
            newf = sum_out(prod, Z)
            factors = [f for f in factors if Z not in f.scope] + [newf]

        prod = factors[0]
        for f in factors[1:]:
            prod = multiply(prod, f)
        prod = prod.normalize()

        out: Dict[str, Tensor] = {}
        if len(query) == 1:
            q = query[0]
            g = prod
            for v in list(g.scope):
                if v != q:
                    g = sum_out(g, v)
            out[q] = g.table
        else:
            out["joint"] = prod.table
        return out
