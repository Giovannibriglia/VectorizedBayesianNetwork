# vbn/inference/exact/variable_elimination.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from vbn.inference.base import InferenceBackend

Tensor = torch.Tensor


@dataclass
class Factor:
    table: Tensor
    scope: List[str]
    cards: Dict[str, int]

    def to(self, device):
        return Factor(self.table.to(device), list(self.scope), dict(self.cards))

    def shaped(self) -> Tensor:
        if not self.scope:
            return self.table.view(1)
        shape = [int(self.cards[v]) for v in self.scope]
        return self.table.view(*shape)

    def normalize_(self):
        t = self.shaped()
        s = t.sum()
        if s > 0:
            self.table = (t / s).reshape(-1)
        return self


def multiply(a: Factor, b: Factor) -> Factor:
    scope = list(a.scope) + [v for v in b.scope if v not in a.scope]
    cards = dict(a.cards)
    for v in b.cards:
        cards[v] = int(b.cards[v])

    def align(f: Factor) -> Tensor:
        t = f.shaped()
        if not scope:
            return t.view(1)
        if not f.scope:
            return t.view(*([1] * len(scope))).expand(*[int(cards[v]) for v in scope])

        present = [v for v in scope if v in f.scope]
        if present:
            perm = [f.scope.index(v) for v in present]
            if perm != list(range(len(perm))):
                t = t.permute(perm)

        full_shape = []
        cur = 0
        for v in scope:
            if v in f.scope:
                full_shape.append(int(cards[v]))
                cur += 1
            else:
                t = t.unsqueeze(cur)
                full_shape.append(int(cards[v]))
                cur += 1
        return t.expand(*full_shape)

    prod = align(a) * align(b)
    return Factor(prod.reshape(-1), scope, cards)


def sum_out(f: Factor, var: str) -> Factor:
    if var not in f.scope:
        return f
    axis = f.scope.index(var)
    t = f.shaped().sum(dim=axis)
    scope = list(f.scope)
    scope.pop(axis)
    cards = dict(f.cards)
    cards.pop(var, None)
    return Factor(t.reshape(-1), scope, cards)


def reduce_evidence(f: Factor, evidence: Dict[str, Tensor]) -> Factor:
    if not evidence:
        return f
    t = f.shaped()
    scope = list(f.scope)
    cards = dict(f.cards)
    for v, val in evidence.items():
        if v not in scope:
            continue
        idx = int(val.view(-1)[0].item())
        axis = scope.index(v)
        t = t.select(dim=axis, index=idx)
        scope.pop(axis)
        cards.pop(v, None)
    return Factor(t.reshape(-1), scope, cards)


def delta_factor(var: str, val: int, card: Dict[str, int], device) -> Factor:
    t = torch.zeros(int(card[var]), device=device, dtype=torch.float32)
    t[int(val)] = 1.0
    return Factor(t.view(-1), [var], {var: int(card[var])})


def _as_probs_from_cpd(cpd: object) -> Tensor:
    # Order of preference: table -> probs -> softmax(logits)
    if hasattr(cpd, "table"):
        return getattr(cpd, "table")
    if hasattr(cpd, "probs"):
        return getattr(cpd, "probs")
    if hasattr(cpd, "logits"):
        return torch.softmax(getattr(cpd, "logits"), dim=-1)
    raise AssertionError("Discrete CPD must provide .table or .probs or .logits")


def discrete_cpd_to_table(bn, X: str, card: Dict[str, int], device) -> Factor:
    """Return Factor for P(X | Pa(X)) with scope Pa(X)+[X], table flat.
    Accepts CPDs exposing .table or .probs or .logits.
    """
    parents = bn.parents.get(X, [])
    K = int(card[X])
    raw = _as_probs_from_cpd(bn.cpd[X]).to(device).float()

    if len(parents) == 0:
        # Expect shape [K]
        if raw.dim() == 1 and raw.numel() == K:
            t = raw
        else:
            # tolerate [1,K] etc.
            t = raw.view(-1)
            assert t.numel() == K, f"Root CPD for {X}: expected {K}, got {t.numel()}"
        return Factor(
            table=t.reshape(-1),
            scope=[X],
            cards={X: K},
        )

    # Conditional: expect either flat [∏cards(pa), K] or something reshape-able to that.
    parent_cards = [int(card[p]) for p in parents]
    rows = int(torch.tensor(parent_cards).prod().item()) if parent_cards else 1

    if raw.dim() == 2 and raw.shape[-1] == K:
        # [rows, K] (rows should be ∏cards(pa))
        if raw.shape[0] != rows:
            # attempt auto-reshape if total elements match
            assert raw.numel() == rows * K, (
                f"CPD for {X} mismatched rows: have {raw.shape}, " f"need [{rows},{K}]"
            )
            raw = raw.view(rows, K)
        t2 = raw
    else:
        # try to reshape generically
        assert (
            raw.numel() == rows * K
        ), f"CPD for {X} not reshapeable to [{rows},{K}], got {tuple(raw.shape)}"
        t2 = raw.view(rows, K)

    return Factor(
        table=t2.reshape(-1),
        scope=list(parents) + [X],
        cards={**{p: int(card[p]) for p in parents}, X: K},
    )


class VariableElimination(InferenceBackend):
    """Exact discrete inference (VE) with tolerant CPD adapters."""

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
        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}

        # collect discrete cards (skip non-discrete nodes)
        card = {
            n: int(spec["card"])
            for n, spec in bn.nodes.items()
            if spec.get("type") == "discrete"
        }

        # build factors
        factors: List[Factor] = []
        for X in bn.topo_order:
            if X not in card:
                continue
            if X in do:
                factors.append(
                    delta_factor(X, int(do[X].view(-1)[0].item()), card, device)
                )
                continue
            factors.append(discrete_cpd_to_table(bn, X, card, device))

        # reduce evidence
        if evidence:
            factors = [reduce_evidence(f, evidence) for f in factors]

        # eliminate hidden vars
        keep = set(query) | set(evidence or {}) | set(do or {})
        elim = [v for v in bn.topo_order if v in card and v not in keep]
        for Z in elim:
            bucket = [f for f in factors if Z in f.scope]
            if not bucket:
                continue
            prod = bucket[0]
            for f in bucket[1:]:
                prod = multiply(prod, f)
            newf = sum_out(prod, Z)
            factors = [f for f in factors if Z not in f.scope] + [newf]

        # multiply remaining & normalize
        prod = factors[0]
        for f in factors[1:]:
            prod = multiply(prod, f)
        prod.normalize_()

        out: Dict[str, Tensor] = {}
        if len(query) == 1:
            g = prod
            for v in list(g.scope):
                if v != query[0]:
                    g = sum_out(g, v)
            out[query[0]] = g.table  # categorical
        else:
            out["joint"] = prod.table
        return out
