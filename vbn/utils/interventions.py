from __future__ import annotations

from typing import Iterable, Optional

import torch

from vbn.core.base import Query


def is_intervened(node: str, query: Query) -> bool:
    return node in query.do


def is_observed(node: str, query: Query) -> bool:
    return node in query.evidence


def get_fixed_value(node: str, query: Query) -> Optional[torch.Tensor]:
    if node in query.do:
        return query.do[node]
    if node in query.evidence:
        return query.evidence[node]
    return None


def effective_parents(node: str, query: Query, parents: Iterable[str]) -> list[str]:
    if is_intervened(node, query):
        return []
    return list(parents)
