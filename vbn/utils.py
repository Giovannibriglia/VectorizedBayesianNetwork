from __future__ import annotations

from typing import Dict, List

import torch

Tensor = torch.Tensor


def to_tensor(x, device=None, dtype=None) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    t = torch.as_tensor(x)
    return t.to(device=device or t.device, dtype=dtype or t.dtype)


def topo_sort(nodes: Dict[str, List[str]]) -> List[str]:
    # nodes: name -> parents
    indeg = {n: len(par) for n, par in nodes.items()}
    out = []
    S = [n for n, d in indeg.items() if d == 0]
    while S:
        n = S.pop()
        out.append(n)
        for c, par in nodes.items():
            if n in par:
                indeg[c] -= 1
                if indeg[c] == 0:
                    S.append(c)
    if len(out) != len(nodes):
        raise ValueError("Graph must be a DAG")
    return out
