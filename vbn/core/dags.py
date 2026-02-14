from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx


class BaseDAG:
    def nodes(self) -> List[str]:
        raise NotImplementedError

    def edges(self) -> List[tuple[str, str]]:
        raise NotImplementedError

    def parents(self, node: str) -> List[str]:
        raise NotImplementedError

    def topological_order(self) -> List[str]:
        raise NotImplementedError


@dataclass
class StaticDAG(BaseDAG):
    graph: nx.DiGraph

    def __post_init__(self) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("DAG must be acyclic")
        self._topo = list(nx.topological_sort(self.graph))
        self._parents: Dict[str, List[str]] = {
            n: list(self.graph.predecessors(n)) for n in self.graph.nodes
        }

    def nodes(self) -> List[str]:
        return list(self.graph.nodes)

    def edges(self) -> List[tuple[str, str]]:
        return list(self.graph.edges)

    def parents(self, node: str) -> List[str]:
        return self._parents.get(node, [])

    def topological_order(self) -> List[str]:
        return list(self._topo)


class TemporalDAG(BaseDAG):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TemporalDAG is a placeholder for future work.")


class DynamicDAG(BaseDAG):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DynamicDAG is a placeholder for future work.")
