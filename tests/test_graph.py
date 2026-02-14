import networkx as nx
import pytest
from vbn.core.dags import StaticDAG


def test_topological_order_and_parents():
    g = nx.DiGraph()
    g.add_edges_from([("a", "b"), ("b", "c")])
    dag = StaticDAG(g)
    assert dag.topological_order() == ["a", "b", "c"]
    assert dag.parents("c") == ["b"]


def test_cycle_detection():
    g = nx.DiGraph()
    g.add_edge("a", "a")
    with pytest.raises(ValueError):
        StaticDAG(g)
