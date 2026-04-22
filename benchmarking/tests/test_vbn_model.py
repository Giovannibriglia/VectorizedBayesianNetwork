from __future__ import annotations

import networkx as nx

from benchmarking.models.vbn import _build_nodes_cpds


def test_build_nodes_cpds_infers_discrete_cardinalities() -> None:
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    domain = {
        "nodes": {
            "A": {"type": "discrete", "states": ["a0", "a1", "a2"]},
            "B": {"type": "discrete", "states": ["b0", "b1"]},
        }
    }

    nodes_cpds = _build_nodes_cpds(
        domain=domain,
        cpd_name="categorical_table",
        cpd_kwargs={},
        dag=dag,
    )

    assert nodes_cpds["A"]["n_classes"] == 3
    assert nodes_cpds["B"]["n_classes"] == 2
    assert nodes_cpds["B"]["parent_n_classes"] == [3]
