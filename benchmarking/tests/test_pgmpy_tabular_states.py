from __future__ import annotations

import networkx as nx
import pandas as pd

from benchmarking.models.config import make_component, ModelBenchmarkConfig
from benchmarking.models.pgmpy import PgmpyBenchmarkModel


def _mle_config() -> ModelBenchmarkConfig:
    return ModelBenchmarkConfig(
        model="pgmpy",
        config_id="test_mle",
        learning=make_component("learning", "mle", kwargs={}),
        cpd=make_component("cpd", "tabular_mle", kwargs={}),
        inference=make_component("inference", "exact_variable_elimination", kwargs={}),
    )


def test_pgmpy_tabular_handles_unseen_declared_states() -> None:
    dag = nx.DiGraph()
    dag.add_edge("A", "B")
    domain = {
        "nodes": {
            "A": {"type": "discrete", "states": ["s0", "s1", "s2"]},
            "B": {"type": "discrete", "states": ["t0", "t1", "t2"]},
        }
    }
    # Training data only contains the first two classes for both variables.
    data_df = pd.DataFrame({"A": [0, 1, 0, 1, 1, 0], "B": [0, 1, 1, 1, 0, 1]})

    model = PgmpyBenchmarkModel(
        dag=dag,
        seed=0,
        domain=domain,
        benchmark_config=_mle_config(),
    )
    model.fit(data_df)

    by_code = model.answer_inference_query({"target": "B", "evidence_values": {"A": 2}})
    assert by_code["ok"] is True
    assert by_code["result"]["k"] == 3
    assert len(by_code["result"]["probs"]) == 3

    by_name = model.answer_inference_query(
        {"target": "B", "evidence_values": {"A": "s2"}}
    )
    assert by_name["ok"] is True
    assert by_name["result"]["k"] == 3
    assert len(by_name["result"]["probs"]) == 3
