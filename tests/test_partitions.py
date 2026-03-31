import importlib

import pandas as pd

report = importlib.import_module("benchmarking.scripts.05_report_results")


def test_compute_partitions_basic():
    rows = []
    methods = ["m1", "m2", "m3"]
    query_keys = ["q1", "q2", "q3", "q4", "q5", "q6"]
    solver_map = {
        "q1": {"m1", "m2", "m3"},
        "q2": {"m1", "m2"},
        "q3": {"m1"},
        "q4": {"m2", "m3"},
        "q5": set(),
        "q6": {"m1", "m3"},
    }
    for q in query_keys:
        for m in methods:
            rows.append(
                {
                    "query_key": q,
                    "query_type": "cpd",
                    "method_id": m,
                    "ok": m in solver_map[q],
                }
            )
    attempts_df = pd.DataFrame(rows)

    partition_sets, inventory_df = report.compute_partitions(
        attempts_df, min_partition_queries=1, max_subsets=None
    )

    assert len(partition_sets["all"]) == 6
    assert len(partition_sets["common"]) == 1

    common_row = inventory_df[inventory_df["partition_name"] == "common"].iloc[0]
    assert int(common_row["n_queries"]) == 1
    assert common_row["partition_type"] == "common"

    subset_inv = inventory_df[inventory_df["partition_type"] == "subset"]
    assert set(subset_inv["solver_set"]) == {
        "m1",
        "m1|m2",
        "m1|m3",
        "m2|m3",
    }
    for _, row in subset_inv.iterrows():
        pname = row["partition_name"]
        assert len(partition_sets[pname]) == 1

    non_common_total = 5
    for _, row in subset_inv.iterrows():
        assert abs(float(row["share_of_non_common"]) - 1.0 / non_common_total) < 1e-9
