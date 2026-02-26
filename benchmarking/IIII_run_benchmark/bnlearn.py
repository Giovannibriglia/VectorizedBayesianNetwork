from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import networkx as nx

from benchmarking.utils import (
    get_dataset_domain_metadata_path,
    get_dataset_queries_dir,
    get_generator_datasets_dir,
    parse_bif_structure,
    read_dataframe,
    read_json,
    read_jsonl,
    select_data_file,
)
from .base import BaseBenchmarkRunner, ProblemAssets, ProblemLoadResult
from .registry import register_benchmark_runner


def _build_dag(nodes: List[str], parents_map: dict) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for child, parents in parents_map.items():
        for parent in parents:
            dag.add_edge(parent, child)
    return dag


@register_benchmark_runner
class BNLearnBenchmarkRunner(BaseBenchmarkRunner):
    generator = "bnlearn"

    def list_problem_dirs(self) -> list[Path]:
        datasets_dir = get_generator_datasets_dir(self.root, self.generator)
        if not datasets_dir.exists():
            return []
        return sorted([p for p in datasets_dir.iterdir() if p.is_dir()])

    def load_problem_assets(self, dataset_dir: Path) -> ProblemLoadResult:
        problem = dataset_dir.name
        logger = logging.getLogger(__name__)

        bif_path = dataset_dir / "model.bif"
        if not bif_path.exists():
            bif_gz = dataset_dir / "model.bif.gz"
            if bif_gz.exists():
                bif_path = bif_gz
            else:
                return ProblemLoadResult(
                    problem=problem,
                    assets=None,
                    skipped=True,
                    reason="Missing model.bif",
                )

        domain_path = get_dataset_domain_metadata_path(
            self.root, self.generator, problem
        )
        if not domain_path.exists():
            return ProblemLoadResult(
                problem=problem,
                assets=None,
                skipped=True,
                reason="Missing domain.json",
            )

        queries_dir = get_dataset_queries_dir(self.root, self.generator, problem)
        meta_path = queries_dir / "queries.json"
        cpd_path = queries_dir / "cpds.jsonl"
        inf_path = queries_dir / "inference.jsonl"
        queries_meta: dict = {}
        if meta_path.exists():
            try:
                queries_meta = read_json(meta_path)
            except Exception as exc:
                return ProblemLoadResult(
                    problem=problem,
                    assets=None,
                    skipped=True,
                    reason=f"Failed to read queries metadata: {type(exc).__name__}: {exc}",
                )

        mode = getattr(self, "mode", "cpds")
        if mode == "cpds":
            have_queries = cpd_path.exists() or (
                isinstance(queries_meta, dict) and "cpd_queries" in queries_meta
            )
            if cpd_path.exists():
                cpd_queries = read_jsonl(cpd_path)
            else:
                cpd_queries = (
                    queries_meta.get("cpd_queries", []) if queries_meta else []
                )
            inf_queries = []
            queries_path = cpd_path if cpd_path.exists() else meta_path
            if not have_queries:
                return ProblemLoadResult(
                    problem=problem,
                    assets=None,
                    skipped=True,
                    reason="Missing cpds.jsonl",
                )
        else:
            have_queries = inf_path.exists() or (
                isinstance(queries_meta, dict) and "inference_queries" in queries_meta
            )
            if inf_path.exists():
                inf_queries = read_jsonl(inf_path)
            else:
                inf_queries = (
                    queries_meta.get("inference_queries", []) if queries_meta else []
                )
            cpd_queries = []
            queries_path = inf_path if inf_path.exists() else meta_path
            if not have_queries:
                return ProblemLoadResult(
                    problem=problem,
                    assets=None,
                    skipped=True,
                    reason="Missing inference.jsonl",
                )

        data_path = select_data_file(dataset_dir, self.seed, logger)
        if data_path is None:
            return ProblemLoadResult(
                problem=problem,
                assets=None,
                skipped=True,
                reason="Missing generated data file",
            )

        try:
            data_df = read_dataframe(data_path)
        except Exception as exc:
            return ProblemLoadResult(
                problem=problem,
                assets=None,
                skipped=True,
                reason=f"Failed to load data: {type(exc).__name__}: {exc}",
            )

        try:
            domain = read_json(domain_path)
        except Exception as exc:
            return ProblemLoadResult(
                problem=problem,
                assets=None,
                skipped=True,
                reason=f"Failed to read domain.json: {type(exc).__name__}: {exc}",
            )

        queries: dict = {
            "cpd_queries": cpd_queries,
            "inference_queries": inf_queries,
        }
        if isinstance(queries_meta, dict) and "ground_truth" in queries_meta:
            queries["ground_truth"] = queries_meta.get("ground_truth")

        try:
            nodes, parents_map = parse_bif_structure(bif_path)
            dag = _build_dag(nodes, parents_map)
        except Exception as exc:
            return ProblemLoadResult(
                problem=problem,
                assets=None,
                skipped=True,
                reason=f"Failed to parse BIF: {type(exc).__name__}: {exc}",
            )

        assets = ProblemAssets(
            problem=problem,
            dag=dag,
            domain=domain,
            data_df=data_df,
            data_path=data_path,
            queries=queries,
            queries_path=queries_path,
            dataset_dir=dataset_dir,
        )
        return ProblemLoadResult(
            problem=problem,
            assets=assets,
            skipped=False,
            reason=None,
        )
