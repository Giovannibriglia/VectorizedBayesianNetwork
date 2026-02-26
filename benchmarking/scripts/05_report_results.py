from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarking.metrics.divergences import _compute_discrete_metrics
from benchmarking.utils import (
    ensure_dir,
    get_generator_datasets_dir,
    get_project_root,
    parse_bif_structure,
    read_json,
)

CPD_TARGET_CATEGORIES = ["markov_blanket", "parent_set", "random_pac"]
CPD_EVIDENCE_STRATEGIES = ["paths", "markov_blanket", "random"]
INF_TARGET_CATEGORIES = [
    "central_hub",
    "separator_cut",
    "peripheral_terminal",
    "random_pac",
]
INF_TASKS = ["prediction", "diagnosis"]
INF_EVIDENCE_MODES = ["empty", "on_manifold", "off_manifold"]
SUMMARY_KEYS = [
    "n",
    "iqm",
    "iqr_std",
    "q1",
    "median",
    "q3",
    "iqm_pm_iqrstd",
]
RECORD_COLUMNS = [
    "model_name",
    "config_id",
    "config_hash",
    "cpd_name",
    "inference_name",
    "learning_name",
    "problem_id",
    "n_nodes",
    "n_edges",
    "query_type",
    "target",
    "target_category",
    "evidence_strategy",
    "evidence_mode",
    "evidence_size",
    "task",
    "skeleton_id",
    "mb_size",
    "parent_size",
    "run_id",
    "seed",
    "generator",
    "timestamp_utc",
    "kl",
    "wass",
    "jsd",
    "jsd_norm",
    "time",
    "batch_enabled",
    "batch_size",
]

METRIC_LABELS = {
    "kl": "KL",
    "wass": "Wasserstein",
    "jsd": "JSD",
    "jsd_norm": "JSD (norm)",
    "time": "Time",
}
PLOT_METRICS = ["kl", "wass", "jsd_norm"]


class GTComputer:
    def __init__(self, *, run_dir: Path, generator: str) -> None:
        self.run_dir = run_dir
        self.generator = generator
        self.project_root = get_project_root()
        self._cache: dict[str, dict] = {}

    def _load_problem(self, problem_id: str) -> dict | None:
        if problem_id in self._cache:
            return self._cache[problem_id]
        dataset_dir = (
            get_generator_datasets_dir(self.project_root, self.generator) / problem_id
        )
        if not dataset_dir.exists():
            return None
        bif_path = dataset_dir / "model.bif"
        if not bif_path.exists():
            bif_gz = dataset_dir / "model.bif.gz"
            if bif_gz.exists():
                bif_path = bif_gz
            else:
                return None
        domain_path = (
            self.project_root
            / "benchmarking"
            / "data"
            / "metadata"
            / self.generator
            / problem_id
            / "domain.json"
        )
        if not domain_path.exists():
            return None
        domain = read_json(domain_path)
        try:
            from pgmpy.inference import VariableElimination
            from pgmpy.readwrite import BIFReader
        except Exception:
            return None
        try:
            model = BIFReader(str(bif_path)).get_model()
            infer = VariableElimination(model)
        except Exception:
            return None

        code_to_state: dict[str, dict[int, str]] = {}
        state_to_code: dict[str, dict[str, int]] = {}
        nodes = domain.get("nodes", {}) if isinstance(domain, dict) else {}
        for node, meta in nodes.items():
            codes = meta.get("codes") or {}
            if not isinstance(codes, dict):
                continue
            inv = {int(v): str(k) for k, v in codes.items() if v is not None}
            code_to_state[node] = inv
            state_to_code[node] = {
                str(k): int(v) for k, v in codes.items() if v is not None
            }

        payload = {
            "infer": infer,
            "code_to_state": code_to_state,
            "state_to_code": state_to_code,
        }
        self._cache[problem_id] = payload
        return payload

    def compute_probs(
        self, *, problem_id: str, target: str, evidence_vals: dict
    ) -> list[float] | None:
        payload = self._load_problem(problem_id)
        if payload is None:
            return None
        infer = payload["infer"]
        code_to_state = payload["code_to_state"]
        state_to_code = payload["state_to_code"]
        if target not in state_to_code:
            return None

        evidence = {}
        for var, value in evidence_vals.items():
            if var not in code_to_state:
                return None
            try:
                code = int(value)
            except Exception:
                return None
            state_label = code_to_state[var].get(code)
            if state_label is None:
                return None
            evidence[var] = state_label

        try:
            query = infer.query([target], evidence=evidence, show_progress=False)
        except Exception:
            return None
        if target not in query.state_names:
            return None
        state_names = list(query.state_names[target])
        values = np.asarray(query.values, dtype=float).reshape(-1)
        label_to_code = state_to_code[target]
        max_code = max(label_to_code.values()) if label_to_code else -1
        if max_code < 0:
            return None
        probs = [0.0] * (max_code + 1)
        for state_label, prob in zip(state_names, values):
            if state_label not in label_to_code:
                continue
            probs[label_to_code[state_label]] = float(prob)
        return probs


def _read_jsonl(path: Path, max_records: int | None = None) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
            if max_records is not None and len(records) >= int(max_records):
                break
    return records


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _is_int_like(value: Any) -> bool:
    if isinstance(value, (int, bool, np.integer)):
        return True
    if isinstance(value, float) and math.isfinite(value):
        return value.is_integer()
    return False


def _get_by_path(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def _extract_evidence_values(query: dict) -> dict:
    if not isinstance(query, dict):
        return {}
    values = query.get("evidence_values")
    if values is None:
        evidence = query.get("evidence")
        if isinstance(evidence, dict):
            values = evidence.get("values")
    if values is None:
        return {}
    if not isinstance(values, dict):
        return {}
    return {k: v for k, v in values.items() if v is not None}


def _join_key(record: dict) -> tuple:
    query = record.get("query") if isinstance(record.get("query"), dict) else record
    if not isinstance(query, dict):
        query = {}
    query_id = query.get("id")
    if query_id:
        return ("id", query_id)
    problem = record.get("problem", {}).get("id")
    qtype = query.get("type") or query.get("query_type")
    index = query.get("index")
    target = query.get("target")
    target_category = query.get("target_category")
    task = query.get("task")
    evidence = query.get("evidence") if isinstance(query.get("evidence"), dict) else {}
    ev_vars = evidence.get("vars") or query.get("evidence_vars") or []
    if not isinstance(ev_vars, list):
        ev_vars = list(ev_vars) if ev_vars is not None else []
    ev_vars = tuple(ev_vars)
    ev_mode = evidence.get("mode") or query.get("evidence_mode")
    n_inst = evidence.get("n_instantiations")
    mc_id = evidence.get("mc_id") or query.get("mc_id")
    skeleton_id = evidence.get("skeleton_id") or query.get("skeleton_id")
    return (
        "fields",
        problem,
        qtype,
        index,
        target,
        target_category,
        task,
        ev_mode,
        n_inst,
        mc_id,
        skeleton_id,
        ev_vars,
    )


def robust_summary(values: list[float]) -> dict:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(cleaned)
    if n == 0:
        return {
            "n": 0,
            "iqm": None,
            "iqr_std": None,
            "q1": None,
            "median": None,
            "q3": None,
            "iqm_pm_iqrstd": None,
        }
    sorted_vals = sorted(cleaned)
    lo = int(n * 0.25)
    hi = int(n * 0.75)
    trimmed = sorted_vals[lo:hi] if hi > lo else sorted_vals
    iqm = float(np.mean(trimmed)) if trimmed else float(np.mean(sorted_vals))
    q1, median, q3 = np.percentile(sorted_vals, [25, 50, 75])
    iqr_std = float((q3 - q1) / 1.349)
    return {
        "n": n,
        "iqm": iqm,
        "iqr_std": iqr_std,
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqm_pm_iqrstd": f"{iqm:.4f} ± {iqr_std:.4f}",
    }


def aggregate_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    rows = []
    if df.empty:
        columns: list[str] = list(group_cols)
        for metric in metric_cols:
            for key in SUMMARY_KEYS:
                columns.append(f"{metric}_{key}")
        return pd.DataFrame(columns=columns)
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in metric_cols:
            summary = robust_summary(group[metric].tolist())
            for key, value in summary.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    return result


def aggregate_time_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        columns: list[str] = list(group_cols)
        for key in SUMMARY_KEYS:
            columns.append(f"time_{key}")
        columns.extend(["time_sum_ms", "time_sum_s"])
        return pd.DataFrame(columns=columns)
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        times = [
            float(v)
            for v in group["time"].tolist()
            if v is not None and math.isfinite(float(v))
        ]
        summary = robust_summary(times)
        for key, value in summary.items():
            row[f"time_{key}"] = value
        row["time_sum_ms"] = float(sum(times)) if times else 0.0
        row["time_sum_s"] = float(row["time_sum_ms"] / 1000.0)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_batching_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = list(group_cols) + [
        "n_queries",
        "fraction_batched",
        "avg_batch_size",
        "time_sum_ms",
        "time_per_query_ms",
        "throughput_qps",
    ]
    if df.empty or "batch_enabled" not in df.columns:
        return pd.DataFrame(columns=columns)
    if df["batch_enabled"].dropna().empty:
        return pd.DataFrame(columns=columns)
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        n_queries = int(len(group))
        batch_flags = group["batch_enabled"].fillna(False).astype(bool)
        n_batched = int(batch_flags.sum())
        row["n_queries"] = n_queries
        row["fraction_batched"] = (
            float(n_batched / n_queries) if n_queries > 0 else None
        )
        batch_sizes = [
            float(v)
            for v in group["batch_size"].tolist()
            if v is not None and math.isfinite(float(v))
        ]
        row["avg_batch_size"] = float(np.mean(batch_sizes)) if batch_sizes else None
        times = [
            float(v)
            for v in group["time"].tolist()
            if v is not None and math.isfinite(float(v))
        ]
        time_sum_ms = float(sum(times)) if times else 0.0
        row["time_sum_ms"] = time_sum_ms
        row["time_per_query_ms"] = (
            float(time_sum_ms / n_queries) if n_queries > 0 else None
        )
        row["throughput_qps"] = (
            float(n_queries / (time_sum_ms / 1000.0)) if time_sum_ms > 0 else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.upper())


def _is_discrete_output(output: dict | None) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get("format") != "categorical_probs":
        return False
    support = output.get("support")
    if isinstance(support, list) and support:
        return all(_is_int_like(v) for v in support)
    return True


def _extract_pred_probs(record: dict) -> tuple[list[float] | None, dict | None]:
    output = record.get("result", {}).get("output")
    if not isinstance(output, dict):
        return None, None
    probs = output.get("probs")
    if not isinstance(probs, list) or not probs:
        return None, output
    return probs, output


def _extract_gt_probs(record: dict, gt_key: str) -> list[float] | None:
    value = _get_by_path(record, gt_key)
    if isinstance(value, list):
        return value
    return None


def _extract_gt_from_line(line: dict, gt_key: str) -> list[float] | None:
    if isinstance(line.get("gt_probs"), list):
        return line.get("gt_probs")
    value = _get_by_path(line, gt_key)
    if isinstance(value, list):
        return value
    output = (
        line.get("result", {}).get("output")
        if isinstance(line.get("result"), dict)
        else None
    )
    if isinstance(output, dict) and isinstance(output.get("probs"), list):
        return output.get("probs")
    return None


def _load_ground_truth_folder(
    run_dir: Path, gt_key: str, max_records: int | None
) -> dict:
    gt_map: dict[tuple, list[float]] = {}
    gt_dir = run_dir / "ground_truth"
    if gt_dir.exists():
        for name in ("cpds.jsonl", "inference.jsonl"):
            path = gt_dir / name
            for record in _read_jsonl(path, max_records=max_records):
                probs = _extract_gt_from_line(record, gt_key)
                if probs is None:
                    continue
                key = _join_key(record)
                gt_map[key] = probs
    sources_path = run_dir / "ground_truth_sources.json"
    if sources_path.exists():
        try:
            sources = read_json(sources_path)
        except Exception:
            sources = {}
        if isinstance(sources, dict):
            for meta in sources.values():
                if isinstance(meta, dict):
                    path_value = meta.get("path")
                else:
                    path_value = meta
                if not path_value:
                    continue
                gt_path = Path(path_value)
                if not gt_path.is_absolute():
                    gt_path = get_project_root() / gt_path
                for record in _read_jsonl(gt_path, max_records=max_records):
                    probs = _extract_gt_from_line(record, gt_key)
                    if probs is None:
                        continue
                    key = _join_key(record)
                    gt_map[key] = probs
    return gt_map


def _compute_graph_stats(run_dir: Path) -> dict[str, dict[str, int]]:
    generator = run_dir.parent.name
    project_root = get_project_root()
    datasets_dir = get_generator_datasets_dir(project_root, generator)
    stats: dict[str, dict[str, int]] = {}
    if not datasets_dir.exists():
        return stats
    for dataset_dir in sorted(datasets_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        problem_id = dataset_dir.name
        bif_path = dataset_dir / "model.bif"
        if not bif_path.exists():
            bif_gz = dataset_dir / "model.bif.gz"
            if bif_gz.exists():
                bif_path = bif_gz
            else:
                continue
        try:
            _, parents_map = parse_bif_structure(bif_path)
        except Exception:
            continue
        children_map: dict[str, set[str]] = {node: set() for node in parents_map}
        for child, parents in parents_map.items():
            for parent in parents:
                children_map.setdefault(parent, set()).add(child)
        mb_sizes: dict[str, int] = {}
        parent_sizes: dict[str, int] = {}
        for node, parents in parents_map.items():
            parent_set = set(parents)
            child_set = children_map.get(node, set())
            spouses = set()
            for child in child_set:
                spouses.update(set(parents_map.get(child, [])))
            spouses.discard(node)
            mb = parent_set | child_set | spouses
            mb_sizes[node] = len(mb)
            parent_sizes[node] = len(parent_set)
        n_nodes = len(parents_map)
        n_edges = sum(len(parents) for parents in parents_map.values())
        stats[problem_id] = {
            "mb_sizes": mb_sizes,
            "parent_sizes": parent_sizes,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }
    return stats


def _build_records(
    *,
    run_dir: Path,
    gt_source: str,
    gt_key: str,
    max_records: int | None,
    eps: float,
    model_filter: set[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    gt_map: dict[tuple, list[float]] = {}
    if gt_source == "folder":
        gt_map = _load_ground_truth_folder(run_dir, gt_key, max_records)

    graph_stats = _compute_graph_stats(run_dir)
    generator = run_dir.parent.name
    gt_computer = (
        GTComputer(run_dir=run_dir, generator=generator)
        if gt_source == "compute"
        else None
    )

    rows: list[dict] = []
    for subdir in ("cpds", "inference"):
        base_dir = run_dir / subdir
        if not base_dir.exists():
            continue
        for path in sorted(base_dir.glob("*.jsonl")):
            for record in _read_jsonl(path, max_records=max_records):
                model_meta = (
                    record.get("model") if isinstance(record.get("model"), dict) else {}
                )
                model_name = model_meta.get("name") or "unknown"
                config_id = model_meta.get("config_id")
                config_hash = model_meta.get("config_hash")
                components = (
                    model_meta.get("components")
                    if isinstance(model_meta.get("components"), dict)
                    else {}
                )
                cpd_meta = (
                    components.get("cpd")
                    if isinstance(components.get("cpd"), dict)
                    else {}
                )
                inference_meta = (
                    components.get("inference")
                    if isinstance(components.get("inference"), dict)
                    else {}
                )
                learning_meta = (
                    components.get("learning")
                    if isinstance(components.get("learning"), dict)
                    else {}
                )
                cpd_name = cpd_meta.get("name")
                inference_name = inference_meta.get("name")
                learning_name = learning_meta.get("name")
                if model_filter:
                    key = f"{model_name}/{config_id or config_hash or 'unknown'}"
                    if (
                        model_name not in model_filter
                        and (config_id or "") not in model_filter
                        and key not in model_filter
                    ):
                        continue

                query = (
                    record.get("query") if isinstance(record.get("query"), dict) else {}
                )
                pred_probs, pred_output = _extract_pred_probs(record)
                if pred_probs is None:
                    continue
                gt_probs: list[float] | None = None
                if gt_source == "embedded":
                    gt_probs = _extract_gt_probs(record, gt_key)
                elif gt_source == "folder":
                    gt_probs = gt_map.get(_join_key(record))
                elif gt_source == "compute":
                    problem_id = record.get("problem", {}).get("id")
                    target = query.get("target")
                    evidence_vals = _extract_evidence_values(query)
                    if problem_id and target and evidence_vals:
                        gt_probs = gt_computer.compute_probs(
                            problem_id=problem_id,
                            target=target,
                            evidence_vals=evidence_vals,
                        )
                if gt_probs is None:
                    continue

                if len(pred_probs) != len(gt_probs):
                    continue

                try:
                    kl, wass, jsd, jsd_norm = _compute_discrete_metrics(
                        gt_probs,
                        pred_probs,
                        eps,
                        compute_jsd=_is_discrete_output(pred_output),
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    continue

                time_ms = None
                result = (
                    record.get("result")
                    if isinstance(record.get("result"), dict)
                    else {}
                )
                timing = result.get("timing_ms")
                if timing is not None:
                    try:
                        time_ms = float(timing)
                    except Exception:
                        time_ms = None

                problem_id = record.get("problem", {}).get("id")
                query_type = query.get("type")
                target = query.get("target")
                target_category = query.get("target_category")
                evidence_strategy = query.get("evidence_strategy")
                evidence = (
                    query.get("evidence")
                    if isinstance(query.get("evidence"), dict)
                    else {}
                )
                if evidence_strategy is None and isinstance(evidence, dict):
                    evidence_strategy = evidence.get("strategy") or evidence.get(
                        "evidence_strategy"
                    )
                evidence_mode = evidence.get("mode") or query.get("evidence_mode")
                ev_vars = evidence.get("vars") or query.get("evidence_vars") or []
                if not isinstance(ev_vars, list):
                    ev_vars = list(ev_vars) if ev_vars is not None else []
                evidence_size = len(ev_vars)
                task = query.get("task")
                skeleton_id = evidence.get("skeleton_id") or query.get("skeleton_id")
                batching = (
                    record.get("batching")
                    if isinstance(record.get("batching"), dict)
                    else {}
                )
                batch_enabled = None
                batch_size = None
                if isinstance(batching, dict) and batching:
                    if "enabled" in batching:
                        batch_enabled = bool(batching.get("enabled"))
                    batch_size = _coerce_int(batching.get("batch_size"))

                problem_meta = (
                    record.get("problem")
                    if isinstance(record.get("problem"), dict)
                    else {}
                )
                n_nodes = _coerce_int(problem_meta.get("n_nodes"))
                n_edges = _coerce_int(problem_meta.get("n_edges"))
                if problem_id and (n_nodes is None or n_edges is None):
                    node_stats = graph_stats.get(problem_id)
                    if node_stats:
                        if n_nodes is None:
                            n_nodes = node_stats.get("n_nodes")
                        if n_edges is None:
                            n_edges = node_stats.get("n_edges")

                mb_size = None
                parent_size = None
                if problem_id and target:
                    node_stats = graph_stats.get(problem_id)
                    if node_stats:
                        mb_size = node_stats.get("mb_sizes", {}).get(target)
                        parent_size = node_stats.get("parent_sizes", {}).get(target)

                run_meta = (
                    record.get("run") if isinstance(record.get("run"), dict) else {}
                )
                run_id = run_meta.get("run_id")
                run_seed = _coerce_int(run_meta.get("seed"))
                run_generator = run_meta.get("generator")
                run_timestamp = run_meta.get("timestamp_utc")

                rows.append(
                    {
                        "model_name": model_name,
                        "config_id": config_id,
                        "config_hash": config_hash,
                        "cpd_name": cpd_name,
                        "inference_name": inference_name,
                        "learning_name": learning_name,
                        "problem_id": problem_id,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "query_type": query_type,
                        "target": target,
                        "target_category": target_category,
                        "evidence_strategy": evidence_strategy,
                        "evidence_mode": evidence_mode,
                        "evidence_size": evidence_size,
                        "task": task,
                        "skeleton_id": skeleton_id,
                        "mb_size": mb_size,
                        "parent_size": parent_size,
                        "run_id": run_id,
                        "seed": run_seed,
                        "generator": run_generator,
                        "timestamp_utc": run_timestamp,
                        "kl": kl,
                        "wass": wass,
                        "jsd": jsd,
                        "jsd_norm": jsd_norm,
                        "time": time_ms,
                        "batch_enabled": batch_enabled,
                        "batch_size": batch_size,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[*RECORD_COLUMNS, "config_key", "method_id"])
        return df, errors
    df["config_key"] = df["config_id"].fillna(
        df["config_hash"].fillna("unknown").astype(str).str[:8]
    )
    df["method_id"] = df["model_name"].astype(str) + "/" + df["config_key"].astype(str)
    return df, errors


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        if list(df.columns):
            df.to_csv(path, index=False)
        else:
            path.write_text("")
        return
    df = df.sort_values(list(df.columns))
    df.to_csv(path, index=False)


def _two_stage_aggregate(
    df: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        stage2_group = [
            "method_id",
            "model_name",
            "config_id",
            "config_hash",
            *(extra_group_cols or []),
            x_col,
        ]
        columns: list[str] = list(stage2_group)
        for metric in metric_cols:
            for key in SUMMARY_KEYS:
                columns.append(f"{metric}_{key}")
        columns.append("dataset_n")
        return pd.DataFrame(columns=columns)
    stage1_rows = []
    extra_group_cols = list(extra_group_cols or [])
    group_cols = [
        "method_id",
        "model_name",
        "config_id",
        "config_hash",
        *extra_group_cols,
        "problem_id",
        x_col,
    ]
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in metric_cols:
            summary = robust_summary(group[metric].tolist())
            row[metric] = summary["iqm"]
        stage1_rows.append(row)
    stage1 = pd.DataFrame(stage1_rows)
    if stage1.empty:
        stage2_group = [
            "method_id",
            "model_name",
            "config_id",
            "config_hash",
            *extra_group_cols,
            x_col,
        ]
        columns: list[str] = list(stage2_group)
        for metric in metric_cols:
            for key in SUMMARY_KEYS:
                columns.append(f"{metric}_{key}")
        columns.append("dataset_n")
        return pd.DataFrame(columns=columns)
    stage2_group = [
        "method_id",
        "model_name",
        "config_id",
        "config_hash",
        *extra_group_cols,
        x_col,
    ]
    rows = []
    for keys, group in stage1.groupby(stage2_group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(stage2_group, keys)}
        for metric in metric_cols:
            summary = robust_summary(group[metric].tolist())
            for key, value in summary.items():
                row[f"{metric}_{key}"] = value
        row["dataset_n"] = int(group["problem_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_error_vs_size(
    df: pd.DataFrame,
    *,
    size_col: str,
    metric: str,
    out_dir: Path,
    title_prefix: str,
    filename_prefix: str,
) -> Path | None:
    if df.empty:
        return None
    plt.figure(figsize=(8, 4.5))
    method_ids = sorted(df["method_id"].dropna().unique())
    for method_id in method_ids:
        group = df[df["method_id"] == method_id]
        group = group[group[size_col].notna()]
        group = group[group[f"{metric}_iqm"].notna()]
        if group.empty:
            continue
        group = group.sort_values(size_col)
        x = group[size_col].astype(int).tolist()
        y = group[f"{metric}_iqm"].tolist()
        yerr = group[f"{metric}_iqr_std"].tolist()
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=method_id)
    plt.title(title_prefix)
    plt.xlabel(size_col)
    plt.ylabel(_metric_label(metric))
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / f"{filename_prefix}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _plot_error_vs_evidence_size(
    df: pd.DataFrame,
    *,
    metric: str,
    out_dir: Path,
    filename_prefix: str,
    mode: str | None = None,
) -> Path | None:
    if df.empty:
        return None
    data = df
    if mode is not None:
        data = data[data["evidence_mode"] == mode]
    data = data[data["evidence_size"].notna()]
    data = data[data[f"{metric}_iqm"].notna()]
    if data.empty:
        return None
    plt.figure(figsize=(8, 4.5))
    method_ids = sorted(data["method_id"].dropna().unique())
    for method_id in method_ids:
        sub = data[data["method_id"] == method_id].sort_values("evidence_size")
        if sub.empty:
            continue
        x = sub["evidence_size"].astype(int).tolist()
        y = sub[f"{metric}_iqm"].tolist()
        yerr = sub[f"{metric}_iqr_std"].tolist()
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=method_id)
    title = f"Inference {_metric_label(metric)} vs Evidence Size"
    if mode is not None:
        title = f"{title} ({mode})"
    plt.title(title)
    plt.xlabel("evidence_size")
    plt.ylabel(_metric_label(metric))
    plt.legend()
    plt.grid(True, alpha=0.3)
    suffix = f"__mode_{mode}" if mode else ""
    out_path = out_dir / f"{filename_prefix}{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _plot_category_bars(
    df: pd.DataFrame,
    *,
    category_col: str,
    metric: str,
    out_dir: Path,
    filename_prefix: str,
    title_prefix: str,
    category_order: list[str],
) -> Path | None:
    if df.empty:
        return None
    data = df[df[category_col].notna()]
    data = data[data[f"{metric}_iqm"].notna()]
    if data.empty:
        return None
    ordered = [c for c in category_order if c in set(data[category_col])]
    ordered += [
        c for c in sorted(data[category_col].dropna().unique()) if c not in ordered
    ]
    method_ids = sorted(data["method_id"].dropna().unique())
    if not method_ids:
        return None
    x = np.arange(len(ordered))
    width = 0.8 / max(1, len(method_ids))
    plt.figure(figsize=(10, 4.5))
    for idx, method_id in enumerate(method_ids):
        sub = data[data["method_id"] == method_id]
        sub = sub[sub[category_col].isin(ordered)].copy()
        if sub.empty:
            continue
        sub[category_col] = pd.Categorical(
            sub[category_col], categories=ordered, ordered=True
        )
        sub = sub.sort_values(category_col)
        y_map = dict(zip(sub[category_col], sub[f"{metric}_iqm"]))
        err_map = dict(zip(sub[category_col], sub[f"{metric}_iqr_std"]))
        y = [y_map.get(cat, np.nan) for cat in ordered]
        yerr = [err_map.get(cat, np.nan) for cat in ordered]
        offset = (idx - (len(method_ids) - 1) / 2) * width
        plt.bar(x + offset, y, width=width, yerr=yerr, capsize=3, label=method_id)
    plt.xticks(x, ordered, rotation=30, ha="right")
    plt.title(title_prefix)
    plt.ylabel(_metric_label(metric))
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    out_path = out_dir / f"{filename_prefix}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _pareto_frontier(points: list[tuple[float, float]]) -> list[bool]:
    is_pareto = []
    for i, (x_i, y_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j) in enumerate(points):
            if i == j:
                continue
            if x_j <= x_i and y_j <= y_i and (x_j < x_i or y_j < y_i):
                dominated = True
                break
        is_pareto.append(not dominated)
    return is_pareto


def _plot_pareto(
    df: pd.DataFrame,
    *,
    metric: str,
    out_dir: Path,
    filename: str,
    title: str,
) -> Path | None:
    if df.empty:
        return None
    sub = df.copy()
    sub = sub[sub[f"{metric}_iqm"].notna() & sub["time_iqm"].notna()]
    if sub.empty:
        return None
    points = list(
        zip(sub["time_iqm"].astype(float), sub[f"{metric}_iqm"].astype(float))
    )
    pareto_mask = _pareto_frontier(points)
    plt.figure(figsize=(7.5, 5))
    pareto_label_added = False
    other_label_added = False
    for (time_val, err_val), is_pareto, method_id, xerr, yerr in zip(
        points,
        pareto_mask,
        sub["method_id"].tolist(),
        sub["time_iqr_std"].tolist(),
        sub[f"{metric}_iqr_std"].tolist(),
    ):
        if is_pareto:
            plt.errorbar(
                time_val,
                err_val,
                xerr=xerr,
                yerr=yerr,
                fmt="o",
                color="C1",
                capsize=3,
                label="Pareto" if not pareto_label_added else None,
            )
            plt.annotate(
                method_id,
                (time_val, err_val),
                textcoords="offset points",
                xytext=(6, 6),
            )
            pareto_label_added = True
        else:
            plt.errorbar(
                time_val,
                err_val,
                xerr=xerr,
                yerr=yerr,
                fmt="x",
                color="0.6",
                capsize=2,
                label="Other" if not other_label_added else None,
            )
            other_label_added = True
    pareto_points = [pt for pt, keep in zip(points, pareto_mask) if keep]
    if pareto_points:
        pareto_points = sorted(pareto_points, key=lambda t: t[0])
        plt.plot(
            [p[0] for p in pareto_points],
            [p[1] for p in pareto_points],
            color="C1",
            alpha=0.5,
        )
    plt.xlabel("IQM time (ms)")
    plt.ylabel(f"IQM {_metric_label(metric)}")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = out_dir / filename
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _write_report_md(
    out_dir: Path, table_paths: list[Path], figure_paths: list[Path]
) -> None:
    lines = ["# Benchmark Report", "", "## Tables", ""]
    for path in table_paths:
        rel = path.relative_to(out_dir)
        lines.append(f"- {rel}")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        rel = path.relative_to(out_dir)
        lines.append(f"- {rel}")
    (out_dir / "report.md").write_text("\n".join(lines))


def _detect_run_mode(run_dir: Path) -> str | None:
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            payload = read_json(meta_path)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            mode = payload.get("mode")
            if mode in {"cpds", "inference"}:
                return mode
    name = run_dir.name
    if name.startswith("benchmark_cpds_"):
        return "cpds"
    if name.startswith("benchmark_inference_"):
        return "inference"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument(
        "--gt_source",
        type=str,
        default="folder",
        choices=["embedded", "folder", "compute"],
    )
    parser.add_argument(
        "--gt_key",
        type=str,
        default="result.ground_truth.output.probs",
    )
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--include_time",
        action="store_true",
        default=True,
        help="Include time tables and plots (default: true)",
    )
    parser.add_argument(
        "--no-include_time",
        dest="include_time",
        action="store_false",
        help="Disable time tables and plots",
    )
    parser.add_argument(
        "--include_pareto",
        action="store_true",
        default=True,
        help="Include Pareto plots (default: true)",
    )
    parser.add_argument(
        "--no-include_pareto",
        dest="include_pareto",
        action="store_false",
        help="Disable Pareto plots",
    )
    parser.add_argument(
        "--pareto_split",
        type=str,
        default="none",
        choices=["none", "mode", "task", "target_category"],
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "report"
    tables_dir = ensure_dir(out_dir / "tables")
    figures_dir = ensure_dir(out_dir / "figures")

    model_filter = None
    if args.models:
        model_filter = {m.strip() for m in args.models.split(",") if m.strip()}

    run_mode = _detect_run_mode(run_dir)
    if run_mode:
        logging.info("Detected run mode: %s", run_mode)

    df, errors = _build_records(
        run_dir=run_dir,
        gt_source=args.gt_source,
        gt_key=args.gt_key,
        max_records=args.max_records,
        eps=float(args.eps),
        model_filter=model_filter,
    )

    if df.empty:
        logging.warning("No records to report. Check GT source and inputs.")

    metric_cols = ["kl", "wass", "jsd", "jsd_norm"]

    tables: list[Path] = []
    figures: list[Path] = []

    flat = df.copy()
    if "problem_id" in flat.columns:
        flat["network_name"] = flat["problem_id"]
    else:
        flat["network_name"] = None
    flat_cols = [
        "network_name",
        "problem_id",
        "n_nodes",
        "n_edges",
        "model_name",
        "config_id",
        "config_hash",
        "method_id",
        "cpd_name",
        "inference_name",
        "learning_name",
        "query_type",
        "run_id",
        "seed",
        "generator",
        "timestamp_utc",
        "kl",
        "wass",
        "jsd",
        "jsd_norm",
        "time",
        "batch_enabled",
        "batch_size",
        "target",
        "target_category",
        "evidence_strategy",
        "evidence_mode",
        "evidence_size",
        "task",
        "skeleton_id",
        "mb_size",
        "parent_size",
    ]
    flat = flat[[col for col in flat_cols if col in flat.columns]]
    path = tables_dir / "flat_results.csv"
    _write_table(flat, path)
    tables.append(path)

    overall = aggregate_table(
        df,
        ["method_id", "model_name", "config_id", "config_hash", "query_type"],
        metric_cols,
    )
    path = tables_dir / "overall_by_model.csv"
    _write_table(overall, path)
    tables.append(path)

    cpd = df[df["query_type"] == "cpd"]
    inference = df[df["query_type"] == "inference"]

    cpd_by_target = aggregate_table(
        cpd,
        ["method_id", "model_name", "config_id", "config_hash", "target_category"],
        metric_cols,
    )
    path = tables_dir / "cpd_by_target_category.csv"
    _write_table(cpd_by_target, path)
    tables.append(path)

    cpd_by_strategy = aggregate_table(
        cpd,
        ["method_id", "model_name", "config_id", "config_hash", "evidence_strategy"],
        metric_cols,
    )
    path = tables_dir / "cpd_by_evidence_strategy.csv"
    _write_table(cpd_by_strategy, path)
    tables.append(path)

    inf_by_target = aggregate_table(
        inference,
        ["method_id", "model_name", "config_id", "config_hash", "target_category"],
        metric_cols,
    )
    path = tables_dir / "inference_by_target_category.csv"
    _write_table(inf_by_target, path)
    tables.append(path)

    inf_by_task = aggregate_table(
        inference,
        ["method_id", "model_name", "config_id", "config_hash", "task"],
        metric_cols,
    )
    path = tables_dir / "inference_by_task.csv"
    _write_table(inf_by_task, path)
    tables.append(path)

    inf_by_mode = aggregate_table(
        inference,
        ["method_id", "model_name", "config_id", "config_hash", "evidence_mode"],
        metric_cols,
    )
    path = tables_dir / "inference_by_evidence_mode.csv"
    _write_table(inf_by_mode, path)
    tables.append(path)

    batching_metrics = aggregate_batching_table(
        inference,
        ["method_id", "model_name", "config_id", "config_hash"],
    )
    if not batching_metrics.empty:
        path = tables_dir / "inference_batching_metrics.csv"
        _write_table(batching_metrics, path)
        tables.append(path)

    cpd_mb = _two_stage_aggregate(cpd, "mb_size", metric_cols)
    path = tables_dir / "cpd_by_mb_size.csv"
    _write_table(cpd_mb, path)
    tables.append(path)

    cpd_parent = _two_stage_aggregate(cpd, "parent_size", metric_cols)
    path = tables_dir / "cpd_by_parent_size.csv"
    _write_table(cpd_parent, path)
    tables.append(path)

    cpd_nodes = _two_stage_aggregate(cpd, "n_nodes", metric_cols)
    path = tables_dir / "cpd_by_n_nodes.csv"
    _write_table(cpd_nodes, path)
    tables.append(path)

    cpd_edges = _two_stage_aggregate(cpd, "n_edges", metric_cols)
    path = tables_dir / "cpd_by_n_edges.csv"
    _write_table(cpd_edges, path)
    tables.append(path)

    inf_ev_size = _two_stage_aggregate(inference, "evidence_size", metric_cols)
    path = tables_dir / "inference_by_evidence_size.csv"
    _write_table(inf_ev_size, path)
    tables.append(path)

    inf_nodes = _two_stage_aggregate(inference, "n_nodes", metric_cols)
    path = tables_dir / "inference_by_n_nodes.csv"
    _write_table(inf_nodes, path)
    tables.append(path)

    inf_edges = _two_stage_aggregate(inference, "n_edges", metric_cols)
    path = tables_dir / "inference_by_n_edges.csv"
    _write_table(inf_edges, path)
    tables.append(path)

    inf_ev_size_mode = _two_stage_aggregate(
        inference, "evidence_size", metric_cols, extra_group_cols=["evidence_mode"]
    )

    time_tables: dict[str, pd.DataFrame] = {}
    if args.include_time:
        overall_time = aggregate_time_table(
            df, ["method_id", "model_name", "config_id", "config_hash", "query_type"]
        )
        path = tables_dir / "overall_time_by_method.csv"
        _write_table(overall_time, path)
        tables.append(path)
        time_tables["overall"] = overall_time

        cpd_time_by_target = aggregate_time_table(
            cpd,
            ["method_id", "model_name", "config_id", "config_hash", "target_category"],
        )
        path = tables_dir / "cpd_time_by_target_category.csv"
        _write_table(cpd_time_by_target, path)
        tables.append(path)
        time_tables["cpd_by_target"] = cpd_time_by_target

        cpd_time_by_strategy = aggregate_time_table(
            cpd,
            [
                "method_id",
                "model_name",
                "config_id",
                "config_hash",
                "evidence_strategy",
            ],
        )
        path = tables_dir / "cpd_time_by_evidence_strategy.csv"
        _write_table(cpd_time_by_strategy, path)
        tables.append(path)
        time_tables["cpd_by_strategy"] = cpd_time_by_strategy

        cpd_time_mb = _two_stage_aggregate(cpd, "mb_size", ["time"])
        path = tables_dir / "cpd_time_by_mb_size.csv"
        _write_table(cpd_time_mb, path)
        tables.append(path)
        time_tables["cpd_mb"] = cpd_time_mb

        cpd_time_parent = _two_stage_aggregate(cpd, "parent_size", ["time"])
        path = tables_dir / "cpd_time_by_parent_size.csv"
        _write_table(cpd_time_parent, path)
        tables.append(path)
        time_tables["cpd_parent"] = cpd_time_parent

        cpd_time_nodes = _two_stage_aggregate(cpd, "n_nodes", ["time"])
        path = tables_dir / "cpd_time_by_n_nodes.csv"
        _write_table(cpd_time_nodes, path)
        tables.append(path)
        time_tables["cpd_nodes"] = cpd_time_nodes

        cpd_time_edges = _two_stage_aggregate(cpd, "n_edges", ["time"])
        path = tables_dir / "cpd_time_by_n_edges.csv"
        _write_table(cpd_time_edges, path)
        tables.append(path)
        time_tables["cpd_edges"] = cpd_time_edges

        cpd_time_ev_size = _two_stage_aggregate(cpd, "evidence_size", ["time"])
        path = tables_dir / "cpd_time_by_evidence_size.csv"
        _write_table(cpd_time_ev_size, path)
        tables.append(path)
        time_tables["cpd_ev_size"] = cpd_time_ev_size

        inf_time_by_target = aggregate_time_table(
            inference,
            ["method_id", "model_name", "config_id", "config_hash", "target_category"],
        )
        path = tables_dir / "inference_time_by_target_category.csv"
        _write_table(inf_time_by_target, path)
        tables.append(path)
        time_tables["inf_by_target"] = inf_time_by_target

        inf_time_by_task = aggregate_time_table(
            inference,
            ["method_id", "model_name", "config_id", "config_hash", "task"],
        )
        path = tables_dir / "inference_time_by_task.csv"
        _write_table(inf_time_by_task, path)
        tables.append(path)
        time_tables["inf_by_task"] = inf_time_by_task

        inf_time_by_mode = aggregate_time_table(
            inference,
            ["method_id", "model_name", "config_id", "config_hash", "evidence_mode"],
        )
        path = tables_dir / "inference_time_by_evidence_mode.csv"
        _write_table(inf_time_by_mode, path)
        tables.append(path)
        time_tables["inf_by_mode"] = inf_time_by_mode

        inf_time_ev_size = _two_stage_aggregate(inference, "evidence_size", ["time"])
        path = tables_dir / "inference_time_by_evidence_size.csv"
        _write_table(inf_time_ev_size, path)
        tables.append(path)
        time_tables["inf_ev_size"] = inf_time_ev_size

        inf_time_nodes = _two_stage_aggregate(inference, "n_nodes", ["time"])
        path = tables_dir / "inference_time_by_n_nodes.csv"
        _write_table(inf_time_nodes, path)
        tables.append(path)
        time_tables["inf_nodes"] = inf_time_nodes

        inf_time_edges = _two_stage_aggregate(inference, "n_edges", ["time"])
        path = tables_dir / "inference_time_by_n_edges.csv"
        _write_table(inf_time_edges, path)
        tables.append(path)
        time_tables["inf_edges"] = inf_time_edges

    skeleton = aggregate_table(
        inference[inference["skeleton_id"].notna()],
        [
            "method_id",
            "model_name",
            "config_id",
            "config_hash",
            "problem_id",
            "skeleton_id",
        ],
        metric_cols,
    )
    path = tables_dir / "inference_by_skeleton.csv"
    _write_table(skeleton, path)
    tables.append(path)

    # Plots
    cpd_size_specs = [
        (cpd_mb, "mb_size", "Markov Blanket Size", "mb_size"),
        (cpd_parent, "parent_size", "Parent Set Size", "parent_size"),
        (cpd_nodes, "n_nodes", "#Nodes", "n_nodes"),
        (cpd_edges, "n_edges", "#Edges", "n_edges"),
    ]
    for data, size_col, size_label, tag in cpd_size_specs:
        for metric in PLOT_METRICS:
            fig = _plot_error_vs_size(
                data,
                size_col=size_col,
                metric=metric,
                out_dir=figures_dir,
                title_prefix=f"CPD {_metric_label(metric)} vs {size_label}",
                filename_prefix=f"cpd_{metric}_vs_{tag}",
            )
            if fig:
                figures.append(fig)

    inf_size_specs = [
        (inf_nodes, "n_nodes", "#Nodes", "n_nodes"),
        (inf_edges, "n_edges", "#Edges", "n_edges"),
    ]
    for data, size_col, size_label, tag in inf_size_specs:
        for metric in PLOT_METRICS:
            fig = _plot_error_vs_size(
                data,
                size_col=size_col,
                metric=metric,
                out_dir=figures_dir,
                title_prefix=f"Inference {_metric_label(metric)} vs {size_label}",
                filename_prefix=f"inference_{metric}_vs_{tag}",
            )
            if fig:
                figures.append(fig)

    modes = [
        m for m in INF_EVIDENCE_MODES if m in set(inf_ev_size_mode["evidence_mode"])
    ]
    modes += [
        m
        for m in sorted(inf_ev_size_mode["evidence_mode"].dropna().unique())
        if m not in modes
    ]
    for mode in modes:
        for metric in PLOT_METRICS:
            fig = _plot_error_vs_evidence_size(
                inf_ev_size_mode,
                metric=metric,
                out_dir=figures_dir,
                filename_prefix=f"inference_{metric}_vs_evidence_size",
                mode=mode,
            )
            if fig:
                figures.append(fig)

    if args.include_pareto:
        pareto_metrics = ["kl", "wass", "jsd_norm"]
        pareto_summary = aggregate_table(
            df,
            ["method_id", "model_name", "config_id", "config_hash", "query_type"],
            [*pareto_metrics, "time"],
        )
        for metric in pareto_metrics:
            sub = pareto_summary[pareto_summary["query_type"] == "cpd"]
            fig = _plot_pareto(
                sub,
                metric=metric,
                out_dir=figures_dir,
                filename=f"pareto_cpd_{metric}_vs_time.png",
                title=f"CPD {_metric_label(metric)} vs Time",
            )
            if fig:
                figures.append(fig)
            sub = pareto_summary[pareto_summary["query_type"] == "inference"]
            fig = _plot_pareto(
                sub,
                metric=metric,
                out_dir=figures_dir,
                filename=f"pareto_inference_{metric}_vs_time.png",
                title=f"Inference {_metric_label(metric)} vs Time",
            )
            if fig:
                figures.append(fig)

        if args.pareto_split != "none":
            split_col = {
                "mode": "evidence_mode",
                "task": "task",
                "target_category": "target_category",
            }[args.pareto_split]
            split_df = df[df[split_col].notna()]
            if not split_df.empty:
                split_summary = aggregate_table(
                    split_df,
                    [
                        "method_id",
                        "model_name",
                        "config_id",
                        "config_hash",
                        "query_type",
                        split_col,
                    ],
                    [*pareto_metrics, "time"],
                )
                for (qtype, split_val), group in split_summary.groupby(
                    ["query_type", split_col], dropna=False
                ):
                    tag = _safe_tag(f"{split_col}_{split_val}")
                    for metric in pareto_metrics:
                        fig = _plot_pareto(
                            group,
                            metric=metric,
                            out_dir=figures_dir,
                            filename=f"pareto_{qtype}_{metric}_vs_time__{tag}.png",
                            title=f"{qtype.capitalize()} {_metric_label(metric)} vs Time ({split_col}={split_val})",
                        )
                        if fig:
                            figures.append(fig)
    if args.include_time:
        fig = _plot_error_vs_size(
            time_tables.get("cpd_mb", pd.DataFrame()),
            size_col="mb_size",
            metric="time",
            out_dir=figures_dir,
            title_prefix="CPD Time vs Markov Blanket Size",
            filename_prefix="cpd_time_vs_mb_size",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("cpd_parent", pd.DataFrame()),
            size_col="parent_size",
            metric="time",
            out_dir=figures_dir,
            title_prefix="CPD Time vs Parent Set Size",
            filename_prefix="cpd_time_vs_parent_size",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("cpd_nodes", pd.DataFrame()),
            size_col="n_nodes",
            metric="time",
            out_dir=figures_dir,
            title_prefix="CPD Time vs #Nodes",
            filename_prefix="cpd_time_vs_n_nodes",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("cpd_edges", pd.DataFrame()),
            size_col="n_edges",
            metric="time",
            out_dir=figures_dir,
            title_prefix="CPD Time vs #Edges",
            filename_prefix="cpd_time_vs_n_edges",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("cpd_ev_size", pd.DataFrame()),
            size_col="evidence_size",
            metric="time",
            out_dir=figures_dir,
            title_prefix="CPD Time vs Evidence Size",
            filename_prefix="cpd_time_vs_evidence_size",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("inf_ev_size", pd.DataFrame()),
            size_col="evidence_size",
            metric="time",
            out_dir=figures_dir,
            title_prefix="Inference Time vs Evidence Size",
            filename_prefix="inference_time_vs_evidence_size",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("inf_nodes", pd.DataFrame()),
            size_col="n_nodes",
            metric="time",
            out_dir=figures_dir,
            title_prefix="Inference Time vs #Nodes",
            filename_prefix="inference_time_vs_n_nodes",
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_size(
            time_tables.get("inf_edges", pd.DataFrame()),
            size_col="n_edges",
            metric="time",
            out_dir=figures_dir,
            title_prefix="Inference Time vs #Edges",
            filename_prefix="inference_time_vs_n_edges",
        )
        if fig:
            figures.append(fig)

    category_specs = [
        (
            cpd_by_target,
            "target_category",
            "CPD",
            "cpd",
            "target_category",
            "Target Category",
            CPD_TARGET_CATEGORIES,
        ),
        (
            inf_by_target,
            "target_category",
            "Inference",
            "inference",
            "target_category",
            "Target Category",
            INF_TARGET_CATEGORIES,
        ),
        (
            inf_by_task,
            "task",
            "Inference",
            "inference",
            "task",
            "Task",
            INF_TASKS,
        ),
        (
            inf_by_mode,
            "evidence_mode",
            "Inference",
            "inference",
            "evidence_mode",
            "Evidence Mode",
            INF_EVIDENCE_MODES,
        ),
    ]
    for (
        data,
        category_col,
        title_prefix,
        file_prefix,
        file_suffix,
        title_suffix,
        category_order,
    ) in category_specs:
        for metric in PLOT_METRICS:
            fig = _plot_category_bars(
                data,
                category_col=category_col,
                metric=metric,
                out_dir=figures_dir,
                filename_prefix=f"{file_prefix}_{metric}_by_{file_suffix}",
                title_prefix=f"{title_prefix} {_metric_label(metric)} by {title_suffix}",
                category_order=category_order,
            )
            if fig:
                figures.append(fig)

    if args.include_time:
        fig = _plot_category_bars(
            time_tables.get("cpd_by_target", pd.DataFrame()),
            category_col="target_category",
            metric="time",
            out_dir=figures_dir,
            filename_prefix="cpd_time_by_target_category",
            title_prefix="CPD Time by Target Category",
            category_order=CPD_TARGET_CATEGORIES,
        )
        if fig:
            figures.append(fig)
        fig = _plot_category_bars(
            time_tables.get("cpd_by_strategy", pd.DataFrame()),
            category_col="evidence_strategy",
            metric="time",
            out_dir=figures_dir,
            filename_prefix="cpd_time_by_evidence_strategy",
            title_prefix="CPD Time by Evidence Strategy",
            category_order=CPD_EVIDENCE_STRATEGIES,
        )
        if fig:
            figures.append(fig)
        fig = _plot_category_bars(
            time_tables.get("inf_by_target", pd.DataFrame()),
            category_col="target_category",
            metric="time",
            out_dir=figures_dir,
            filename_prefix="inference_time_by_target_category",
            title_prefix="Inference Time by Target Category",
            category_order=INF_TARGET_CATEGORIES,
        )
        if fig:
            figures.append(fig)
        fig = _plot_category_bars(
            time_tables.get("inf_by_task", pd.DataFrame()),
            category_col="task",
            metric="time",
            out_dir=figures_dir,
            filename_prefix="inference_time_by_task",
            title_prefix="Inference Time by Task",
            category_order=INF_TASKS,
        )
        if fig:
            figures.append(fig)
        fig = _plot_category_bars(
            time_tables.get("inf_by_mode", pd.DataFrame()),
            category_col="evidence_mode",
            metric="time",
            out_dir=figures_dir,
            filename_prefix="inference_time_by_evidence_mode",
            title_prefix="Inference Time by Evidence Mode",
            category_order=INF_EVIDENCE_MODES,
        )
        if fig:
            figures.append(fig)

    if not figures:
        figures = sorted(figures_dir.glob("*.png"))
    _write_report_md(out_dir, tables, figures)

    if errors:
        logging.warning("Encountered %s metric errors", len(errors))


if __name__ == "__main__":
    main()
