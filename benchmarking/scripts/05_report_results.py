from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarking.utils import (
    ensure_dir,
    get_generator_datasets_dir,
    get_project_root,
    parse_bif_structure,
    read_json,
)

try:  # optional
    from scipy.stats import wasserstein_distance as _scipy_wasserstein
except Exception:  # pragma: no cover
    _scipy_wasserstein = None


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


def _normalize_probs(probs: Iterable[float], eps: float) -> np.ndarray:
    arr = np.asarray(list(probs), dtype=float)
    arr = np.clip(arr, eps, 1.0)
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0:
        return np.full_like(arr, 1.0 / len(arr))
    return arr / total


def kl_divergence(p: Iterable[float], q: Iterable[float], eps: float) -> float:
    p_arr = _normalize_probs(p, eps)
    q_arr = _normalize_probs(q, eps)
    return float(np.sum(p_arr * np.log(p_arr / q_arr)))


def wasserstein_distance(p: Iterable[float], q: Iterable[float], eps: float) -> float:
    p_arr = _normalize_probs(p, eps)
    q_arr = _normalize_probs(q, eps)
    k = len(p_arr)
    xs = np.arange(k, dtype=float)
    if _scipy_wasserstein is not None:
        return float(_scipy_wasserstein(xs, xs, p_arr, q_arr))
    cdf_p = np.cumsum(p_arr)
    cdf_q = np.cumsum(q_arr)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


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
        return pd.DataFrame()
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


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _extract_pred_probs(record: dict) -> list[float] | None:
    output = record.get("result", {}).get("output")
    if not isinstance(output, dict):
        return None
    if output.get("format") != "categorical_probs":
        return None
    probs = output.get("probs")
    if not isinstance(probs, list) or not probs:
        return None
    return probs


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
        stats[problem_id] = {
            "mb_sizes": mb_sizes,
            "parent_sizes": parent_sizes,
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
                pred_probs = _extract_pred_probs(record)
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
                    kl = kl_divergence(gt_probs, pred_probs, eps)
                    wass = wasserstein_distance(gt_probs, pred_probs, eps)
                except Exception as exc:
                    errors.append(str(exc))
                    continue

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

                mb_size = None
                parent_size = None
                if problem_id and target:
                    node_stats = graph_stats.get(problem_id)
                    if node_stats:
                        mb_size = node_stats.get("mb_sizes", {}).get(target)
                        parent_size = node_stats.get("parent_sizes", {}).get(target)

                rows.append(
                    {
                        "model_name": model_name,
                        "config_id": config_id,
                        "config_hash": config_hash,
                        "problem_id": problem_id,
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
                        "kl": kl,
                        "wass": wass,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, errors
    df["config_key"] = df["config_id"].fillna(
        df["config_hash"].fillna("unknown").astype(str).str[:8]
    )
    df["method_id"] = df["model_name"].astype(str) + "/" + df["config_key"].astype(str)
    return df, errors


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text("")
        return
    df = df.sort_values(list(df.columns))
    df.to_csv(path, index=False)


def _two_stage_aggregate(
    df: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    stage1_rows = []
    group_cols = [
        "method_id",
        "model_name",
        "config_id",
        "config_hash",
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
        return pd.DataFrame()
    stage2_group = ["method_id", "model_name", "config_id", "config_hash", x_col]
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
        if group.empty:
            continue
        group = group.sort_values(size_col)
        x = group[size_col].astype(int).tolist()
        y = group[f"{metric}_iqm"].tolist()
        yerr = group[f"{metric}_iqr_std"].tolist()
        plt.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=method_id)
    plt.title(title_prefix)
    plt.xlabel(size_col)
    plt.ylabel(metric.upper())
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
    title = f"Inference {metric.upper()} vs Evidence Size"
    if mode is not None:
        title = f"{title} ({mode})"
    plt.title(title)
    plt.xlabel("evidence_size")
    plt.ylabel(metric.upper())
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
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    out_path = out_dir / f"{filename_prefix}.png"
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
        return

    metric_cols = ["kl", "wass"]

    tables: list[Path] = []
    figures: list[Path] = []

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

    cpd_mb = _two_stage_aggregate(cpd, "mb_size", metric_cols)
    path = tables_dir / "cpd_by_mb_size.csv"
    _write_table(cpd_mb, path)
    tables.append(path)

    cpd_parent = _two_stage_aggregate(cpd, "parent_size", metric_cols)
    path = tables_dir / "cpd_by_parent_size.csv"
    _write_table(cpd_parent, path)
    tables.append(path)

    inf_ev_size = aggregate_table(
        inference,
        ["method_id", "model_name", "config_id", "config_hash", "evidence_size"],
        metric_cols,
    )
    path = tables_dir / "inference_by_evidence_size.csv"
    _write_table(inf_ev_size, path)
    tables.append(path)

    inf_ev_size_mode = aggregate_table(
        inference,
        [
            "method_id",
            "model_name",
            "config_id",
            "config_hash",
            "evidence_mode",
            "evidence_size",
        ],
        metric_cols,
    )

    if not inference.empty:
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
    fig = _plot_error_vs_size(
        cpd_mb,
        size_col="mb_size",
        metric="kl",
        out_dir=figures_dir,
        title_prefix="CPD KL vs Markov Blanket Size",
        filename_prefix="cpd_kl_vs_mb_size",
    )
    if fig:
        figures.append(fig)
    fig = _plot_error_vs_size(
        cpd_mb,
        size_col="mb_size",
        metric="wass",
        out_dir=figures_dir,
        title_prefix="CPD Wasserstein vs Markov Blanket Size",
        filename_prefix="cpd_wass_vs_mb_size",
    )
    if fig:
        figures.append(fig)
    fig = _plot_error_vs_size(
        cpd_parent,
        size_col="parent_size",
        metric="kl",
        out_dir=figures_dir,
        title_prefix="CPD KL vs Parent Set Size",
        filename_prefix="cpd_kl_vs_parent_size",
    )
    if fig:
        figures.append(fig)
    fig = _plot_error_vs_size(
        cpd_parent,
        size_col="parent_size",
        metric="wass",
        out_dir=figures_dir,
        title_prefix="CPD Wasserstein vs Parent Set Size",
        filename_prefix="cpd_wass_vs_parent_size",
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
        fig = _plot_error_vs_evidence_size(
            inf_ev_size_mode,
            metric="kl",
            out_dir=figures_dir,
            filename_prefix="inference_kl_vs_evidence_size",
            mode=mode,
        )
        if fig:
            figures.append(fig)
        fig = _plot_error_vs_evidence_size(
            inf_ev_size_mode,
            metric="wass",
            out_dir=figures_dir,
            filename_prefix="inference_wass_vs_evidence_size",
            mode=mode,
        )
        if fig:
            figures.append(fig)

    fig = _plot_category_bars(
        cpd_by_target,
        category_col="target_category",
        metric="kl",
        out_dir=figures_dir,
        filename_prefix="cpd_kl_by_target_category",
        title_prefix="CPD KL by Target Category",
        category_order=CPD_TARGET_CATEGORIES,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        cpd_by_target,
        category_col="target_category",
        metric="wass",
        out_dir=figures_dir,
        filename_prefix="cpd_wass_by_target_category",
        title_prefix="CPD Wasserstein by Target Category",
        category_order=CPD_TARGET_CATEGORIES,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        inf_by_target,
        category_col="target_category",
        metric="kl",
        out_dir=figures_dir,
        filename_prefix="inference_kl_by_target_category",
        title_prefix="Inference KL by Target Category",
        category_order=INF_TARGET_CATEGORIES,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        inf_by_target,
        category_col="target_category",
        metric="wass",
        out_dir=figures_dir,
        filename_prefix="inference_wass_by_target_category",
        title_prefix="Inference Wasserstein by Target Category",
        category_order=INF_TARGET_CATEGORIES,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        inf_by_task,
        category_col="task",
        metric="kl",
        out_dir=figures_dir,
        filename_prefix="inference_kl_by_task",
        title_prefix="Inference KL by Task",
        category_order=INF_TASKS,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        inf_by_task,
        category_col="task",
        metric="wass",
        out_dir=figures_dir,
        filename_prefix="inference_wass_by_task",
        title_prefix="Inference Wasserstein by Task",
        category_order=INF_TASKS,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        inf_by_mode,
        category_col="evidence_mode",
        metric="kl",
        out_dir=figures_dir,
        filename_prefix="inference_kl_by_evidence_mode",
        title_prefix="Inference KL by Evidence Mode",
        category_order=INF_EVIDENCE_MODES,
    )
    if fig:
        figures.append(fig)
    fig = _plot_category_bars(
        inf_by_mode,
        category_col="evidence_mode",
        metric="wass",
        out_dir=figures_dir,
        filename_prefix="inference_wass_by_evidence_mode",
        title_prefix="Inference Wasserstein by Evidence Mode",
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
