from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - best-effort fallback

    def tqdm(iterable, **kwargs):  # type: ignore[no-redef]
        return iterable


from benchmarking.metrics.divergences import (
    _compute_discrete_metrics,
    jensen_shannon_divergence_samples,
    jensen_shannon_divergence_samples_normalized,
    wasserstein_distance_samples,
)
from benchmarking.utils import (
    ensure_dir,
    get_generator_datasets_dir,
    get_project_root,
    read_json,
)
from benchmarking.utils_errors import classify_error
from benchmarking.utils_logging import setup_logging

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


@dataclass(frozen=True)
class SummaryStyle:
    name: str
    keys: tuple[str, ...]
    center_key: str
    spread_key: str
    pm_key: str | None
    center_label: str
    spread_label: str


SUMMARY_STYLES: dict[str, SummaryStyle] = {
    "robust": SummaryStyle(
        name="robust",
        keys=(
            "n",
            "iqm",
            "min",
            "max",
            "iqr",
            "q1",
            "median",
            "q3",
            "iqm_pm_iqr",
            "iqr_std",
            "iqm_pm_iqrstd",
            "iqm_low_iqrstd_clipped",
            "iqm_high_iqrstd_clipped",
            "iqm_range_iqrstd_clipped",
        ),
        center_key="iqm",
        spread_key="iqr",
        pm_key="iqm_pm_iqr",
        center_label="IQM",
        spread_label="IQR",
    ),
    "mean": SummaryStyle(
        name="mean",
        keys=("n", "mean", "std", "mean_pm_std"),
        center_key="mean",
        spread_key="std",
        pm_key="mean_pm_std",
        center_label="Mean",
        spread_label="Std",
    ),
}
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
    "evidence_vars",
    "task",
    "target_set",
    "skeleton_id",
    "mb_size",
    "parent_size",
    "mc_id",
    "query_id",
    "query_index",
    "run_id",
    "seed",
    "generator",
    "timestamp_utc",
    "kl",
    "wass",
    "jsd",
    "jsd_norm",
    "time",
    "mean_abs_err",
    "std_abs_err",
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
NON_NEGATIVE_METRICS = {
    "kl",
    "wass",
    "jsd",
    "jsd_norm",
    "time",
    "mean_abs_err",
    "std_abs_err",
}


class GTComputer:
    def __init__(
        self, *, run_dir: Path, generator: str, bundle_dir: Path | None = None
    ) -> None:
        self.run_dir = run_dir
        self.generator = generator
        self.bundle_dir = bundle_dir
        self.project_root = get_project_root()
        self._cache: dict[str, dict] = {}

    def _load_problem(self, problem_id: str) -> dict | None:
        if problem_id in self._cache:
            return self._cache[problem_id]
        if self.bundle_dir is not None:
            dataset_dir = (
                Path(self.bundle_dir) / "datasets" / self.generator / problem_id
            )
        else:
            dataset_dir = (
                get_generator_datasets_dir(self.project_root, self.generator)
                / problem_id
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
        domain_path = dataset_dir / "domain.json"
        if not domain_path.exists():
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


def _format_number(value: Any) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NaN"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.4f}"
    return str(value)


def _df_to_md_table(df: pd.DataFrame) -> list[str]:
    if df.empty or not list(df.columns):
        return []
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = [_format_number(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_md_table(df: pd.DataFrame, path: Path) -> None:
    lines = _df_to_md_table(df)
    if not lines:
        path.write_text("")
        return
    path.write_text("\n".join(lines) + "\n")


def _format_bin_edge(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _compute_bin_edges(values: pd.Series, n_bins: int = 4) -> list[float] | None:
    cleaned = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if cleaned.empty:
        return None
    uniq = np.unique(cleaned)
    if len(uniq) < 2:
        return None
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(cleaned, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return None
    return list(edges)


def _apply_bins(
    df: pd.DataFrame, *, col: str, n_bins: int = 4
) -> tuple[pd.DataFrame, list[str] | None]:
    edges = _compute_bin_edges(df[col], n_bins=n_bins)
    bin_col = f"{col}_bin"
    if not edges or len(edges) < 2:
        df[bin_col] = np.nan
        return df, None
    labels: list[str] = []
    for idx in range(len(edges) - 1):
        left = _format_bin_edge(edges[idx])
        right = _format_bin_edge(edges[idx + 1])
        left_bracket = "[" if idx == 0 else "("
        labels.append(f"{left_bracket}{left},{right}]")
    df[bin_col] = pd.cut(
        df[col],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    return df, labels


def _aggregate_success(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = list(group_cols) + ["n_total", "n_ok", "n_error", "success_rate"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        ok_vals = group["ok"].dropna()
        if ok_vals.empty:
            row.update({"n_total": 0, "n_ok": 0, "n_error": 0, "success_rate": np.nan})
            rows.append(row)
            continue
        ok_bool = ok_vals.astype(bool)
        n_total = int(len(ok_bool))
        n_ok = int(ok_bool.sum())
        n_error = int(n_total - n_ok)
        success_rate = float(n_ok / n_total) if n_total > 0 else np.nan
        row.update(
            {
                "n_total": n_total,
                "n_ok": n_ok,
                "n_error": n_error,
                "success_rate": success_rate,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_query_key(row: pd.Series | dict) -> tuple | None:
    payload = row if isinstance(row, dict) else row.to_dict()
    query_type = payload.get("query_type") or payload.get("type")
    problem_id = payload.get("problem_id") or payload.get("problem")
    if not query_type or not problem_id:
        return None
    skeleton_id = payload.get("skeleton_id")
    target = payload.get("target")
    task = payload.get("task") if query_type == "inference" else None
    if skeleton_id:
        return ("skeleton", query_type, problem_id, skeleton_id, target, task)
    query_id = payload.get("query_id") or payload.get("id")
    if query_id:
        return ("id", query_type, problem_id, query_id)
    evidence_vars = payload.get("evidence_vars")
    if isinstance(evidence_vars, list):
        evidence_vars = tuple(evidence_vars)
    target_set = payload.get("target_set")
    if isinstance(target_set, list):
        target_set = tuple(target_set)
    return (
        "fields",
        query_type,
        problem_id,
        target,
        payload.get("target_category"),
        task,
        payload.get("evidence_mode"),
        payload.get("evidence_strategy"),
        payload.get("evidence_size"),
        payload.get("mc_id"),
        payload.get("query_index"),
        evidence_vars,
        target_set,
    )


def _solver_set_str(solver_set: frozenset[str]) -> str:
    if not solver_set:
        return ""
    return "|".join(sorted(solver_set))


def _slugify_method_id(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "method"


def _ensemble_slug(methods: list[str], used: set[str] | None = None) -> str:
    slugs = [_slugify_method_id(method) for method in methods]
    base = "__".join(slugs)
    slug = base
    if len(slug) > 120:
        hash8 = hashlib.md5("|".join(methods).encode("utf-8")).hexdigest()[:8]
        head = "__".join(slugs[:2]) if slugs else "ensemble"
        slug = f"{head}__{hash8}"
    if used is None:
        return slug
    candidate = slug
    if candidate in used:
        hash8 = hashlib.md5("|".join(methods).encode("utf-8")).hexdigest()[:8]
        if not candidate.endswith(hash8):
            candidate = f"{candidate}__{hash8}"
    used.add(candidate)
    return candidate


def compute_partitions(
    attempts_df: pd.DataFrame,
    *,
    min_partition_queries: int,
    max_subsets: int | None,
) -> tuple[dict[str, set], pd.DataFrame]:
    partition_sets: dict[str, set] = {"all": set(), "common": set()}
    subset_df = attempts_df[attempts_df["query_key"].notna()].copy()
    methods = sorted(subset_df["method_id"].dropna().unique())
    if subset_df.empty or not methods:
        inventory_rows = [
            {
                "partition_name": "all",
                "partition_type": "all",
                "solver_set": "",
                "n_queries": 0,
                "share_of_total": 0.0,
                "share_of_non_common": np.nan,
            },
            {
                "partition_name": "common",
                "partition_type": "common",
                "solver_set": "",
                "n_queries": 0,
                "share_of_total": 0.0,
                "share_of_non_common": np.nan,
            },
        ]
        return partition_sets, pd.DataFrame(inventory_rows)

    ok_series = subset_df[["query_key", "method_id", "ok"]].copy()
    ok_series["ok"] = ok_series["ok"].fillna(False).astype(bool)
    ok_matrix = (
        ok_series.groupby(["query_key", "method_id"])["ok"]
        .max()
        .unstack(fill_value=False)
    )
    for method in methods:
        if method not in ok_matrix.columns:
            ok_matrix[method] = False
    ok_matrix = ok_matrix[methods]

    solver_sets: dict[tuple, frozenset[str]] = {}
    for query_key, row in ok_matrix.iterrows():
        solved = frozenset(method for method in methods if bool(row.get(method, False)))
        solver_sets[query_key] = solved

    all_keys = set(ok_matrix.index)
    common_set = frozenset(methods)
    common_keys = {
        key for key, solver_set in solver_sets.items() if solver_set == common_set
    }
    non_common_keys = all_keys - common_keys
    total_queries = len(all_keys)
    non_common_total = len(non_common_keys)

    partition_sets["all"] = all_keys
    partition_sets["common"] = common_keys

    counts: dict[frozenset[str], int] = {}
    for query_key in non_common_keys:
        solver_set = solver_sets.get(query_key, frozenset())
        if not solver_set:
            continue
        counts[solver_set] = counts.get(solver_set, 0) + 1

    subset_items = [
        (solver_set, count)
        for solver_set, count in counts.items()
        if count >= int(min_partition_queries)
    ]
    subset_items.sort(key=lambda item: (-item[1], _solver_set_str(item[0])))
    if max_subsets is not None:
        subset_items = subset_items[: int(max_subsets)]

    used_slugs: set[str] = set()
    inventory_rows = [
        {
            "partition_name": "all",
            "partition_type": "all",
            "solver_set": "",
            "n_queries": len(all_keys),
            "share_of_total": (
                float(len(all_keys) / total_queries) if total_queries else 0.0
            ),
            "share_of_non_common": np.nan,
        },
        {
            "partition_name": "common",
            "partition_type": "common",
            "solver_set": "",
            "n_queries": len(common_keys),
            "share_of_total": (
                float(len(common_keys) / total_queries) if total_queries else 0.0
            ),
            "share_of_non_common": np.nan,
        },
    ]

    for solver_set, count in subset_items:
        label = _solver_set_str(solver_set)
        ensemble_slug = _ensemble_slug(sorted(solver_set), used_slugs)
        name = f"subset_{ensemble_slug}"
        keys = {key for key in non_common_keys if solver_sets.get(key) == solver_set}
        partition_sets[name] = keys
        share_total = float(count / total_queries) if total_queries else 0.0
        share_non_common = float(count / non_common_total) if non_common_total else 0.0
        inventory_rows.append(
            {
                "partition_name": name,
                "partition_type": "subset",
                "solver_set": label,
                "n_queries": int(len(keys)),
                "share_of_total": share_total,
                "share_of_non_common": share_non_common,
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    return partition_sets, inventory_df


def _format_method_list(methods: list[str]) -> str:
    if not methods:
        return ""
    return "|".join(methods)


def _write_partition_inventory(
    out_dir: Path,
    *,
    mode_label: str,
    inventory_df: pd.DataFrame,
    methods: list[str],
) -> Path:
    columns = [
        "partition_name",
        "partition_type",
        "solver_set",
        "n_queries",
        "share_of_total",
        "share_of_non_common",
    ]
    inv = inventory_df.copy()
    if inv.empty:
        inv = pd.DataFrame(columns=columns)
    inv = inv[[col for col in columns if col in inv.columns]]

    path = out_dir / "partition_inventory.csv"
    _write_table(inv, path)

    lines = [f"# Partition inventory ({mode_label})", ""]
    lines.append("## Methods")
    lines.append("")
    lines.append(f"- n_methods: {len(methods)}")
    lines.append(f"- methods_considered: {', '.join(methods) if methods else ''}")
    lines.append("")
    lines.append("## Partitions")
    lines.append("")
    table_lines = _df_to_md_table(inv)
    if table_lines:
        lines.extend(table_lines)
    else:
        lines.append("(no partitions)")
    path.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n")
    return path


def _write_report_index(
    out_dir: Path,
    *,
    run_dir: Path,
    mode_label: str,
    methods: list[str],
    inventory_df: pd.DataFrame,
    problem_id: str | None = None,
) -> Path:
    total_queries = 0
    common_queries = 0
    if not inventory_df.empty and "partition_name" in inventory_df.columns:
        all_row = inventory_df[inventory_df["partition_name"] == "all"]
        if not all_row.empty:
            try:
                total_queries = int(all_row.iloc[0]["n_queries"])
            except Exception:
                total_queries = 0
        common_row = inventory_df[inventory_df["partition_name"] == "common"]
        if not common_row.empty:
            try:
                common_queries = int(common_row.iloc[0]["n_queries"])
            except Exception:
                common_queries = 0
    non_common_queries = max(total_queries - common_queries, 0)

    lines = [f"# Report: {run_dir.name}", ""]
    if problem_id:
        lines.append(f"- problem_id: {problem_id}")
    lines.append(f"- mode: {mode_label}")
    lines.append(f"- methods considered: {', '.join(methods) if methods else '(none)'}")
    lines.append(f"- n_total_queries: {total_queries}")
    lines.append(f"- n_common_queries: {common_queries}")
    lines.append(f"- n_non_common_queries: {non_common_queries}")
    lines.append("")
    lines.append("## Partitions")
    lines.append("")

    if inventory_df.empty:
        lines.append("(no partitions)")
    else:
        display_df = inventory_df.copy()
        if "partition_name" in display_df.columns:
            display_df["partition"] = display_df["partition_name"].apply(
                lambda name: f"[{name}]({name}/)" if name else ""
            )
        display_cols = [
            col
            for col in [
                "partition",
                "partition_type",
                "solver_set",
                "n_queries",
                "share_of_total",
                "share_of_non_common",
            ]
            if col in display_df.columns
        ]
        table_lines = _df_to_md_table(display_df[display_cols])
        if table_lines:
            lines.extend(table_lines)
        else:
            lines.append("(no partitions)")

    if not inventory_df.empty:
        lines.append("")
        lines.append("## Partition Links")
        lines.append("")
        for _, row in inventory_df.iterrows():
            name = row.get("partition_name", "")
            if not name:
                continue
            links = [
                f"[tables]({name}/tables/)",
                f"[figures]({name}/figures/)",
            ]
            if isinstance(name, str) and name.startswith("subset_"):
                links.append(f"[subset_meta.json]({name}/subset_meta.json)")
            lines.append(f"- `{name}`: " + " | ".join(links))

        subset_rows = inventory_df[
            inventory_df["partition_name"].astype(str).str.startswith("subset_")
        ]
        if not subset_rows.empty:
            subset_rows = subset_rows.sort_values(
                ["n_queries", "partition_name"], ascending=[False, True]
            )
            top_k = subset_rows.head(10)
            lines.append("")
            lines.append("## Top subsets")
            lines.append("")
            for _, row in top_k.iterrows():
                name = row.get("partition_name", "")
                solver_set = row.get("solver_set", "")
                n_queries = row.get("n_queries", "")
                lines.append(
                    f"- `{name}` ({n_queries} queries, solver_set={solver_set})"
                )

    path = out_dir / "index.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def _write_root_report_index(
    out_dir: Path,
    *,
    run_dir: Path,
    summary_style: SummaryStyle,
    problem_ids: list[str],
    problem_categories: list[str] | None = None,
) -> Path:
    problem_categories = problem_categories or []
    lines = [f"# Report: {run_dir.name}", ""]
    lines.append(f"- summary_style: {summary_style.name}")
    lines.append("- aggregate: [aggregate](aggregate/)")
    lines.append(f"- n_problems: {len(problem_ids)}")
    lines.append(f"- n_problem_categories: {len(problem_categories)}")
    lines.append("")
    lines.append("## Per-problem")
    lines.append("")
    if not problem_ids:
        lines.append("(no problems)")
    else:
        for problem_id in problem_ids:
            lines.append(f"- [{problem_id}](single/{problem_id}/)")
    lines.append("")
    lines.append("## Per-category")
    lines.append("")
    if not problem_categories:
        lines.append("(no categories)")
    else:
        for category in problem_categories:
            lines.append(f"- [{category}](by_category/{category}/)")
    path = out_dir / "index.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def _generate_report_set(
    *,
    df_mode: pd.DataFrame,
    attempts_mode: pd.DataFrame,
    out_dir: Path,
    run_dir: Path,
    mode_label: str,
    summary_style: SummaryStyle,
    include_time: bool,
    include_pareto: bool,
    pareto_split: str,
    include_coverage: bool,
    allowed_query_types: set[str] | None,
    methods: list[str],
    min_partition_queries: int,
    max_subsets: int | None,
    include_all_methods_in_subsets: bool,
    problem_id: str | None = None,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    partition_sets, inventory_df = compute_partitions(
        attempts_mode,
        min_partition_queries=int(min_partition_queries),
        max_subsets=max_subsets,
    )

    all_keys = (
        set(attempts_mode["query_key"].dropna().unique())
        if not attempts_mode.empty
        else set()
    )
    if "all" not in partition_sets:
        partition_sets["all"] = all_keys
    if "common" not in partition_sets:
        partition_sets["common"] = set()

    generate_report_for_partition(
        df=df_mode,
        attempts_df=attempts_mode,
        out_dir=out_dir / "all",
        summary_style=summary_style,
        include_time=include_time,
        include_pareto=include_pareto,
        pareto_split=pareto_split,
        include_coverage=include_coverage,
        allowed_query_types=allowed_query_types,
        methods_to_show=methods,
    )

    _write_partition_inventory(
        out_dir,
        mode_label=mode_label,
        inventory_df=inventory_df,
        methods=methods,
    )

    partition_iter = inventory_df.iterrows()
    if not inventory_df.empty:
        label = f"{problem_id or 'aggregate'} partitions"
        partition_iter = tqdm(
            inventory_df.iterrows(),
            total=len(inventory_df),
            desc=label,
            leave=False,
        )

    for _, row in partition_iter:
        partition_name = row.get("partition_name")
        if not partition_name or partition_name == "all":
            continue
        keys = partition_sets.get(partition_name, set())
        partition_out = out_dir / str(partition_name)
        df_part = (
            df_mode[df_mode["query_key"].isin(keys)] if not df_mode.empty else df_mode
        )
        attempts_part = (
            attempts_mode[attempts_mode["query_key"].isin(keys)]
            if not attempts_mode.empty
            else attempts_mode
        )
        partition_type = row.get("partition_type")
        solver_label = row.get("solver_set") or ""
        solver_set = [s for s in str(solver_label).split("|") if s]
        if (
            partition_type == "subset"
            and not include_all_methods_in_subsets
            and solver_set
        ):
            methods_to_show = solver_set
        else:
            methods_to_show = methods

        generate_report_for_partition(
            df=df_part,
            attempts_df=attempts_part,
            out_dir=partition_out,
            summary_style=summary_style,
            include_time=include_time,
            include_pareto=include_pareto,
            pareto_split=pareto_split,
            include_coverage=True,
            allowed_query_types=allowed_query_types,
            methods_to_show=methods_to_show,
        )

        if partition_type == "subset":
            n_queries = int(row.get("n_queries", 0))
            share_of_total = float(row.get("share_of_total", 0.0))
            share_of_non_common = float(row.get("share_of_non_common", 0.0))
            meta = {
                "solver_set": solver_set,
                "n_queries": n_queries,
                "share_of_total": share_of_total,
                "share_of_non_common": share_of_non_common,
                "notes": "Queries in ALL but not in COMMON with exact solver ensemble",
            }
            (partition_out / "subset_meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True)
            )

    _write_report_index(
        out_dir,
        run_dir=run_dir,
        mode_label=mode_label,
        methods=methods,
        inventory_df=inventory_df,
        problem_id=problem_id,
    )
    return inventory_df


def _build_success_rate_line_table(
    df: pd.DataFrame,
    *,
    x_col: str,
    n_bins: int = 4,
    discrete_max: int = 20,
) -> pd.DataFrame:
    columns = [
        "model",
        "mode",
        "x_bin_left",
        "x_bin_right",
        "x_mid",
        "success_rate",
        "n_attempts",
        "n_ok",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    data = df[df["ok"].notna() & df[x_col].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    series = pd.to_numeric(data[x_col], errors="coerce")
    data = data[series.notna()].copy()
    data[x_col] = series.dropna().astype(float)
    if data.empty:
        return pd.DataFrame(columns=columns)

    use_discrete = False
    if x_col == "evidence_size":
        unique_vals = sorted(data[x_col].unique())
        if (
            unique_vals
            and len(unique_vals) <= discrete_max
            and all(_is_int_like(v) for v in unique_vals)
        ):
            use_discrete = True

    mode_col = "mode" if "mode" in data.columns else "query_type"
    rows: list[dict] = []
    if use_discrete:
        grouped = data.groupby(["method_id", mode_col, x_col], dropna=False)
        for (method_id, mode, x_val), group in grouped:
            ok_vals = group["ok"].astype(bool)
            n_attempts = int(len(ok_vals))
            if n_attempts == 0:
                success_rate = np.nan
            else:
                success_rate = float(ok_vals.mean())
            rows.append(
                {
                    "model": method_id,
                    "mode": mode,
                    "x_bin_left": float(x_val),
                    "x_bin_right": float(x_val),
                    "x_mid": float(x_val),
                    "success_rate": success_rate,
                    "n_attempts": n_attempts,
                    "n_ok": int(ok_vals.sum()),
                }
            )
        return pd.DataFrame(rows, columns=columns)

    edges = _compute_bin_edges(data[x_col], n_bins=n_bins)
    if not edges or len(edges) < 2:
        return pd.DataFrame(columns=columns)

    data["_bin_index"] = pd.cut(
        data[x_col],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    grouped = data.groupby(["method_id", mode_col, "_bin_index"], dropna=False)
    for (method_id, mode, bin_idx), group in grouped:
        if bin_idx is None or (isinstance(bin_idx, float) and math.isnan(bin_idx)):
            continue
        bin_idx = int(bin_idx)
        if bin_idx < 0 or bin_idx >= len(edges) - 1:
            continue
        ok_vals = group["ok"].astype(bool)
        n_attempts = int(len(ok_vals))
        success_rate = float(ok_vals.mean()) if n_attempts > 0 else np.nan
        left = float(edges[bin_idx])
        right = float(edges[bin_idx + 1])
        rows.append(
            {
                "model": method_id,
                "mode": mode,
                "x_bin_left": left,
                "x_bin_right": right,
                "x_mid": float((left + right) / 2.0),
                "success_rate": success_rate,
                "n_attempts": n_attempts,
                "n_ok": int(ok_vals.sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_coverage_line_table(
    df: pd.DataFrame,
    *,
    x_col: str,
    n_bins: int = 4,
    discrete_max: int = 20,
) -> pd.DataFrame:
    columns = ["model", "mode", "x_bin_left", "x_bin_right", "x_mid", "n_attempts"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    data = df[df[x_col].notna()].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    series = pd.to_numeric(data[x_col], errors="coerce")
    data = data[series.notna()].copy()
    data[x_col] = series.dropna().astype(float)
    if data.empty:
        return pd.DataFrame(columns=columns)

    use_discrete = False
    if x_col == "evidence_size":
        unique_vals = sorted(data[x_col].unique())
        if (
            unique_vals
            and len(unique_vals) <= discrete_max
            and all(_is_int_like(v) for v in unique_vals)
        ):
            use_discrete = True

    mode_col = "mode" if "mode" in data.columns else "query_type"
    rows: list[dict] = []
    if use_discrete:
        grouped = data.groupby(["method_id", mode_col, x_col], dropna=False)
        for (method_id, mode, x_val), group in grouped:
            n_attempts = int(len(group))
            rows.append(
                {
                    "model": method_id,
                    "mode": mode,
                    "x_bin_left": float(x_val),
                    "x_bin_right": float(x_val),
                    "x_mid": float(x_val),
                    "n_attempts": n_attempts,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    edges = _compute_bin_edges(data[x_col], n_bins=n_bins)
    if not edges or len(edges) < 2:
        return pd.DataFrame(columns=columns)
    data["_bin_index"] = pd.cut(
        data[x_col],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    grouped = data.groupby(["method_id", mode_col, "_bin_index"], dropna=False)
    for (method_id, mode, bin_idx), group in grouped:
        if bin_idx is None or (isinstance(bin_idx, float) and math.isnan(bin_idx)):
            continue
        bin_idx = int(bin_idx)
        if bin_idx < 0 or bin_idx >= len(edges) - 1:
            continue
        left = float(edges[bin_idx])
        right = float(edges[bin_idx + 1])
        rows.append(
            {
                "model": method_id,
                "mode": mode,
                "x_bin_left": left,
                "x_bin_right": right,
                "x_mid": float((left + right) / 2.0),
                "n_attempts": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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


def _metric_lower_bound(metric: str | None) -> float | None:
    if metric and metric in NON_NEGATIVE_METRICS:
        return 0.0
    return None


def _sanitize_numeric_values(
    values: list[float], *, lower_bound: float | None = None
) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            continue
        if not math.isfinite(numeric):
            continue
        if lower_bound is not None and numeric < lower_bound:
            numeric = lower_bound
        cleaned.append(float(numeric))
    return cleaned


def robust_summary(values: list[float], *, lower_bound: float | None = None) -> dict:
    cleaned = _sanitize_numeric_values(values, lower_bound=lower_bound)
    n = len(cleaned)
    if n == 0:
        return {
            "n": 0,
            "iqm": None,
            "min": None,
            "max": None,
            "iqr": None,
            "iqr_std": None,
            "q1": None,
            "median": None,
            "q3": None,
            "iqm_pm_iqr": None,
            "iqm_pm_iqrstd": None,
            "iqm_low_iqrstd_clipped": None,
            "iqm_high_iqrstd_clipped": None,
            "iqm_range_iqrstd_clipped": None,
        }
    sorted_vals = sorted(cleaned)
    lo = int(n * 0.25)
    hi = int(n * 0.75)
    trimmed = sorted_vals[lo:hi] if hi > lo else sorted_vals
    iqm = float(np.mean(trimmed)) if trimmed else float(np.mean(sorted_vals))
    observed_min = float(sorted_vals[0])
    observed_max = float(sorted_vals[-1])
    q1, median, q3 = np.percentile(sorted_vals, [25, 50, 75])
    iqr = float(q3 - q1)
    iqr_std = float((q3 - q1) / 1.349)
    low_iqrstd_clipped = float(max(observed_min, iqm - iqr_std))
    high_iqrstd_clipped = float(min(observed_max, iqm + iqr_std))
    return {
        "n": n,
        "iqm": iqm,
        "min": observed_min,
        "max": observed_max,
        "iqr": iqr,
        "iqr_std": iqr_std,
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqm_pm_iqr": f"{iqm:.4f} ± {iqr:.4f}",
        "iqm_pm_iqrstd": f"{iqm:.4f} ± {iqr_std:.4f}",
        "iqm_low_iqrstd_clipped": low_iqrstd_clipped,
        "iqm_high_iqrstd_clipped": high_iqrstd_clipped,
        "iqm_range_iqrstd_clipped": (
            f"[{low_iqrstd_clipped:.4f}, {high_iqrstd_clipped:.4f}]"
        ),
    }


def mean_summary(values: list[float], *, lower_bound: float | None = None) -> dict:
    cleaned = _sanitize_numeric_values(values, lower_bound=lower_bound)
    n = len(cleaned)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "mean_pm_std": None}
    mean = float(np.mean(cleaned))
    std = float(np.std(cleaned, ddof=0))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "mean_pm_std": f"{mean:.4f} ± {std:.4f}",
    }


def summarize(
    values: list[float], style: SummaryStyle, *, metric: str | None = None
) -> dict:
    lower_bound = _metric_lower_bound(metric)
    if style.name == "mean":
        return mean_summary(values, lower_bound=lower_bound)
    return robust_summary(values, lower_bound=lower_bound)


def _sanitize_metric_value(value: float | None, *, metric: str) -> float | None:
    if value is None:
        return None
    cleaned = _sanitize_numeric_values([value], lower_bound=_metric_lower_bound(metric))
    if not cleaned:
        return None
    return float(cleaned[0])


def aggregate_table(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    *,
    summary_style: SummaryStyle,
) -> pd.DataFrame:
    rows = []
    if df.empty:
        columns: list[str] = list(group_cols)
        for metric in metric_cols:
            for key in summary_style.keys:
                columns.append(f"{metric}_{key}")
        return pd.DataFrame(columns=columns)
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        for metric in metric_cols:
            summary = summarize(group[metric].tolist(), summary_style, metric=metric)
            for key, value in summary.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    result = pd.DataFrame(rows)
    return result


def aggregate_time_table(
    df: pd.DataFrame, group_cols: list[str], *, summary_style: SummaryStyle
) -> pd.DataFrame:
    if df.empty:
        columns: list[str] = list(group_cols)
        for key in summary_style.keys:
            columns.append(f"time_{key}")
        columns.extend(
            [
                "n_queries",
                "n_timed_queries",
                "time_sum_ms",
                "time_sum_s",
                "time_per_query_ms",
            ]
        )
        return pd.DataFrame(columns=columns)
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        lower_bound = _metric_lower_bound("time")
        times = [
            max(float(v), lower_bound) if lower_bound is not None else float(v)
            for v in group["time"].tolist()
            if v is not None and math.isfinite(float(v))
        ]
        summary = summarize(times, summary_style, metric="time")
        for key, value in summary.items():
            row[f"time_{key}"] = value
        n_queries = int(len(group))
        n_timed_queries = int(len(times))
        row["n_queries"] = n_queries
        row["n_timed_queries"] = n_timed_queries
        row["time_sum_ms"] = float(sum(times)) if times else 0.0
        row["time_sum_s"] = float(row["time_sum_ms"] / 1000.0)
        row["time_per_query_ms"] = (
            float(row["time_sum_ms"] / n_queries) if n_queries > 0 else None
        )
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
        lower_bound = _metric_lower_bound("time")
        times = [
            max(float(v), lower_bound) if lower_bound is not None else float(v)
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


def _to_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _metric_errorbars(
    data: pd.DataFrame,
    *,
    metric: str,
    summary_style: SummaryStyle,
) -> np.ndarray | list[float] | None:
    center_col = f"{metric}_{summary_style.center_key}"
    if center_col not in data:
        return None
    center = pd.to_numeric(data[center_col], errors="coerce")
    if center.dropna().empty:
        return None
    if summary_style.name == "robust":
        low_col = f"{metric}_iqm_low_iqrstd_clipped"
        high_col = f"{metric}_iqm_high_iqrstd_clipped"
        if low_col in data and high_col in data:
            low = pd.to_numeric(data[low_col], errors="coerce")
            high = pd.to_numeric(data[high_col], errors="coerce")
            if low.notna().all() and high.notna().all():
                lower = (center - low).clip(lower=0.0)
                upper = (high - center).clip(lower=0.0)
                return np.vstack(
                    [lower.to_numpy(dtype=float), upper.to_numpy(dtype=float)]
                )
    spread_col = f"{metric}_{summary_style.spread_key}"
    if spread_col not in data:
        return None
    spread = pd.to_numeric(data[spread_col], errors="coerce")
    if spread.dropna().empty:
        return None
    return spread.fillna(0.0).clip(lower=0.0).astype(float).tolist()


def _metric_errorbar_point(
    row: pd.Series,
    *,
    metric: str,
    summary_style: SummaryStyle,
) -> np.ndarray | float | None:
    center_col = f"{metric}_{summary_style.center_key}"
    center = _to_finite_float(row.get(center_col))
    if center is None:
        return None
    if summary_style.name == "robust":
        low = _to_finite_float(row.get(f"{metric}_iqm_low_iqrstd_clipped"))
        high = _to_finite_float(row.get(f"{metric}_iqm_high_iqrstd_clipped"))
        if low is not None and high is not None:
            lower = max(0.0, center - low)
            upper = max(0.0, high - center)
            return np.array([[lower], [upper]], dtype=float)
    spread = _to_finite_float(row.get(f"{metric}_{summary_style.spread_key}"))
    if spread is None:
        return None
    return max(0.0, float(spread))


def _ax_has_labeled_artists(ax: plt.Axes) -> bool:
    handles, labels = ax.get_legend_handles_labels()
    labels_filtered = [
        label for label in labels if label and not str(label).startswith("_")
    ]
    return len(labels_filtered) > 0


def _finalize_and_save_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    out_path: Path,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    add_legend: bool = True,
    logger: logging.Logger | None = None,
) -> bool:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if add_legend:
        has_labels = _ax_has_labeled_artists(ax)
        has_data = has_labels
    else:
        has_data = ax.has_data()

    if not has_data:
        plt.close(fig)
        logger = logger or logging.getLogger(__name__)
        logger.info("Skipping empty plot: %s (no data after filtering)", out_path.name)
        return False

    if add_legend and _ax_has_labeled_artists(ax):
        ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


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


def _extract_pred_samples(
    record: dict, *, seed: int, n_samples: int
) -> tuple[list[float] | None, dict | None]:
    output = record.get("result", {}).get("output")
    if not isinstance(output, dict):
        return None, None
    samples = output.get("samples")
    if isinstance(samples, list) and samples:
        return samples, output
    mean = output.get("mean")
    std = output.get("std")
    if mean is None or std is None:
        return None, output
    try:
        mean = float(mean)
        std = float(std)
    except Exception:
        return None, output
    if not math.isfinite(mean) or not math.isfinite(std):
        return None, output
    if std < 0:
        return None, output
    rng = np.random.default_rng(int(seed))
    draws = rng.normal(mean, std, size=int(n_samples)).astype(float)
    return draws.tolist(), output


def _extract_gt_probs(record: dict, gt_key: str) -> list[float] | None:
    value = _get_by_path(record, gt_key)
    if isinstance(value, list):
        return value
    return None


def _extract_gt_samples(
    record: dict, gt_key: str
) -> tuple[list[float] | None, float | None, float | None]:
    if isinstance(record.get("gt_samples"), list):
        return record.get("gt_samples"), record.get("gt_mean"), record.get("gt_std")
    value = _get_by_path(record, gt_key)
    if isinstance(value, list):
        return value, record.get("gt_mean"), record.get("gt_std")
    return None, None, None


def _extract_gt_from_line(line: dict, gt_key: str) -> dict | None:
    if isinstance(line.get("gt_samples"), list):
        return {
            "samples": line.get("gt_samples"),
            "mean": line.get("gt_mean"),
            "std": line.get("gt_std"),
        }
    if isinstance(line.get("gt_probs"), list):
        return {"probs": line.get("gt_probs")}
    value = _get_by_path(line, gt_key)
    if isinstance(value, list):
        return {"probs": value}
    output = (
        line.get("result", {}).get("output")
        if isinstance(line.get("result"), dict)
        else None
    )
    if isinstance(output, dict) and isinstance(output.get("probs"), list):
        return {"probs": output.get("probs")}
    return None


def _load_ground_truth_folder(
    run_dir: Path,
    gt_key: str,
    max_records: int | None,
    *,
    bundle_dir: Path | None = None,
) -> dict:
    gt_map: dict[tuple, dict] = {}
    gt_dir = run_dir / "ground_truth"
    if gt_dir.exists():
        for name in ("cpds.jsonl", "inference.jsonl"):
            path = gt_dir / name
            for record in _read_jsonl(path, max_records=max_records):
                entry = _extract_gt_from_line(record, gt_key)
                if entry is None:
                    continue
                key = _join_key(record)
                gt_map[key] = entry
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
                    if bundle_dir is not None:
                        candidate = Path(bundle_dir) / gt_path
                        if candidate.exists():
                            gt_path = candidate
                        else:
                            gt_path = get_project_root() / gt_path
                    else:
                        gt_path = get_project_root() / gt_path
                for record in _read_jsonl(gt_path, max_records=max_records):
                    entry = _extract_gt_from_line(record, gt_key)
                    if entry is None:
                        continue
                    key = _join_key(record)
                    gt_map[key] = entry
    return gt_map


def _read_run_metadata(run_dir: Path) -> dict:
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        payload = read_json(meta_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_result_files(run_dir: Path) -> list[Path]:
    results_root = run_dir / "results"
    if results_root.exists():
        return sorted([p for p in results_root.glob("**/*.jsonl") if p.is_file()])
    files: list[Path] = []
    for subdir in ("cpds", "inference"):
        base_dir = run_dir / subdir
        if not base_dir.exists():
            continue
        files.extend(sorted(base_dir.glob("*.jsonl")))
    return files


def _compute_graph_stats(run_dir: Path) -> dict[str, dict[str, int]]:
    generator = run_dir.parent.name
    project_root = get_project_root()
    stats: dict[str, dict[str, int]] = {}

    def _ensure_entry(problem_id: str) -> dict[str, int]:
        entry = stats.get(problem_id)
        if entry is None:
            entry = {"mb_sizes": {}, "parent_sizes": {}}
            stats[problem_id] = entry
        return entry

    # Priority 1: from run JSONL records
    for path in _iter_result_files(run_dir):
        for record in _read_jsonl(path, max_records=None):
            problem_meta = (
                record.get("problem") if isinstance(record.get("problem"), dict) else {}
            )
            problem_id = (
                problem_meta.get("id")
                or problem_meta.get("problem_id")
                or record.get("problem_id")
                or record.get("problem")
            )
            if not problem_id:
                continue
            entry = _ensure_entry(str(problem_id))
            n_nodes = _coerce_int(problem_meta.get("n_nodes"))
            n_edges = _coerce_int(problem_meta.get("n_edges"))
            if n_nodes is not None:
                entry["n_nodes"] = n_nodes
            if n_edges is not None:
                entry["n_edges"] = n_edges

    # Priority 2: bundle datasets (domain + download metadata)
    bundle_dir = _read_run_metadata(run_dir).get("bundle_dir")
    if bundle_dir:
        dataset_root = Path(bundle_dir) / "datasets" / generator
        if dataset_root.exists():
            for problem_dir in sorted(dataset_root.iterdir()):
                if not problem_dir.is_dir():
                    continue
                problem_id = problem_dir.name
                entry = _ensure_entry(problem_id)
                if entry.get("n_nodes") is None:
                    domain_path = problem_dir / "domain.json"
                    if domain_path.exists():
                        try:
                            domain = read_json(domain_path)
                        except Exception:
                            domain = {}
                        nodes = (
                            domain.get("nodes", {}) if isinstance(domain, dict) else {}
                        )
                        if isinstance(nodes, dict) and nodes:
                            entry["n_nodes"] = int(len(nodes))
                if entry.get("n_edges") is None:
                    for name in ("dataset.json", "download.json"):
                        meta_path = problem_dir / name
                        if not meta_path.exists():
                            continue
                        try:
                            meta = read_json(meta_path)
                        except Exception:
                            meta = {}
                        if isinstance(meta, dict):
                            n_edges = _coerce_int(meta.get("n_edges"))
                            if n_edges is not None:
                                entry["n_edges"] = n_edges
                                break
                            edges = meta.get("edges")
                            if isinstance(edges, list):
                                entry["n_edges"] = int(len(edges))
                                break

    # Priority 3: legacy metadata JSON (no R parsing)
    meta_root = project_root / "benchmarking" / "data" / "metadata" / generator
    if meta_root.exists():
        for problem_dir in sorted(meta_root.iterdir()):
            if not problem_dir.is_dir():
                continue
            problem_id = problem_dir.name
            entry = _ensure_entry(problem_id)
            if entry.get("n_nodes") is None:
                domain_path = problem_dir / "domain.json"
                if domain_path.exists():
                    try:
                        domain = read_json(domain_path)
                    except Exception:
                        domain = {}
                    nodes = domain.get("nodes", {}) if isinstance(domain, dict) else {}
                    if isinstance(nodes, dict) and nodes:
                        entry["n_nodes"] = int(len(nodes))
            if entry.get("n_edges") is None:
                for name in ("dataset.json", "download.json"):
                    meta_path = problem_dir / name
                    if not meta_path.exists():
                        continue
                    try:
                        meta = read_json(meta_path)
                    except Exception:
                        meta = {}
                    if isinstance(meta, dict):
                        n_edges = _coerce_int(meta.get("n_edges"))
                        if n_edges is not None:
                            entry["n_edges"] = n_edges
                            break
                        edges = meta.get("edges")
                        if isinstance(edges, list):
                            entry["n_edges"] = int(len(edges))
                            break

    return stats


def _load_problem_categories(run_dir: Path) -> dict[str, str]:
    generator = run_dir.parent.name
    meta_path = get_project_root() / "benchmarking" / "metadata" / f"{generator}.json"
    if not meta_path.exists():
        return {}
    try:
        payload = read_json(meta_path)
    except Exception:
        logging.warning("Failed to read generator metadata: %s", meta_path)
        return {}
    if not isinstance(payload, dict):
        return {}

    categories: dict[str, str] = {}
    for problem_id, meta in payload.items():
        if not isinstance(meta, dict):
            continue
        category = meta.get("category")
        if category is None:
            continue
        category_value = str(category).strip()
        if not category_value:
            continue
        categories[str(problem_id).strip().lower()] = category_value
    return categories


def _attach_problem_categories(
    df: pd.DataFrame,
    attempts_df: pd.DataFrame,
    problem_categories: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _attach(frame: pd.DataFrame) -> pd.DataFrame:
        if "problem_category" in frame.columns:
            return frame
        if frame.empty:
            frame["problem_category"] = pd.Series(dtype=object)
            return frame
        if "problem_id" not in frame.columns:
            frame["problem_category"] = pd.Series(index=frame.index, dtype=object)
            return frame
        problem_ids = frame["problem_id"]
        mapped = problem_ids.astype(str).str.lower().map(problem_categories)
        mapped = mapped.where(problem_ids.notna(), other=np.nan)
        frame["problem_category"] = mapped
        return frame

    return _attach(df), _attach(attempts_df)


def _build_records(
    *,
    run_dir: Path,
    gt_source: str,
    gt_key: str,
    max_records: int | None,
    eps: float,
    model_filter: set[str] | None,
    graph_stats: dict[str, dict[str, int]] | None = None,
    bundle_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    gt_map: dict[tuple, dict] = {}
    generator = run_dir.parent.name
    if bundle_dir is None:
        meta = _read_run_metadata(run_dir)
        bundle_value = meta.get("bundle_dir")
        if bundle_value:
            bundle_dir = Path(bundle_value)
    if gt_source == "folder":
        gt_map = _load_ground_truth_folder(
            run_dir, gt_key, max_records, bundle_dir=bundle_dir
        )

    if graph_stats is None:
        graph_stats = _compute_graph_stats(run_dir)
    gt_computer = (
        GTComputer(run_dir=run_dir, generator=generator, bundle_dir=bundle_dir)
        if gt_source == "compute"
        else None
    )

    rows: list[dict] = []
    for path in _iter_result_files(run_dir):
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
                components.get("cpd") if isinstance(components.get("cpd"), dict) else {}
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

            query = record.get("query") if isinstance(record.get("query"), dict) else {}
            gt_entry: dict | None = None
            if gt_source == "embedded":
                gt_samples, gt_mean, gt_std = _extract_gt_samples(record, gt_key)
                if gt_samples is not None:
                    gt_entry = {
                        "samples": gt_samples,
                        "mean": gt_mean,
                        "std": gt_std,
                    }
                else:
                    gt_probs = _extract_gt_probs(record, gt_key)
                    if gt_probs is not None:
                        gt_entry = {"probs": gt_probs}
            elif gt_source == "folder":
                gt_entry = gt_map.get(_join_key(record))
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
                    if gt_probs is not None:
                        gt_entry = {"probs": gt_probs}

            if gt_entry is None:
                continue

            kl = wass = jsd = jsd_norm = float("nan")
            mean_abs_err = None
            std_abs_err = None
            pred_output = None

            if "probs" in gt_entry:
                pred_probs, pred_output = _extract_pred_probs(record)
                if pred_probs is None:
                    continue
                gt_probs = gt_entry.get("probs") or []
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
            elif "samples" in gt_entry:
                gt_samples = gt_entry.get("samples") or []
                if not isinstance(gt_samples, list) or not gt_samples:
                    continue
                gt_mean = gt_entry.get("mean")
                gt_std = gt_entry.get("std")
                if gt_mean is None or gt_std is None:
                    arr = np.asarray(gt_samples, dtype=float)
                    if arr.size > 0 and np.isfinite(arr).any():
                        gt_mean = float(np.mean(arr))
                        gt_std = float(np.std(arr, ddof=0))
                query_id = query.get("id") or record.get("query", {}).get("id")
                seed_src = str(query_id or _join_key(record))
                seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
                pred_samples, pred_output = _extract_pred_samples(
                    record, seed=seed, n_samples=len(gt_samples)
                )
                if pred_samples is None:
                    continue
                wass = wasserstein_distance_samples(gt_samples, pred_samples)
                jsd = jensen_shannon_divergence_samples(gt_samples, pred_samples)
                jsd_norm = jensen_shannon_divergence_samples_normalized(
                    gt_samples, pred_samples
                )
                arr_pred = np.asarray(pred_samples, dtype=float)
                if arr_pred.size > 0 and np.isfinite(arr_pred).any():
                    pred_mean = float(np.mean(arr_pred))
                    pred_std = float(np.std(arr_pred, ddof=0))
                    if gt_mean is not None:
                        mean_abs_err = float(abs(pred_mean - float(gt_mean)))
                    if gt_std is not None:
                        std_abs_err = float(abs(pred_std - float(gt_std)))
            else:
                continue

            time_ms = None
            result = (
                record.get("result") if isinstance(record.get("result"), dict) else {}
            )
            timing = result.get("timing_ms")
            if timing is not None:
                try:
                    time_ms = float(timing)
                except Exception:
                    time_ms = None

            problem_id = record.get("problem", {}).get("id")
            query_type = query.get("type") or query.get("query_type")
            if not query_type:
                mode_val = record.get("mode") or record.get("run", {}).get("mode")
                if mode_val == "cpds":
                    query_type = "cpd"
                elif mode_val:
                    query_type = str(mode_val)
            target = query.get("target")
            target_category = query.get("target_category")
            target_set = query.get("target_set")
            evidence_strategy = query.get("evidence_strategy")
            evidence = (
                query.get("evidence") if isinstance(query.get("evidence"), dict) else {}
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
            query_id = query.get("id")
            query_index = query.get("index")
            mc_id = evidence.get("mc_id") or query.get("mc_id")
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
                record.get("problem") if isinstance(record.get("problem"), dict) else {}
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

            run_meta = record.get("run") if isinstance(record.get("run"), dict) else {}
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
                    "target_set": target_set,
                    "skeleton_id": skeleton_id,
                    "mb_size": mb_size,
                    "parent_size": parent_size,
                    "mc_id": mc_id,
                    "query_id": query_id,
                    "query_index": query_index,
                    "evidence_vars": ev_vars,
                    "run_id": run_id,
                    "seed": run_seed,
                    "generator": run_generator,
                    "timestamp_utc": run_timestamp,
                    "kl": _sanitize_metric_value(kl, metric="kl"),
                    "wass": _sanitize_metric_value(wass, metric="wass"),
                    "jsd": _sanitize_metric_value(jsd, metric="jsd"),
                    "jsd_norm": _sanitize_metric_value(jsd_norm, metric="jsd_norm"),
                    "time": _sanitize_metric_value(time_ms, metric="time"),
                    "mean_abs_err": _sanitize_metric_value(
                        mean_abs_err, metric="mean_abs_err"
                    ),
                    "std_abs_err": _sanitize_metric_value(
                        std_abs_err, metric="std_abs_err"
                    ),
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


def _build_attempts(
    *,
    run_dir: Path,
    max_records: int | None,
    model_filter: set[str] | None,
    graph_stats: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    if graph_stats is None:
        graph_stats = _compute_graph_stats(run_dir)
    rows: list[dict] = []
    for path in _iter_result_files(run_dir):
        for record in _read_jsonl(path, max_records=max_records):
            model_meta = (
                record.get("model") if isinstance(record.get("model"), dict) else {}
            )
            model_name = (
                model_meta.get("name")
                or model_meta.get("alias")
                or model_meta.get("backend")
                or "unknown"
            )
            model_alias = model_meta.get("alias")
            model_backend = model_meta.get("backend")
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

            run_meta = record.get("run") if isinstance(record.get("run"), dict) else {}
            run_id = run_meta.get("run_id")
            run_seed = _coerce_int(run_meta.get("seed"))
            run_generator = run_meta.get("generator")
            run_timestamp = run_meta.get("timestamp_utc")
            mode = record.get("mode") or run_meta.get("mode")

            query = record.get("query") if isinstance(record.get("query"), dict) else {}
            query_type = query.get("type") or query.get("query_type")
            if not query_type and mode:
                query_type = "cpd" if mode == "cpds" else str(mode)

            target = query.get("target")
            target_category = query.get("target_category")
            target_set = query.get("target_set")
            evidence_strategy = query.get("evidence_strategy")
            evidence = (
                query.get("evidence") if isinstance(query.get("evidence"), dict) else {}
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
            query_id = query.get("id")
            query_index = query.get("index")
            mc_id = evidence.get("mc_id") or query.get("mc_id")
            task = query.get("task")
            skeleton_id = evidence.get("skeleton_id") or query.get("skeleton_id")

            problem_meta = (
                record.get("problem") if isinstance(record.get("problem"), dict) else {}
            )
            problem_id = problem_meta.get("id")
            n_nodes = _coerce_int(problem_meta.get("n_nodes"))
            n_edges = _coerce_int(problem_meta.get("n_edges"))
            if problem_id and (n_nodes is None or n_edges is None):
                node_stats = graph_stats.get(problem_id)
                if node_stats:
                    if n_nodes is None:
                        n_nodes = node_stats.get("n_nodes")
                    if n_edges is None:
                        n_edges = node_stats.get("n_edges")

            result = (
                record.get("result") if isinstance(record.get("result"), dict) else {}
            )
            ok_val = result.get("ok")
            if ok_val is None:
                if result.get("output") is not None:
                    ok = True
                elif result.get("error_type") or result.get("error_msg"):
                    ok = False
                else:
                    ok = None
            else:
                ok = bool(ok_val)
            error_type = result.get("error_type")
            error_msg = result.get("error_msg")
            error_stage = result.get("error_stage")
            is_oom = result.get("is_oom")
            error_signature = result.get("error_signature")
            if ok is False:
                if error_msg is not None and not isinstance(error_msg, str):
                    error_msg = str(error_msg)
                if not error_type:
                    error_type = "ModelError" if error_msg else "UnknownError"
                if error_signature is None or is_oom is None:
                    info = classify_error(error_type, error_msg)
                    error_signature = error_signature or info["error_signature"]
                    if is_oom is None:
                        is_oom = info["is_oom"]
            if error_signature is None and ok is False:
                error_signature = "unknown"

            rows.append(
                {
                    "model_name": model_name,
                    "model_alias": model_alias,
                    "model_backend": model_backend,
                    "config_id": config_id,
                    "config_hash": config_hash,
                    "problem_id": problem_id,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "query_type": query_type,
                    "mode": mode or query_type,
                    "target": target,
                    "target_category": target_category,
                    "evidence_strategy": evidence_strategy,
                    "evidence_mode": evidence_mode,
                    "evidence_size": evidence_size,
                    "task": task,
                    "target_set": target_set,
                    "skeleton_id": skeleton_id,
                    "mc_id": mc_id,
                    "query_id": query_id,
                    "query_index": query_index,
                    "evidence_vars": ev_vars,
                    "run_id": run_id,
                    "seed": run_seed,
                    "generator": run_generator,
                    "timestamp_utc": run_timestamp,
                    "ok": ok,
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "error_stage": error_stage,
                    "is_oom": is_oom,
                    "error_signature": error_signature,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "model_alias",
                "model_backend",
                "config_id",
                "config_hash",
                "problem_id",
                "n_nodes",
                "n_edges",
                "query_type",
                "mode",
                "target",
                "target_category",
                "evidence_strategy",
                "evidence_mode",
                "evidence_size",
                "evidence_vars",
                "task",
                "target_set",
                "skeleton_id",
                "mc_id",
                "query_id",
                "query_index",
                "run_id",
                "seed",
                "generator",
                "timestamp_utc",
                "ok",
                "error_type",
                "error_msg",
                "error_stage",
                "is_oom",
                "error_signature",
                "config_key",
                "method_id",
            ]
        )
    df["config_key"] = df["config_id"].fillna(
        df["config_hash"].fillna("unknown").astype(str).str[:8]
    )
    df["method_id"] = df["model_name"].astype(str) + "/" + df["config_key"].astype(str)
    return df


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        if list(df.columns):
            df.to_csv(path, index=False)
        else:
            path.write_text("")
        return
    df = df.sort_values(list(df.columns))
    df.to_csv(path, index=False)


def _write_table_with_md(df: pd.DataFrame, path: Path) -> None:
    _write_table(df, path)
    _write_md_table(df, path.with_suffix(".md"))


def _two_stage_aggregate(
    df: pd.DataFrame,
    x_col: str,
    metric_cols: list[str],
    extra_group_cols: list[str] | None = None,
    *,
    summary_style: SummaryStyle,
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
            for key in summary_style.keys:
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
            summary = summarize(group[metric].tolist(), summary_style, metric=metric)
            row[metric] = summary.get(summary_style.center_key)
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
            for key in summary_style.keys:
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
            summary = summarize(group[metric].tolist(), summary_style, metric=metric)
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
    summary_style: SummaryStyle,
    out_dir: Path,
    title_prefix: str,
    filename_prefix: str,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        center_col = f"{metric}_{summary_style.center_key}"
        # spread_col = f"{metric}_{summary_style.spread_key}"
        pre_rows = int(len(df))
        filtered = df[df[size_col].notna() & df[center_col].notna()]
        post_rows = int(len(filtered))
        out_path = out_dir / f"{filename_prefix}.png"
        if filtered.empty:
            logging.getLogger(__name__).info(
                "Skipping empty plot %s: rows=%s -> after filter(%s+%s)=%s",
                out_path.name,
                pre_rows,
                size_col,
                center_col,
                post_rows,
            )
            return None
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.gca()
        method_ids = sorted(filtered["method_id"].dropna().unique())
        for method_id in method_ids:
            group = filtered[filtered["method_id"] == method_id]
            if group.empty:
                continue
            group = group.sort_values(size_col)
            x = group[size_col].astype(int).tolist()
            y = pd.to_numeric(group[center_col], errors="coerce").astype(float).tolist()
            yerr = _metric_errorbars(group, metric=metric, summary_style=summary_style)
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=method_id)
        if metric in NON_NEGATIVE_METRICS:
            ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title_prefix,
            xlabel=size_col,
            ylabel=_metric_label(metric),
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename_prefix, exc)
        if fig is not None:
            plt.close(fig)
        return None


def _plot_error_vs_evidence_size(
    df: pd.DataFrame,
    *,
    metric: str,
    summary_style: SummaryStyle,
    out_dir: Path,
    filename_prefix: str,
    mode: str | None = None,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        data = df
        if mode is not None:
            data = data[data["evidence_mode"] == mode]
        center_col = f"{metric}_{summary_style.center_key}"
        # spread_col = f"{metric}_{summary_style.spread_key}"
        pre_rows = int(len(data))
        data = data[data["evidence_size"].notna() & data[center_col].notna()]
        out_path = out_dir / f"{filename_prefix}{'__mode_' + mode if mode else ''}.png"
        post_rows = int(len(data))
        if data.empty:
            logging.getLogger(__name__).info(
                "Skipping empty plot %s: rows=%s -> after filter(evidence_size+%s)=%s",
                out_path.name,
                pre_rows,
                center_col,
                post_rows,
            )
            return None
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.gca()
        method_ids = sorted(data["method_id"].dropna().unique())
        for method_id in method_ids:
            sub = data[data["method_id"] == method_id].sort_values("evidence_size")
            if sub.empty:
                continue
            x = sub["evidence_size"].astype(int).tolist()
            y = pd.to_numeric(sub[center_col], errors="coerce").astype(float).tolist()
            yerr = _metric_errorbars(sub, metric=metric, summary_style=summary_style)
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3, label=method_id)
        title = f"Inference {_metric_label(metric)} vs Evidence Size"
        if mode is not None:
            title = f"{title} ({mode})"
        if metric in NON_NEGATIVE_METRICS:
            ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title,
            xlabel="evidence_size",
            ylabel=_metric_label(metric),
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename_prefix, exc)
        if fig is not None:
            plt.close(fig)
        return None


def _plot_category_bars(
    df: pd.DataFrame,
    *,
    category_col: str,
    metric: str,
    summary_style: SummaryStyle,
    out_dir: Path,
    filename_prefix: str,
    title_prefix: str,
    category_order: list[str],
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        center_col = f"{metric}_{summary_style.center_key}"
        data = df[df[category_col].notna()]
        data = data[data[center_col].notna()]
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
        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.gca()
        for idx, method_id in enumerate(method_ids):
            sub = data[data["method_id"] == method_id]
            sub = sub[sub[category_col].isin(ordered)].copy()
            if sub.empty:
                continue
            sub[category_col] = pd.Categorical(
                sub[category_col], categories=ordered, ordered=True
            )
            sub = sub.sort_values(category_col)
            sub = sub.drop_duplicates(subset=[category_col], keep="first")
            row_map = {cat: row for cat, row in sub.set_index(category_col).iterrows()}
            y: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            has_errorbars = False
            for cat in ordered:
                row = row_map.get(cat)
                if row is None:
                    y.append(np.nan)
                    lower.append(np.nan)
                    upper.append(np.nan)
                    continue
                center = _to_finite_float(row.get(center_col))
                y.append(center if center is not None else np.nan)
                err = _metric_errorbar_point(
                    row, metric=metric, summary_style=summary_style
                )
                if isinstance(err, np.ndarray):
                    low = float(err[0, 0])
                    high = float(err[1, 0])
                    has_errorbars = True
                elif err is not None:
                    val = float(err)
                    low = val
                    high = val
                    has_errorbars = True
                else:
                    low = np.nan
                    high = np.nan
                lower.append(low)
                upper.append(high)
            yerr = (
                np.vstack([np.array(lower, dtype=float), np.array(upper, dtype=float)])
                if has_errorbars
                else None
            )
            offset = (idx - (len(method_ids) - 1) / 2) * width
            ax.bar(x + offset, y, width=width, yerr=yerr, capsize=3, label=method_id)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=30, ha="right")
        if metric in NON_NEGATIVE_METRICS:
            ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", alpha=0.3)
        out_path = out_dir / f"{filename_prefix}.png"
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title_prefix,
            ylabel=_metric_label(metric),
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename_prefix, exc)
        if fig is not None:
            plt.close(fig)
        return None


def plot_success_rate_bar(
    df: pd.DataFrame,
    *,
    category_col: str,
    out_dir: Path,
    filename_prefix: str,
    title_prefix: str,
    category_order: list[str] | None = None,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        data = df[df[category_col].notna()]
        data = data[data["success_rate"].notna()]
        if data.empty:
            return None
        ordered: list[str]
        if category_order:
            ordered = [c for c in category_order if c in set(data[category_col])]
            ordered += [
                c
                for c in sorted(data[category_col].dropna().unique())
                if c not in ordered
            ]
        else:
            ordered = sorted(data[category_col].dropna().unique())
        method_ids = sorted(data["method_id"].dropna().unique())
        if not method_ids:
            return None
        x = np.arange(len(ordered))
        width = 0.8 / max(1, len(method_ids))
        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.gca()
        for idx, method_id in enumerate(method_ids):
            sub = data[data["method_id"] == method_id]
            sub = sub[sub[category_col].isin(ordered)].copy()
            if sub.empty:
                continue
            sub[category_col] = pd.Categorical(
                sub[category_col], categories=ordered, ordered=True
            )
            sub = sub.sort_values(category_col)
            y_map = dict(zip(sub[category_col], sub["success_rate"]))
            y = [y_map.get(cat, np.nan) for cat in ordered]
            offset = (idx - (len(method_ids) - 1) / 2) * width
            ax.bar(x + offset, y, width=width, label=method_id)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        out_path = out_dir / f"{filename_prefix}.png"
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title_prefix,
            ylabel="Success Rate",
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename_prefix, exc)
        if fig is not None:
            plt.close(fig)
        return None


def plot_success_rate_line(
    df: pd.DataFrame,
    *,
    x_label: str,
    out_dir: Path,
    filename: str,
    title: str,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        data = df[df["success_rate"].notna()]
        if data.empty:
            return None
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.gca()
        for model in sorted(data["model"].dropna().unique()):
            sub = data[data["model"] == model].sort_values("x_mid")
            if sub.empty:
                continue
            ax.errorbar(
                sub["x_mid"].astype(float),
                sub["success_rate"],
                fmt="-o",
                capsize=3,
                label=model,
            )
        out_path = out_dir / filename
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title,
            xlabel=x_label,
            ylabel="Success Rate",
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename, exc)
        if fig is not None:
            plt.close(fig)
        return None


def plot_coverage_line(
    df: pd.DataFrame,
    *,
    x_label: str,
    out_dir: Path,
    filename: str,
    title: str,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        data = df[df["n_attempts"].notna()]
        if data.empty:
            return None
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.gca()
        for model in sorted(data["model"].dropna().unique()):
            sub = data[data["model"] == model].sort_values("x_mid")
            if sub.empty:
                continue
            ax.errorbar(
                sub["x_mid"].astype(float),
                sub["n_attempts"].astype(float),
                fmt="-o",
                capsize=3,
                label=model,
            )
        out_path = out_dir / filename
        ax.grid(True, alpha=0.3)
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title,
            xlabel=x_label,
            ylabel="n_queries",
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename, exc)
        if fig is not None:
            plt.close(fig)
        return None


def _plot_error_type_distribution(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    filename: str,
    title: str,
    top_k: int = 6,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        data = df[df["error_type"].notna()]
        if data.empty:
            return None
        top_types = data["error_type"].value_counts().head(top_k).index.tolist()
        data = data.copy()
        data["error_type_plot"] = data["error_type"].where(
            data["error_type"].isin(top_types), "Other"
        )
        pivot = (
            data.groupby(["method_id", "error_type_plot"]).size().unstack(fill_value=0)
        )
        if pivot.empty:
            return None
        pivot = pivot.sort_index()
        fig = plt.figure(figsize=(10, 4.5))
        ax = fig.gca()
        bottoms = np.zeros(len(pivot))
        for col in pivot.columns:
            values = pivot[col].values
            ax.bar(pivot.index, values, bottom=bottoms, label=str(col))
            bottoms = bottoms + values
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        out_path = out_dir / filename
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title,
            ylabel="Error Count",
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename, exc)
        if fig is not None:
            plt.close(fig)
        return None


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
    summary_style: SummaryStyle,
    out_dir: Path,
    filename: str,
    title: str,
) -> Path | None:
    if df.empty:
        return None
    fig = None
    try:
        sub = df.copy()
        metric_center = f"{metric}_{summary_style.center_key}"
        # metric_spread = f"{metric}_{summary_style.spread_key}"
        time_center = f"time_{summary_style.center_key}"
        # time_spread = f"time_{summary_style.spread_key}"
        sub = sub[sub[metric_center].notna() & sub[time_center].notna()]
        if sub.empty:
            return None
        points = list(
            zip(sub[time_center].astype(float), sub[metric_center].astype(float))
        )
        pareto_mask = _pareto_frontier(points)
        fig = plt.figure(figsize=(7.5, 5))
        ax = fig.gca()
        pareto_label_added = False
        other_label_added = False
        for (_, row), is_pareto in zip(sub.iterrows(), pareto_mask):
            time_val = float(row[time_center])
            err_val = float(row[metric_center])
            method_id = row.get("method_id")
            xerr = _metric_errorbar_point(
                row, metric="time", summary_style=summary_style
            )
            yerr = _metric_errorbar_point(
                row, metric=metric, summary_style=summary_style
            )
            if is_pareto:
                ax.errorbar(
                    time_val,
                    err_val,
                    xerr=xerr,
                    yerr=yerr,
                    fmt="o",
                    color="C1",
                    capsize=3,
                    label="Pareto" if not pareto_label_added else None,
                )
                ax.annotate(
                    method_id,
                    (time_val, err_val),
                    textcoords="offset points",
                    xytext=(6, 6),
                )
                pareto_label_added = True
            else:
                ax.errorbar(
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
            ax.plot(
                [p[0] for p in pareto_points],
                [p[1] for p in pareto_points],
                color="C1",
                alpha=0.5,
            )
        if "time" in NON_NEGATIVE_METRICS:
            ax.set_xlim(left=0.0)
        if metric in NON_NEGATIVE_METRICS:
            ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        out_path = out_dir / filename
        saved = _finalize_and_save_plot(
            fig,
            ax,
            out_path,
            title=title,
            xlabel=f"{summary_style.center_label} time (ms)",
            ylabel=f"{summary_style.center_label} {_metric_label(metric)}",
            add_legend=True,
        )
        return out_path if saved else None
    except Exception as exc:
        logging.warning("Plot failed (%s): %s", filename, exc)
        if fig is not None:
            plt.close(fig)
        return None


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


def _normalize_query_type(value: str | None) -> str | None:
    if value is None:
        return None
    if value == "cpds":
        return "cpd"
    return value


def _infer_query_type(
    run_mode: str | None, df: pd.DataFrame, attempts_df: pd.DataFrame
) -> tuple[str | None, str]:
    if run_mode == "cpds":
        return "cpd", "cpds"
    if run_mode == "inference":
        return "inference", "inference"

    qtypes: set[str] = set()
    for frame in (df, attempts_df):
        if frame is not None and not frame.empty and "query_type" in frame.columns:
            qtypes.update(
                _normalize_query_type(qt)
                for qt in frame["query_type"].dropna().unique()
            )
    qtypes.discard(None)
    if len(qtypes) == 1:
        qt = next(iter(qtypes))
        label = "cpds" if qt == "cpd" else str(qt)
        return qt, label
    if not qtypes:
        return None, "unknown"
    qt = sorted(qtypes)[0]
    label = "cpds" if qt == "cpd" else str(qt)
    logging.warning("Multiple query types detected; defaulting partitions to %s", qt)
    return qt, label


def generate_report_for_partition(
    *,
    df: pd.DataFrame,
    attempts_df: pd.DataFrame,
    out_dir: Path,
    summary_style: SummaryStyle,
    include_time: bool,
    include_pareto: bool,
    pareto_split: str,
    include_coverage: bool,
    allowed_query_types: set[str] | None = None,
    methods_to_show: list[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    if allowed_query_types:
        if not df.empty:
            df = df[df["query_type"].isin(allowed_query_types)]
        if not attempts_df.empty:
            attempts_df = attempts_df[
                attempts_df["query_type"].isin(allowed_query_types)
            ]
    if methods_to_show:
        methods_set = set(methods_to_show)
        if not df.empty:
            df = df[df["method_id"].isin(methods_set)]
        if not attempts_df.empty:
            attempts_df = attempts_df[attempts_df["method_id"].isin(methods_set)]

    tables_dir = ensure_dir(out_dir / "tables")
    figures_dir = ensure_dir(out_dir / "figures")

    if df.empty:
        logging.warning("No records to report. Check GT source and inputs.")
    if attempts_df.empty:
        logging.warning("No attempt records found for success/error reporting.")

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
        summary_style=summary_style,
    )
    path = tables_dir / "overall_by_model.csv"
    _write_table(overall, path)
    tables.append(path)

    cpd = df[df["query_type"] == "cpd"]
    inference = df[df["query_type"] == "inference"]
    include_cpd = allowed_query_types is None or "cpd" in allowed_query_types
    include_inference = (
        allowed_query_types is None or "inference" in allowed_query_types
    )

    if include_cpd:
        cpd_by_target = aggregate_table(
            cpd,
            ["method_id", "model_name", "config_id", "config_hash", "target_category"],
            metric_cols,
            summary_style=summary_style,
        )
        path = tables_dir / "cpd_by_target_category.csv"
        _write_table(cpd_by_target, path)
        tables.append(path)

        cpd_by_strategy = aggregate_table(
            cpd,
            [
                "method_id",
                "model_name",
                "config_id",
                "config_hash",
                "evidence_strategy",
            ],
            metric_cols,
            summary_style=summary_style,
        )
        path = tables_dir / "cpd_by_evidence_strategy.csv"
        _write_table(cpd_by_strategy, path)
        tables.append(path)

        cpd_mb = _two_stage_aggregate(
            cpd, "mb_size", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "cpd_by_mb_size.csv"
        _write_table(cpd_mb, path)
        tables.append(path)

        cpd_parent = _two_stage_aggregate(
            cpd, "parent_size", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "cpd_by_parent_size.csv"
        _write_table(cpd_parent, path)
        tables.append(path)

        cpd_nodes = _two_stage_aggregate(
            cpd, "n_nodes", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "cpd_by_n_nodes.csv"
        _write_table(cpd_nodes, path)
        tables.append(path)

        cpd_edges = _two_stage_aggregate(
            cpd, "n_edges", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "cpd_by_n_edges.csv"
        _write_table(cpd_edges, path)
        tables.append(path)

    if include_inference:
        inf_by_target = aggregate_table(
            inference,
            ["method_id", "model_name", "config_id", "config_hash", "target_category"],
            metric_cols,
            summary_style=summary_style,
        )
        path = tables_dir / "inference_by_target_category.csv"
        _write_table(inf_by_target, path)
        tables.append(path)

        inf_by_task = aggregate_table(
            inference,
            ["method_id", "model_name", "config_id", "config_hash", "task"],
            metric_cols,
            summary_style=summary_style,
        )
        path = tables_dir / "inference_by_task.csv"
        _write_table(inf_by_task, path)
        tables.append(path)

        inf_by_mode = aggregate_table(
            inference,
            ["method_id", "model_name", "config_id", "config_hash", "evidence_mode"],
            metric_cols,
            summary_style=summary_style,
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

        inf_ev_size = _two_stage_aggregate(
            inference, "evidence_size", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "inference_by_evidence_size.csv"
        _write_table(inf_ev_size, path)
        tables.append(path)

        inf_nodes = _two_stage_aggregate(
            inference, "n_nodes", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "inference_by_n_nodes.csv"
        _write_table(inf_nodes, path)
        tables.append(path)

        inf_edges = _two_stage_aggregate(
            inference, "n_edges", metric_cols, summary_style=summary_style
        )
        path = tables_dir / "inference_by_n_edges.csv"
        _write_table(inf_edges, path)
        tables.append(path)

        inf_ev_size_mode = _two_stage_aggregate(
            inference,
            "evidence_size",
            metric_cols,
            extra_group_cols=["evidence_mode"],
            summary_style=summary_style,
        )

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
            summary_style=summary_style,
        )
        path = tables_dir / "inference_by_skeleton.csv"
        _write_table(skeleton, path)
        tables.append(path)

    time_tables: dict[str, pd.DataFrame] = {}
    if include_time:
        overall_time = aggregate_time_table(
            df,
            ["method_id", "model_name", "config_id", "config_hash", "query_type"],
            summary_style=summary_style,
        )
        path = tables_dir / "overall_time_by_method.csv"
        _write_table(overall_time, path)
        tables.append(path)
        time_tables["overall"] = overall_time

        if include_cpd:
            cpd_time_by_target = aggregate_time_table(
                cpd,
                [
                    "method_id",
                    "model_name",
                    "config_id",
                    "config_hash",
                    "target_category",
                ],
                summary_style=summary_style,
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
                summary_style=summary_style,
            )
            path = tables_dir / "cpd_time_by_evidence_strategy.csv"
            _write_table(cpd_time_by_strategy, path)
            tables.append(path)
            time_tables["cpd_by_strategy"] = cpd_time_by_strategy

            cpd_time_mb = _two_stage_aggregate(
                cpd, "mb_size", ["time"], summary_style=summary_style
            )
            path = tables_dir / "cpd_time_by_mb_size.csv"
            _write_table(cpd_time_mb, path)
            tables.append(path)
            time_tables["cpd_mb"] = cpd_time_mb

            cpd_time_parent = _two_stage_aggregate(
                cpd, "parent_size", ["time"], summary_style=summary_style
            )
            path = tables_dir / "cpd_time_by_parent_size.csv"
            _write_table(cpd_time_parent, path)
            tables.append(path)
            time_tables["cpd_parent"] = cpd_time_parent

            cpd_time_nodes = _two_stage_aggregate(
                cpd, "n_nodes", ["time"], summary_style=summary_style
            )
            path = tables_dir / "cpd_time_by_n_nodes.csv"
            _write_table(cpd_time_nodes, path)
            tables.append(path)
            time_tables["cpd_nodes"] = cpd_time_nodes

            cpd_time_edges = _two_stage_aggregate(
                cpd, "n_edges", ["time"], summary_style=summary_style
            )
            path = tables_dir / "cpd_time_by_n_edges.csv"
            _write_table(cpd_time_edges, path)
            tables.append(path)
            time_tables["cpd_edges"] = cpd_time_edges

            cpd_time_ev_size = _two_stage_aggregate(
                cpd, "evidence_size", ["time"], summary_style=summary_style
            )
            path = tables_dir / "cpd_time_by_evidence_size.csv"
            _write_table(cpd_time_ev_size, path)
            tables.append(path)
            time_tables["cpd_ev_size"] = cpd_time_ev_size

        if include_inference:
            inf_time_by_target = aggregate_time_table(
                inference,
                [
                    "method_id",
                    "model_name",
                    "config_id",
                    "config_hash",
                    "target_category",
                ],
                summary_style=summary_style,
            )
            path = tables_dir / "inference_time_by_target_category.csv"
            _write_table(inf_time_by_target, path)
            tables.append(path)
            time_tables["inf_by_target"] = inf_time_by_target

            inf_time_by_task = aggregate_time_table(
                inference,
                ["method_id", "model_name", "config_id", "config_hash", "task"],
                summary_style=summary_style,
            )
            path = tables_dir / "inference_time_by_task.csv"
            _write_table(inf_time_by_task, path)
            tables.append(path)
            time_tables["inf_by_task"] = inf_time_by_task

            inf_time_by_mode = aggregate_time_table(
                inference,
                [
                    "method_id",
                    "model_name",
                    "config_id",
                    "config_hash",
                    "evidence_mode",
                ],
                summary_style=summary_style,
            )
            path = tables_dir / "inference_time_by_evidence_mode.csv"
            _write_table(inf_time_by_mode, path)
            tables.append(path)
            time_tables["inf_by_mode"] = inf_time_by_mode

            inf_time_ev_size = _two_stage_aggregate(
                inference, "evidence_size", ["time"], summary_style=summary_style
            )
            path = tables_dir / "inference_time_by_evidence_size.csv"
            _write_table(inf_time_ev_size, path)
            tables.append(path)
            time_tables["inf_ev_size"] = inf_time_ev_size

            inf_time_nodes = _two_stage_aggregate(
                inference, "n_nodes", ["time"], summary_style=summary_style
            )
            path = tables_dir / "inference_time_by_n_nodes.csv"
            _write_table(inf_time_nodes, path)
            tables.append(path)
            time_tables["inf_nodes"] = inf_time_nodes

            inf_time_edges = _two_stage_aggregate(
                inference, "n_edges", ["time"], summary_style=summary_style
            )
            path = tables_dir / "inference_time_by_n_edges.csv"
            _write_table(inf_time_edges, path)
            tables.append(path)
            time_tables["inf_edges"] = inf_time_edges

    # Success rates + errors
    success_target = pd.DataFrame()
    success_strategy = pd.DataFrame()
    success_task = pd.DataFrame()
    success_mode = pd.DataFrame()
    success_ev_size = pd.DataFrame()
    success_line_nodes = pd.DataFrame()
    success_line_edges = pd.DataFrame()
    success_line_ev_size = pd.DataFrame()
    success_line_ev_size_by_mode: dict[str, pd.DataFrame] = {}
    coverage_nodes = pd.DataFrame()
    coverage_edges = pd.DataFrame()
    coverage_ev_size = pd.DataFrame()
    error_df = pd.DataFrame()
    if not attempts_df.empty:
        base_group = [
            "method_id",
            "model_name",
            "config_id",
            "config_hash",
            "query_type",
        ]
        success_by_model = _aggregate_success(attempts_df, base_group)
        path = tables_dir / "success_rate_by_model.csv"
        _write_table_with_md(success_by_model, path)
        tables.append(path)

        success_line_nodes = _build_success_rate_line_table(
            attempts_df, x_col="n_nodes", n_bins=4
        )
        path = tables_dir / "success_rate_vs_nodes.csv"
        _write_table_with_md(success_line_nodes, path)
        tables.append(path)

        success_line_edges = _build_success_rate_line_table(
            attempts_df, x_col="n_edges", n_bins=4
        )
        path = tables_dir / "success_rate_vs_edges.csv"
        _write_table_with_md(success_line_edges, path)
        tables.append(path)

        success_target = _aggregate_success(
            attempts_df[attempts_df["target_category"].notna()],
            base_group + ["target_category"],
        )
        path = tables_dir / "success_rate_by_category.csv"
        _write_table_with_md(success_target, path)
        tables.append(path)

        success_strategy = _aggregate_success(
            attempts_df[attempts_df["evidence_strategy"].notna()],
            base_group + ["evidence_strategy"],
        )
        path = tables_dir / "success_rate_by_evidence_strategy.csv"
        _write_table_with_md(success_strategy, path)
        tables.append(path)

        success_task = _aggregate_success(
            attempts_df[attempts_df["task"].notna()],
            base_group + ["task"],
        )
        path = tables_dir / "success_rate_by_task.csv"
        _write_table_with_md(success_task, path)
        tables.append(path)

        success_mode = _aggregate_success(
            attempts_df[attempts_df["evidence_mode"].notna()],
            base_group + ["evidence_mode"],
        )
        path = tables_dir / "success_rate_by_evidence_mode.csv"
        _write_table_with_md(success_mode, path)
        tables.append(path)

        success_ev_size = _aggregate_success(
            attempts_df[attempts_df["evidence_size"].notna()],
            base_group + ["evidence_size"],
        )
        path = tables_dir / "success_rate_by_evidence_size.csv"
        _write_table_with_md(success_ev_size, path)
        tables.append(path)

        mode_series = attempts_df.get("mode")
        if mode_series is None or mode_series.isna().all():
            mode_series = attempts_df["query_type"]
        inference_mask = mode_series == "inference"
        success_line_ev_size = _build_success_rate_line_table(
            attempts_df[inference_mask],
            x_col="evidence_size",
            n_bins=4,
        )
        path = tables_dir / "success_rate_vs_evidence_size.csv"
        _write_table_with_md(success_line_ev_size, path)
        tables.append(path)

        evidence_modes = [
            m
            for m in INF_EVIDENCE_MODES
            if m in set(attempts_df["evidence_mode"].dropna().unique())
        ]
        evidence_modes += [
            m
            for m in sorted(attempts_df["evidence_mode"].dropna().unique())
            if m not in evidence_modes
        ]
        for mode in evidence_modes:
            mode_mask = inference_mask & (attempts_df["evidence_mode"] == mode)
            table = _build_success_rate_line_table(
                attempts_df[mode_mask],
                x_col="evidence_size",
                n_bins=4,
            )
            success_line_ev_size_by_mode[str(mode)] = table
            tag = _safe_tag(str(mode))
            path = tables_dir / f"success_rate_vs_evidence_size__mode_{tag}.csv"
            _write_table_with_md(table, path)
            tables.append(path)

        if include_coverage:
            coverage_nodes = _build_coverage_line_table(
                attempts_df, x_col="n_nodes", n_bins=4
            )
            path = tables_dir / "coverage_vs_nodes.csv"
            _write_table_with_md(coverage_nodes, path)
            tables.append(path)

            coverage_edges = _build_coverage_line_table(
                attempts_df, x_col="n_edges", n_bins=4
            )
            path = tables_dir / "coverage_vs_edges.csv"
            _write_table_with_md(coverage_edges, path)
            tables.append(path)

            coverage_ev_size = _build_coverage_line_table(
                attempts_df[inference_mask],
                x_col="evidence_size",
                n_bins=4,
            )
            path = tables_dir / "coverage_vs_evidence_size.csv"
            _write_table_with_md(coverage_ev_size, path)
            tables.append(path)

        error_df = attempts_df[~attempts_df["ok"]].copy()
        if not error_df.empty:
            error_df["error_type"] = error_df["error_type"].fillna("UnknownError")
            error_df["error_signature"] = error_df["error_signature"].fillna("unknown")
            counts = (
                error_df.groupby(["method_id", "query_type", "error_type"])
                .size()
                .reset_index(name="count")
            )
            totals = counts.groupby(["method_id", "query_type"])["count"].transform(
                "sum"
            )
            counts["share"] = counts["count"] / totals
            top_errors = counts.rename(
                columns={
                    "method_id": "model",
                    "query_type": "mode",
                }
            )[["model", "mode", "error_type", "count", "share"]]
            path = tables_dir / "top_errors_by_model.csv"
            _write_table_with_md(top_errors, path)
            tables.append(path)

            sig_counts = (
                error_df.groupby(["method_id", "query_type", "error_signature"])
                .size()
                .reset_index(name="count")
            )
            sig_totals = sig_counts.groupby(["method_id", "query_type"])[
                "count"
            ].transform("sum")
            sig_counts["share"] = sig_counts["count"] / sig_totals
            examples = (
                error_df.groupby(["method_id", "query_type", "error_signature"])[
                    "error_msg"
                ]
                .apply(lambda s: next((str(v) for v in s if pd.notna(v)), None))
                .reset_index(name="example")
            )
            sig_counts = sig_counts.merge(
                examples, on=["method_id", "query_type", "error_signature"], how="left"
            )
            sig_counts["example"] = sig_counts["example"].fillna("")
            top_sigs = sig_counts.rename(
                columns={
                    "method_id": "model",
                    "query_type": "mode",
                }
            )[["model", "mode", "error_signature", "count", "share", "example"]]
            path = tables_dir / "top_error_signatures.csv"
            _write_table_with_md(top_sigs, path)
            tables.append(path)

    # Plots
    if include_cpd:
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
                    summary_style=summary_style,
                    out_dir=figures_dir,
                    title_prefix=f"CPD {_metric_label(metric)} vs {size_label}",
                    filename_prefix=f"cpd_{metric}_vs_{tag}",
                )
                if fig:
                    figures.append(fig)

    if include_inference:
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
                    summary_style=summary_style,
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
                    summary_style=summary_style,
                    out_dir=figures_dir,
                    filename_prefix=f"inference_{metric}_vs_evidence_size",
                    mode=mode,
                )
                if fig:
                    figures.append(fig)

    if include_pareto:
        pareto_metrics = ["kl", "wass", "jsd_norm"]
        pareto_summary = aggregate_table(
            df,
            ["method_id", "model_name", "config_id", "config_hash", "query_type"],
            [*pareto_metrics, "time"],
            summary_style=summary_style,
        )
        if include_cpd:
            for metric in pareto_metrics:
                sub = pareto_summary[pareto_summary["query_type"] == "cpd"]
                fig = _plot_pareto(
                    sub,
                    metric=metric,
                    summary_style=summary_style,
                    out_dir=figures_dir,
                    filename=f"pareto_cpd_{metric}_vs_time.png",
                    title=f"CPD {_metric_label(metric)} vs Time",
                )
                if fig:
                    figures.append(fig)
        if include_inference:
            for metric in pareto_metrics:
                sub = pareto_summary[pareto_summary["query_type"] == "inference"]
                fig = _plot_pareto(
                    sub,
                    metric=metric,
                    summary_style=summary_style,
                    out_dir=figures_dir,
                    filename=f"pareto_inference_{metric}_vs_time.png",
                    title=f"Inference {_metric_label(metric)} vs Time",
                )
                if fig:
                    figures.append(fig)

        if pareto_split != "none":
            split_col = {
                "mode": "evidence_mode",
                "task": "task",
                "target_category": "target_category",
            }[pareto_split]
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
                    summary_style=summary_style,
                )
                for (qtype, split_val), group in split_summary.groupby(
                    ["query_type", split_col], dropna=False
                ):
                    tag = _safe_tag(f"{split_col}_{split_val}")
                    for metric in pareto_metrics:
                        fig = _plot_pareto(
                            group,
                            metric=metric,
                            summary_style=summary_style,
                            out_dir=figures_dir,
                            filename=f"pareto_{qtype}_{metric}_vs_time__{tag}.png",
                            title=f"{qtype.capitalize()} {_metric_label(metric)} vs Time ({split_col}={split_val})",
                        )
                        if fig:
                            figures.append(fig)

    if include_time:
        if include_cpd:
            fig = _plot_error_vs_size(
                time_tables.get("cpd_mb", pd.DataFrame()),
                size_col="mb_size",
                metric="time",
                summary_style=summary_style,
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
                summary_style=summary_style,
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
                summary_style=summary_style,
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
                summary_style=summary_style,
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
                summary_style=summary_style,
                out_dir=figures_dir,
                title_prefix="CPD Time vs Evidence Size",
                filename_prefix="cpd_time_vs_evidence_size",
            )
            if fig:
                figures.append(fig)
        if include_inference:
            fig = _plot_error_vs_size(
                time_tables.get("inf_ev_size", pd.DataFrame()),
                size_col="evidence_size",
                metric="time",
                summary_style=summary_style,
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
                summary_style=summary_style,
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
                summary_style=summary_style,
                out_dir=figures_dir,
                title_prefix="Inference Time vs #Edges",
                filename_prefix="inference_time_vs_n_edges",
            )
            if fig:
                figures.append(fig)

    category_specs = []
    if include_cpd:
        category_specs.append(
            (
                cpd_by_target,
                "target_category",
                "CPD",
                "cpd",
                "target_category",
                "Target Category",
                CPD_TARGET_CATEGORIES,
            )
        )
    if include_inference:
        category_specs.extend(
            [
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
        )
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
                summary_style=summary_style,
                out_dir=figures_dir,
                filename_prefix=f"{file_prefix}_{metric}_by_{file_suffix}",
                title_prefix=f"{title_prefix} {_metric_label(metric)} by {title_suffix}",
                category_order=category_order,
            )
            if fig:
                figures.append(fig)

    if include_time:
        if include_cpd:
            fig = _plot_category_bars(
                time_tables.get("cpd_by_target", pd.DataFrame()),
                category_col="target_category",
                metric="time",
                summary_style=summary_style,
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
                summary_style=summary_style,
                out_dir=figures_dir,
                filename_prefix="cpd_time_by_evidence_strategy",
                title_prefix="CPD Time by Evidence Strategy",
                category_order=CPD_EVIDENCE_STRATEGIES,
            )
            if fig:
                figures.append(fig)
        if include_inference:
            fig = _plot_category_bars(
                time_tables.get("inf_by_target", pd.DataFrame()),
                category_col="target_category",
                metric="time",
                summary_style=summary_style,
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
                summary_style=summary_style,
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
                summary_style=summary_style,
                out_dir=figures_dir,
                filename_prefix="inference_time_by_evidence_mode",
                title_prefix="Inference Time by Evidence Mode",
                category_order=INF_EVIDENCE_MODES,
            )
            if fig:
                figures.append(fig)

    # Success rate line plots2 for numeric axes
    if not success_line_nodes.empty:
        for mode in sorted(success_line_nodes["mode"].dropna().unique()):
            sub = success_line_nodes[success_line_nodes["mode"] == mode]
            tag = "cpd" if mode == "cpds" else str(mode)
            title_label = "CPD" if mode == "cpds" else str(mode).capitalize()
            fig = plot_success_rate_line(
                sub,
                x_label="n_nodes",
                out_dir=figures_dir,
                filename=f"{tag}_success_rate_vs_n_nodes.png",
                title=f"{title_label} Success Rate vs #Nodes",
            )
            if fig:
                figures.append(fig)

    if not success_line_edges.empty:
        for mode in sorted(success_line_edges["mode"].dropna().unique()):
            sub = success_line_edges[success_line_edges["mode"] == mode]
            tag = "cpd" if mode == "cpds" else str(mode)
            title_label = "CPD" if mode == "cpds" else str(mode).capitalize()
            fig = plot_success_rate_line(
                sub,
                x_label="n_edges",
                out_dir=figures_dir,
                filename=f"{tag}_success_rate_vs_n_edges.png",
                title=f"{title_label} Success Rate vs #Edges",
            )
            if fig:
                figures.append(fig)

    if not success_line_ev_size.empty:
        fig = plot_success_rate_line(
            success_line_ev_size,
            x_label="evidence_size",
            out_dir=figures_dir,
            filename="inference_success_rate_vs_evidence_size.png",
            title="Inference Success Rate vs Evidence Size",
        )
        if fig:
            figures.append(fig)
    if success_line_ev_size_by_mode:
        for mode, table in success_line_ev_size_by_mode.items():
            if table.empty:
                continue
            tag = _safe_tag(str(mode))
            fig = plot_success_rate_line(
                table,
                x_label="evidence_size",
                out_dir=figures_dir,
                filename=f"inference_success_rate_vs_evidence_size__mode_{tag}.png",
                title=f"Inference Success Rate vs Evidence Size (mode={mode})",
            )
            if fig:
                figures.append(fig)

    if not success_target.empty:
        for qtype in sorted(success_target["query_type"].dropna().unique()):
            sub = success_target[success_target["query_type"] == qtype]
            order = CPD_TARGET_CATEGORIES if qtype == "cpd" else INF_TARGET_CATEGORIES
            fig = plot_success_rate_bar(
                sub,
                category_col="target_category",
                out_dir=figures_dir,
                filename_prefix=f"{qtype}_success_rate_by_target_category",
                title_prefix=f"{qtype.capitalize()} Success Rate by Target Category",
                category_order=order,
            )
            if fig:
                figures.append(fig)

    if not success_strategy.empty:
        for qtype in sorted(success_strategy["query_type"].dropna().unique()):
            sub = success_strategy[success_strategy["query_type"] == qtype]
            order = CPD_EVIDENCE_STRATEGIES if qtype == "cpd" else None
            fig = plot_success_rate_bar(
                sub,
                category_col="evidence_strategy",
                out_dir=figures_dir,
                filename_prefix=f"{qtype}_success_rate_by_evidence_strategy",
                title_prefix=f"{qtype.capitalize()} Success Rate by Evidence Strategy",
                category_order=order,
            )
            if fig:
                figures.append(fig)

    if not success_task.empty:
        sub = success_task[success_task["query_type"] == "inference"]
        fig = plot_success_rate_bar(
            sub,
            category_col="task",
            out_dir=figures_dir,
            filename_prefix="inference_success_rate_by_task",
            title_prefix="Inference Success Rate by Task",
            category_order=INF_TASKS,
        )
        if fig:
            figures.append(fig)

    if not success_mode.empty:
        sub = success_mode[success_mode["query_type"] == "inference"]
        fig = plot_success_rate_bar(
            sub,
            category_col="evidence_mode",
            out_dir=figures_dir,
            filename_prefix="inference_success_rate_by_evidence_mode",
            title_prefix="Inference Success Rate by Evidence Mode",
            category_order=INF_EVIDENCE_MODES,
        )
        if fig:
            figures.append(fig)

    if include_coverage:
        if not coverage_nodes.empty:
            for mode in sorted(coverage_nodes["mode"].dropna().unique()):
                sub = coverage_nodes[coverage_nodes["mode"] == mode]
                tag = "cpd" if mode == "cpds" else str(mode)
                title_label = "CPD" if mode == "cpds" else str(mode).capitalize()
                fig = plot_coverage_line(
                    sub,
                    x_label="n_nodes",
                    out_dir=figures_dir,
                    filename=f"{tag}_coverage_vs_n_nodes.png",
                    title=f"{title_label} Coverage vs #Nodes",
                )
                if fig:
                    figures.append(fig)
        if not coverage_edges.empty:
            for mode in sorted(coverage_edges["mode"].dropna().unique()):
                sub = coverage_edges[coverage_edges["mode"] == mode]
                tag = "cpd" if mode == "cpds" else str(mode)
                title_label = "CPD" if mode == "cpds" else str(mode).capitalize()
                fig = plot_coverage_line(
                    sub,
                    x_label="n_edges",
                    out_dir=figures_dir,
                    filename=f"{tag}_coverage_vs_n_edges.png",
                    title=f"{title_label} Coverage vs #Edges",
                )
                if fig:
                    figures.append(fig)
        if not coverage_ev_size.empty:
            fig = plot_coverage_line(
                coverage_ev_size,
                x_label="evidence_size",
                out_dir=figures_dir,
                filename="inference_coverage_vs_evidence_size.png",
                title="Inference Coverage vs Evidence Size",
            )
            if fig:
                figures.append(fig)

    # Error plots2
    if not error_df.empty:
        for qtype in sorted(error_df["query_type"].dropna().unique()):
            sub = error_df[error_df["query_type"] == qtype]
            fig = _plot_error_type_distribution(
                sub,
                out_dir=figures_dir,
                filename=f"errors_by_type_{qtype}.png",
                title=f"{qtype.capitalize()} Error Type Distribution",
            )
            if fig:
                figures.append(fig)

    if not figures:
        figures = sorted(figures_dir.glob("*.png"))
    _write_report_md(out_dir, tables, figures)
    return tables, figures


def _generate_partition_reports(
    *,
    df: pd.DataFrame,
    attempts_df: pd.DataFrame,
    out_dir: Path,
    summary_style: SummaryStyle,
    include_time: bool,
    include_pareto: bool,
    pareto_split: str,
    include_all_methods_in_subsets: bool,
    allowed_query_types: set[str] | None,
    min_partition_queries: int,
    max_subsets: int | None,
    mode_label: str,
    run_dir: Path,
    problem_id: str | None = None,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    partition_sets, inventory_df = compute_partitions(
        attempts_df,
        min_partition_queries=int(min_partition_queries),
        max_subsets=max_subsets,
    )

    all_keys = (
        set(attempts_df["query_key"].dropna().unique())
        if not attempts_df.empty
        else set()
    )
    partition_sets.setdefault("all", all_keys)
    partition_sets.setdefault("common", set())

    methods = sorted(
        attempts_df["method_id"].dropna().unique()
        if not attempts_df.empty
        else df.get("method_id", pd.Series(dtype=object)).dropna().unique()
    )

    generate_report_for_partition(
        df=df,
        attempts_df=attempts_df,
        out_dir=out_dir / "all",
        summary_style=summary_style,
        include_time=include_time,
        include_pareto=include_pareto,
        pareto_split=pareto_split,
        include_coverage=False,
        allowed_query_types=allowed_query_types,
        methods_to_show=methods,
    )

    _write_partition_inventory(
        out_dir,
        mode_label=mode_label,
        inventory_df=inventory_df,
        methods=methods,
    )

    for _, row in inventory_df.iterrows():
        partition_name = row.get("partition_name")
        if not partition_name or partition_name == "all":
            continue
        keys = partition_sets.get(partition_name, set())
        partition_out = out_dir / str(partition_name)
        df_part = df[df["query_key"].isin(keys)] if not df.empty else df
        attempts_part = (
            attempts_df[attempts_df["query_key"].isin(keys)]
            if not attempts_df.empty
            else attempts_df
        )
        partition_type = row.get("partition_type")
        solver_label = row.get("solver_set") or ""
        solver_set = [s for s in str(solver_label).split("|") if s]
        if (
            partition_type == "subset"
            and not include_all_methods_in_subsets
            and solver_set
        ):
            methods_to_show = solver_set
        else:
            methods_to_show = methods

        generate_report_for_partition(
            df=df_part,
            attempts_df=attempts_part,
            out_dir=partition_out,
            summary_style=summary_style,
            include_time=include_time,
            include_pareto=include_pareto,
            pareto_split=pareto_split,
            include_coverage=True,
            allowed_query_types=allowed_query_types,
            methods_to_show=methods_to_show,
        )

        if partition_type == "subset":
            n_queries = int(row.get("n_queries", 0))
            share_of_total = float(row.get("share_of_total", 0.0))
            share_of_non_common = float(row.get("share_of_non_common", 0.0))
            meta = {
                "solver_set": solver_set,
                "n_queries": n_queries,
                "share_of_total": share_of_total,
                "share_of_non_common": share_of_non_common,
                "notes": "Queries in ALL but not in COMMON with exact solver ensemble",
            }
            (partition_out / "subset_meta.json").write_text(
                json.dumps(meta, indent=2, sort_keys=True)
            )

    _write_report_index(
        out_dir,
        run_dir=run_dir,
        mode_label=mode_label,
        methods=methods,
        inventory_df=inventory_df,
        problem_id=problem_id,
    )

    return inventory_df


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
        help="Include time tables and plots2 (default: true)",
    )
    parser.add_argument(
        "--no-include_time",
        dest="include_time",
        action="store_false",
        help="Disable time tables and plots2",
    )
    parser.add_argument(
        "--include_pareto",
        action="store_true",
        default=True,
        help="Include Pareto plots2 (default: true)",
    )
    parser.add_argument(
        "--no-include_pareto",
        dest="include_pareto",
        action="store_false",
        help="Disable Pareto plots2",
    )
    parser.add_argument(
        "--pareto_split",
        type=str,
        default="none",
        choices=["none", "mode", "task", "target_category"],
    )
    parser.add_argument(
        "--min_partition_queries",
        type=int,
        default=1,
        help="Minimum queries required to emit a subset partition (default: 1)",
    )
    parser.add_argument(
        "--max_subsets",
        type=int,
        default=None,
        help="Limit subset partitions to top-K by size (default: all)",
    )
    parser.add_argument(
        "--max_partitions",
        type=int,
        default=None,
        help="Deprecated alias for --max_subsets",
    )
    parser.add_argument(
        "--include_all_methods_in_subsets",
        action="store_true",
        default=False,
        help="Include all methods in subset reports (default: false)",
    )
    parser.add_argument(
        "--summary_style",
        type=str,
        default="robust",
        choices=sorted(SUMMARY_STYLES),
        help="Summary style for aggregate metrics (robust or mean)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(level=args.log_level)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "report"
    aggregate_dir = ensure_dir(out_dir / "aggregate")
    single_dir = ensure_dir(out_dir / "single")
    by_category_dir = ensure_dir(out_dir / "by_category")

    summary_style = SUMMARY_STYLES.get(args.summary_style)
    if summary_style is None:
        raise SystemExit(f"Unknown summary_style: {args.summary_style}")

    run_meta = _read_run_metadata(run_dir)
    bundle_dir = run_meta.get("bundle_dir")
    bundle_path = Path(bundle_dir).resolve() if bundle_dir else None

    logging.info("Report run_dir=%s", run_dir)
    logging.info("Report out_dir=%s", out_dir)
    if bundle_path is not None:
        logging.info("Bundle dir=%s", bundle_path)
    logging.info("Summary style=%s", summary_style.name)

    model_filter = None
    if args.models:
        model_filter = {m.strip() for m in args.models.split(",") if m.strip()}

    run_mode = _detect_run_mode(run_dir)
    if run_mode:
        logging.info("Detected run mode: %s", run_mode)

    max_subsets = args.max_subsets
    if max_subsets is None and args.max_partitions is not None:
        max_subsets = args.max_partitions

    graph_stats = _compute_graph_stats(run_dir)
    df, errors = _build_records(
        run_dir=run_dir,
        gt_source=args.gt_source,
        gt_key=args.gt_key,
        max_records=args.max_records,
        eps=float(args.eps),
        model_filter=model_filter,
        graph_stats=graph_stats,
        bundle_dir=bundle_path,
    )
    attempts_df = _build_attempts(
        run_dir=run_dir,
        max_records=args.max_records,
        model_filter=model_filter,
        graph_stats=graph_stats,
    )
    category_map = _load_problem_categories(run_dir)
    df, attempts_df = _attach_problem_categories(df, attempts_df, category_map)
    if category_map:
        logging.info(
            "Loaded %s problem categories from static metadata", len(category_map)
        )

    if not df.empty and "query_type" in df.columns:
        df["query_type"] = df["query_type"].apply(_normalize_query_type)
    if not attempts_df.empty and "query_type" in attempts_df.columns:
        attempts_df["query_type"] = attempts_df["query_type"].apply(
            _normalize_query_type
        )

    if not df.empty:
        df["query_key"] = df.apply(build_query_key, axis=1)
    if not attempts_df.empty:
        attempts_df["query_key"] = attempts_df.apply(build_query_key, axis=1)

    query_type_filter, mode_label = _infer_query_type(run_mode, df, attempts_df)
    if query_type_filter:
        df_mode = df[df["query_type"] == query_type_filter] if not df.empty else df
        attempts_mode = (
            attempts_df[attempts_df["query_type"] == query_type_filter]
            if not attempts_df.empty
            else attempts_df
        )
    else:
        df_mode = df
        attempts_mode = attempts_df

    allowed_query_types = {query_type_filter} if query_type_filter else None

    _generate_partition_reports(
        df=df_mode,
        attempts_df=attempts_mode,
        out_dir=aggregate_dir,
        summary_style=summary_style,
        include_time=args.include_time,
        include_pareto=args.include_pareto,
        pareto_split=args.pareto_split,
        include_all_methods_in_subsets=args.include_all_methods_in_subsets,
        allowed_query_types=allowed_query_types,
        min_partition_queries=int(args.min_partition_queries),
        max_subsets=max_subsets,
        mode_label=mode_label,
        run_dir=run_dir,
    )

    problem_ids = sorted(
        set(df_mode.get("problem_id", pd.Series(dtype=object)).dropna().unique())
        | set(
            attempts_mode.get("problem_id", pd.Series(dtype=object)).dropna().unique()
        )
    )
    if problem_ids:
        for problem_id in tqdm(problem_ids, desc="Per-problem reports"):
            df_problem = (
                df_mode[df_mode["problem_id"] == problem_id]
                if not df_mode.empty
                else df_mode
            )
            attempts_problem = (
                attempts_mode[attempts_mode["problem_id"] == problem_id]
                if not attempts_mode.empty
                else attempts_mode
            )
            _generate_partition_reports(
                df=df_problem,
                attempts_df=attempts_problem,
                out_dir=single_dir / str(problem_id),
                summary_style=summary_style,
                include_time=args.include_time,
                include_pareto=args.include_pareto,
                pareto_split=args.pareto_split,
                include_all_methods_in_subsets=args.include_all_methods_in_subsets,
                allowed_query_types=allowed_query_types,
                min_partition_queries=int(args.min_partition_queries),
                max_subsets=max_subsets,
                mode_label=mode_label,
                run_dir=run_dir,
                problem_id=str(problem_id),
            )

    problem_categories = sorted(
        set(
            df_mode.get("problem_category", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
        )
        | set(
            attempts_mode.get("problem_category", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
        )
    )
    if problem_categories:
        for category in tqdm(problem_categories, desc="Per-category reports"):
            df_category = (
                df_mode[df_mode["problem_category"] == category]
                if not df_mode.empty
                else df_mode
            )
            attempts_category = (
                attempts_mode[attempts_mode["problem_category"] == category]
                if not attempts_mode.empty
                else attempts_mode
            )
            _generate_partition_reports(
                df=df_category,
                attempts_df=attempts_category,
                out_dir=by_category_dir / str(category),
                summary_style=summary_style,
                include_time=args.include_time,
                include_pareto=args.include_pareto,
                pareto_split=args.pareto_split,
                include_all_methods_in_subsets=args.include_all_methods_in_subsets,
                allowed_query_types=allowed_query_types,
                min_partition_queries=int(args.min_partition_queries),
                max_subsets=max_subsets,
                mode_label=mode_label,
                run_dir=run_dir,
            )

    _write_root_report_index(
        out_dir,
        run_dir=run_dir,
        summary_style=summary_style,
        problem_ids=problem_ids,
        problem_categories=problem_categories,
    )

    if errors:
        logging.warning("Encountered %s metric errors", len(errors))
    return


if __name__ == "__main__":
    main()
