from __future__ import annotations

import gc
import hashlib
import json
import logging
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import pandas as pd
from tqdm.auto import tqdm
from vbn.utils.device_logging import log_device

from benchmarking.bundles import BenchmarkBundle
from benchmarking.models import get_benchmark_model
from benchmarking.models.config import apply_overrides, config_hash
from benchmarking.models.presets import get_preset_config
from benchmarking.utils import ensure_dir, get_generator_out_dir, timed_call, write_json
from benchmarking.utils_errors import (
    classify_error,
    ErrorSummary,
    render_error_summary_md,
)
from benchmarking.utils_logging import setup_logging


@dataclass(frozen=True)
class ProblemAssets:
    problem: str
    dag: Any
    domain: dict
    data_df: pd.DataFrame
    data_path: Path
    queries: dict
    queries_path: Path
    dataset_dir: Path


@dataclass(frozen=True)
class ProblemLoadResult:
    problem: str
    assets: ProblemAssets | None
    skipped: bool
    reason: str | None


def _batch_group_key(
    query: dict, default_dataset_id: str | None = None
) -> tuple[str | None, str | None, str | None, str] | None:
    if not isinstance(query, dict):
        return None
    evidence = query.get("evidence") if isinstance(query.get("evidence"), dict) else {}
    skeleton_id = evidence.get("skeleton_id") or query.get("skeleton_id")
    if not skeleton_id:
        return None
    dataset_id = query.get("dataset_id") or default_dataset_id
    target = query.get("target")
    task = query.get("task")
    return (dataset_id, target, task, skeleton_id)


def _iter_inference_batches(
    queries: list[dict],
    batch_size: int,
    *,
    default_dataset_id: str | None = None,
) -> Iterator[list[tuple[int, dict]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_size == 1:
        for idx, query in enumerate(queries):
            yield [(idx, query)]
        return

    current: list[tuple[int, dict]] = []
    current_key: tuple[str | None, str | None, str | None, str] | None = None

    def _flush_group(group: list[tuple[int, dict]]) -> Iterable[list[tuple[int, dict]]]:
        for start in range(0, len(group), batch_size):
            yield group[start : start + batch_size]

    for idx, query in enumerate(queries):
        key = _batch_group_key(query, default_dataset_id=default_dataset_id)
        if key is None:
            if current:
                yield from _flush_group(current)
                current = []
                current_key = None
            yield [(idx, query)]
            continue
        if current_key is None:
            current_key = key
        if key != current_key:
            if current:
                yield from _flush_group(current)
            current = []
            current_key = key
        current.append((idx, query))

    if current:
        yield from _flush_group(current)


def _should_use_batched_inference(model: Any, batch_size: int, chunk_size: int) -> bool:
    return (
        batch_size > 1
        and chunk_size > 1
        and bool(getattr(model, "supports_batched_inference_queries", False))
    )


def _execute_inference_chunk(
    model: Any, queries: list[dict], *, batch_size: int
) -> list[dict]:
    if _should_use_batched_inference(model, batch_size, len(queries)):
        return list(model.answer_inference_queries(queries))
    return [model.answer_inference_query(query) for query in queries]


class _P2Quantile:
    def __init__(self, q: float = 0.5) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError("q must be between 0 and 1")
        self.q = float(q)
        self._init: list[float] = []
        self.n = 0
        self.qs: list[float] = []
        self.ni: list[int] = []
        self.np: list[float] = []
        self.dn: list[float] = []

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        if len(self._init) < 5:
            self._init.append(x)
            if len(self._init) == 5:
                self._init.sort()
                self.qs = list(self._init)
                self.ni = [1, 2, 3, 4, 5]
                self.np = [
                    1.0,
                    1.0 + 2.0 * self.q,
                    1.0 + 4.0 * self.q,
                    3.0 + 2.0 * self.q,
                    5.0,
                ]
                self.dn = [0.0, self.q / 2.0, self.q, (1.0 + self.q) / 2.0, 1.0]
            return

        k = 0
        if x < self.qs[0]:
            self.qs[0] = x
            k = 0
        elif x >= self.qs[4]:
            self.qs[4] = x
            k = 3
        else:
            for i in range(4):
                if self.qs[i] <= x < self.qs[i + 1]:
                    k = i
                    break

        for i in range(k + 1, 5):
            self.ni[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        for i in range(1, 4):
            d = self.np[i] - self.ni[i]
            if (d >= 1 and self.ni[i + 1] - self.ni[i] > 1) or (
                d <= -1 and self.ni[i - 1] - self.ni[i] < -1
            ):
                di = 1 if d > 0 else -1
                qhat = self.qs[i] + (
                    di
                    / (self.ni[i + 1] - self.ni[i - 1])
                    * (
                        (self.ni[i] - self.ni[i - 1] + di)
                        * (self.qs[i + 1] - self.qs[i])
                        / (self.ni[i + 1] - self.ni[i])
                        + (self.ni[i + 1] - self.ni[i] - di)
                        * (self.qs[i] - self.qs[i - 1])
                        / (self.ni[i] - self.ni[i - 1])
                    )
                )
                if self.qs[i - 1] < qhat < self.qs[i + 1]:
                    self.qs[i] = qhat
                else:
                    self.qs[i] = self.qs[i] + di * (self.qs[i + di] - self.qs[i]) / (
                        self.ni[i + di] - self.ni[i]
                    )
                self.ni[i] += di

    def value(self) -> float | None:
        if self.n == 0:
            return None
        if len(self._init) < 5:
            return float(sorted(self._init)[len(self._init) // 2])
        return float(self.qs[2])


class _StreamingStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.median = _P2Quantile(0.5)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += float(value)
        self.median.add(float(value))

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0, "avg_ms": None, "median_ms": None}
        return {
            "count": int(self.count),
            "avg_ms": float(self.total / self.count),
            "median_ms": self.median.value(),
        }


class BaseBenchmarkRunner(ABC):
    generator: str

    def __init__(
        self,
        *,
        root: Path,
        bundle: BenchmarkBundle,
        seed: int,
        mode: str,
        models: list[str],
        model_kwargs: dict | None = None,
        model_configs: dict[str, str] | None = None,
        model_aliases: dict[str, str] | None = None,
        config_overrides: dict | None = None,
        max_problems: int | None = None,
        store_full_query: bool = False,
        progress: bool = True,
        batch_size_queries: int = 1,
        log_level: str = "INFO",
    ) -> None:
        if not getattr(self, "generator", None):
            raise ValueError("Benchmark runner must define 'generator'.")
        self.root = Path(root).resolve()
        self.bundle = bundle
        self.seed = int(seed)
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in {"cpds", "inference"}:
            raise ValueError(
                "mode must be one of {'cpds','inference'} " f"(got '{mode}')"
            )
        self.mode = resolved_mode
        self.models = list(models)
        self.model_kwargs = dict(model_kwargs or {})
        self.model_configs = dict(model_configs or {})
        self.model_aliases = dict(model_aliases or {})
        self.config_overrides = dict(config_overrides or {})
        self.max_problems = max_problems
        self.store_full_query = bool(store_full_query)
        self.progress = bool(progress)
        self.batch_size_queries = int(batch_size_queries)
        if self.batch_size_queries < 1:
            raise ValueError("batch_size_queries must be >= 1")
        self.log_level = str(log_level).upper()
        self.logger = logging.getLogger(__name__)
        if self.bundle.spec.generator != self.generator:
            raise ValueError(
                f"Bundle generator mismatch: {self.bundle.spec.generator} != {self.generator}"
            )
        if self.bundle.spec.mode != self.mode:
            raise ValueError(
                f"Bundle mode mismatch: {self.bundle.spec.mode} != {self.mode}"
            )

    def _resolve_model_device(self, model) -> object | None:
        device = getattr(model, "device", None)
        if device is None:
            vbn_model = getattr(model, "_vbn", None)
            device = getattr(vbn_model, "device", None)
        if device is None:
            return None
        try:
            import torch

            if isinstance(device, str):
                return torch.device(device)
        except Exception:
            return None
        return device

    @abstractmethod
    def list_problem_dirs(self) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def load_problem_assets(self, dataset_dir: Path) -> ProblemLoadResult:
        raise NotImplementedError

    def _init_output_dir(self) -> tuple[Path, str]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_root = ensure_dir(get_generator_out_dir(self.root, self.generator))
        run_dir = ensure_dir(out_root / f"benchmark_{self.mode}_{timestamp}")
        ensure_dir(run_dir / "results")
        ensure_dir(run_dir / "logs")
        ensure_dir(run_dir / "configs")
        ensure_dir(run_dir / "errors")
        return run_dir, timestamp

    def _hash_kwargs(self) -> str:
        payload = json.dumps(self.model_kwargs, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def _git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.root),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            return None
        return None

    def _write_run_metadata(
        self,
        run_dir: Path,
        *,
        timestamp: str,
        datasets_run: list[str],
        counts: dict[str, Any] | None = None,
        git_commit: str | None = None,
    ) -> None:
        backends: list[str] = []
        preset_names: list[str] = []
        model_entries: list[dict] = []
        for model_name in self.models:
            base_model = self._base_model_name(model_name)
            config_id = self.model_configs.get(model_name, "default")
            backends.append(base_model)
            preset_names.append(config_id)
            model_entries.append(
                {
                    "alias": model_name,
                    "backend": base_model,
                    "preset": config_id,
                }
            )

        unique_backends = sorted(set(backends))
        unique_presets = sorted(set(preset_names))
        payload = {
            "mode": self.mode,
            "timestamp": timestamp,
            "generator": self.generator,
            "seed": int(self.seed),
            "bundle_dir": str(self.bundle.paths.root),
            "datasets_run": list(datasets_run),
            "models": model_entries,
        }
        if counts is not None:
            payload["counts"] = counts
        if git_commit:
            payload["git_commit"] = git_commit
        if self.bundle.spec.seeds:
            payload["bundle_seeds"] = dict(self.bundle.spec.seeds)
        if len(unique_backends) > 1:
            payload["preset_backends"] = unique_backends
        if len(unique_presets) > 1:
            payload["preset_names"] = unique_presets
        if unique_backends:
            payload["preset_backend"] = unique_backends[0]
        if unique_presets:
            payload["preset_name"] = unique_presets[0]
        write_json(run_dir / "run_metadata.json", payload)

    def _safe_model_tag(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def _base_model_name(self, alias: str) -> str:
        return self.model_aliases.get(alias, alias)

    def _model_info(
        self,
        model,
        *,
        config,
        config_hash_value: str,
        alias: str,
        backend: str,
    ) -> dict:
        return {
            "alias": alias,
            "backend": backend,
            "name": getattr(model, "name", "unknown"),
            "version": getattr(model, "version", None),
            "family": getattr(model, "family", "unknown"),
            "kwargs_hash": self._hash_kwargs(),
            "config_id": config.config_id,
            "config_hash": config_hash_value,
            "components": {
                "learning": {
                    "name": config.learning.name,
                    "key": config.learning.key,
                },
                "cpd": {
                    "name": config.cpd.name,
                    "key": config.cpd.key,
                },
                "inference": {
                    "name": config.inference.name,
                    "key": config.inference.key,
                },
            },
        }

    def _resolve_model_configs(self) -> tuple[dict[str, Any], dict[str, str]]:
        configs: Dict[str, Any] = {}
        hashes: Dict[str, str] = {}
        unknown_models = set(self.model_configs) - set(self.models)
        if unknown_models:
            raise ValueError(
                f"Config provided for unknown models: {sorted(unknown_models)}"
            )
        allowed_override_keys = set(self.models) | set(self.model_aliases.values())
        unknown_overrides = set(self.config_overrides) - allowed_override_keys
        if unknown_overrides:
            raise ValueError(
                f"Config overrides provided for unknown models: {sorted(unknown_overrides)}"
            )
        for model_name in self.models:
            config_id = self.model_configs.get(model_name, "default")
            base_model = self._base_model_name(model_name)
            base_config = get_preset_config(base_model, self.mode, config_id)
            base_overrides = self.config_overrides.get(base_model)
            alias_overrides = self.config_overrides.get(model_name)
            for label, overrides in (
                (base_model, base_overrides),
                (model_name, alias_overrides),
            ):
                if overrides is not None and not isinstance(overrides, dict):
                    raise ValueError(
                        f"Config overrides for '{label}' must be a JSON object"
                    )
            resolved = base_config
            if base_overrides:
                resolved = apply_overrides(resolved, base_overrides)
            if alias_overrides:
                resolved = apply_overrides(resolved, alias_overrides)
            configs[model_name] = resolved
            hashes[model_name] = config_hash(resolved)
        return configs, hashes

    def _problem_info(self, problem: str, dag) -> dict:
        return {
            "id": problem,
            "n_nodes": int(dag.number_of_nodes()),
            "n_edges": int(dag.number_of_edges()),
        }

    def _compact_query(self, query: dict, qtype: str, index: int, problem: str) -> dict:
        evidence = (
            query.get("evidence") if isinstance(query.get("evidence"), dict) else {}
        )
        vars_list = evidence.get("vars") or query.get("evidence_vars") or []
        evidence_strategy = query.get("evidence_strategy")
        if evidence_strategy is None and isinstance(evidence, dict):
            evidence_strategy = evidence.get("strategy") or evidence.get(
                "evidence_strategy"
            )
        evidence_mode = evidence.get("mode") or query.get("evidence_mode")
        values = evidence.get("values") if evidence else query.get("evidence_values")
        if values is None:
            values_kind = "none"
            n_instantiations = 0
        elif isinstance(values, dict):
            values_kind = "empty" if not values else "values"
            n_instantiations = 1
        else:
            values_kind = "unknown"
            n_instantiations = 1

        payload = {
            "type": qtype,
            "index": int(index),
            "id": f"{problem}::{qtype}::{index}",
            "target": query.get("target"),
            "target_category": query.get("target_category"),
            "evidence_strategy": evidence_strategy,
            "evidence_vars": list(vars_list),
            "evidence_mode": evidence_mode,
            "task": query.get("task") if qtype == "inference" else None,
            "evidence": {
                "vars": list(vars_list),
                "strategy": evidence_strategy,
                "mode": evidence_mode,
                "values_kind": values_kind,
                "n_instantiations": int(n_instantiations),
                "mc_id": query.get("mc_id"),
                "skeleton_id": query.get("skeleton_id"),
            },
        }
        if self.store_full_query:
            payload["full"] = query
        return payload

    def _normalize_output(self, result: dict | None) -> dict | None:
        if result is None:
            return None
        if isinstance(result, dict) and result.get("format"):
            output = dict(result)
            if output.get("format") in {"normal_params", "samples_1d"}:
                return output
            if (
                output.get("format") == "categorical_probs"
                and output.get("support") is None
                and output.get("k") is not None
            ):
                output["support"] = list(range(int(output["k"])))
            return output
        if isinstance(result, dict) and "probs" in result:
            k = result.get("k")
            if k is None and result.get("probs") is not None:
                k = len(result.get("probs") or [])
            output = {
                "format": "categorical_probs",
                "k": k,
                "probs": result.get("probs"),
                "support": result.get("support"),
            }
            if output.get("support") is None and k is not None:
                output["support"] = list(range(int(k)))
            return output
        return result

    def _stage_for_mode(self) -> dict[str, str]:
        if self.mode == "cpds":
            return {
                "query": "cpd_query",
                "batch": "cpd_query",
            }
        return {
            "query": "inference_query",
            "batch": "inference_batch",
        }

    def run_all(self) -> Path:
        run_dir, run_timestamp = self._init_output_dir()
        model_configs, model_config_hashes = self._resolve_model_configs()
        configs_dir = run_dir / "configs"
        for model_name, config in model_configs.items():
            model_tag = self._safe_model_tag(model_name)
            snapshot = config.to_dict()
            snapshot["config_hash"] = model_config_hashes[model_name]
            snapshot["run_key"] = config.run_key()
            write_json(configs_dir / f"{model_tag}.json", snapshot)

        run_id = run_dir.name
        timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        log_path = run_dir / "logs" / "run.log"
        setup_logging(level=self.log_level, log_path=log_path)
        self.logger = logging.getLogger(f"benchmark.{self.generator}")

        problem_dirs = self.list_problem_dirs()
        if self.max_problems is not None:
            problem_dirs = problem_dirs[: int(self.max_problems)]
        total_problems = len(problem_dirs)

        self.logger.info(
            "Benchmark run: bundle=%s mode=%s generator=%s models=%s problems=%s out=%s",
            self.bundle.paths.root,
            self.mode,
            self.generator,
            ",".join(self.models),
            total_problems,
            run_dir,
        )

        model_stats: Dict[str, dict] = {}
        error_summary = ErrorSummary()
        summary: dict = {
            "generator": self.generator,
            "seed": int(self.seed),
            "timestamp": timestamp_utc,
            "run_id": run_id,
            "mode": self.mode,
            "progress": bool(self.progress),
            "models": {},
            "run": {"progress": bool(self.progress)},
        }
        ground_truth_sources: dict[str, dict] = {}
        run_cpds = self.mode == "cpds"
        run_inference = self.mode == "inference"
        stage_map = self._stage_for_mode()
        results_root = ensure_dir(run_dir / "results")
        problem_counts: dict[str, dict[str, int]] = {}

        for model_name in self.models:
            model_stats[model_name] = {
                "problems": {"ok": 0, "skipped": 0, "error": 0},
                "queries": {
                    "cpd": {"ok": 0, "error": 0},
                    "inference": {"ok": 0, "error": 0},
                },
                "timing_ms": {"per_problem": {}},
            }
            model_stats[model_name]["_timing"] = {
                "cpd": _StreamingStats(),
                "inference": _StreamingStats(),
            }

        def _model_meta_placeholder(
            alias: str,
            backend: str,
            config,
            config_hash_value: str,
        ) -> dict:
            if config is None:
                components = {}
                config_id = None
            else:
                components = {
                    "learning": {
                        "name": config.learning.name,
                        "key": config.learning.key,
                    },
                    "cpd": {"name": config.cpd.name, "key": config.cpd.key},
                    "inference": {
                        "name": config.inference.name,
                        "key": config.inference.key,
                    },
                }
                config_id = config.config_id
            return {
                "alias": alias,
                "backend": backend,
                "name": backend,
                "version": None,
                "family": "unknown",
                "kwargs_hash": self._hash_kwargs(),
                "config_id": config_id,
                "config_hash": config_hash_value,
                "components": components,
            }

        def _write_failure_records(
            handle,
            *,
            queries: list[dict],
            qtype: str,
            model_meta: dict,
            problem_meta: dict,
            run_meta: dict,
            error_type: str,
            error_msg: str,
            error_stage: str,
        ) -> int:
            info = classify_error(error_type, error_msg)
            is_oom = info["is_oom"]
            error_signature = info["error_signature"]
            for q_index, query in enumerate(queries):
                record = {
                    "mode": self.mode,
                    "run": run_meta,
                    "model": model_meta,
                    "problem": problem_meta,
                    "query": self._compact_query(
                        query, qtype, q_index, problem_meta["id"]
                    ),
                    "result": {
                        "ok": False,
                        "error_type": error_type,
                        "error_msg": error_msg,
                        "error_stage": error_stage,
                        "is_oom": is_oom,
                        "error_signature": error_signature,
                        "timing_ms": 0.0,
                        "output": None,
                    },
                }
                if qtype == "inference":
                    record["batching"] = {
                        "enabled": False,
                        "batch_size": 1,
                        "skeleton_id": (query.get("evidence") or {}).get("skeleton_id")
                        or query.get("skeleton_id"),
                    }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            return len(queries)

        problem_iter = (
            tqdm(problem_dirs, desc="Problems", unit="problem")
            if self.progress
            else problem_dirs
        )
        for idx, dataset_dir in enumerate(problem_iter, start=1):
            load_result = self.load_problem_assets(dataset_dir)
            problem = load_result.problem
            if load_result.skipped or load_result.assets is None:
                for model_name in self.models:
                    model_stats[model_name]["problems"]["skipped"] += 1
                self.logger.warning(
                    "Skipping problem %s (%s/%s): %s",
                    problem,
                    idx,
                    total_problems,
                    load_result.reason,
                )
                continue

            assets = load_result.assets
            dag = assets.dag
            n_nodes = int(dag.number_of_nodes())
            n_edges = int(dag.number_of_edges())
            cpd_queries = assets.queries.get("cpd_queries", []) if run_cpds else []
            inf_queries = (
                assets.queries.get("inference_queries", []) if run_inference else []
            )
            problem_counts[problem] = {
                "cpd": int(len(cpd_queries)),
                "inference": int(len(inf_queries)),
                "total": int(len(cpd_queries) + len(inf_queries)),
            }
            gt_meta = assets.queries.get("ground_truth")
            if isinstance(gt_meta, dict):
                ground_truth_sources[problem] = dict(gt_meta)

            self.logger.info(
                "Problem %s (%s/%s): nodes=%s edges=%s cpd=%s inf=%s",
                problem,
                idx,
                total_problems,
                n_nodes,
                n_edges,
                len(cpd_queries),
                len(inf_queries),
            )

            problem_results_dir = ensure_dir(results_root / problem)
            model_iter = (
                tqdm(self.models, desc=f"{problem} models", unit="model", leave=False)
                if self.progress
                else self.models
            )

            for model_name in model_iter:
                model_start = time.perf_counter()
                ok_cpd = err_cpd = ok_inf = err_inf = 0
                cpd_timing_sum = 0.0
                inf_timing_sum = 0.0
                cpd_err_logged = 0
                inf_err_logged = 0
                max_logged_query_errors = 3

                base_model = self._base_model_name(model_name)
                model_config = model_configs[model_name]
                model_config_hash = model_config_hashes[model_name]
                model_meta: dict
                result_path = (
                    problem_results_dir / f"{self._safe_model_tag(model_name)}.jsonl"
                )

                run_meta = {
                    "generator": self.generator,
                    "seed": int(self.seed),
                    "timestamp_utc": timestamp_utc,
                    "run_id": run_id,
                    "mode": self.mode,
                    "progress": bool(self.progress),
                }
                problem_meta = self._problem_info(problem, assets.dag)

                with result_path.open("w", encoding="utf-8") as handle:
                    try:
                        model_cls = get_benchmark_model(base_model)
                        model = model_cls(
                            dag=assets.dag,
                            seed=self.seed,
                            domain=assets.domain,
                            benchmark_config=model_config,
                            **self.model_kwargs,
                        )
                        model_meta = self._model_info(
                            model,
                            config=model_config,
                            config_hash_value=model_config_hash,
                            alias=model_name,
                            backend=base_model,
                        )
                    except Exception as exc:
                        info = classify_error(type(exc).__name__, str(exc))
                        error_summary.add(
                            model=model_name,
                            problem=problem,
                            error_type=type(exc).__name__,
                            error_signature=info["error_signature"],
                            error_stage="model_init",
                            error_msg=str(exc),
                            is_oom=info["is_oom"],
                        )
                        model_stats[model_name]["problems"]["error"] += 1
                        model_meta = _model_meta_placeholder(
                            model_name, base_model, model_config, model_config_hash
                        )
                        if run_cpds:
                            err_cpd += _write_failure_records(
                                handle,
                                queries=cpd_queries,
                                qtype="cpd",
                                model_meta=model_meta,
                                problem_meta=problem_meta,
                                run_meta=run_meta,
                                error_type=type(exc).__name__,
                                error_msg=str(exc),
                                error_stage="model_init",
                            )
                        if run_inference:
                            err_inf += _write_failure_records(
                                handle,
                                queries=inf_queries,
                                qtype="inference",
                                model_meta=model_meta,
                                problem_meta=problem_meta,
                                run_meta=run_meta,
                                error_type=type(exc).__name__,
                                error_msg=str(exc),
                                error_stage="model_init",
                            )
                        model_stats[model_name]["queries"]["cpd"]["error"] += err_cpd
                        model_stats[model_name]["queries"]["inference"][
                            "error"
                        ] += err_inf
                        continue

                    try:
                        model_device = self._resolve_model_device(model)
                        log_device(self.logger, phase="fit_start", device=model_device)
                        _, _ = timed_call(
                            model.fit, assets.data_df, progress=self.progress
                        )
                    except Exception as exc:
                        info = classify_error(type(exc).__name__, str(exc))
                        error_summary.add(
                            model=model_name,
                            problem=problem,
                            error_type=type(exc).__name__,
                            error_signature=info["error_signature"],
                            error_stage="fit",
                            error_msg=str(exc),
                            is_oom=info["is_oom"],
                        )
                        model_stats[model_name]["problems"]["error"] += 1
                        if run_cpds:
                            err_cpd += _write_failure_records(
                                handle,
                                queries=cpd_queries,
                                qtype="cpd",
                                model_meta=model_meta,
                                problem_meta=problem_meta,
                                run_meta=run_meta,
                                error_type=type(exc).__name__,
                                error_msg=str(exc),
                                error_stage="fit",
                            )
                        if run_inference:
                            err_inf += _write_failure_records(
                                handle,
                                queries=inf_queries,
                                qtype="inference",
                                model_meta=model_meta,
                                problem_meta=problem_meta,
                                run_meta=run_meta,
                                error_type=type(exc).__name__,
                                error_msg=str(exc),
                                error_stage="fit",
                            )
                        model_stats[model_name]["queries"]["cpd"]["error"] += err_cpd
                        model_stats[model_name]["queries"]["inference"][
                            "error"
                        ] += err_inf
                        continue

                    if run_cpds:
                        cpd_bar = (
                            tqdm(
                                cpd_queries,
                                desc=f"{problem} | {model_name} | CPD",
                                leave=False,
                            )
                            if self.progress
                            else None
                        )
                        cpd_iter = cpd_bar if cpd_bar is not None else cpd_queries
                        for q_index, query in enumerate(cpd_iter):
                            start = time.perf_counter()
                            response = {}
                            try:
                                response = model.answer_cpd_query(query)
                                response = dict(response or {})
                                ok = bool(response.get("ok"))
                                error_msg = response.get("error")
                                error_type = None if not error_msg else "ModelError"
                                output = self._normalize_output(response.get("result"))
                            except Exception as exc:
                                if (
                                    self.logger.isEnabledFor(logging.DEBUG)
                                    and cpd_err_logged < max_logged_query_errors
                                ):
                                    self.logger.debug(
                                        "CPD query error model=%s problem=%s idx=%s: %s",
                                        model_name,
                                        problem,
                                        q_index,
                                        exc,
                                    )
                                    cpd_err_logged += 1
                                ok = False
                                error_type = type(exc).__name__
                                error_msg = str(exc)
                                output = None
                            if error_msg is not None and not isinstance(error_msg, str):
                                error_msg = str(error_msg)
                            if not ok and not error_type:
                                error_type = "ModelError"
                            error_stage = stage_map["query"]
                            is_oom = None
                            error_signature = None
                            if not ok and (error_msg or error_type):
                                info = classify_error(error_type, error_msg)
                                is_oom = info["is_oom"]
                                error_signature = info["error_signature"]
                                error_summary.add(
                                    model=model_name,
                                    problem=problem,
                                    error_type=error_type,
                                    error_signature=error_signature,
                                    error_stage=error_stage,
                                    error_msg=error_msg,
                                    is_oom=is_oom,
                                )
                            elapsed = (time.perf_counter() - start) * 1000.0
                            cpd_timing_sum += elapsed
                            if ok:
                                ok_cpd += 1
                            else:
                                err_cpd += 1
                            model_stats[model_name]["_timing"]["cpd"].add(
                                float(elapsed)
                            )

                            record = {
                                "mode": self.mode,
                                "run": run_meta,
                                "model": model_meta,
                                "problem": problem_meta,
                                "query": self._compact_query(
                                    query, "cpd", q_index, problem
                                ),
                                "result": {
                                    "ok": bool(ok),
                                    "error_type": error_type,
                                    "error_msg": error_msg,
                                    "error_stage": error_stage if not ok else None,
                                    "is_oom": is_oom if not ok else None,
                                    "error_signature": (
                                        error_signature if not ok else None
                                    ),
                                    "timing_ms": float(elapsed),
                                    "output": output,
                                },
                            }
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                            if cpd_bar is not None:
                                avg_ms = cpd_timing_sum / max(1, ok_cpd + err_cpd)
                                cpd_bar.set_postfix(
                                    ok=ok_cpd, err=err_cpd, avg=f"{avg_ms:.2f}"
                                )
                        if cpd_bar is not None:
                            cpd_bar.close()

                    if run_inference:
                        inf_bar = (
                            tqdm(
                                total=len(inf_queries),
                                desc=f"{problem} | {model_name} | Inference",
                                leave=False,
                            )
                            if self.progress
                            else None
                        )
                        batch_size_queries = int(getattr(self, "batch_size_queries", 1))
                        for chunk in _iter_inference_batches(
                            inf_queries,
                            batch_size_queries,
                            default_dataset_id=problem,
                        ):
                            if not chunk:
                                continue
                            batch_queries = [query for _, query in chunk]
                            use_batch = _should_use_batched_inference(
                                model, batch_size_queries, len(batch_queries)
                            )
                            if use_batch:
                                start = time.perf_counter()
                                responses: list[dict] | None = None
                                batch_error: Exception | None = None
                                try:
                                    responses = _execute_inference_chunk(
                                        model,
                                        batch_queries,
                                        batch_size=batch_size_queries,
                                    )
                                    if len(responses) != len(batch_queries):
                                        raise ValueError(
                                            "Batched inference returned unexpected response count"
                                        )
                                except Exception as exc:
                                    batch_error = exc
                                    responses = None
                                elapsed = (time.perf_counter() - start) * 1000.0
                                per_query_ms = elapsed / max(1, len(batch_queries))
                                if batch_error is not None:
                                    if (
                                        self.logger.isEnabledFor(logging.DEBUG)
                                        and inf_err_logged < max_logged_query_errors
                                    ):
                                        self.logger.debug(
                                            "Inference batch error model=%s problem=%s: %s",
                                            model_name,
                                            problem,
                                            batch_error,
                                        )
                                        inf_err_logged += 1
                                    for q_index, query in chunk:
                                        error_type = type(batch_error).__name__
                                        error_msg = str(batch_error)
                                        if error_msg is not None and not isinstance(
                                            error_msg, str
                                        ):
                                            error_msg = str(error_msg)
                                        ok = False
                                        output = None
                                        error_stage = stage_map["batch"]
                                        info = classify_error(error_type, error_msg)
                                        is_oom = info["is_oom"]
                                        error_signature = info["error_signature"]
                                        error_summary.add(
                                            model=model_name,
                                            problem=problem,
                                            error_type=error_type,
                                            error_signature=error_signature,
                                            error_stage=error_stage,
                                            error_msg=error_msg,
                                            is_oom=is_oom,
                                        )
                                        inf_timing_sum += per_query_ms
                                        err_inf += 1
                                        model_stats[model_name]["_timing"][
                                            "inference"
                                        ].add(float(per_query_ms))
                                        skeleton_id = (query.get("evidence") or {}).get(
                                            "skeleton_id"
                                        ) or query.get("skeleton_id")
                                        record = {
                                            "mode": self.mode,
                                            "run": run_meta,
                                            "model": model_meta,
                                            "problem": problem_meta,
                                            "query": self._compact_query(
                                                query, "inference", q_index, problem
                                            ),
                                            "result": {
                                                "ok": bool(ok),
                                                "error_type": error_type,
                                                "error_msg": error_msg,
                                                "error_stage": error_stage,
                                                "is_oom": is_oom,
                                                "error_signature": error_signature,
                                                "timing_ms": float(per_query_ms),
                                                "output": output,
                                            },
                                            "batching": {
                                                "enabled": True,
                                                "batch_size": int(len(batch_queries)),
                                                "skeleton_id": skeleton_id,
                                            },
                                        }
                                        handle.write(
                                            json.dumps(record, sort_keys=True) + "\n"
                                        )
                                else:
                                    for (q_index, query), response in zip(
                                        chunk, responses or []
                                    ):
                                        if isinstance(response, dict):
                                            response = dict(response or {})
                                        else:
                                            response = {}
                                        ok = bool(response.get("ok"))
                                        error_msg = response.get("error")
                                        error_type = (
                                            None if not error_msg else "ModelError"
                                        )
                                        output = self._normalize_output(
                                            response.get("result")
                                        )
                                        if error_msg is not None and not isinstance(
                                            error_msg, str
                                        ):
                                            error_msg = str(error_msg)
                                        if not ok and not error_type:
                                            error_type = "ModelError"
                                        error_stage = stage_map["query"]
                                        is_oom = None
                                        error_signature = None
                                        if not ok and (error_msg or error_type):
                                            info = classify_error(error_type, error_msg)
                                            is_oom = info["is_oom"]
                                            error_signature = info["error_signature"]
                                            error_summary.add(
                                                model=model_name,
                                                problem=problem,
                                                error_type=error_type,
                                                error_signature=error_signature,
                                                error_stage=error_stage,
                                                error_msg=error_msg,
                                                is_oom=is_oom,
                                            )
                                        inf_timing_sum += per_query_ms
                                        if ok:
                                            ok_inf += 1
                                        else:
                                            err_inf += 1
                                        model_stats[model_name]["_timing"][
                                            "inference"
                                        ].add(float(per_query_ms))
                                        skeleton_id = (query.get("evidence") or {}).get(
                                            "skeleton_id"
                                        ) or query.get("skeleton_id")
                                        record = {
                                            "mode": self.mode,
                                            "run": run_meta,
                                            "model": model_meta,
                                            "problem": problem_meta,
                                            "query": self._compact_query(
                                                query, "inference", q_index, problem
                                            ),
                                            "result": {
                                                "ok": bool(ok),
                                                "error_type": error_type,
                                                "error_msg": error_msg,
                                                "error_stage": (
                                                    error_stage if not ok else None
                                                ),
                                                "is_oom": is_oom if not ok else None,
                                                "error_signature": (
                                                    error_signature if not ok else None
                                                ),
                                                "timing_ms": float(per_query_ms),
                                                "output": output,
                                            },
                                            "batching": {
                                                "enabled": True,
                                                "batch_size": int(len(batch_queries)),
                                                "skeleton_id": skeleton_id,
                                            },
                                        }
                                        handle.write(
                                            json.dumps(record, sort_keys=True) + "\n"
                                        )
                            else:
                                for q_index, query in chunk:
                                    start = time.perf_counter()
                                    response = {}
                                    try:
                                        response = model.answer_inference_query(query)
                                        response = dict(response or {})
                                        ok = bool(response.get("ok"))
                                        error_msg = response.get("error")
                                        error_type = (
                                            None if not error_msg else "ModelError"
                                        )
                                        output = self._normalize_output(
                                            response.get("result")
                                        )
                                    except Exception as exc:
                                        if (
                                            self.logger.isEnabledFor(logging.DEBUG)
                                            and inf_err_logged < max_logged_query_errors
                                        ):
                                            self.logger.debug(
                                                "Inference query error model=%s problem=%s idx=%s: %s",
                                                model_name,
                                                problem,
                                                q_index,
                                                exc,
                                            )
                                            inf_err_logged += 1
                                        ok = False
                                        error_type = type(exc).__name__
                                        error_msg = str(exc)
                                        output = None
                                    if error_msg is not None and not isinstance(
                                        error_msg, str
                                    ):
                                        error_msg = str(error_msg)
                                    if not ok and not error_type:
                                        error_type = "ModelError"
                                    error_stage = stage_map["query"]
                                    is_oom = None
                                    error_signature = None
                                    if not ok and (error_msg or error_type):
                                        info = classify_error(error_type, error_msg)
                                        is_oom = info["is_oom"]
                                        error_signature = info["error_signature"]
                                        error_summary.add(
                                            model=model_name,
                                            problem=problem,
                                            error_type=error_type,
                                            error_signature=error_signature,
                                            error_stage=error_stage,
                                            error_msg=error_msg,
                                            is_oom=is_oom,
                                        )
                                    elapsed = (time.perf_counter() - start) * 1000.0
                                    inf_timing_sum += elapsed
                                    if ok:
                                        ok_inf += 1
                                    else:
                                        err_inf += 1
                                    model_stats[model_name]["_timing"]["inference"].add(
                                        float(elapsed)
                                    )

                                    skeleton_id = (query.get("evidence") or {}).get(
                                        "skeleton_id"
                                    ) or query.get("skeleton_id")
                                    record = {
                                        "mode": self.mode,
                                        "run": run_meta,
                                        "model": model_meta,
                                        "problem": problem_meta,
                                        "query": self._compact_query(
                                            query, "inference", q_index, problem
                                        ),
                                        "result": {
                                            "ok": bool(ok),
                                            "error_type": error_type,
                                            "error_msg": error_msg,
                                            "error_stage": (
                                                error_stage if not ok else None
                                            ),
                                            "is_oom": is_oom if not ok else None,
                                            "error_signature": (
                                                error_signature if not ok else None
                                            ),
                                            "timing_ms": float(elapsed),
                                            "output": output,
                                        },
                                        "batching": {
                                            "enabled": False,
                                            "batch_size": 1,
                                            "skeleton_id": skeleton_id,
                                        },
                                    }
                                    handle.write(
                                        json.dumps(record, sort_keys=True) + "\n"
                                    )
                            if inf_bar is not None:
                                avg_ms = inf_timing_sum / max(1, ok_inf + err_inf)
                                inf_bar.update(len(chunk))
                                inf_bar.set_postfix(
                                    ok=ok_inf, err=err_inf, avg=f"{avg_ms:.2f}"
                                )
                        if inf_bar is not None:
                            inf_bar.close()

                    total_ms = (time.perf_counter() - model_start) * 1000.0
                    model_stats[model_name]["problems"]["ok"] += 1
                    model_stats[model_name]["queries"]["cpd"]["ok"] += ok_cpd
                    model_stats[model_name]["queries"]["cpd"]["error"] += err_cpd
                    model_stats[model_name]["queries"]["inference"]["ok"] += ok_inf
                    model_stats[model_name]["queries"]["inference"]["error"] += err_inf
                    model_stats[model_name]["timing_ms"]["per_problem"][problem] = (
                        float(total_ms)
                    )

            del assets
            gc.collect()

        for model_name in self.models:
            stats = model_stats[model_name]
            timing = stats.pop("_timing")
            cpd_summary = timing["cpd"].summary()
            inf_summary = timing["inference"].summary()
            stats["queries"]["cpd"].update(
                {
                    "avg_ms": cpd_summary["avg_ms"],
                    "median_ms": cpd_summary["median_ms"],
                }
            )
            stats["queries"]["inference"].update(
                {
                    "avg_ms": inf_summary["avg_ms"],
                    "median_ms": inf_summary["median_ms"],
                }
            )
            stats["timing_ms"]["total_ms"] = float(
                sum(stats["timing_ms"]["per_problem"].values())
                if stats["timing_ms"]["per_problem"]
                else 0.0
            )
            summary["models"][model_name] = stats

        if ground_truth_sources:
            summary["ground_truth"] = ground_truth_sources
            write_json(run_dir / "ground_truth_sources.json", ground_truth_sources)

        summary_path = run_dir / "summary.json"
        write_json(summary_path, summary)

        counts_payload = {
            "n_problems": int(total_problems),
            "queries_per_problem": problem_counts,
            "queries_total": int(
                sum(entry.get("total", 0) for entry in problem_counts.values())
            ),
        }
        self._write_run_metadata(
            run_dir,
            timestamp=run_timestamp,
            datasets_run=[p.name for p in problem_dirs],
            counts=counts_payload,
            git_commit=self._git_commit(),
        )

        errors_payload = error_summary.to_dict()
        write_json(run_dir / "errors" / "errors_summary.json", errors_payload)
        (run_dir / "errors" / "errors_summary.md").write_text(
            render_error_summary_md(errors_payload)
        )
        return run_dir
