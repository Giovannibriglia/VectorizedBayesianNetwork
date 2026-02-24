from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm.auto import tqdm

from benchmarking.models import get_benchmark_model
from benchmarking.utils import ensure_dir, get_generator_out_dir, timed_call, write_json


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
        seed: int,
        models: list[str],
        model_kwargs: dict | None = None,
        max_problems: int | None = None,
        store_full_query: bool = False,
        progress: bool = True,
    ) -> None:
        if not getattr(self, "generator", None):
            raise ValueError("Benchmark runner must define 'generator'.")
        self.root = Path(root).resolve()
        self.seed = int(seed)
        self.models = list(models)
        self.model_kwargs = dict(model_kwargs or {})
        self.max_problems = max_problems
        self.store_full_query = bool(store_full_query)
        self.progress = bool(progress)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def list_problem_dirs(self) -> list[Path]:
        raise NotImplementedError

    @abstractmethod
    def load_problem_assets(self, dataset_dir: Path) -> ProblemLoadResult:
        raise NotImplementedError

    def _init_output_dir(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_root = ensure_dir(get_generator_out_dir(self.root, self.generator))
        run_dir = ensure_dir(out_root / f"benchmark_{timestamp}")
        ensure_dir(run_dir / "cpds")
        ensure_dir(run_dir / "inference")
        ensure_dir(run_dir / "logs")
        return run_dir

    def _hash_kwargs(self) -> str:
        payload = json.dumps(self.model_kwargs, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def _model_info(self, model) -> dict:
        return {
            "name": getattr(model, "name", "unknown"),
            "version": getattr(model, "version", None),
            "family": getattr(model, "family", "unknown"),
            "kwargs_hash": self._hash_kwargs(),
        }

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
            "task": query.get("task") if qtype == "inference" else None,
            "evidence": {
                "vars": list(vars_list),
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

    def run_all(self) -> Path:
        run_dir = self._init_output_dir()
        run_id = run_dir.name
        timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        log_path = run_dir / "logs" / "run.log"
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        self.logger.info("Benchmark output: %s", run_dir)
        problem_dirs = self.list_problem_dirs()
        if self.max_problems is not None:
            problem_dirs = problem_dirs[: int(self.max_problems)]
        total_problems = len(problem_dirs)
        self.logger.info(
            "Run config: generator=%s seed=%s models=%s problems=%s",
            self.generator,
            self.seed,
            ",".join(self.models),
            total_problems,
        )
        model_files: Dict[str, dict] = {}
        model_stats: Dict[str, dict] = {}
        summary: dict = {
            "generator": self.generator,
            "seed": int(self.seed),
            "timestamp": timestamp_utc,
            "run_id": run_id,
            "progress": bool(self.progress),
            "models": {},
            "run": {"progress": bool(self.progress)},
        }

        for model_name in self.models:
            cpd_path = run_dir / "cpds" / f"{model_name}.jsonl"
            inf_path = run_dir / "inference" / f"{model_name}.jsonl"
            model_files[model_name] = {
                "cpd": cpd_path.open("w", encoding="utf-8"),
                "inf": inf_path.open("w", encoding="utf-8"),
            }
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

        for idx, dataset_dir in enumerate(problem_dirs, start=1):
            load_result = self.load_problem_assets(dataset_dir)
            problem = load_result.problem
            if load_result.skipped or load_result.assets is None:
                for model_name in self.models:
                    model_stats[model_name]["problems"]["skipped"] += 1
                self.logger.info(
                    "Skipping problem %s (%s/%s): %s",
                    problem,
                    idx,
                    len(problem_dirs),
                    load_result.reason,
                )
                continue

            assets = load_result.assets
            dag = assets.dag
            n_nodes = int(dag.number_of_nodes())
            n_edges = int(dag.number_of_edges())
            cpd_queries = assets.queries.get("cpd_queries", [])
            inf_queries = assets.queries.get("inference_queries", [])
            self.logger.info(
                "Starting problem %s (%s/%s)",
                problem,
                idx,
                len(problem_dirs),
            )
            self.logger.info("DAG size: |V|=%s |E|=%s", n_nodes, n_edges)
            self.logger.info("Data shape: %s", assets.data_df.shape)
            self.logger.info("#CPD queries: %s", len(cpd_queries))
            self.logger.info("#Inference queries: %s", len(inf_queries))

            for model_name in self.models:
                model_start = time.perf_counter()
                ok_cpd = 0
                err_cpd = 0
                ok_inf = 0
                err_inf = 0
                cpd_timing_sum = 0.0
                inf_timing_sum = 0.0

                try:
                    model_cls = get_benchmark_model(model_name)
                    model = model_cls(
                        dag=assets.dag,
                        seed=self.seed,
                        domain=assets.domain,
                        **self.model_kwargs,
                    )
                except Exception as exc:
                    reason = f"Model init failed: {type(exc).__name__}: {exc}"
                    self.logger.info("[%s] %s", model_name, reason)
                    model_stats[model_name]["problems"]["error"] += 1
                    continue

                try:
                    self.logger.info("[%s] Fit start", model_name)
                    _, fit_ms = timed_call(
                        model.fit, assets.data_df, progress=self.progress
                    )
                    self.logger.info("[%s] Fit done in %.3f ms", model_name, fit_ms)
                except Exception as exc:
                    reason = f"Fit failed: {type(exc).__name__}: {exc}"
                    self.logger.info("[%s] %s", model_name, reason)
                    model_stats[model_name]["problems"]["error"] += 1
                    continue

                run_meta = {
                    "generator": self.generator,
                    "seed": int(self.seed),
                    "timestamp_utc": timestamp_utc,
                    "run_id": run_id,
                    "progress": bool(self.progress),
                }
                model_meta = self._model_info(model)
                problem_meta = self._problem_info(problem, assets.dag)

                cpd_desc = f"{self.generator}/{problem} | {model_name} | CPD"
                cpd_bar = (
                    tqdm(cpd_queries, desc=cpd_desc, leave=False)
                    if self.progress
                    else None
                )
                cpd_iter = cpd_bar if cpd_bar is not None else cpd_queries
                for q_index, query in enumerate(cpd_iter):
                    start = time.perf_counter()
                    try:
                        response = model.answer_cpd_query(query)
                        response = dict(response or {})
                        ok = bool(response.get("ok"))
                        error_msg = response.get("error")
                        error_type = None if not error_msg else "ModelError"
                        output = self._normalize_output(response.get("result"))
                    except Exception as exc:
                        ok = False
                        error_type = type(exc).__name__
                        error_msg = str(exc)
                        output = None
                    elapsed = (time.perf_counter() - start) * 1000.0
                    cpd_timing_sum += elapsed
                    if ok:
                        ok_cpd += 1
                    else:
                        err_cpd += 1
                    model_stats[model_name]["_timing"]["cpd"].add(float(elapsed))

                    record = {
                        "run": run_meta,
                        "model": model_meta,
                        "problem": problem_meta,
                        "query": self._compact_query(query, "cpd", q_index, problem),
                        "result": {
                            "ok": bool(ok),
                            "error_type": error_type,
                            "error_msg": error_msg,
                            "timing_ms": float(elapsed),
                            "output": output,
                        },
                    }
                    model_files[model_name]["cpd"].write(
                        json.dumps(record, sort_keys=True) + "\n"
                    )

                    avg_ms = cpd_timing_sum / max(1, ok_cpd + err_cpd)
                    if cpd_bar is not None:
                        cpd_bar.set_postfix(ok=ok_cpd, err=err_cpd, avg=f"{avg_ms:.2f}")
                    if not ok and error_msg:
                        message = (
                            f"{self.generator}/{problem} [{model_name}] CPD query "
                            f"{q_index} error: {error_type}: {error_msg}"
                        )
                        if cpd_bar is not None:
                            tqdm.write(message)
                        else:
                            self.logger.warning(message)
                if cpd_bar is not None:
                    cpd_bar.close()

                inf_desc = f"{self.generator}/{problem} | {model_name} | Inference"
                inf_bar = (
                    tqdm(inf_queries, desc=inf_desc, leave=False)
                    if self.progress
                    else None
                )
                inf_iter = inf_bar if inf_bar is not None else inf_queries
                for q_index, query in enumerate(inf_iter):
                    start = time.perf_counter()
                    try:
                        response = model.answer_inference_query(query)
                        response = dict(response or {})
                        ok = bool(response.get("ok"))
                        error_msg = response.get("error")
                        error_type = None if not error_msg else "ModelError"
                        output = self._normalize_output(response.get("result"))
                    except Exception as exc:
                        ok = False
                        error_type = type(exc).__name__
                        error_msg = str(exc)
                        output = None
                    elapsed = (time.perf_counter() - start) * 1000.0
                    inf_timing_sum += elapsed
                    if ok:
                        ok_inf += 1
                    else:
                        err_inf += 1
                    model_stats[model_name]["_timing"]["inference"].add(float(elapsed))

                    record = {
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
                            "timing_ms": float(elapsed),
                            "output": output,
                        },
                    }
                    model_files[model_name]["inf"].write(
                        json.dumps(record, sort_keys=True) + "\n"
                    )

                    avg_ms = inf_timing_sum / max(1, ok_inf + err_inf)
                    if inf_bar is not None:
                        inf_bar.set_postfix(ok=ok_inf, err=err_inf, avg=f"{avg_ms:.2f}")
                    if not ok and error_msg:
                        message = (
                            f"{self.generator}/{problem} [{model_name}] Inference query "
                            f"{q_index} error: {error_type}: {error_msg}"
                        )
                        if inf_bar is not None:
                            tqdm.write(message)
                        else:
                            self.logger.warning(message)
                if inf_bar is not None:
                    inf_bar.close()

                total_ms = (time.perf_counter() - model_start) * 1000.0
                model_stats[model_name]["problems"]["ok"] += 1
                model_stats[model_name]["queries"]["cpd"]["ok"] += ok_cpd
                model_stats[model_name]["queries"]["cpd"]["error"] += err_cpd
                model_stats[model_name]["queries"]["inference"]["ok"] += ok_inf
                model_stats[model_name]["queries"]["inference"]["error"] += err_inf
                model_stats[model_name]["timing_ms"]["per_problem"][problem] = float(
                    total_ms
                )

                self.logger.info(
                    "[%s] CPDs done: ok=%s err=%s avg=%.2fms",
                    model_name,
                    ok_cpd,
                    err_cpd,
                    cpd_timing_sum / max(1, ok_cpd + err_cpd),
                )
                self.logger.info(
                    "[%s] Inference done: ok=%s err=%s avg=%.2fms",
                    model_name,
                    ok_inf,
                    err_inf,
                    inf_timing_sum / max(1, ok_inf + err_inf),
                )
                model_files[model_name]["cpd"].flush()
                model_files[model_name]["inf"].flush()

            del assets
            gc.collect()

        for model_name in self.models:
            model_files[model_name]["cpd"].close()
            model_files[model_name]["inf"].close()
            stats = model_stats[model_name]
            timing = stats.pop("_timing")
            cpd_summary = timing["cpd"].summary()
            inf_summary = timing["inference"].summary()
            stats["queries"]["cpd"].update(
                {"avg_ms": cpd_summary["avg_ms"], "median_ms": cpd_summary["median_ms"]}
            )
            stats["queries"]["inference"].update(
                {"avg_ms": inf_summary["avg_ms"], "median_ms": inf_summary["median_ms"]}
            )
            stats["timing_ms"]["total_ms"] = float(
                sum(stats["timing_ms"]["per_problem"].values())
                if stats["timing_ms"]["per_problem"]
                else 0.0
            )
            summary["models"][model_name] = stats

        summary_path = run_dir / "summary.json"
        write_json(summary_path, summary)
        return run_dir
