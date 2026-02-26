from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .config import ModelBenchmarkConfig


class BaseBenchmarkModel(ABC):
    name: str
    supports_batched_inference_queries: bool = False

    def __init__(
        self,
        *,
        dag,
        seed: int,
        domain: dict,
        benchmark_config: "ModelBenchmarkConfig | None" = None,
        **kwargs: Any,
    ) -> None:
        if not getattr(self, "name", None):
            raise ValueError("Benchmark model must define a non-empty 'name'.")
        self.dag = dag
        self.seed = int(seed)
        self.domain = domain
        self.benchmark_config = benchmark_config
        self.model_kwargs = dict(kwargs)

    @abstractmethod
    def fit(
        self, data_df: pd.DataFrame, *, progress: bool = True, **kwargs: Any
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def answer_cpd_query(self, query: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def answer_inference_query(self, query: dict) -> dict:
        raise NotImplementedError

    def answer_inference_queries(self, queries: list[dict]) -> list[dict]:
        return [self.answer_inference_query(query) for query in queries]

    def supports(self) -> dict:
        return {
            "can_fit": True,
            "can_answer_cpd": True,
            "can_answer_inference": True,
        }

    def _timed(self, fn, *args, **kwargs) -> tuple[float, Any]:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        return (end - start) * 1000.0, result
