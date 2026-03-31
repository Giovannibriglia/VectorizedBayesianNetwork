from __future__ import annotations

from benchmarking.IIII_run_benchmark.base import (
    _execute_inference_chunk,
    _iter_inference_batches,
)


def _make_query(skeleton_id: str, value: int) -> dict:
    return {
        "dataset_id": "dataset",
        "query_type": "inference",
        "target": "lung",
        "task": "prediction",
        "skeleton_id": skeleton_id,
        "evidence_mode": "on_manifold",
        "evidence_vars": ["smoke"],
        "evidence_values": {"smoke": value},
    }


class DummyBatchModel:
    supports_batched_inference_queries = True

    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0

    def answer_inference_query(self, query: dict) -> dict:
        self.single_calls += 1
        return {
            "ok": True,
            "error": None,
            "result": {"format": "categorical_probs", "probs": [0.5, 0.5], "k": 2},
        }

    def answer_inference_queries(self, queries: list[dict]) -> list[dict]:
        self.batch_calls += 1
        return [
            {
                "ok": True,
                "error": None,
                "result": {
                    "format": "categorical_probs",
                    "probs": [0.5, 0.5],
                    "k": 2,
                },
            }
            for _ in queries
        ]


class DummyNoBatchModel(DummyBatchModel):
    supports_batched_inference_queries = False

    def answer_inference_queries(self, queries: list[dict]) -> list[dict]:
        raise AssertionError("Batch path should not be used")


def test_iter_inference_batches_grouping() -> None:
    queries = [
        _make_query("s1", 0),
        _make_query("s1", 1),
        _make_query("s1", 2),
        _make_query("s2", 3),
    ]
    batches = list(
        _iter_inference_batches(queries, batch_size=2, default_dataset_id="dataset")
    )
    sizes = [len(batch) for batch in batches]
    assert sizes == [2, 1, 1]
    indices = [idx for batch in batches for idx, _ in batch]
    assert indices == list(range(len(queries)))


def test_execute_inference_chunk_batched() -> None:
    model = DummyBatchModel()
    queries = [_make_query("s1", 0), _make_query("s1", 1)]
    responses = _execute_inference_chunk(model, queries, batch_size=4)
    assert model.batch_calls == 1
    assert model.single_calls == 0
    assert len(responses) == len(queries)


def test_execute_inference_chunk_fallback() -> None:
    model = DummyNoBatchModel()
    queries = [_make_query("s1", 0), _make_query("s1", 1)]
    responses = _execute_inference_chunk(model, queries, batch_size=4)
    assert model.batch_calls == 0
    assert model.single_calls == len(queries)
    assert len(responses) == len(queries)


def test_execute_inference_chunk_batch_size_one() -> None:
    model = DummyBatchModel()
    queries = [_make_query("s1", 0), _make_query("s1", 1)]
    responses = _execute_inference_chunk(model, queries, batch_size=1)
    assert model.batch_calls == 0
    assert model.single_calls == len(queries)
    assert len(responses) == len(queries)
