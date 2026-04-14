from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkBundleSpec:
    bundle_id: str
    mode: str
    created_at: str
    generator: str
    seeds: dict[str, int] = field(default_factory=dict)
    dataset_ids: list[str] = field(default_factory=list)
    query_generation: dict[str, Any] = field(default_factory=dict)
    data_generation: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "id": self.bundle_id,
            "mode": self.mode,
            "created_at": self.created_at,
            "generator": self.generator,
            "seeds": dict(self.seeds),
            "dataset_ids": list(self.dataset_ids),
            "query_generation": dict(self.query_generation),
            "data_generation": dict(self.data_generation),
            "artifacts": dict(self.artifacts),
            "config": dict(self.config),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "BenchmarkBundleSpec":
        return cls(
            bundle_id=str(payload.get("id") or payload.get("bundle_id") or ""),
            mode=str(payload.get("mode") or ""),
            created_at=str(payload.get("created_at") or ""),
            generator=str(payload.get("generator") or ""),
            seeds=dict(payload.get("seeds") or {}),
            dataset_ids=list(payload.get("dataset_ids") or []),
            query_generation=dict(payload.get("query_generation") or {}),
            data_generation=dict(payload.get("data_generation") or {}),
            artifacts=dict(payload.get("artifacts") or {}),
            config=dict(payload.get("config") or {}),
        )


@dataclass(frozen=True)
class BenchmarkBundlePaths:
    root: Path
    datasets: Path
    queries: Path
    ground_truth: Path
    metadata: Path


@dataclass(frozen=True)
class BenchmarkProblemPaths:
    dataset_dir: Path
    queries_dir: Path
    ground_truth_dir: Path
    queries_meta: Path
    cpds_path: Path
    inference_path: Path
    ground_truth_path: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "queries_dir": str(self.queries_dir),
            "ground_truth_dir": str(self.ground_truth_dir),
            "queries_meta": str(self.queries_meta),
            "cpds_path": str(self.cpds_path),
            "inference_path": str(self.inference_path),
            "ground_truth_path": str(self.ground_truth_path),
        }


@dataclass
class BenchmarkBundle:
    spec: BenchmarkBundleSpec
    paths: BenchmarkBundlePaths

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        generator: str,
        seed: int | None,
        timestamp: str | None = None,
        root: Path,
        bundle_id: str | None = None,
    ) -> "BenchmarkBundle":
        mode = str(mode).strip().lower()
        if mode not in {"cpds", "inference"}:
            raise ValueError(f"mode must be cpds or inference (got '{mode}')")
        generator = str(generator).strip()
        if not generator:
            raise ValueError("generator must be non-empty")
        created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        bundle_id = bundle_id or f"benchmark_{mode}_{timestamp}"
        root = Path(root).resolve() / bundle_id
        paths = _build_paths(root)
        for path in (paths.datasets, paths.queries, paths.ground_truth):
            path.mkdir(parents=True, exist_ok=True)
        seeds: dict[str, int] = {}
        if seed is not None:
            seeds["bundle"] = int(seed)
        spec = BenchmarkBundleSpec(
            bundle_id=bundle_id,
            mode=mode,
            created_at=created_at,
            generator=generator,
            seeds=seeds,
        )
        bundle = cls(spec=spec, paths=paths)
        bundle.save_metadata()
        return bundle

    @classmethod
    def load(cls, path: Path) -> "BenchmarkBundle":
        path = Path(path).resolve()
        meta_path = path
        if path.is_dir():
            meta_path = path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Bundle metadata not found: {meta_path}")
        payload = json.loads(meta_path.read_text())
        spec = BenchmarkBundleSpec.from_dict(payload)
        bundle_root = meta_path.parent
        paths = _build_paths(bundle_root)
        return cls(spec=spec, paths=paths)

    def save_metadata(self) -> Path:
        payload = self.spec.to_dict()
        self.paths.metadata.parent.mkdir(parents=True, exist_ok=True)
        self.paths.metadata.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return self.paths.metadata

    def list_problems(self) -> list[str]:
        if self.spec.dataset_ids:
            return list(self.spec.dataset_ids)
        datasets_root = self.paths.datasets / self.spec.generator
        if not datasets_root.exists():
            return []
        return sorted([p.name for p in datasets_root.iterdir() if p.is_dir()])

    def problem_paths(self, problem_id: str) -> BenchmarkProblemPaths:
        problem_id = str(problem_id)
        dataset_dir = self.paths.datasets / self.spec.generator / problem_id
        queries_dir = self.paths.queries / self.spec.generator / problem_id
        ground_truth_dir = self.paths.ground_truth / self.spec.generator / problem_id
        return BenchmarkProblemPaths(
            dataset_dir=dataset_dir,
            queries_dir=queries_dir,
            ground_truth_dir=ground_truth_dir,
            queries_meta=queries_dir / "queries.json",
            cpds_path=queries_dir / "cpds.jsonl",
            inference_path=queries_dir / "inference.jsonl",
            ground_truth_path=ground_truth_dir / "ground_truth.jsonl",
        )

    def update_artifact(self, problem_id: str, entries: dict[str, Any]) -> None:
        artifacts = dict(self.spec.artifacts)
        problems = dict(artifacts.get("problems") or {})
        problem_entry = dict(problems.get(problem_id) or {})
        problem_entry.update(entries)
        problems[problem_id] = problem_entry
        artifacts["problems"] = problems
        artifacts.setdefault(
            "datasets_dir", _rel_path(self.paths.root, self.paths.datasets)
        )
        artifacts.setdefault(
            "queries_dir", _rel_path(self.paths.root, self.paths.queries)
        )
        artifacts.setdefault(
            "ground_truth_dir", _rel_path(self.paths.root, self.paths.ground_truth)
        )
        self._replace_spec(artifacts=artifacts)

    def set_dataset_ids(self, dataset_ids: list[str]) -> None:
        self._replace_spec(dataset_ids=list(dataset_ids))

    def set_query_generation(self, payload: dict[str, Any]) -> None:
        self._replace_spec(query_generation=dict(payload))

    def set_data_generation(self, payload: dict[str, Any]) -> None:
        self._replace_spec(data_generation=dict(payload))

    def update_seeds(self, entries: dict[str, int]) -> None:
        seeds = dict(self.spec.seeds)
        seeds.update({k: int(v) for k, v in entries.items() if v is not None})
        self._replace_spec(seeds=seeds)

    def _replace_spec(
        self,
        *,
        bundle_id: str | None = None,
        mode: str | None = None,
        created_at: str | None = None,
        generator: str | None = None,
        seeds: dict[str, int] | None = None,
        dataset_ids: list[str] | None = None,
        query_generation: dict[str, Any] | None = None,
        data_generation: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.spec = BenchmarkBundleSpec(
            bundle_id=bundle_id or self.spec.bundle_id,
            mode=mode or self.spec.mode,
            created_at=created_at or self.spec.created_at,
            generator=generator or self.spec.generator,
            seeds=seeds if seeds is not None else self.spec.seeds,
            dataset_ids=(
                dataset_ids if dataset_ids is not None else self.spec.dataset_ids
            ),
            query_generation=(
                query_generation
                if query_generation is not None
                else self.spec.query_generation
            ),
            data_generation=(
                data_generation
                if data_generation is not None
                else self.spec.data_generation
            ),
            artifacts=artifacts if artifacts is not None else self.spec.artifacts,
            config=config if config is not None else self.spec.config,
        )


def _build_paths(root: Path) -> BenchmarkBundlePaths:
    root = Path(root).resolve()
    return BenchmarkBundlePaths(
        root=root,
        datasets=root / "datasets",
        queries=root / "queries",
        ground_truth=root / "ground_truth",
        metadata=root / "metadata.json",
    )


def _rel_path(root: Path, path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(path)


def resolve_bundle_dir(
    *,
    bundle_dir: str | None,
    bundle_name: str | None,
    bundle_root: Path,
) -> Path | None:
    if bundle_dir:
        return Path(bundle_dir).resolve()
    if bundle_name:
        return (Path(bundle_root).resolve() / bundle_name).resolve()
    return None


def find_latest_bundle(
    *,
    bundle_root: Path,
    mode: str | None = None,
    generator: str | None = None,
) -> Path | None:
    bundle_root = Path(bundle_root).resolve()
    if not bundle_root.exists():
        return None
    candidates: list[Path] = []
    for path in bundle_root.iterdir():
        if not path.is_dir():
            continue
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text())
        except Exception:
            continue
        if mode and str(payload.get("mode")) != str(mode):
            continue
        if generator and str(payload.get("generator")) != str(generator):
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
