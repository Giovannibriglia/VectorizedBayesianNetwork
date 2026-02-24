from __future__ import annotations

import argparse
import importlib
import json
import logging

from benchmarking.utils import get_project_root


def _parse_model_kwargs(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--model-kwargs must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--model-kwargs must be a JSON object")
    return payload


def _parse_models(values: list[str]) -> list[str]:
    models: list[str] = []
    for item in values:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        models.extend(parts)
    deduped = []
    seen = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        deduped.append(model)
    if not deduped:
        raise SystemExit("--models must include at least one model")
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="Comma-separated or repeatable list of models",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="JSON object of model-specific kwargs",
    )
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument(
        "--store_full_query",
        action="store_true",
        help="Store full query payloads in JSONL outputs",
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Enable tqdm/progress bars (default)",
    )
    progress_group.add_argument(
        "--no-progress",
        "--no_progress",
        dest="progress",
        action="store_false",
        help="Disable tqdm/progress bars",
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

    project_root = get_project_root()
    logging.info("Benchmark root: %s", project_root)

    model_kwargs = _parse_model_kwargs(args.model_kwargs)
    models = _parse_models(args.models)

    module = importlib.import_module("benchmarking.IIII_run_benchmark")
    runner_cls = module.get_benchmark_runner(args.generator)
    runner = runner_cls(
        root=project_root,
        seed=args.seed,
        models=models,
        model_kwargs=model_kwargs,
        max_problems=args.max_problems,
        store_full_query=args.store_full_query,
        progress=args.progress,
    )

    out_dir = runner.run_all()
    logging.info("Benchmark complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
