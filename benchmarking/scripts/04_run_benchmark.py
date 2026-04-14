from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

from benchmarking.bundles import BenchmarkBundle, find_latest_bundle, resolve_bundle_dir
from benchmarking.utils import get_project_root
from benchmarking.utils_logging import setup_logging


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


def _parse_config_overrides(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--config-overrides must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--config-overrides must be a JSON object")
    return payload


def _parse_config_selection(raw: str | None) -> tuple[str, dict[str, str]]:
    if not raw:
        raw = "default"
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        tokens = ["default"]
    selections: dict[str, str] = {}
    default_config: str | None = None
    for token in tokens:
        if ":" in token:
            model, config_id = token.split(":", 1)
            model = model.strip()
            config_id = config_id.strip()
            if not model or not config_id:
                raise SystemExit(
                    "--config entries must look like model:config_id when using pairs"
                )
            selections[model] = config_id
        else:
            if default_config is None:
                default_config = token
            elif token != default_config:
                raise SystemExit(
                    "--config cannot include multiple default config values"
                )
    return default_config or "default", selections


def _parse_model_specs(values: list[str]) -> list[dict]:
    models: list[str] = []
    for item in values:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        models.extend(parts)
    deduped: list[str] = []
    seen = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        deduped.append(model)
    if not deduped:
        raise SystemExit("--models must include at least one model")

    specs: list[dict] = []
    for token in deduped:
        if ":" in token:
            base, config_id = token.split(":", 1)
            base = base.strip()
            config_id = config_id.strip()
            if not base or not config_id:
                raise SystemExit(
                    "--models entries with ':' must look like model:config_id"
                )
            specs.append(
                {
                    "alias": token,
                    "base": base,
                    "config_id": config_id,
                }
            )
        else:
            specs.append({"alias": token, "base": token, "config_id": None})
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["cpds", "inference"],
        help="Benchmark mode: cpds or inference.",
    )
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="Comma-separated or repeatable list of models (supports model:config_id)",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="JSON object of model-specific kwargs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Model config preset (single value or model:config_id pairs)",
    )
    parser.add_argument(
        "--config-overrides",
        type=str,
        default=None,
        help="JSON object of per-component config overrides",
    )
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument(
        "--batch_size_queries",
        type=int,
        default=1,
        help="Max number of inference queries to batch together (default: 1 = no batching)",
    )
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
    parser.add_argument("--bundle_dir", type=str, default=None)
    parser.add_argument("--bundle", type=str, default=None)
    parser.add_argument(
        "--bundle_root",
        type=str,
        default=None,
        help="Root directory for benchmark bundles (default: benchmarking/data/benchmarks)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Resolve bundle, models, and output paths; then exit.",
    )

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    project_root = get_project_root()
    logging.info("Benchmark root: %s", project_root)

    bundle_root = (
        Path(args.bundle_root).resolve()
        if args.bundle_root
        else project_root / "benchmarking" / "data" / "benchmarks"
    )
    bundle_path = resolve_bundle_dir(
        bundle_dir=args.bundle_dir, bundle_name=args.bundle, bundle_root=bundle_root
    )
    if bundle_path is None:
        bundle_path = find_latest_bundle(
            bundle_root=bundle_root, mode=args.mode, generator=args.generator
        )
        if bundle_path is None:
            raise SystemExit(
                f"No bundle found under {bundle_root}. Run 01_download_data first or pass --bundle_dir."
            )
    if not bundle_path.exists():
        raise SystemExit(f"Bundle not found: {bundle_path}")
    bundle = BenchmarkBundle.load(bundle_path)
    if bundle.spec.mode != args.mode or bundle.spec.generator != args.generator:
        raise SystemExit(
            f"Bundle metadata mismatch. bundle={bundle.paths.root} mode={bundle.spec.mode} generator={bundle.spec.generator}"
        )
    logging.info("Resolved bundle: %s", bundle.paths.root)

    model_kwargs = _parse_model_kwargs(args.model_kwargs)
    model_specs = _parse_model_specs(args.models)
    models = [spec["alias"] for spec in model_specs]
    model_aliases = {spec["alias"]: spec["base"] for spec in model_specs}
    default_config, per_model_configs = _parse_config_selection(args.config)
    model_configs: dict[str, str] = {}
    for spec in model_specs:
        alias = spec["alias"]
        base = spec["base"]
        explicit = spec["config_id"]
        if explicit:
            model_configs[alias] = explicit
        else:
            model_configs[alias] = per_model_configs.get(base, default_config)
    config_overrides = _parse_config_overrides(args.config_overrides)

    module = importlib.import_module("benchmarking.IIII_run_benchmark")
    runner_cls = module.get_benchmark_runner(args.generator)
    runner = runner_cls(
        root=project_root,
        bundle=bundle,
        seed=args.seed,
        mode=args.mode,
        models=models,
        model_kwargs=model_kwargs,
        model_configs=model_configs,
        model_aliases=model_aliases,
        config_overrides=config_overrides,
        max_problems=args.max_problems,
        store_full_query=args.store_full_query,
        progress=args.progress,
        batch_size_queries=args.batch_size_queries,
        log_level=args.log_level,
    )

    if args.dry_run:
        problems = bundle.list_problems()
        if args.max_problems is not None:
            problems = problems[: int(args.max_problems)]
        logging.info("Dry run: bundle=%s", bundle.paths.root)
        logging.info("Dry run: mode=%s generator=%s", args.mode, args.generator)
        logging.info("Dry run: problems=%s", len(problems))
        logging.info("Dry run: models=%s", ",".join(models))
        return

    out_dir = runner.run_all()
    logging.info("Benchmark complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
