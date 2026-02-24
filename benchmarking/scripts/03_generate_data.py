from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path


def _parse_generator_kwargs(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--generator-kwargs must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--generator-kwargs must be a JSON object")
    return payload


def _parse_kv_pairs(pairs: list[str] | None) -> dict:
    if not pairs:
        return {}
    kwargs: dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--kw must be key=value (got '{item}')")
        key, value_raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--kw key cannot be empty (got '{item}')")
        try:
            value = json.loads(value_raw)
        except json.JSONDecodeError:
            value = value_raw
        kwargs[key] = value
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generation_strategy", type=str, default="default")
    parser.add_argument(
        "--generator-kwargs",
        type=str,
        default=None,
        help="JSON object of generator-specific kwargs (e.g. '{\"batch_size\": 4096}')",
    )
    parser.add_argument(
        "--kw",
        action="append",
        default=None,
        help="Repeatable generator kwarg in key=value form (overrides JSON)",
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

    project_root = Path(__file__).resolve().parents[2]
    logging.info("Benchmark root: %s", project_root)

    generator_kwargs = _parse_generator_kwargs(args.generator_kwargs)
    generator_kwargs.update(_parse_kv_pairs(args.kw))
    if generator_kwargs:
        logging.info("Generator kwargs: %s", generator_kwargs)

    module = importlib.import_module("benchmarking.03_data_generation")
    generator_cls = module.get_data_generator(args.generator)
    generator = generator_cls(
        root_path=project_root,
        seed=args.seed,
        n_samples=args.n_samples,
        generation_strategy=args.generation_strategy,
        generator_kwargs=generator_kwargs,
    )

    dataset_dirs = generator.list_dataset_dirs()
    logging.info(
        "Found %s dataset(s) under %s", len(dataset_dirs), generator.datasets_dir
    )
    if not dataset_dirs:
        logging.warning("No datasets found. Run 01_download_data first.")

    outputs = generator.generate_all()
    logging.info("Generated %s data metadata file(s)", len(outputs))
    for output in outputs:
        logging.info("Wrote %s", output)


if __name__ == "__main__":
    main()
