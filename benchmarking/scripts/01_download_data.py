from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path

from benchmarking.bundles import BenchmarkBundle, resolve_bundle_dir
from benchmarking.utils import get_project_root
from benchmarking.utils_logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, default="bnlearn")
    parser.add_argument(
        "--mode",
        type=str,
        default="cpds",
        choices=["cpds", "inference"],
        help="Benchmark bundle mode (cpds or inference).",
    )
    parser.add_argument("--networks", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bundle_dir", type=str, default=None)
    parser.add_argument("--bundle", type=str, default=None)
    parser.add_argument(
        "--bundle_root",
        type=str,
        default=None,
        help="Root directory for benchmark bundles (default: benchmarking/data/benchmarks)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    if args.networks is not None and len(args.networks) == 0:
        args.networks = None

    setup_logging(level=args.log_level)
    project_root = get_project_root()

    bundle_root = (
        Path(args.bundle_root).resolve()
        if args.bundle_root
        else project_root / "benchmarking" / "data" / "benchmarks"
    )
    bundle_path = resolve_bundle_dir(
        bundle_dir=args.bundle_dir, bundle_name=args.bundle, bundle_root=bundle_root
    )
    if bundle_path is None:
        bundle = BenchmarkBundle.create(
            mode=args.mode,
            generator=args.generator,
            seed=args.seed,
            root=bundle_root,
        )
    elif bundle_path.exists():
        bundle = BenchmarkBundle.load(bundle_path)
    else:
        bundle = BenchmarkBundle.create(
            mode=args.mode,
            generator=args.generator,
            seed=args.seed,
            root=bundle_path.parent,
            bundle_id=bundle_path.name,
        )
    if bundle.spec.mode != args.mode or bundle.spec.generator != args.generator:
        raise SystemExit(
            f"Bundle metadata mismatch. bundle={bundle.paths.root} mode={bundle.spec.mode} generator={bundle.spec.generator}"
        )
    logging.info("Resolved bundle: %s", bundle.paths.root)

    module = importlib.import_module("benchmarking.I_data_download")
    downloader_cls = module.get_downloader(args.generator)
    downloader = downloader_cls(root_path=project_root, seed=args.seed, bundle=bundle)

    downloader.download(
        datasets=args.networks,
        force=args.force,
    )


if __name__ == "__main__":
    main()
