from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, default="bnlearn")
    parser.add_argument("--networks", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.networks is not None and len(args.networks) == 0:
        args.networks = None

    project_root = Path(__file__).resolve().parents[2]

    module = importlib.import_module("benchmarking.01_data_download")
    downloader_cls = module.get_downloader(args.generator)
    downloader = downloader_cls(root_path=project_root, seed=args.seed)

    downloader.download(
        datasets=args.networks,
        force=args.force,
    )


if __name__ == "__main__":
    main()
