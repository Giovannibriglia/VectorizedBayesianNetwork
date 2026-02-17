from __future__ import annotations

import argparse
from pathlib import Path

from benchmarking.data_generation import get_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", type=str, default="bnlearn")
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--networks", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.networks is not None and len(args.networks) == 0:
        args.networks = None

    project_root = Path(__file__).resolve().parents[2]

    # Module execution keeps imports clean; registry keeps generators extensible.
    generator_cls = get_generator(args.generator)
    generator = generator_cls(root_path=project_root, seed=args.seed)

    generator.generate(
        n_samples=args.n_samples,
        networks=args.networks,
        force=args.force,
    )


if __name__ == "__main__":
    main()
