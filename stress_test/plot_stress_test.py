import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from run import plot_metrics_grid


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_logs(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid log format in {path}")
    return data


def split_inference_metrics(metrics: Sequence[str]) -> Tuple[List[str], List[str]]:
    distribution_set = {"kl", "js", "ws", "inference_time"}
    point_set = {"mse", "mae", "r2"}
    dist = [m for m in metrics if m in distribution_set]
    point = [m for m in metrics if m in point_set]
    return dist, point


def derive_metrics_from_logs(logs: Dict[str, Any]) -> List[str]:
    metrics = logs.get("metrics", {})
    return list(metrics.keys())


def plot_for_run_dir(run_dir: Path, show_fill_between: bool = True) -> None:
    cpd_logs_path = run_dir / "cpd_logs.json"
    inf_logs_path = run_dir / "inference_logs.json"
    if not cpd_logs_path.exists() or not inf_logs_path.exists():
        raise FileNotFoundError(
            f"Missing log files in {run_dir}. Expected cpd_logs.json and inference_logs.json."
        )

    cpd_logs = load_logs(cpd_logs_path)
    inf_logs = load_logs(inf_logs_path)

    config_path = run_dir / "config.json"
    if config_path.exists():
        config = load_json(config_path)
        cfg_block = config.get("config", {})
        cpd_metrics = cfg_block.get("metrics") or derive_metrics_from_logs(cpd_logs)
        inf_metrics = cfg_block.get("inference_metrics") or derive_metrics_from_logs(
            inf_logs
        )
        n_states = config.get("n_states")
        n_actions = config.get("n_actions")
        title_prefix = (
            f"#states: {n_states}, #actions: {n_actions}"
            if n_states is not None and n_actions is not None
            else run_dir.name
        )
    else:
        cpd_metrics = derive_metrics_from_logs(cpd_logs)
        inf_metrics = derive_metrics_from_logs(inf_logs)
        title_prefix = run_dir.name

    dist_metrics, point_metrics = split_inference_metrics(inf_metrics)

    plot_metrics_grid(
        logs=cpd_logs,
        metrics=cpd_metrics,
        title=f"{title_prefix} - CPD metrics",
        out_path=run_dir / "cpd_metrics.png",
        show_fill_between=show_fill_between,
    )
    plot_metrics_grid(
        logs=inf_logs,
        metrics=dist_metrics,
        title=f"{title_prefix} - Inference distribution metrics",
        out_path=run_dir / "inference_distribution_metrics.png",
        show_fill_between=show_fill_between,
    )
    if point_metrics:
        plot_metrics_grid(
            logs=inf_logs,
            metrics=point_metrics,
            title=f"{title_prefix} - Inference point metrics",
            out_path=run_dir / "inference_point_metrics.png",
            show_fill_between=show_fill_between,
        )


def find_run_dirs(root: Path) -> List[Path]:
    if (root / "cpd_logs.json").exists() and (root / "inference_logs.json").exists():
        return [root]

    run_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "cpd_logs.json").exists() and (
            child / "inference_logs.json"
        ).exists():
            run_dirs.append(child)
    return run_dirs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stress-test metrics from a saved benchmark directory."
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        help="Path to benchmark directory (or a directory containing run subfolders).",
    )
    parser.add_argument(
        "--show-fill-between",
        type=str,
        help="Path to benchmark directory (or a directory containing run subfolders).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    root = Path(args.benchmark_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found in {root}. Expected cpd_logs.json and inference_logs.json."
        )

    for run_dir in run_dirs:
        plot_for_run_dir(run_dir, args.show_fill_between)


if __name__ == "__main__":
    main()
