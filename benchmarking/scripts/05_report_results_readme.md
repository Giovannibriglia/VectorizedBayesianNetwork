# Report Results (Step 05)

This step summarizes a completed benchmark run by joining predictions with ground truth, computing metrics (KL/Wasserstein/JSD), and producing tables and plots. It also reports success rates and error breakdowns using per-query `ok` fields written during benchmarking.

## CLI

```bash
python -m benchmarking.scripts.05_report_results \
  --run_dir benchmarking/out/<generator>/benchmark_<mode>_<timestamp> \
  --summary_style robust
```

Key flags:

- `--run_dir` (required): path to a benchmark run folder.
- `--out_dir` (optional): output folder (default: `<run_dir>/report`).
- `--gt_source` (default: `folder`): `embedded|folder|compute`.
- `--gt_key` (default: `result.ground_truth.output.probs`).
- `--summary_style` (default: `robust`): `robust` (IQM ± IQRStd) or `mean` (mean ± std).
- `--include_time` / `--no-include_time`: include time tables/plots.
- `--include_pareto` / `--no-include_pareto`: include Pareto plots.
- `--pareto_split`: `none|mode|task|target_category`.
- `--min_partition_queries`: minimum size for subset partitions.
- `--max_subsets`: limit subset partitions.
- `--include_all_methods_in_subsets`: show all methods in subset reports.

## Output Structure

```
<run_dir>/report/
  index.md
  aggregate/
    partition_inventory.csv
    partition_inventory.md
    all/ tables/ figures/
    common/ tables/ figures/
    subset_*/ tables/ figures/
  single/
    <problem_id>/
      index.md
      partition_inventory.csv
      partition_inventory.md
      all/ tables/ figures/
      common/ tables/ figures/
      subset_*/ tables/ figures/
  by_category/
    <category>/
      index.md
      partition_inventory.csv
      partition_inventory.md
      all/ tables/ figures/
      common/ tables/ figures/
      subset_*/ tables/ figures/
```

The same solvability partitions (all/common/subset_*) are generated for aggregate, per-problem, and per-network-category views.
Category folders are derived from `benchmarking/metadata/<generator>.json` (for `bnlearn`, this includes labels like `small`, `medium`, `very_large`, `gaussian_medium`, `clgaussian_small`, etc.).

## Summary Style

- `robust`: IQM ± IQRStd (current default).
- `mean`: mean ± std.

The selected style affects aggregate tables and plot annotations.

## Notes

- Ground-truth computation with pgmpy (`--gt_source compute`) is best-effort and requires evidence values in the stored query payloads.
- If GT cannot be resolved for a query, that record is skipped.
- Success rate is `n_ok / n_total` for each slice, based on per-query `result.ok`.
