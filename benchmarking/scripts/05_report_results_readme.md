# 05: Report Results

This step summarizes a completed benchmark run by joining predictions with ground truth, computing metrics (KL/Wasserstein/JSD), and producing tables and plots sliced by query categories.

---

## CLI Usage

```bash
python -m benchmarking.scripts.05_report_results \
    --run_dir benchmarking/out/<generator>/benchmark_<mode>_<timestamp>
```

### Flags

- `--run_dir` (required): path to a single benchmark run folder.
- `--out_dir` (optional): output folder (default: `<run_dir>/report`).
- `--gt_source` (optional): where ground truth comes from (default: `folder`).
  - `embedded`: use `--gt_key` path in each result record.
  - `folder`: use `<run_dir>/ground_truth/{cpds,inference}.jsonl` if present, else `ground_truth_sources.json` links (created by step 04).
  - `compute`: compute exact GT with pgmpy (best-effort; requires evidence values).
- `--gt_key` (optional): JSON path for embedded GT (default: `result.ground_truth.output.probs`).
- `--models` (optional): comma-separated filter of model names/config ids.
- `--max_records` (optional): limit records for debugging.
- `--eps` (optional): smoothing epsilon for KL (default: `1e-12`).
- `--include_time` / `--no-include_time`: include time tables and plots (default: enabled).
- `--include_pareto` / `--no-include_pareto`: include Pareto plots (default: enabled).
- `--pareto_split` (optional): `none|mode|task|target_category` (default: `none`).

---

## Outputs

```
<run_dir>/report/
  tables/
  figures/
  report.md
```

The reporter auto-detects the run mode (`cpds` or `inference`) from `run_metadata.json`
or the run directory name, and will still emit placeholder tables for missing categories.

### Tables

All tables include numeric columns per metric (`kl`, `wass`, `jsd`, `jsd_norm`):

- `*_iqm`, `*_iqr_std`, `*_n`
- `*_iqm_pm_iqrstd`
- `*_q1`, `*_median`, `*_q3`

Tables produced in `tables/`:

- `overall_by_model.csv`
- `cpd_by_target_category.csv`
- `cpd_by_evidence_strategy.csv`
- `inference_by_target_category.csv`
- `inference_by_task.csv`
- `inference_by_evidence_mode.csv`
- `cpd_by_mb_size.csv` (two-stage aggregation: dataset IQM, then across datasets)
- `cpd_by_parent_size.csv` (two-stage aggregation)
- `inference_by_evidence_size.csv`
- `inference_by_skeleton.csv` (per-skeleton aggregation for MC queries)
- `overall_time_by_method.csv`
- `cpd_time_by_target_category.csv`
- `cpd_time_by_evidence_strategy.csv`
- `cpd_time_by_mb_size.csv`
- `cpd_time_by_parent_size.csv`
- `cpd_time_by_evidence_size.csv`
- `inference_time_by_target_category.csv`
- `inference_time_by_task.csv`
- `inference_time_by_evidence_mode.csv`
- `inference_time_by_evidence_size.csv`

### Figures

All plots are saved as PNG in `figures/`:

- **CPD**: KL/Wass/JSD(norm) vs Markov blanket size (one plot per metric, one line per method)
- **CPD**: KL/Wass/JSD(norm) vs parent set size (one plot per metric, one line per method)
- **Inference**: KL/Wass/JSD(norm) vs evidence size (one plot per metric per evidence mode, one line per method)
- **Category bars** (grouped bars per method):
  - CPD target category (KL/Wass/JSD(norm))
  - Inference target category (KL/Wass/JSD(norm))
  - Inference task (KL/Wass/JSD(norm))
  - Inference evidence mode (KL/Wass/JSD(norm))
  - CPD time by target/evidence strategy
  - Inference time by target/task/mode

Efficiency (Pareto) plots:

- `pareto_cpd_kl_vs_time.png`
- `pareto_cpd_wass_vs_time.png`
- `pareto_cpd_jsd_norm_vs_time.png`
- `pareto_inference_kl_vs_time.png`
- `pareto_inference_wass_vs_time.png`
- `pareto_inference_jsd_norm_vs_time.png`

Optional stratified plots depend on `--pareto_split`.

---

## Ground Truth Alignment

Predictions are joined to GT using:

1. `query.id` (preferred), else
2. tuple of model-independent query fields:
   `(problem.id, query.type, query.index, target, category, task, evidence mode/vars, mc_id, skeleton_id)`

---

## Notes

- Ground-truth computation with pgmpy is best-effort and requires evidence values in the stored query payloads.
- If GT cannot be resolved for a query, that record is skipped.
- MB size and parent set size are computed from the dataset DAG once per problem and cached.
- JSD is computed as `0.5 * KL(P || M) + 0.5 * KL(Q || M)` with `M = (P + Q) / 2` (natural log).
- Normalized JSD is `jsd_norm = jsd / log(2)` and is in `[0, 1]` (0 best, 1 worst).
- JSD/JSD norm are computed only for discrete distributions; continuous outputs get `NaN` for these fields.
