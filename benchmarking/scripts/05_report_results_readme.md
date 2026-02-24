# 05: Report Results

This step summarizes a completed benchmark run by joining predictions with ground truth, computing metrics (KL/Wasserstein), and producing tables and plots sliced by query categories.

---

## CLI Usage

```bash
python -m benchmarking.scripts.05_report_results \
    --run_dir benchmarking/out/<generator>/benchmark_<timestamp>
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

---

## Outputs

```
<run_dir>/report/
  tables/
  figures/
  report.md
```

### Tables

All tables include numeric columns per metric (`kl`, `wass`):

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

### Figures

All plots are saved as PNG in `figures/`:

- **CPD**: KL/Wass vs Markov blanket size (one plot per metric, one line per method)
- **CPD**: KL/Wass vs parent set size (one plot per metric, one line per method)
- **Inference**: KL/Wass vs evidence size (one plot per metric per evidence mode, one line per method)
- **Category bars** (grouped bars per method):
  - CPD target category
  - Inference target category
  - Inference task
  - Inference evidence mode

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
