# Benchmarking Framework

This benchmarking suite compares VBN and other probabilistic graphical model libraries across a curated set of Bayesian networks. It is designed as a staged, reproducible pipeline so each step can be validated independently and extended over time.

---

## Pipeline Overview

```
01_download_data
        ↓
02_generate_benchmark_queries
        ↓
03_generate_data
        ↓
04_run_benchmark
        ↓
05_report_results
```

---

## End-to-End Procedures (CPDs and Inference)

If your environment does not expose `python`, replace `python` with `python3`.

### CPDs benchmark

```bash
python -m benchmarking.scripts.01_download_data --generator bnlearn --mode cpds --seed 42 --bundle benchmark_cpds_local
```
```bash
python -m benchmarking.scripts.02_generate_benchmark_queries --generator bnlearn --mode cpds --seed 42 --n_queries_cpds 1024 --bundle benchmark_cpds_local
```
```bash
python -m benchmarking.scripts.03_generate_data --generator bnlearn --n_samples 9192 --seed 42 --bundle benchmark_cpds_local
```
```bash
python -m benchmarking.scripts.04_run_benchmark --generator bnlearn --seed 42 --mode cpds --bundle benchmark_cpds_local --batch_size_queries 256 --models vbn,pgmpy,numpyro,gpytorch,pyro
```
```bash
CPDS_RUN_DIR=$(ls -td benchmarking/out/bnlearn/benchmark_cpds_* | head -n 1)

python -m benchmarking.scripts.05_report_results --run_dir "$CPDS_RUN_DIR" --summary_style robust --cmap tab20
```

### Inference benchmark

```bash
python -m benchmarking.scripts.01_download_data --generator bnlearn --mode inference --seed 42 --bundle benchmark_inference_local
```
```bash
python -m benchmarking.scripts.02_generate_benchmark_queries --generator bnlearn --mode inference --seed 42 --n_queries_inference 5120 --generator-kwargs '{"n_mc": 256}' --bundle benchmark_inference_local
```
```bash
python -m benchmarking.scripts.03_generate_data --generator bnlearn --n_samples 9192 --seed 42 --bundle benchmark_inference_local
```
```bash
python -m benchmarking.scripts.04_run_benchmark --generator bnlearn --seed 42 --mode inference --bundle benchmark_inference_local --batch_size_queries 256 --models pgmpy:pgmpy_mle_ei,pgmpy:pgmpy_bdeu_ei,pgmpy:pgmpy_gaussian_exact,vbn:vbn_lg_rao,vbn:vbn_lg_exact,vbn:vbn_ct_ce,pyro:pyro_lw,pyro:pyro_ais,numpyro:numpyro_lw,numpyro:numpyro_ais,gpytorch:gpytorch_forward,gpytorch:gpytorch_posterior
```
```bash
INFERENCE_RUN_DIR=$(ls -td benchmarking/out/bnlearn/benchmark_inference_* | head -n 1)

python -m benchmarking.scripts.05_report_results --run_dir "$INFERENCE_RUN_DIR" --summary_style robust --cmap tab20
```

---

## Benchmark Bundles (Authoritative Layout)

Steps 01–03 create a **benchmark bundle** under `benchmarking/data/benchmarks/`. All datasets, queries, ground truth, and bundle metadata live in that folder.

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/
  metadata.json
  datasets/
    <generator>/<problem_id>/...
  queries/
    <generator>/<problem_id>/cpds.jsonl|inference.jsonl
    <generator>/<problem_id>/queries.json
  ground_truth/
    <generator>/<problem_id>/ground_truth.jsonl
  logs/
    queries/...
    datasets/...
```

`mode` is `cpds` or `inference`. The bundle metadata records generator, seeds, dataset selection, query/data generation parameters, and paths to artifacts.

---

## Last Validated Usage (2026-04-13, Inference)

Latest end-to-end run currently in this repository:

- Bundle: `benchmark_inference_20260407_125322`
- Run directory: `benchmarking/out/bnlearn/benchmark_inference_20260413_190932`
- Networks: 31
- Total queries: 286,720 inference queries
- Models: `pgmpy:pgmpy_mle_ei`, `pgmpy:pgmpy_bdeu_ei`, `vbn:vbn_lg_rao`, `vbn:vbn_lg_exact`, `vbn:vbn_ct_ce`

Reproduce that exact flow:

```bash
python -m benchmarking.scripts.01_download_data \
  --generator bnlearn \
  --mode inference \
  --seed 42

python -m benchmarking.scripts.02_generate_benchmark_queries \
  --generator bnlearn \
  --mode inference \
  --seed 42 \
  --n_queries_inference 10240 \
  --generator-kwargs '{"n_mc": 256}' \
  --bundle benchmark_inference_20260407_125322

python -m benchmarking.scripts.03_generate_data \
  --generator bnlearn \
  --n_samples 4096 \
  --seed 42 \
  --bundle benchmark_inference_20260407_125322

python -m benchmarking.scripts.04_run_benchmark \
  --generator bnlearn \
  --seed 42 \
  --mode inference \
  --bundle benchmark_inference_20260407_125322 \
  --batch_size_queries 256 \
  --models pgmpy:pgmpy_mle_ei,pgmpy:pgmpy_bdeu_ei,vbn:vbn_lg_rao,vbn:vbn_lg_exact,vbn:vbn_ct_ce

python -m benchmarking.scripts.05_report_results \
  --run_dir benchmarking/out/bnlearn/benchmark_inference_20260413_190932 \
  --summary_style robust \
  --cmap tab20
```

---

## 1) Data Download

**Purpose.** Download and prepare dataset artifacts for a specific generator and store them in a bundle.

```bash
python -m benchmarking.scripts.01_download_data \
  --generator bnlearn \
  --mode cpds
```

Outputs:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/datasets/<generator>/<problem>/
  model.bif
  dataset.json
  download.json
```

Developer guide: `benchmarking/scripts/01_download_data_readme.md`.

---

## 2) Query Generation

**Purpose.** Generate deterministic benchmark query sets for each dataset in the bundle.

```bash
python -m benchmarking.scripts.02_generate_benchmark_queries \
  --generator bnlearn \
  --mode cpds \
  --seed 42 \
  --n_queries_cpds 1024 \
  --generator-kwargs '{"n_mc": 1}' \
  --bundle benchmark_cpds_YYYYMMDD_HHMMSS
```

Outputs:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/queries/<generator>/<problem>/
  cpds.jsonl | inference.jsonl
  queries.json
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/ground_truth/<generator>/<problem>/ground_truth.jsonl
```

Developer guide: `benchmarking/scripts/02_generate_benchmark_queries_readme.md`.

---

## 3) Learning Data Generation

**Purpose.** Generate tabular datasets (i.i.d. samples) for model fitting and CPD learning.

```bash
python -m benchmarking.scripts.03_generate_data \
  --generator bnlearn \
  --n_samples 4096 \
  --seed 42 \
  --bundle benchmark_cpds_YYYYMMDD_HHMMSS
```

Outputs:

```
benchmarking/data/benchmarks/benchmark_<mode>_<timestamp>/datasets/<generator>/<problem>/
  data_<strategy>_n<n_samples>_seed<seed>.(parquet|csv|pkl)
  data_generation.json
  domain.json
```

Developer guide: `benchmarking/scripts/03_generate_data_readme.md`.

---

## 4) Run Benchmark

**Purpose.** Fit registered models on learning data and run CPD + inference query workloads.

```bash
python -m benchmarking.scripts.04_run_benchmark \
  --generator bnlearn \
  --seed 42 \
  --mode cpds \
  --bundle benchmark_cpds_YYYYMMDD_HHMMSS \
  --models vbn,pgmpy,numpyro,gpytorch,pyro
```

Outputs:

```
benchmarking/out/<generator>/benchmark_<mode>_<timestamp>/
  run_metadata.json
  configs/
  logs/
  errors/
  results/<problem_id>/<method_id>.jsonl
```

Use `--dry_run` to print the resolved bundle, models, and output dir without running.

Developer guide: `benchmarking/scripts/04_run_benchmark_readme.md`.

---

## 5) Report Results

**Purpose.** Aggregate metrics, success rates, and plots for a benchmark run.

```bash
python -m benchmarking.scripts.05_report_results \
  --run_dir benchmarking/out/bnlearn/benchmark_cpds_YYYYMMDD_HHMMSS \
  --summary_style robust \
  --cmap tab20
```

Outputs:

```
<run_dir>/report/
  index.md
  aggregate/...
  single/<problem_id>/...
```

Use `--summary_style mean` for mean ± std instead of IQM ± IQRStd. Use `--cmap <name>` to choose a matplotlib colormap (method colors are deterministic and consistent across generated figures).

Developer guide: `benchmarking/scripts/05_report_results_readme.md`.
