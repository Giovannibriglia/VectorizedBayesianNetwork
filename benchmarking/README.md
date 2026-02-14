# Benchmarking VBN vs pgmpy vs BAMT

This suite compares parameter learning, inference, and sampling accuracy and speed on discrete Bayesian networks from the bnlearn repository. Networks, queries, and tool configurations are configurable via JSON/CLI.

## Layout
- `bnrepo/metadata.json` — network metadata (name, url, size).
- `bnrepo/bif/` — downloaded `.bif` files.
- `exact_reference/` — BIF loader and exact enumeration/sampling from ground-truth CPTs.
- `metrics/` — KL/Wasserstein/TV utilities for CPDs and posteriors, plus timing helpers.
- `runners/` — wrappers for VBN, pgmpy, and BAMT to fit, infer, and sample.
- `scripts/` — end-to-end scripts: download BIFs, generate synthetic data, run benchmarks, summarize results.
- `results/` — raw JSONL runs, CSV tables, and optional plots.
- `tests_benchmarking/` — sanity tests for exact reference and state ordering.

## Quickstart (ASIA/ALARM)
All commands below are executed **from the repo root**.

```bash
# 0) Dependencies (in your virtualenv)
pip install -r benchmarking/requirements.txt

# 1) Download BIFs
python benchmarking/scripts/00_download_bifs.py --networks asia alarm
# (also works as a module) python -m benchmarking.scripts.00_download_bifs --networks asia alarm

# 2) Generate synthetic train/val/test splits + queries (default CSV)
python benchmarking/scripts/01_generate_data.py --networks asia alarm --n-train 5000 --n-test 1000 --format csv
# Parquet (requires pyarrow or fastparquet):
# python benchmarking/scripts/01_generate_data.py --networks asia alarm --format parquet
#   add --strict-format to error if parquet engine is missing (otherwise falls back to CSV once).
# Module form: python -m benchmarking.scripts.01_generate_data --networks asia alarm

# 3) Run all tool configurations (small nets first)
python benchmarking/scripts/02_run_all.py --networks asia alarm --out benchmarking/results/raw/asia_alarm.jsonl
# Module form: python -m benchmarking.scripts.02_run_all --networks asia alarm

# 4) Summaries to CSV tables
python benchmarking/scripts/03_summarize.py --input benchmarking/results/raw/asia_alarm.jsonl --out-prefix benchmarking/results/tables/asia_alarm
```

Supported networks out of the box: `asia`, `alarm`. To add more, append entries in `benchmarking/bnrepo/metadata.json` and download/generate again (HAILFINDER/ANDES/BARLEY are already listed).

## Notes
- Exact reference uses enumeration over CPT factors with optional `do` interventions; evidence size is kept small by default for large networks.
- VBN is run with `mle_softmax` learning and Monte Carlo posterior; pgmpy uses VariableElimination; BAMT uses sampling-based posteriors.
- Network selection, query budgets, and batch sizes are JSON-configurable; see inline defaults in scripts.
- Additional networks (HAILFINDER, ANDES, BARLEY) are supported but may require tighter query budgets to keep exact inference feasible.
