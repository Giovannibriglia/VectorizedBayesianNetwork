# Benchmark Query Generation (Step 02)

## 1) Pipeline Overview

The benchmarking pipeline produces three data components under a gitignored `benchmarking/data/` tree:

- **Datasets**: downloaded/converted model artifacts per generator and problem.
- **Metadata**: derived, per-problem artifacts such as domain mappings and capability flags.
- **Queries**: generated benchmark queries (CPD and inference) for each problem.

Directory layout:

```
benchmarking/data/
  datasets/<generator>/<problem>/
  metadata/<generator>/<problem>/
  queries/<generator>/<problem>/
  queries/log/<generator>/
```

Notes:

- `benchmarking/data/` is generated locally and should not be committed.
- Static metadata shipped with the repo lives under `benchmarking/metadata/`.

## 2) Query Generation Script and CLI

Entry point:

```bash
python -m benchmarking.scripts.02_generate_benchmark_queries \
  --generator bnlearn \
  --seed 42 \
  --n_queries_cpds 64 \
  --n_queries_inference 128 \
  --generator-kwargs '{"n_mc": 32}'
```

Required flags:

- `--generator`
- `--seed`
- `--n_queries_cpds`
- `--n_queries_inference`

Generator kwargs:

- Pass JSON via `--generator-kwargs`, for example `{"n_mc": 32}`.
- The parsed kwargs are forwarded to the generator and stored in the output payload as `generator_kwargs`.

Exact-count constraint:

- CPD query count equals `n_queries_cpds`.
- Inference query count equals `n_queries_inference` and refers to **instantiated** queries (after MC expansion).
- If `n_mc` is set, inference queries are generated in contiguous blocks per skeleton. The number of skeletons is derived from `n_queries_inference / n_mc` and the final skeleton may be a partial block if there is a remainder.

## 3) BNLearn Generator (`bnlearn`)

### Targets

CPD target categories:

- big Markov blanket
- big parent set
- random (PAC-diverse)

Inference target categories:

- central hubs
- separators/cut nodes
- peripheral/terminal
- random (PAC-diverse)

Inference tasks:

- prediction (evidence biased toward ancestors)
- diagnosis (evidence biased toward descendants)

### Evidence

CPD evidence strategies:

- paths
- markov_blanket
- random

Evidence subset sampling rule (CPD and inference on-manifold):

- Sample subset size `k` uniformly from `{0..|P(Q)|}`.
- Sample a size-`k` subset uniformly without replacement.

Inference evidence modes:

- empty (prior)
- on-manifold (MC instantiations)
- off-manifold with full evidence (MC instantiations)

Evidence values must be numeric:

- Discrete variables use the domain mapping (state label → integer code).
- Variable identifiers remain strings (BN node names).

### bnlearn metadata

Dataset types and download URLs are defined in:

- `benchmarking/metadata/bnlearn.json`

Some gaussian/clgaussian datasets do not have `.bif` artifacts and may be skipped depending on capabilities.

## 4) Coverage Metrics

The generator logs and stores coverage metrics for both target selection and evidence:

CPD coverage:

- Markov blanket size and parent-set coverage.
- Evidence size and variable coverage within evidence pools.

Inference coverage:

- Role coverage (hubs, separators, peripheral/terminal).
- Evidence mode counts and empty rate.
- Instantiated vs skeleton counts.

Generation is deterministic and reproducible given the same seed and downloaded artifacts.

## 5) File Examples

```
benchmarking/data/
  datasets/bnlearn/asia/model.bif
  metadata/bnlearn/asia/domain.json
  queries/bnlearn/asia/queries.json
  queries/log/bnlearn/asia_seed0.log
```
