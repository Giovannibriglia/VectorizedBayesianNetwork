## Computational Complexity (Batched)

This section makes **time / memory complexity explicit** for each method, using a **batch of queries** for inference/sampling.

### Notation (symbols used below)

**Graph / structure**
- `N`: number of nodes (random variables)
- `E`: number of edges
- `Pa(i)`: parent set of node `i`
- `d_i = |Pa(i)|`: in-degree (number of parents) of node `i`
- `d̄`: average in-degree
- `topo(N,E)`: topological traversal cost, typically `O(N+E)`

**Data / training**
- `M`: number of training samples (rows in dataset)
- `B`: mini-batch size during fit/update
- `T`: number of optimizer steps (e.g., epochs × steps-per-epoch)
- `H`: hidden width of a neural CPD (order-of-magnitude parameter)
- `K`: number of mixture components (MDN)
- `D_x`: dimension of conditioning vector (parents/features used as input)
- `D_y`: dimension of target variable (`1` for scalar target)

**Inference / sampling (batched queries)**
- `Q`: number of queries in a batch
- `S`: number of Monte Carlo particles/samples per query used by inference/sampling
- `R`: number of Gibbs sweeps (full passes over variables)
- `L`: HMC leapfrog steps per sample
- `A`: acceptance rate (HMC), affects effective sample count

**Discrete / binned**
- `C`: number of bins/classes for a binned categorical variable (softmax CPD)

**KDE**
- `M_eff`: effective number of stored points used by KDE (can be `M` or a buffer subset)
- `BW`: bandwidth/scales (treated as constant-cost unless learned per-dimension)

> Notes:
> - Complexities are given **per call** (fit/update/infer/sample) and assume torch vectorization.
> - Constants depend on implementation details (network depth, feature engineering, etc.).
> - For inference methods, the dominant cost is typically repeated evaluation of CPDs and graph traversal.

---

## Learning Methods (Orchestration)

### `node_wise`
Trains each node CPD independently (with parent features as inputs).

- **Time:** `Σ_i time_fit(CPD_i on M samples)`
- **Memory:** `Σ_i memory(CPD_i params + optimizer state)` plus minibatch tensors `O(B·(D_x+D_y))`

Node-wise orchestration adds only:
- **Time overhead:** `O(N + E)` per epoch (building parent inputs / topological bookkeeping)
- **Memory overhead:** negligible beyond per-CPD training state

### `amortized` (in progress)
A shared model learns multiple CPDs (or a shared encoder across nodes).

- **Time:** roughly `O(T · B · cost_shared_forward)` (plus heads per node)
- **Memory:** shared parameters + per-node heads; can be lower than node-wise if heavy encoders are shared

---

## CPDs (Training + Evaluation)

Below, “**eval**” refers to computing `log_prob(y | parents)` or producing `pdf/samples` for a given conditioning batch.

### `gaussian_nn` (Neural Gaussian CPD)
Predicts `μ(x), σ(x)` (diagonal Gaussian).

- **Train time:** `O(T · B · H · (D_x + D_y))` (order-of-magnitude; depends on depth)
- **Train memory:** `O(params + optimizer_state + activations(B,H))`
- **Eval time (per conditioning batch):** `O(B · H · D_x)` for forward + `O(B · D_y)` for Gaussian log_prob
- **Sampling time:** `O(B · D_y)` after forward pass

### `softmax_nn` (Binned categorical CPD)
Predicts logits over `C` bins/classes.

- **Train time:** `O(T · B · (H · D_x + H · C))`
- **Eval time:** `O(B · (H · D_x + H · C))` + softmax/logsumexp
- **Sampling time:** `O(B · C)` (categorical sample from logits)

### `mdn` (Mixture Density Network CPD)
Outputs `K` mixture components (weights + params).

- **Train time:** `O(T · B · (H · D_x + H · K · D_y))` + mixture logsumexp
- **Eval time:** `O(B · (H · D_x + K · D_y))`
- **Sampling time:** `O(B · (K + D_y))` (sample component + sample Gaussian)

### `kde` (Conditional Gaussian KDE CPD)
Evaluates kernel contributions from stored samples.

- **Fit time:** typically `O(M)` to store data (plus preprocessing); if bandwidth learned, add its optimizer cost
- **Eval time:** `O(B · M_eff · D)` where `D = D_x + D_y` (distance + kernel)
- **Memory:** `O(M_eff · D)` for stored points (+ bandwidth params)

> KDE can become the dominant bottleneck when `M_eff` is large; replay buffers or prototypes reduce `M_eff`.

---

## Inference (Posterior) Methods — Batched Queries

We assume `vbn.infer_posterior(query_batch)` where a **query batch** has size `Q`, and each query asks for:
- target node(s): `Tq` (usually 1)
- evidence set: `Ev` (nodes with observed values)

### Common parameters for inference
- `S`: number of Monte Carlo particles per query (or per posterior estimate)
- `C_eval`: average cost to evaluate all required CPDs once for a single particle
  - For neural CPDs: `~ Σ_i O(H·D_x)` on visited nodes
  - For KDE CPDs: `~ Σ_i O(M_eff·D)`

### `monte_carlo_marginalization`
Approximates posteriors by sampling latent variables and scoring via CPDs.

- **Time:** `O(Q · S · (topo(N,E) + C_eval))`
  - If only a subgraph is needed (Markov blanket / ancestors of targets ∪ evidence), replace `N,E` by subgraph size.
- **Memory:** `O(Q · S · (|latent| + |targets|))` to store particles/samples, plus temporary tensors

Returned outputs:
- **Samples:** `O(Q · S · D_y)` memory
- **PDF estimate:** depends on representation (histogram / KDE / param fit); often `O(Q · S)` extra

### `importance_sampling`
Samples from a proposal `q(z)` and reweights to approximate `p(z|e)`.

- **Time:** `O(Q · S · (topo(N,E) + C_eval))` + `O(Q · S)` for weights normalization
- **Memory:** `O(Q · S)` for weights + particle storage

> Importance sampling is sensitive to weight degeneracy; effective sample size (ESS) can be far smaller than `S`.

### SVGP (placeholder, MC fallback)
Currently treated as MC fallback in complexity. When SVGP is enabled for certain CPDs:
- **Per SVGP node eval:** typically `O(B · M_ind)` or `O(B · M_ind²)` depending on factorization/whitening
- `M_ind`: number of inducing points

---

## Sampling Methods — Batched Queries

We assume `vbn.sample(query_batch, n_samples=S)` where we generate `S` samples per query, batch size `Q`.

### `ancestral`
Topological sampling from the joint given evidence (clamping evidence nodes).

- **Time:** `O(Q · S · (topo(N,E) + C_sample))`
  - `C_sample` is CPD sampling cost across nodes (often similar to eval cost)
- **Memory:** `O(Q · S · N)` if returning full trajectories; can be reduced if returning only targets

### `gibbs` (simple conditional sampling)
Iteratively resample each non-evidence node from `p(x_i | mb(i))`.

Let `R` be the number of sweeps and `MB_i` denote Markov blanket size.

- **Time:** `O(Q · S · R · Σ_i cost_sample(CPD_i | MB_i))`
  - For neural CPDs: typically `O(Q · S · R · Σ_i H·D_x(MB_i))`
  - For KDE CPDs: can be `O(Q · S · R · Σ_i M_eff·D)`
- **Memory:** `O(Q · S · N)` for current chain state (plus returned samples)

> Gibbs mixes slowly in highly coupled graphs; `R` can dominate runtime.

### HMC (placeholder, ancestral fallback)
When implemented for continuous subgraphs:
- **Time:** `O(Q · S · L · C_grad)` where `C_grad` is cost of evaluating log joint + gradients
- **Memory:** `O(Q · S · N_cont)` for positions/momenta + autodiff activations

---

## Update Policies (Complexity)

Update policies primarily change **(a)** which data are fed into `cpd.update(...)` and **(b)** how many optimizer steps are applied.

Let `B_u` be update batch size and `T_u` the number of update steps.

### `online_sgd`
- **Time:** `O(T_u · B_u · cost_update_step)` (per CPD)
- **Memory:** optimizer state + gradients (same order as training)

### `ema`
Maintains an exponential moving average of parameters or statistics.

- **Time:** `O(#params)` per update step (plus any forward passes if required)
- **Memory:** `O(#params)` extra for EMA shadow copy

### `streaming_stats`
Maintains running estimates (means/vars or sufficient stats).

- **Time:** `O(B_u · D)` per update batch (very fast; no backprop)
- **Memory:** `O(D)` for stored stats (per node)

### `replay_buffer`
Stores recent samples and updates from replayed mini-batches.

Parameters:
- `max_size`: buffer capacity `N_buf`
- `replay_ratio`: how many replay samples per new sample (or per update step)

- **Buffer maintenance time:** `O(B_u)` amortized (enqueue/dequeue)
- **Update time:** `O(T_u · B_u · replay_ratio · cost_update_step)`
- **Memory:** `O(N_buf · D)` for stored transitions/features (per node or global, depending on design)

---

## Parameters Reference (What affects complexity)

### CPD-level parameters
- `H`: hidden size / network width (`gaussian_nn`, `softmax_nn`, `mdn`)
- `K`: mixture components (`mdn`)
- `C`: number of bins/classes (`softmax_nn`)
- `M_eff`: number of KDE reference points used during evaluation (`kde`)
- `D_x`: number of conditioning features (parents)
- `D_y`: target dimension

### Inference parameters
- `Q`: number of queries in a batch
- `S`: number of particles / samples per query
- `subgraph`: whether inference restricts to ancestors/Markov blanket (reduces effective `N,E`)
- `proposal`: choice of proposal distribution (importance sampling)

### Sampling parameters
- `S`: number of samples per query
- `R`: number of Gibbs sweeps
- `L`: HMC leapfrog steps (when enabled)

### Update parameters
- `B_u`: update batch size
- `T_u`: number of update steps
- `N_buf`: replay buffer capacity
- `replay_ratio`: replay intensity
- `alpha`: EMA / streaming mixing coefficient

---

## Practical Takeaways

- **Best scaling (neural CPDs):** `gaussian_nn` / `softmax_nn` / `mdn` typically scale with `O(Q·S·(N+E))` times a neural forward cost; GPU batching is effective.
- **Potential bottleneck:** `kde` evaluation scales with `M_eff`; keep `M_eff` bounded (buffers/prototypes) for large datasets.
- **Inference dominates runtime:** when `S` is large, `O(Q·S·...)` quickly dominates; prefer subgraph restriction and batched tensorization.
- **Gibbs/HMC** add extra multiplicative factors (`R`, `L`) and are most useful when they significantly improve posterior quality per sample.
