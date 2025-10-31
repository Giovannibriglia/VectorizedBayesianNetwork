from __future__ import annotations

import numpy as np
import torch

from vbn import VBN
from vbn.io import load_state, save_state

# 1) Define a tiny BN: A -> X -> R with A -> R
#    A: discrete {0,1}; X,R: Gaussian (dim=1)
nodes = {
    "A": {"type": "discrete", "card": 2},
    "X": {"type": "gaussian", "dim": 1},
    "R": {"type": "gaussian", "dim": 1},
}
parents = {
    "A": [],
    "X": ["A"],
    "R": ["A", "X"],
}

# 2) Generate synthetic training data
#    A ~ Bernoulli(0.4)
#    X = 2*A + ε,  ε ~ N(0, 0.25)
#    R = -1 + 0.5*X + 1.0*A + η,  η ~ N(0, 0.3)
N = 20_000
rng = np.random.default_rng(0)
A_np = rng.binomial(1, 0.4, size=(N, 1)).astype(np.int64)
eps = rng.normal(0.0, 0.5, size=(N, 1))
X_np = 2.0 * A_np + eps
eta = rng.normal(0.0, 0.5477, size=(N, 1))  # std≈sqrt(0.3)
R_np = -1.0 + 0.5 * X_np + 1.0 * A_np + eta

data = {
    "A": torch.from_numpy(A_np),
    "X": torch.from_numpy(X_np).float(),
    "R": torch.from_numpy(R_np).float(),
}

# 3) Build & train a VBN
bn = VBN(
    nodes=nodes,
    parents=parents,
    device="cuda" if torch.cuda.is_available() else "cpu",
    inference_method="lw",  # likelihood weighting
    seed=123,
)
# map learners explicitly to be restorable later
bn.set_learners(
    {
        "A": "mle",  # categorical MLE
        "X": "linear_gaussian",  # linear-Gaussian regressor
        "R": "linear_gaussian",  # linear-Gaussian regressor
    }
)
bn.fit(data)  # default steps=0 for linear/mle, which is fine here

# 4) Do a reference posterior query before saving (e.g., E[R | do(A=1)])
with torch.no_grad():
    post_pre = bn.posterior(
        query="R",
        do={"A": torch.tensor([1], device=bn.device)},
        n_samples=8192,
        method="lw",
    )
    # The API may return a dict or tensor depending on your backend;
    # we handle two common cases:
    if isinstance(post_pre, dict) and "samples" in post_pre:
        r_samples_pre = post_pre["samples"].float().view(-1)
    elif torch.is_tensor(post_pre):
        r_samples_pre = post_pre.float().view(-1)
    else:
        # fall back to sampling from the BN under do()
        r_samples_pre = (
            bn.sample(n_samples=8192, do={"A": torch.tensor([1], device=bn.device)})[
                "R"
            ]
            .float()
            .view(-1)
        )

    mu_pre = r_samples_pre.mean().item()
    std_pre = r_samples_pre.std(unbiased=False).item()
print(f"[pre-save] E[R|do(A=1)] ≈ {mu_pre:.4f} ± {std_pre:.4f}")

# 5) Save to disk
ckpt_path = "trained_vbn.pt"
save_state(bn, ckpt_path)

# 6) Rebuild a fresh BN skeleton (no training), then load the state
bn2 = VBN(
    nodes=nodes,
    parents=parents,
    device=bn.device,
    inference_method=None,  # will be restored
    seed=123,
)
# You can set a dummy learner map; load_state will overwrite with the saved one.
bn2.set_learners({"A": "mle", "X": "linear_gaussian", "R": "linear_gaussian"})
# Materialize trainer/CPDs before injecting state
bn2._ensure_trainer()
load_state(bn2, ckpt_path)

# 7) Re-run the same posterior on the reloaded BN
with torch.no_grad():
    post_post = bn2.posterior(
        query="R",
        do={"A": torch.tensor([1], device=bn2.device)},
        n_samples=8192,
        method=bn2.inference_method or "lw",
    )
    if isinstance(post_post, dict) and "samples" in post_post:
        r_samples_post = post_post["samples"].float().view(-1)
    elif torch.is_tensor(post_post):
        r_samples_post = post_post.float().view(-1)
    else:
        r_samples_post = (
            bn2.sample(n_samples=8192, do={"A": torch.tensor([1], device=bn2.device)})[
                "R"
            ]
            .float()
            .view(-1)
        )

    mu_post = r_samples_post.mean().item()
    std_post = r_samples_post.std(unbiased=False).item()
print(f"[post-load] E[R|do(A=1)] ≈ {mu_post:.4f} ± {std_post:.4f}")

# 8) Quick sanity check (should be very close)
abs_diff = abs(mu_pre - mu_post)
print(f"Δ mean ≈ {abs_diff:.6f} (should be ~0)")
