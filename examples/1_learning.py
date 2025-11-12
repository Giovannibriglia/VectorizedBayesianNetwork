#!/usr/bin/env python3
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vbn import VBN


# ----------------- utils -----------------
def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parents_of(vbn: VBN, node: str) -> list[str]:
    return list(vbn.dag.predecessors(node))


@torch.no_grad()
def sample_child(vbn: VBN, child: str, X_ctx: torch.Tensor, n: int) -> torch.Tensor:
    y = vbn._nodes[child].sample(X_ctx, n=n)  # [n,B,D] or [n,B]
    return y if y.dim() == 3 else y.unsqueeze(-1)


@torch.no_grad()
def check_nll_discrete(vbn: VBN, df: pd.DataFrame, child: str) -> tuple[float, float]:
    """Returns (NLL, Accuracy)."""
    ps = parents_of(vbn, child)
    X = torch.tensor(df[ps].values, dtype=torch.float32, device=vbn.device)
    y = torch.tensor(df[child].values, dtype=torch.long, device=vbn.device).view(-1, 1)
    head = vbn._nodes[child]
    # NLL
    nll = (-head.log_prob(X, y)).mean().item()
    # Accuracy (argmax over probs)
    logits = (
        head.net(X) if hasattr(head, "net") else head.linear(X)
    )  # softmax_nn vs mle_softmax
    pred = logits.argmax(dim=-1)
    acc = (pred == y.squeeze(-1)).float().mean().item()
    return float(nll), float(acc)


@torch.no_grad()
def check_nll_continuous(vbn: VBN, df: pd.DataFrame, child: str) -> float:
    """NLL for models that implement log_prob."""
    ps = parents_of(vbn, child)
    X = torch.tensor(df[ps].values, dtype=torch.float32, device=vbn.device)
    y = torch.tensor(df[child].values, dtype=torch.float32, device=vbn.device).view(
        -1, 1
    )
    head = vbn._nodes[child]
    return float((-head.log_prob(X, y)).mean().item())


def _median_heuristic(x: torch.Tensor) -> float:
    """Median pairwise distance (scalar) as RBF bandwidth."""
    x = x.flatten()
    if x.numel() < 2:
        return 1.0
    diffs = torch.cdist(x.view(-1, 1), x.view(-1, 1), p=2)
    med = diffs[diffs > 0].median()
    return float(med.item() if med.numel() > 0 else 1.0)


@torch.no_grad()
def mmd_rbf(a: torch.Tensor, b: torch.Tensor, sigma: float | None = None) -> float:
    """a,b: [N, D]; returns scalar MMD^2."""
    N = a.shape[0]
    if sigma is None:
        sigma = _median_heuristic(torch.cat([a, b], dim=0))
        sigma = max(sigma, 1e-6)

    def _k(x, y):
        d2 = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-d2 / (2.0 * sigma * sigma))

    Kxx = _k(a, a)
    Kyy = _k(b, b)
    Kxy = _k(a, b)
    m = max(1, N - 1)
    term = (
        (Kxx.sum() - Kxx.trace()) / (N * m)
        + (Kyy.sum() - Kyy.trace()) / (N * m)
        - 2.0 * Kxy.mean()
    )
    return float(term.item())


def c2st_accuracy(
    real: torch.Tensor, fake: torch.Tensor, epochs: int = 10, hidden: int = 32
) -> float:
    """
    Train a tiny classifier to separate real vs fake y (shuffled mixture).
    Return holdout accuracy; 0.5 ~ indistinguishable (good).
    """
    device = real.device
    X = torch.cat([real, fake], dim=0).to(device).float()  # [N, D]
    y = torch.cat(
        [
            torch.ones(real.size(0), 1, device=device),
            torch.zeros(fake.size(0), 1, device=device),
        ],
        dim=0,
    )

    # train/test split
    N = X.size(0)
    idx = torch.randperm(N, device=device)
    ntr = int(0.7 * N)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    clf = nn.Sequential(
        nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)
    ).to(
        device
    )  # <<< ensure model on same device
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    clf.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = clf(Xtr)  # all CUDA now
        loss = bce(logits, ytr)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(Xte)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == yte).float().mean().item()
    return float(acc)


# ----------------- plotting -----------------
def _density_from_samples(samples: np.ndarray, bins=120, range=None):
    hist, edges = np.histogram(samples, bins=bins, range=range, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, hist


def plot_discrete(
    ctx_rows: np.ndarray, labels: list[str], samples_list: list[np.ndarray]
):
    B = ctx_rows.shape[0]
    K = int(max(s.max() for s in samples_list) + 1)
    fig, axes = plt.subplots(1, B, figsize=(4 * B, 3), squeeze=False)
    axes = axes.ravel()
    for b in range(B):
        ax = axes[b]
        width = 0.8 / len(samples_list)
        xs = np.arange(K)
        for i, (label, samp) in enumerate(zip(labels, samples_list)):
            hist = np.bincount(samp[:, b], minlength=K) / samp.shape[0]
            ax.bar(xs + i * width, hist, width=width, alpha=0.8, label=label)
        ax.set_xticks(xs + 0.4, [str(k) for k in range(K)])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"parents {ctx_rows[b].tolist()}")
        ax.set_ylabel("P(y|x)")
        ax.legend(fontsize=8)
    fig.suptitle("Discrete P(y|x): samples", y=1.02)
    fig.tight_layout()
    plt.show()


def plot_continuous(
    ctx_rows: np.ndarray, labels: list[str], samples_list: list[np.ndarray]
):
    B = ctx_rows.shape[0]
    fig, axes = plt.subplots(1, B, figsize=(5 * B, 3), squeeze=False)
    axes = axes.ravel()
    gmin = min(s.min() for s in samples_list)
    gmax = max(s.max() for s in samples_list)
    rng = (float(gmin), float(gmax))
    for b in range(B):
        ax = axes[b]
        for label, samp in zip(labels, samples_list):
            c, h = _density_from_samples(samp[:, b], bins=120, range=rng)
            ax.plot(c, h, label=label)
        ax.set_title(f"parents {ctx_rows[b].tolist()}")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Continuous p(y|x): sampled densities", y=1.02)
    fig.tight_layout()
    plt.show()


# ----------------- datasets -----------------
def make_discrete_df(N=4000, K=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, K, size=N)
    b = rng.integers(0, K, size=N)
    c = (a + 2 * b + rng.integers(0, K // 2 + 1, size=N)) % K
    return pd.DataFrame({"feature_0": a, "feature_1": b, "feature_2": c})


def make_continuous_df(N=6000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 5, size=N).astype(np.float32)
    b = rng.integers(0, 5, size=N).astype(np.float32)
    c = np.sin(a - 0.5 * b) + 0.25 * rng.normal(size=N)
    return pd.DataFrame(
        {"feature_0": a, "feature_1": b, "feature_2": c.astype(np.float32)}
    )


# ----------------- main -----------------
if __name__ == "__main__":
    device = device_auto()
    G = nx.DiGraph()
    G.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    # === DISCRETE: mle_softmax vs softmax_nn ===
    K = 5
    df_tr = make_discrete_df(4000, K, seed=0)
    df_up = make_discrete_df(800, K, seed=1)
    df_te = make_discrete_df(800, K, seed=2)

    ctx = np.array([[0, 0], [1, 3], [4, 2]], dtype=np.float32)
    X_ctx_dis = torch.tensor(ctx, dtype=torch.float32, device=device)

    disc_methods = [
        ("mle_softmax", dict(num_classes=K, lr=0.1, epochs=100, batch_size=512)),
        (
            "softmax_nn",
            dict(
                num_classes=K, hidden=128, depth=2, lr=1e-3, epochs=100, batch_size=512
            ),
        ),
    ]
    disc_labels, disc_samples = [], []
    print("\n[Discrete scores]")
    for name, kw in disc_methods:
        v = VBN(G, seed=0, device=device)
        v.set_learning_method(name, **kw)
        v.fit(df_tr)
        v.update(df_up)
        nll, acc = check_nll_discrete(v, df_te, "feature_2")
        print(f"  {name:12s} | NLL={nll:.4f} | Acc={acc:.3f}")
        y_s = (
            sample_child(v, "feature_2", X_ctx_dis, n=4000)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )
        disc_labels.append(name)
        disc_samples.append(y_s)
    plot_discrete(ctx, disc_labels, disc_samples)

    # === CONTINUOUS: groups ===
    df_tr_c = make_continuous_df(6000, seed=0)
    df_up_c = make_continuous_df(1200, seed=1)
    df_te_c = make_continuous_df(1200, seed=2)

    ctx_c = np.array([[0.0, 0.0], [1.0, 3.0], [4.0, 2.0]], dtype=np.float32)
    X_ctx_con = torch.tensor(ctx_c, dtype=torch.float32, device=device)

    likelihood_methods = [
        ("linear_gaussian", dict()),
        ("gaussian_nn", dict(hidden=64, lr=1e-3, epochs=50, batch_size=512)),
        ("mdn", dict(n_components=5, hidden=64, lr=1e-3, epochs=50, batch_size=512)),
        ("flow_rnvp", dict(n_layers=4, hidden=128, lr=1e-3, epochs=50, batch_size=512)),
        ("kde_gaussian", dict(rule="silverman", chunk_size=8192)),
        (
            "kde_gaussian_diff",
            dict(lr=5e-2, epochs=50, batch_size=1024, chunk_size=8192),
        ),
    ]
    implicit_methods = [
        (
            "implicit_c_mmd",
            dict(z_dim=8, hidden=128, epochs=50, batch_size=512, sigma=1.0),
        ),
        (
            "implicit_c_wgan",
            dict(z_dim=8, hidden=128, epochs=50, batch_size=512, critic_steps=3),
        ),
    ]

    # Likelihood CPDs — NLL + samples
    cont_labels, cont_samples = [], []
    print("\n[Continuous scores — likelihood models]")
    for name, kw in likelihood_methods:
        v = VBN(G, seed=0, device=device)
        v.set_learning_method(name, **kw)
        v.fit(df_tr_c)
        v.update(df_up_c)
        nll = check_nll_continuous(v, df_te_c, "feature_2")
        print(f"  {name:18s} | NLL={nll:.4f}")
        y_s = (
            sample_child(v, "feature_2", X_ctx_con, n=2048)
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )
        cont_labels.append(name)
        cont_samples.append(y_s)
    plot_continuous(ctx_c, cont_labels, cont_samples)

    # Implicit CPDs — MMD + C2ST + samples
    imp_labels, imp_samples = [], []
    print("\n[Continuous scores — implicit models] (lower MMD, C2ST≈0.5 is better)")
    # Build a real conditional sample set per context from test data
    # For fairness, build real y by nearest-neighbor in parent space to chosen contexts
    ps = ["feature_0", "feature_1"]
    X_te = torch.tensor(df_te_c[ps].values, dtype=torch.float32, device=device)
    y_te = torch.tensor(
        df_te_c["feature_2"].values, dtype=torch.float32, device=device
    ).view(-1, 1)

    def real_match(ctx_batch: torch.Tensor, k: int = 2048):
        """
        For each context xb, pick the k nearest test-parent rows and
        return their y's. Clamps k to the available test size.
        Returns: [k_eff, B]
        """
        k_eff = min(k, X_te.shape[0])  # <= number of test rows
        ys = []
        for xb in ctx_batch:  # xb: [2]
            d = torch.norm(X_te - xb.view(1, -1), dim=-1)  # [N_test]
            idx = torch.topk(d, k=k_eff, largest=False).indices
            ys.append(y_te[idx])
        return torch.stack(ys, dim=1).squeeze(-1)  # [k_eff, B]

    Y_real = real_match(X_ctx_con, k=2048)  # [N_real, B]
    for name, kw in implicit_methods:
        v = VBN(G, seed=0, device=device)
        v.set_learning_method(name, **kw)
        v.fit(df_tr_c)
        v.update(df_up_c)
        y_samp = sample_child(v, "feature_2", X_ctx_con, n=2048).squeeze(-1)  # [N,B]
        # Scores per context, then average
        mmds, c2sts = [], []
        for b in range(Y_real.shape[1]):
            real_b = Y_real[:, b]
            fake_b = y_samp[:, b]
            mmds.append(mmd_rbf(real_b.view(-1, 1), fake_b.view(-1, 1)))
            c2sts.append(
                c2st_accuracy(
                    real_b.view(-1, 1), fake_b.view(-1, 1), epochs=50, hidden=32
                )
            )
        print(f"  {name:18s} | MMD={np.mean(mmds):.4f} | C2ST={np.mean(c2sts):.3f}")
        imp_labels.append(name)
        imp_samples.append(y_samp.detach().cpu().numpy())
    plot_continuous(
        ctx_c, likelihood_methods + implicit_methods, cont_samples + imp_samples
    )

    print("\nDone.")
