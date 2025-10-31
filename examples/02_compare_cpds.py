from __future__ import annotations

import math
from typing import Dict, List

import torch

from vbn import VBN

N = 5


def make_binned_feature(x: torch.Tensor, B: int, device: str):
    # x: (N,1) float
    xv = x.view(-1)
    lo = torch.quantile(xv, 0.01)
    hi = torch.quantile(xv, 0.99)
    edges = torch.linspace(lo, hi, B + 1, device=x.device)
    xv = xv.clamp(min=edges[0].item(), max=edges[-1].item() - 1e-7)
    labels = torch.bucketize(xv, edges) - 1  # (N,), in [0, B-1]
    mids = 0.5 * (edges[:-1] + edges[1:])
    return labels.long(), edges, mids


@torch.no_grad()
def bn_E_R_do_S_from_Rdisc(bn, s_vals, mids, n: int = 16384):
    """
    Estimate E[R | do(S=s)] using discrete head R_disc with bin midpoints.
    We sample R_disc and take the midpoint expectation.
    """
    out = {}
    for s in s_vals:
        samp = bn.sample(n_samples=n, do={"S": torch.tensor([s], device=bn.device)})
        # samp["R_disc"]: (n,) or (n,1)
        R_disc = samp["R_disc"].view(-1).long()
        est = mids[R_disc].mean().item()
        out[s] = float(est)
    return out


def run_mle_discrete_R(nodes_base, parents_base, train_dict, gt, device, BX=20, BR=20):
    # 1) Discretize X and R
    X_disc, x_edges, x_mids = make_binned_feature(train_dict["X"], B=BX, device=device)
    R_disc, r_edges, r_mids = make_binned_feature(train_dict["R"], B=BR, device=device)

    train = dict(train_dict)
    train["X_disc"] = X_disc
    train["R_disc"] = R_disc

    # 2) Augment graph: S -> X_disc -> R_disc
    nodes = dict(nodes_base)
    nodes["X_disc"] = {"type": "discrete", "card": BX}
    nodes["R_disc"] = {"type": "discrete", "card": BR}

    parents = dict(parents_base)
    parents["X_disc"] = ["S"]  # optional but recommended
    parents["R_disc"] = ["X_disc"]

    # 3) Learners: all MLE on the discrete chain; keep X/R continuous nodes unused here
    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map={"S": "mle", "X_disc": "mle", "R_disc": "mle"},
        default_steps_svgp_kde=0,
        default_steps_others=0,
    )
    bn.fit(train)

    # 4) Estimate E[R | do(S=s)] from R_disc via bin midpoints
    @torch.no_grad()
    def estimate(s_vals, n=16384):
        out = {}
        for s in s_vals:
            samp = bn.sample(n_samples=n, do={"S": torch.tensor([s], device=device)})
            idx = samp["R_disc"].view(-1).long()
            out[s] = float(r_mids[idx].mean().item())
        return out

    s_vals = list(range(nodes_base["S"]["card"]))
    est = estimate(s_vals)
    rmse = math.sqrt(sum((est[s] - gt[s]) ** 2 for s in s_vals) / len(s_vals))
    # proxy NLL on R_disc|X_disc
    nll = (
        -bn.cpd["R_disc"]
        .log_prob(train["R_disc"], {"X_disc": train["X_disc"]})
        .mean()
        .item()
    )

    print(
        f"\n[R|X_disc: MLE (BX={BX}, BR={BR})]  RMSE vs true do-curve: {rmse:.4f} | (train) NLL R_disc|X_disc: {nll:.4f}"
    )
    for s in s_vals:
        print(f"  s={s}:  E_hat={est[s]: .4f} (true {gt[s]: .4f})")


def make_data(N: int, device: str) -> Dict[str, torch.Tensor]:
    S = torch.randint(0, 5, (N,), device=device)
    mu_s = 0.8 * S.float() - 1.0
    X = mu_s + 0.4 * torch.randn(N, device=device)
    fX = torch.sin(2 * X) + 0.25 * (X**2)
    R = fX + 0.10 * torch.randn(N, device=device)
    return {"S": S, "X": X.unsqueeze(-1), "R": R.unsqueeze(-1)}


@torch.no_grad()
def true_E_R_do_S(s_vals: List[int], n_mc: int, device: str) -> Dict[int, float]:
    out = {}
    for s in s_vals:
        mu_s = 0.8 * float(s) - 1.0
        X = mu_s + 0.4 * torch.randn(n_mc, device=device)
        out[s] = float(
            (torch.sin(2 * X) + 0.25 * (X**2) + 0.10 * torch.randn_like(X))
            .mean()
            .item()
        )
    return out


@torch.no_grad()
def bn_E_R_do_S(bn: VBN, s_vals: List[int], n: int = 16384) -> Dict[int, float]:
    out = {}
    for s in s_vals:
        samp = bn.sample(n_samples=n, do={"S": torch.tensor([s], device=bn.device)})
        R = samp["R"].reshape(n, -1)
        out[s] = float(R.mean().item())
    return out


def run(label: str, learner_map: Dict[str, str], train, gt, nodes, parents, device):
    bn = VBN(
        nodes,
        parents,
        device=device,
        learner_map=learner_map,
    )
    bn.fit(train, steps=5000)
    s_vals = list(range(N))
    est = bn_E_R_do_S(bn, s_vals)
    rmse = math.sqrt(sum((est[s] - gt[s]) ** 2 for s in s_vals) / len(s_vals))
    nll_R = -bn.cpd["R"].log_prob(train["R"], {"X": train["X"]}).mean().item()
    print(
        f"\n[{label}]  RMSE vs true do-curve: {rmse:.4f} | (train) NLL R|X: {nll_R:.4f}"
    )
    for s in s_vals:
        print(f"  s={s}:  E_hat={est[s]: .4f} (true {gt[s]: .4f})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = make_data(32000, device)
    nodes = {
        "S": {"type": "discrete", "card": 5},
        "X": {"type": "gaussian", "dim": 1},
        "R": {"type": "gaussian", "dim": 1},
    }
    parents = {"S": [], "X": ["S"], "R": ["X"]}
    gt = true_E_R_do_S(list(range(N)), n_mc=32000, device=device)

    run(
        "R|X: linear_gaussian",
        {"S": "mle", "X": "linear_gaussian", "R": "linear_gaussian"},
        train,
        gt,
        nodes,
        parents,
        device,
    )
    run(
        "R|X: kde",
        {"S": "mle", "X": "linear_gaussian", "R": "kde"},
        train,
        gt,
        nodes,
        parents,
        device,
    )
    run(
        "R|X: gp_svgp",
        {"S": "mle", "X": "linear_gaussian", "R": "gp_svgp"},
        train,
        gt,
        nodes,
        parents,
        device,
    )
    run_mle_discrete_R(nodes, parents, train, gt, device, BX=20, BR=20)


if __name__ == "__main__":
    main()
