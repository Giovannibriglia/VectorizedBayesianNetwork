# Prediction = cause -> effect
# Diagnosis  = effect -> cause
from __future__ import annotations

import argparse
import itertools
from typing import Dict

import networkx as nx
import pandas as pd
import torch
from _common import auto_device, format_prob, print_env_header, seed_all
from vbn import defaults, VBN

# Ground-truth CPTs (binary variables)
P_EXPOSURE = {0: 0.8, 1: 0.2}
P_DISEASE_GIVEN_EXPOSURE = {
    0: {0: 0.95, 1: 0.05},
    1: {0: 0.70, 1: 0.30},
}
P_FEVER_GIVEN_DISEASE = {
    0: {0: 0.90, 1: 0.10},
    1: {0: 0.20, 1: 0.80},
}
P_COUGH_GIVEN_DISEASE = {
    0: {0: 0.92, 1: 0.08},
    1: {0: 0.30, 1: 0.70},
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Prediction vs diagnosis for a discrete Bayesian network."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def _sample_bernoulli(p: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return (torch.rand(p.shape, generator=gen) < p).long()


def sample_ground_truth(n: int, seed: int) -> pd.DataFrame:
    gen = torch.Generator().manual_seed(seed)
    exposure = _sample_bernoulli(torch.full((n, 1), float(P_EXPOSURE[1])), gen).squeeze(
        -1
    )
    p_disease = torch.where(
        exposure == 1,
        torch.full((n,), float(P_DISEASE_GIVEN_EXPOSURE[1][1])),
        torch.full((n,), float(P_DISEASE_GIVEN_EXPOSURE[0][1])),
    )
    disease = _sample_bernoulli(p_disease.unsqueeze(-1), gen).squeeze(-1)
    p_fever = torch.where(
        disease == 1,
        torch.full((n,), float(P_FEVER_GIVEN_DISEASE[1][1])),
        torch.full((n,), float(P_FEVER_GIVEN_DISEASE[0][1])),
    )
    fever = _sample_bernoulli(p_fever.unsqueeze(-1), gen).squeeze(-1)
    p_cough = torch.where(
        disease == 1,
        torch.full((n,), float(P_COUGH_GIVEN_DISEASE[1][1])),
        torch.full((n,), float(P_COUGH_GIVEN_DISEASE[0][1])),
    )
    cough = _sample_bernoulli(p_cough.unsqueeze(-1), gen).squeeze(-1)
    return pd.DataFrame(
        {
            "Exposure": exposure.numpy(),
            "Disease": disease.numpy(),
            "Fever": fever.numpy(),
            "Cough": cough.numpy(),
        }
    )


def exact_conditional(event: Dict[str, int], evidence: Dict[str, int]) -> float:
    total = 0.0
    matched = 0.0
    for e, d, f, c in itertools.product([0, 1], repeat=4):
        p = (
            P_EXPOSURE[e]
            * P_DISEASE_GIVEN_EXPOSURE[e][d]
            * P_FEVER_GIVEN_DISEASE[d][f]
            * P_COUGH_GIVEN_DISEASE[d][c]
        )
        assign = {"Exposure": e, "Disease": d, "Fever": f, "Cough": c}
        if all(assign[k] == v for k, v in evidence.items()):
            total += p
            if all(assign[k] == v for k, v in event.items()):
                matched += p
    if total == 0.0:
        raise ValueError("Evidence has zero probability under the CPTs.")
    return matched / total


def importance_sample_joint(
    vbn: VBN, evidence: Dict[str, int], n_samples: int
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    samples: Dict[str, torch.Tensor] = {}
    batch_size = 1
    with torch.no_grad():
        for node in vbn.dag.topological_order():
            parents = vbn.dag.parents(node)
            if parents:
                parent_tensor = torch.cat([samples[p] for p in parents], dim=-1)
            else:
                parent_tensor = torch.zeros(batch_size, 0, device=vbn.device)
            samples[node] = vbn.nodes[node].sample(parent_tensor, n_samples)

        log_weights = torch.zeros(batch_size, n_samples, device=vbn.device)
        for node, value in evidence.items():
            value_tensor = torch.tensor([[value]], device=vbn.device).expand(
                batch_size, n_samples, 1
            )
            parents = vbn.dag.parents(node)
            if parents:
                parent_tensor = torch.cat([samples[p] for p in parents], dim=-1)
            else:
                parent_tensor = torch.zeros(batch_size, 0, device=vbn.device)
            log_weights = log_weights + vbn.nodes[node].log_prob(
                value_tensor, parent_tensor
            )
        weights = torch.softmax(log_weights, dim=-1)
    return samples, weights


def weighted_prob(
    samples: Dict[str, torch.Tensor],
    weights: torch.Tensor,
    event: Dict[str, int],
) -> float:
    indicator = torch.ones_like(weights)
    for node, value in event.items():
        node_samples = samples[node].squeeze(-1)
        indicator = indicator * (node_samples == value).float()
    return float((weights * indicator).sum().item())


def print_compare(label: str, exact: float, inferred: float) -> None:
    err = abs(exact - inferred)
    print(
        f"{label}: exact={format_prob(exact)} | vbn={format_prob(inferred)} | "
        f"abs_err={format_prob(err)}"
    )


def main() -> None:
    args = _parse_args()
    seed_all(args.seed)
    device = auto_device()
    print_env_header("06_prediction_vs_diagnosis", device)

    df = sample_ground_truth(n=3000, seed=args.seed)

    g = nx.DiGraph()
    g.add_edges_from(
        [("Exposure", "Disease"), ("Disease", "Fever"), ("Disease", "Cough")]
    )

    fit_conf = {"epochs": 50, "batch_size": 1024, "lr": 1e-3, "weight_decay": 0.0}
    node_cpd = {**defaults.cpd("softmax_nn"), "n_classes": 2, "fit": dict(fit_conf)}
    vbn = VBN(g, seed=args.seed, device=device)
    vbn.set_learning_method(
        method=defaults.learning("node_wise"),
        nodes_cpds={
            "Exposure": dict(node_cpd),
            "Disease": dict(node_cpd),
            "Fever": dict(node_cpd),
            "Cough": dict(node_cpd),
        },
    )
    vbn.fit(df, verbosity=0)

    n_samples = 5000

    print("\nPrediction (forward inference)")
    evidence_pred = {"Exposure": 1}
    samples_pred, weights_pred = importance_sample_joint(vbn, evidence_pred, n_samples)
    exact_ff = exact_conditional({"Fever": 1, "Cough": 1}, evidence_pred)
    vbn_ff = weighted_prob(samples_pred, weights_pred, {"Fever": 1, "Cough": 1})
    print_compare("P(Fever=1, Cough=1 | Exposure=1)", exact_ff, vbn_ff)

    exact_f = exact_conditional({"Fever": 1}, evidence_pred)
    vbn_f = weighted_prob(samples_pred, weights_pred, {"Fever": 1})
    print_compare("P(Fever=1 | Exposure=1)", exact_f, vbn_f)

    print("\nDiagnosis (backward inference)")
    evidence_diag = {"Fever": 1, "Cough": 1}
    samples_diag, weights_diag = importance_sample_joint(vbn, evidence_diag, n_samples)
    exact_d = exact_conditional({"Disease": 1}, evidence_diag)
    vbn_d = weighted_prob(samples_diag, weights_diag, {"Disease": 1})
    print_compare("P(Disease=1 | Fever=1, Cough=1)", exact_d, vbn_d)

    evidence_diag_single = {"Fever": 1}
    samples_diag_single, weights_diag_single = importance_sample_joint(
        vbn, evidence_diag_single, n_samples
    )
    exact_d_single = exact_conditional({"Disease": 1}, evidence_diag_single)
    vbn_d_single = weighted_prob(
        samples_diag_single, weights_diag_single, {"Disease": 1}
    )
    print_compare("P(Disease=1 | Fever=1)", exact_d_single, vbn_d_single)


if __name__ == "__main__":
    main()
