import networkx as nx
import pandas as pd
import torch
from vbn.core import CausalBayesNet, merge_learnparams
from vbn.plotting import (
    draw_graph,
    plot_continuous_gaussian,
    plot_discrete_posteriors,
    plot_discrete_table,
    plot_lg_params,
)

# Build a mixed graph
G = nx.DiGraph([("X", "Y"), ("Z", "Y"), ("Y", "A"), ("X", "A")])
types = {"X": "discrete", "Z": "discrete", "Y": "discrete", "A": "continuous"}
cards = {"X": 3, "Z": 2, "Y": 4}

bn = CausalBayesNet(G, types, cards)

# Fake data
N = 20000
data = {
    "X": torch.randint(0, 3, (N,)),
    "Z": torch.randint(0, 2, (N,)),
    "Y": torch.randint(0, 4, (N,)),
}
# append a dict
bn.add_data(
    {
        "X": torch.randint(0, 3, (1000,)),
        "Z": torch.randint(0, 2, (1000,)),
        "Y": torch.randint(0, 4, (1000,)),
    },
    update_params=True,
)

# append a DataFrame
df = pd.DataFrame({"X": [0, 1, 2], "Z": [1, 0, 1], "Y": [3, 2, 1]})
bn.add_data(df, update_params=True)

# Learn discrete MLE (tabular)
lp_disc = bn.fit_discrete_mle(data, laplace_alpha=0.5)

# Save
bn.save_params(lp_disc, "lp_disc.td.pt")

# Load (device-mapped)
lp_disc = bn.load_params(
    "lp_disc.td.pt",
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# Learn continuous MLP (uses discrete parents via one-hots if needed)
lp_cont = bn.fit_continuous_mlp(data, epochs=10, hidden=64)
print("**************")

# DISCRETE exact with do:
postY_do = bn.infer_discrete_exact(
    lp_disc, evidence={"Z": torch.tensor(1)}, query=["Y"], do={"X": torch.tensor(2)}
)
print({k: v.shape for k, v in postY_do.items()})  # {'Y': torch.Size([4])}

# CONTINUOUS gaussian with do: note query must be ['A'] here
lp_cont_lg = bn.materialize_lg_from_cont_mlp(lp_cont, data=data)
mu_do, cov_do = bn.infer_continuous_gaussian(
    lp_cont_lg,
    evidence={"A": torch.tensor(1.0)},
    query=["A"],
    do={
        "A": torch.tensor(0.5)
    },  # or intervene on A or any continuous node in your model
)
print(mu_do, cov_do.shape)  # {'A': tensor(...)} torch.Size([1,1])


print("**************")

# DISCRETE exact with do:
postY_do = bn.infer_discrete_exact(
    lp_disc, evidence={"Z": torch.tensor(1)}, query=["Y"], do={"X": torch.tensor(2)}
)
print(postY_do)

# DISCRETE approx with do (works with tables or discrete-MLPs):
postY_approx_do = bn.infer_discrete_approx(
    merge_learnparams(lp_disc),  # or pass lp with discrete MLPs
    evidence={"Z": torch.tensor([0, 1])},  # batched B=2
    query=["Y"],
    do={"X": torch.tensor([2, 2])},
    num_samples=4096,
)
print(postY_approx_do)

# CONTINUOUS gaussian with do (exact canonical; cuts edges):
lp_cont_lg = bn.materialize_lg_from_cont_mlp(lp_cont, data=data)
mu_do, cov_do = bn.infer_continuous_gaussian(
    lp_cont_lg,
    evidence={"A": torch.tensor(1.0)},
    query=["B", "C"],
    do={"B": torch.tensor(0.5)},
)
print(mu_do, cov_do)

# CONTINUOUS approx with do (samples, no weighting for do vars):
lp_all = merge_learnparams(lp_disc, lp_cont)  # combine discrete+continuous
postA_do = bn.infer_continuous_approx(
    lp_all, evidence={"Z": torch.tensor(1)}, query=["A"], do={"X": torch.tensor(2)}
)
print(postA_do)


# 1) Graph
figG = draw_graph(bn.meta)
figG.savefig("bn_graph.png", dpi=500)

# 2) Discrete CPD
tblY = lp_disc.discrete_tables["Y"]
figCPD = plot_discrete_table(tblY, "Y")
figCPD.savefig("cpd.png", dpi=500)

# 3) Discrete posterior
postY = bn.infer_discrete_exact(lp_disc, evidence={"X": torch.tensor(1)}, query=["Y"])
figs = plot_discrete_posteriors(postY)  # dict of figures
for s, fig in figs.items():
    fig.savefig(f"{s}.png", dpi=500)

# 4) Continuous Gaussian posterior (after LG inference)
lp_cont = bn.fit_continuous_mlp(data, epochs=5)
lp_lg = bn.materialize_lg_from_cont_mlp(lp_cont, data=data)
mu, Sigma = bn.infer_continuous_gaussian(
    lp_lg, evidence={"A": torch.tensor(1.0)}, query=["A"]
)
fig1d = plot_continuous_gaussian(mu, Sigma, query_order=["A"], dims=["A"])
fig1d.savefig("cont_post.png", dpi=500)

# 5) Inspect LG params
figW, figS = plot_lg_params(lp_lg.lg)
figW.savefig("figW.png", dpi=500)
figS.savefig("figS.png", dpi=500)
