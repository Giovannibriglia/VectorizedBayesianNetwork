import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from vbn import VBN

# toy: y = sin(a - b) + 0.1*eps (continuous child)
N = 2000
rng = np.random.default_rng(0)
a = rng.integers(0, 5, size=N).astype(np.float32)
b = rng.integers(0, 5, size=N).astype(np.float32)
c = np.sin(a - b) + 0.1 * rng.normal(size=N)
df = pd.DataFrame({"feature_0": a, "feature_1": b, "feature_2": c})

G = nx.DiGraph()
G.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

# try MMD generator
vbn = VBN(G, seed=0, device="cuda")
vbn.set_learning_method("kde_gaussian_diff")
vbn.fit(df)
print("✓ MMD implicit generator trained")

# sample conditioned on first 3 parent rows
X = torch.tensor(
    df[["feature_0", "feature_1"]].values[:3], dtype=torch.float32, device=vbn.device
)
y_samp = vbn._nodes["feature_2"].sample(X, n=5)  # [5,3,1]
print(y_samp.squeeze(-1).detach().cpu().numpy())

vbn.set_inference_method("montecarlo.lw")
pdf, samples = vbn.infer_posterior("feature_2")

p = pdf["weights"].squeeze(-1).detach().cpu().numpy()
s = samples["feature_2"].squeeze(-1).detach().cpu().numpy()

print(p.shape, s.shape)
plt.figure(dpi=500)
plt.plot(p)
plt.show()

plt.figure(dpi=500)
plt.plot(s)
plt.show()
