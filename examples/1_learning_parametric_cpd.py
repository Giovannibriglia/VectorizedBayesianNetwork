import networkx as nx
import numpy as np
import pandas as pd

from vbn import VBN

if __name__ == "__main__":

    def generate_random_dataset(N: int, d: int, columns=None):
        cols = columns or [f"feature_{i}" for i in range(d)]
        X = np.random.randint(0, 5, (N, d))
        return pd.DataFrame(X, columns=cols)

    G = nx.DiGraph()
    G.add_edges_from([("feature_0", "feature_2"), ("feature_1", "feature_2")])

    vbn = VBN(G, seed=0, t=1, device="cuda")

    # All nodes trained with the SAME estimator
    vbn.set_learning_method("flow_rnvp")
    df = generate_random_dataset(1000, 3, ["feature_0", "feature_1", "feature_2"])
    vbn.fit(df)
    print("3) Train done")
    print(vbn._nodes)

    df_new = generate_random_dataset(200, 3, ["feature_0", "feature_1", "feature_2"])
    vbn.update(df_new)
    print("4) Update done")
    print(vbn._nodes)
