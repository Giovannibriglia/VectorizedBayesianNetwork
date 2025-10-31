from __future__ import annotations

from typing import Dict, Optional

import torch

from .ancestral import _flat_index_select_cpt
from .base import BaseSampler, Tensor


def _cpd_discrete_probs(
    bn, node: str, parents_vals: Dict[str, Tensor], n: int, device
) -> torch.Tensor:
    """
    Returns per-sample class probabilities [n, K] for a DISCRETE node.
    Tries, in order:
      1) cpd.table (canonical CPT)
      2) cpd.logits -> softmax
      3) cpd.forward(...) -> logits -> softmax
    """
    cpd = bn.cpd[node]
    parents = bn.parents.get(node, [])
    K = int(bn.nodes[node]["card"])

    # 1) Fast path: canonical CPT table
    if hasattr(cpd, "table"):
        if parents:
            parent_cards = [bn.nodes[p]["card"] for p in parents]
            idxs = [parents_vals[p].view(-1).long().to(device) for p in parents]
            probs = _flat_index_select_cpt(cpd.table, parent_cards, idxs, K, device)
        else:
            probs = cpd.table.to(device).view(1, K).expand(n, K)
        return probs.clamp_min(1e-12)

    # Build parent design for logits-based CPDs (if any)
    if parents:
        cols = [parents_vals[p].to(device).view(n, -1).float() for p in parents]
        X = torch.cat(cols, dim=1) if cols else torch.zeros((n, 0), device=device)
    else:
        X = torch.zeros((n, 0), device=device)

    # 2) logits attribute
    if hasattr(cpd, "logits"):
        logits = cpd.logits.to(device)
        if logits.dim() == 1:
            logits = logits.view(1, -1).expand(n, -1)
        return torch.softmax(logits, dim=-1).clamp_min(1e-12)

    # 3) forward() -> logits
    if hasattr(cpd, "forward"):
        logits = cpd.forward(X)
        if logits.dim() == 1:
            logits = logits.view(1, -1).expand(n, -1)
        return torch.softmax(logits, dim=-1).clamp_min(1e-12)

    raise RuntimeError(
        f"Cannot extract discrete probs for node '{node}' CPD={type(cpd).__name__}"
    )


class RaoBlackwellizedSampler(BaseSampler):
    """
    Mixed sampler:
      - Sample all DISCRETE nodes ancestrally (schema-driven detection)
      - For Linear-Gaussian nodes:
          * if return_gaussian_params=True: return {'mean':[n,D], 'var':[n,D]}
          * else: draw from N(mean, var) (QMC if enabled)
    """

    def sample(
        self,
        bn,
        n: int = 1024,
        do: Optional[Dict[str, Tensor]] = None,
        return_gaussian_params: bool = False,
        **kw,
    ) -> Dict[str, Tensor]:
        device = self.device
        do = {k: v.to(device) for k, v in (do or {}).items()}
        out: Dict[str, Tensor] = {}

        # 1) Sample ALL discrete nodes first (schema-based detection)
        for node in bn.topo_order:
            if bn.nodes.get(node, {}).get("type") != "discrete":
                continue
            if node in do:
                out[node] = do[node].view(1, 1).long().expand(n, 1)
                continue

            parents = bn.parents.get(node, [])
            # ensure discrete parents are present (topo order guarantees they are already sampled)
            probs = _cpd_discrete_probs(bn, node, out, n, device)
            cat = torch.distributions.Categorical(probs=probs)
            out[node] = cat.sample().view(n, 1)

        # 2) Handle linear-Gaussian nodes conditioned on sampled parents
        for node in bn.topo_order:
            cpd = bn.cpd[node]
            is_lg = all(hasattr(cpd, k) for k in ("W", "b", "sigma2"))
            if not is_lg:
                continue

            parents = bn.parents.get(node, [])

            # Intervene?
            if node in do:
                if return_gaussian_params:
                    out[node] = {
                        "mean": do[node].view(1, -1).expand(n, -1).float(),
                        "var": torch.zeros(n, 1, device=device),
                    }
                else:
                    out[node] = do[node].view(1, -1).expand(n, -1).float()
                continue

            # Build design matrix X from parents; if a parent is a dict(mean,var),
            # use the mean for the design and keep its var to propagate uncertainty.
            cols = []
            parent_var_contribs = []  # list of (W_slice, var_tensor or None)
            offset = 0
            for p in parents:
                val = out[p]
                if isinstance(val, dict):
                    mu_p = val["mean"].to(device).view(n, -1).float()
                    var_p = val.get("var", None)
                    if var_p is not None:
                        var_p = var_p.to(device).view(n, -1).float()
                    d_p = mu_p.shape[1]
                    cols.append(mu_p)
                    # remember slice of W for this parent
                    W_slice = cpd.W.to(device)[offset : offset + d_p].view(1, d_p)
                    parent_var_contribs.append((W_slice, var_p))
                    offset += d_p
                else:
                    t = val.to(device)
                    if t.dim() == 1:
                        t = t.view(-1, 1)
                    t = t.float().view(n, -1)
                    d_p = t.shape[1]
                    cols.append(t)
                    W_slice = cpd.W.to(device)[offset : offset + d_p].view(1, d_p)
                    parent_var_contribs.append((W_slice, None))
                    offset += d_p

            X = torch.cat(cols, dim=1) if cols else torch.zeros((n, 0), device=device)

            # mean and base variance (scalar output assumed by this CPD)
            mean = (X @ cpd.W.to(device) + cpd.b.to(device)).view(n, -1)  # [n,1]
            var = cpd.sigma2.to(device).view(1).expand_as(mean)  # [n,1]

            # Propagate parent uncertainty:  var += sum_j (W_j^2 * Var[parent_j])
            # (only for parents that were LG with provided var)
            extra = torch.zeros_like(var)
            cur = 0
            for W_slice, var_p in parent_var_contribs:
                d_p = W_slice.shape[1]
                if var_p is not None:
                    # (n,d_p) · (1,d_p)^2 → (n,1)
                    extra = extra + (var_p * (W_slice**2)).sum(dim=1, keepdim=True)
                cur += d_p
            var = var + extra

            if return_gaussian_params:
                out[node] = {"mean": mean, "var": var}
            else:
                D = mean.shape[1]
                z = self._standard_normal(n, D) * var.sqrt()
                out[node] = mean + z

        return out
