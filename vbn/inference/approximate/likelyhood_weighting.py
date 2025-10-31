from __future__ import annotations

from typing import Dict

import torch

from vbn.inference.base import InferenceBackend


class LikelihoodWeighting(InferenceBackend):
    """Generic LW over the BN's topological order.
    - evidence: variables to clamp via weight accumulation
    - do: intervene by replacing CPD sampling with provided value(s)
    Returns empirical moments for queried vars.
    """

    def __init__(self, device="cpu", **kwargs):
        super().__init__(device, **kwargs)

    def posterior(self, bn, query, evidence=None, do=None, n_samples=4096):
        device = self.device
        N = int(n_samples)

        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}
        # need_node = set(query) | set(evidence.keys()) | set(do.keys())

        def _broadcast_obs(x: torch.Tensor, D: int, is_disc: bool) -> torch.Tensor:
            # -> [N, D]
            if x.ndim == 0:
                x = x.view(1, 1)
            elif x.ndim == 1:
                x = x.view(1, -1)
            x = x.expand(N, D).clone()
            return x.long() if is_disc else x.float()

        def _broadcast_parents_for_logprob(
            pars: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for k, v in pars.items():
                if v.ndim == 1:
                    v = v.view(1, -1)
                if v.shape[0] == 1 and N > 1:
                    v = v.expand(N, v.shape[1]).clone()
                out[k] = v
            return out

        weights = torch.ones(N, device=device)
        samples: Dict[str, torch.Tensor] = {}

        # --- on-demand materialization ---
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def _ensure_value(var: str):
            # already sampled?
            if var in samples:
                return samples[var]

            spec = bn.nodes[var]
            is_disc = spec["type"] == "discrete"
            D = 1 if is_disc else int(spec.get("dim", 1))

            # do: clamp
            if var in do:
                vN = _broadcast_obs(do[var], D, is_disc)
                samples[var] = vN
                return vN

            # evidence: weight & clamp
            if var in evidence:
                # ensure parents first for likelihood
                req_pars = bn.parents.get(var, [])
                pars = {p: _ensure_value(p) for p in req_pars}
                pars_b = _broadcast_parents_for_logprob(pars)
                yN = _broadcast_obs(evidence[var], D, is_disc)
                lp = bn.cpd[var].log_prob(yN, pars_b)
                if lp.ndim > 1:
                    lp = lp.squeeze(-1)
                weights.mul_(lp.exp().clamp_min(1e-32))
                samples[var] = yN
                return yN

            # otherwise: simulate from CPD, but *first* ensure parents
            req_pars = bn.parents.get(var, [])
            pars = {p: _ensure_value(p) for p in req_pars}  # recursive
            v = bn.cpd[var].sample(pars, n_samples=1).squeeze(1)
            v = v.long() if is_disc else v.float()
            samples[var] = v
            return v

        # --- single forward scan: only touch items that are explicitly constrained ---
        # Evidence and do must be “ensured” to accumulate weights early.
        for node in bn.topo_order:
            if node in do or node in evidence:
                _ensure_value(node)

        # --- build outputs (ensure queried vars exist) ---
        out: Dict[str, torch.Tensor] = {}
        for q in query:
            val = _ensure_value(q)
            spec_q = bn.nodes[q]
            is_disc_q = spec_q["type"] == "discrete"
            if is_disc_q:
                Kq = int(spec_q.get("card", getattr(bn.cpd[q], "K", 0)))
                if val.ndim == 1:
                    val = val.view(-1, 1)
                if val.dtype not in (torch.long, torch.int64):
                    # treat as probs/logits -> argmax to indices
                    if val.ndim == 2 and val.shape[1] > 1:
                        idx = val.argmax(dim=1).long()
                    else:
                        idx = val.view(-1).round().long()
                else:
                    idx = val.view(-1)
                # normalize weights
                Z = weights.sum().clamp_min(1e-32)
                w = (weights / Z).detach()
                hist = torch.zeros(Kq, device=device)
                hist.index_add_(0, idx, w)
                out[q] = hist / hist.sum().clamp_min(1e-32)
            else:
                v = val.float()
                if v.ndim == 1:
                    v = v.view(-1, 1)
                Z = weights.sum().clamp_min(1e-32)
                w = (weights / Z).detach().view(-1, 1)
                mean = (w * v).sum(dim=0)
                ex2 = (w * (v * v)).sum(dim=0)
                var = (ex2 - mean * mean).clamp_min(0.0)
                out[q] = {"mean": mean, "var": var}

        return out
