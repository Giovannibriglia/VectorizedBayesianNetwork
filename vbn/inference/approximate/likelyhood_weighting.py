from __future__ import annotations

from typing import Dict, Optional, Sequence

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

    def posterior(
        self,
        bn,
        query: Sequence[str],
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        n_samples: int = 4096,
    ) -> Dict[str, torch.Tensor]:
        """
        Likelihood Weighting (LW) posterior.

        • Discrete nodes are always handled (sampled / weighted).
        • Continuous nodes are skipped unless they appear in evidence, do, or query.
        • 'do' fixes a node's value and bypasses its CPD.
        • Evidence updates weights via CPD.log_prob(evidence, parents_broadcasted_to_N).
        • Outputs:
            - discrete scalar queries -> normalized histogram [K]
            - continuous queries     -> {'mean': [D], 'var': [D]}
        """
        device = self.device
        N = int(n_samples)

        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}
        need_node = set(query) | set(evidence.keys()) | set(do.keys())

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
            """Only for log_prob during evidence weighting. Sampling uses original parents."""
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

        # ---- ancestral pass ----
        for node in bn.topo_order:
            spec = bn.nodes[node]
            ntype = spec["type"]  # 'discrete' | 'gaussian'
            is_disc = ntype == "discrete"
            D = 1 if is_disc else int(spec.get("dim", 1))

            # parents collected from what's already in samples (do/evidence/simulated)
            parents = {p: samples[p] for p in bn.parents.get(node, []) if p in samples}

            # (1) do-intervention: fix value (broadcast to [N,D]) and continue
            if node in do:
                vN = _broadcast_obs(do[node], D, is_disc)
                samples[node] = vN
                continue

            # (2) evidence: update weights using likelihood under CPD
            if node in evidence:
                yN = _broadcast_obs(evidence[node], D, is_disc)
                parents_b = _broadcast_parents_for_logprob(parents)
                lp = bn.cpd[node].log_prob(yN, parents_b)  # [N] or [N,1]
                if lp.ndim > 1:
                    lp = lp.squeeze(-1)
                weights *= lp.exp().clamp_min(1e-32)
                samples[node] = yN
                continue

            # (3) simulate if discrete OR requested in query; else skip continuous to save work
            if is_disc or (node in query) or (node in need_node):
                # IMPORTANT: for sampling, pass ORIGINAL parents (no broadcast)
                samp = bn.cpd[node].sample(parents, n_samples=1).squeeze(1)  # [N, D]
                samples[node] = samp.long() if is_disc else samp.float()
            else:
                continue

        # ---- normalize weights ----
        Z = weights.sum().clamp_min(1e-32)
        w = (weights / Z).detach()  # [N]

        # ---- build outputs ----
        out: Dict[str, torch.Tensor] = {}
        for q in query:
            spec_q = bn.nodes[q]
            is_disc_q = spec_q["type"] == "discrete"
            # Dq = 1 if is_disc_q else int(spec_q.get("dim", 1))
            Kq = (
                int(spec_q.get("card", getattr(bn.cpd[q], "K", 0)))
                if is_disc_q
                else None
            )

            val = samples.get(q, None)
            if val is None:
                # lazy sample if skipped earlier (use ORIGINAL parents for sampling)
                parents_q = {
                    p: samples[p] for p in bn.parents.get(q, []) if p in samples
                }
                v = bn.cpd[q].sample(parents_q, n_samples=1).squeeze(1)
                val = v.long() if is_disc_q else v.float()

            if is_disc_q:
                if Kq is None or Kq <= 0:
                    raise RuntimeError(f"Missing cardinality for discrete node {q}.")

                # ensure 2D indices [rows,1] if possible
                if isinstance(val, torch.Tensor):
                    if val.ndim == 1:
                        val = val.view(-1, 1)
                    # collapse accidental over-expansion (e.g., [N*r,1] where r>1)
                    if val.shape[0] != w.numel() and (val.shape[0] % w.numel()) == 0:
                        r = val.shape[0] // w.numel()
                        val = val.view(w.numel(), r, 1)[:, 0, :]

                # Case A: indices [N,1]
                if (
                    val.dtype in (torch.long, torch.int64)
                    and val.ndim == 2
                    and val.shape[1] == 1
                ):
                    idx = val.view(-1).long()
                    if idx.numel() != w.numel():
                        if idx.numel() == 1:
                            idx = idx.expand(w.numel())
                        elif (idx.numel() % w.numel()) == 0:
                            r = idx.numel() // w.numel()
                            idx = idx.view(w.numel(), r)[:, 0]
                        else:
                            raise RuntimeError(
                                f"LW: mismatch idx({idx.numel()}) vs weights({w.numel()}) for {q}"
                            )
                    hist = torch.zeros(Kq, device=device)
                    hist.index_add_(0, idx, w)
                    out[q] = hist / hist.sum().clamp_min(1e-32)
                    continue

                # Case B: probability rows [N,K]
                if val.dtype.is_floating_point and val.ndim == 2 and val.shape[1] == Kq:
                    wcol = w.view(-1, 1)
                    hist = (wcol * val).sum(dim=0)  # [K]
                    out[q] = hist / hist.sum().clamp_min(1e-32)
                    continue

                # Case C: logits/prob s with wrong width -> argmax to indices
                if val.ndim == 2 and val.shape[1] > 1:
                    idx = val.argmax(dim=1).view(-1, 1).long()
                    hist = torch.zeros(Kq, device=device)
                    hist.index_add_(0, idx.view(-1), w)
                    out[q] = hist / hist.sum().clamp_min(1e-32)
                    continue

                # Case D: scalar replicated
                if val.ndim == 0:
                    idx = val.view(1).long().expand(w.numel())
                    hist = torch.zeros(Kq, device=device)
                    hist.index_add_(0, idx, w)
                    out[q] = hist / hist.sum().clamp_min(1e-32)
                    continue

                raise TypeError(
                    f"LW: unsupported discrete value shape for {q}: {tuple(val.shape)} dtype={val.dtype}"
                )

            # ----- continuous (or vector) → weighted mean / var -----
            v = val.float()
            if v.ndim == 1:
                v = v.view(-1, 1)
            # guard: if someone over-expanded v (e.g., [N*r, D]), collapse
            if v.shape[0] != w.numel():
                if v.shape[0] % w.numel() == 0:
                    r = v.shape[0] // w.numel()
                    v = v.view(w.numel(), r, v.shape[1]).mean(dim=1)
                else:
                    raise RuntimeError(
                        f"LW: unexpected shape for continuous {q}: {tuple(v.shape)} vs weights {w.numel()}"
                    )

            wcol = w.view(-1, 1)
            mean = (wcol * v).sum(dim=0)
            ex2 = (wcol * (v * v)).sum(dim=0)
            var = (ex2 - mean * mean).clamp_min(0.0)
            out[q] = {"mean": mean, "var": var}

        return out
