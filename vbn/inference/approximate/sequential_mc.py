from __future__ import annotations

from typing import Dict, Optional

import torch

from vbn.inference.base import InferenceBackend


class SMC(InferenceBackend):
    """Sequential Monte Carlo with systematic resampling.
    - Improves over plain LW by resampling when ESS falls below a threshold.
    - Supports evidence and do(·).
    """

    def __init__(self, ess_threshold: float = 0.5, device="cpu", **kwargs):
        super().__init__(device, **kwargs)
        self.ess_threshold = ess_threshold

    def _ess(self, w):
        w = w / w.sum().clamp_min(1e-12)
        return 1.0 / (w * w).sum()

    def _resample(self, xdict: Dict[str, torch.Tensor], w):
        N = w.shape[0]
        w = (w / w.sum().clamp_min(1e-12)).detach()
        # systematic resampling
        u0 = torch.rand(1, device=self.device) / N
        cum = torch.cumsum(w, dim=0)
        idx = torch.searchsorted(cum, (u0 + torch.arange(N, device=self.device) / N))
        xdict = {k: v[idx] for k, v in xdict.items()}
        w = torch.full_like(w, 1.0 / N)
        return xdict, w

    def posterior(
        self,
        bn,
        query,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        n_samples: int = 10_000,
    ):
        device = self.device
        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}
        N = n_samples
        order = bn.topo_order
        w = torch.ones(N, device=device)
        samples = {}
        for node in order:
            if node in do:
                v = do[node].to(device)
                # shape [N, 1] for discrete, [N, D] for continuous
                if hasattr(bn.cpd[node], "K"):  # heuristic: discrete
                    v = v.view(1, 1).long().expand(N, 1)
                else:
                    v = v.view(1, -1).expand(N, -1).float()
                samples[node] = v
            else:
                parents = {p: samples[p] for p in bn.parents[node]}
                samp = bn.cpd[node].sample(parents, n_samples=1).squeeze(1)
                if hasattr(bn.cpd[node], "K"):
                    samp = samp.long().view(-1, 1)
                samples[node] = samp

            if evidence and node in evidence:
                y = evidence[node].to(device)
                # score weights
                lp = bn.cpd[node].log_prob(y, {p: samples[p] for p in bn.parents[node]})
                w *= lp.exp().clamp_min(1e-12)
                # clamp sample to evidence with proper shape
                if hasattr(bn.cpd[node], "K"):
                    y = y.view(1, 1).long().expand(N, 1)
                else:
                    y = y.view(1, -1).expand(N, -1).float()
                samples[node] = y

            # resample if ESS low
            if self._ess(w) < self.ess_threshold * N:
                samples, w = self._resample(samples, w)
        w = (w / w.sum().clamp_min(1e-12)).detach()
        out = {}
        for q in query:
            val = samples[q]
            if (
                val.dtype in (torch.long, torch.int64)
                and val.dim() == 2
                and val.shape[1] == 1
                and hasattr(bn.cpd[q], "K")
            ):
                K = bn.cpd[q].K
                hist = torch.zeros(K, device=device)
                idx = val.view(-1).long()
                if idx.numel() != w.numel():
                    if idx.numel() == 1:
                        idx = idx.expand(w.numel())
                    else:
                        raise RuntimeError(
                            f"SMC: mismatch idx({idx.numel()}) vs weights({w.numel()}) for {q}"
                        )
                hist.index_add_(0, idx, w)
                out[q] = hist
            else:
                out[q] = (w.view(-1, 1) * val).sum(dim=0)
        return out

    @torch.no_grad()
    def sample(
        self,
        bn,
        n: int,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        **kw,
    ):
        """
        Conditional sampling via SMC with evidence & interventions.
        Ensures evidence tensors are broadcast to particle batch N before scoring.
        """
        device = bn.device
        evidence = {k: v.to(device) for k, v in (evidence or {}).items()}
        do = {k: v.to(device) for k, v in (do or {}).items()}
        order = bn.topo_order

        N = int(n)
        w = torch.ones(N, device=device)
        samples: Dict[str, torch.Tensor] = {}

        for node in order:
            spec = bn.nodes[node]
            is_disc = spec.get("type") == "discrete"
            D = 1 if is_disc else int(spec.get("dim", 1))

            if node in do:
                v = do[node]
                # broadcast intervention to [N,1] or [N,D]
                if is_disc:
                    v = v.view(1, 1).long().expand(N, 1)
                else:
                    v = v.view(1, D).float().expand(N, D)
                samples[node] = v
            else:
                # sample from CPD given current parents
                parents = {p: samples[p] for p in bn.parents.get(node, [])}
                samp = bn.cpd[node].sample(parents, n_samples=1).squeeze(1)
                if is_disc:
                    samp = samp.long().view(-1, 1)
                samples[node] = samp

            if node in evidence:
                # --- broadcast evidence to particle batch ---
                y = evidence[node]
                if is_disc:
                    yB = y.view(1, 1).long().expand(N, 1)
                else:
                    yB = y.view(1, D).float().expand(N, D)

                # score weights with broadcasted parents & targets
                parents = {p: samples[p] for p in bn.parents.get(node, [])}
                lp = bn.cpd[node].log_prob(
                    yB, parents
                )  # shape [N] (or [N,1] -> squeeze)
                if lp.ndim > 1:
                    lp = lp.squeeze(-1)
                w *= lp.exp().clamp_min(1e-12)

                # clamp stored sample to evidence values
                samples[node] = yB

            # resample if ESS low
            if self._ess(w) < 0.5 * N:
                samples, w = self._resample(samples, w)

        return samples
