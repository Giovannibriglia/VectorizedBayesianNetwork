from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from vbn.inference.base import _as_B1, BaseInferencer


class ABCInferencer(BaseInferencer):
    @torch.no_grad()
    def infer_posterior(
        self,
        query: str,
        evidence: Optional[Dict[str, torch.Tensor]] = None,
        do: Optional[Dict[str, torch.Tensor]] = None,
        num_samples: int = 4096,
        keep_top_k: int = 1024,
        dist: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]
        ] = None,
        **kwargs,
    ):
        ev = {k: _as_B1(v, self.device).float() for k, v in (evidence or {}).items()}
        do_ = {k: _as_B1(v, self.device).float() for k, v in (do or {}).items()}

        self._check_caps({n: ("has_sample",) for n in self.topo if n not in do_})

        B = 1
        for d in list(ev.values()) + list(do_.values()):
            B = max(B, d.shape[0])

        N = num_samples
        assign: Dict[str, torch.Tensor] = {}  # node -> [B,N]

        def parents_flat(node: str) -> torch.Tensor:
            ps = self.parents[node]
            if not ps:
                return torch.empty((B * N, 0), device=self.device)
            xs = [assign[p].float().unsqueeze(-1) for p in ps]  # [B,N,1]
            X = torch.cat(xs, dim=-1)  # [B,N,|Pa|]
            return X.reshape(B * N, -1)  # [B*N, |Pa|]

        # forward sample
        for n in self.topo:
            if n in do_:
                assign[n] = do_[n].expand(B, N)  # [B,N]
                continue
            Xf = parents_flat(n)
            y = self.nodes[n].sample(Xf, n=1).squeeze(0)  # [B*N,1]
            assign[n] = y.reshape(B, N)  # [B,N]

        def default_dist(
            sim: Dict[str, torch.Tensor], obs: Dict[str, torch.Tensor]
        ) -> torch.Tensor:
            if not obs:
                return torch.zeros(B, N, device=self.device)
            d = 0.0
            for k, v in obs.items():
                sv = sim[k].float()  # [B,N]
                vb = v.expand(B, N)  # [B,N]
                d = d + F.mse_loss(sv, vb, reduction="none")  # [B,N]
            return d

        dist_fn = dist or default_dist
        D = dist_fn(assign, ev)  # [B,N]

        K = min(keep_top_k, N)
        idx = torch.topk(-D, k=K, dim=-1).indices  # [B,K]
        q_all = assign[query]  # [B,N]
        q_kept = torch.gather(q_all, dim=-1, index=idx)  # [B,K]

        pdf = {"kept": torch.tensor(K, device=self.device)}
        samples = {query: q_kept.detach()}  # [B,K]
        return pdf, samples
