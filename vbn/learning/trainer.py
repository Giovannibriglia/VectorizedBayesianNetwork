from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from . import CPD_REGISTRY

Tensor = torch.Tensor


@dataclass
class NodeSpec:
    name: str
    kind: str  # one of CPD_REGISTRY keys
    y_shape: int  # card for discrete / dim for continuous


class VBNParallelTrainer(nn.Module):
    def __init__(
        self,
        vbn,
        node_specs: List[NodeSpec],
        device: Optional[str | torch.device] = None,
    ):
        super().__init__()
        self.vbn = vbn
        self.device = torch.device(device or vbn.device)
        self.cpds = nn.ModuleDict()

        for spec in node_specs:
            kind = spec.kind

            # Build both views of parents once
            par_dims: Dict[str, int] = {}
            par_cards: Dict[str, int] = {}
            for p in vbn.parents.get(spec.name, []):
                pinfo = vbn.nodes[p]
                if pinfo.get("type") == "discrete":
                    par_dims[p] = 1
                    par_cards[p] = int(pinfo["card"])
                else:
                    par_dims[p] = int(pinfo.get("dim", 1))
                    # no entry in par_cards for continuous parents

            # Instantiate exactly once, with the right kwargs/parents
            if kind == "mle":
                # guard: all parents must be discrete
                for p in vbn.parents.get(spec.name, []):
                    if vbn.nodes[p].get("type") != "discrete":
                        raise ValueError(
                            f"MLECategoricalCPD('{spec.name}') requires all parents discrete; "
                            f"got continuous parent '{p}'. Discretize it (e.g., X→X_disc)."
                        )
                cpd = CPD_REGISTRY[kind](
                    name=spec.name,
                    parents=par_cards,  # cardinalities
                    card_y=spec.y_shape,  # number of categories for Y
                    device=self.device,
                )

            elif kind in {"linear_gaussian", "gp_svgp"}:
                cpd = CPD_REGISTRY[kind](
                    name=spec.name,
                    parents=par_dims,  # feature dims (discrete coded as 1)
                    out_dim=spec.y_shape,  # output dim
                    device=self.device,
                )

            elif kind == "kde":
                cpd = CPD_REGISTRY[kind](
                    name=spec.name,
                    parents=par_dims,  # feature dims
                    y_dim=spec.y_shape,  # output dim
                    device=self.device,
                )

            else:
                raise ValueError(f"Unknown CPD kind: {kind}")

            self.cpds[spec.name] = cpd
            self.vbn.set_cpd(spec.name, cpd)

        self.to(self.device)

    def _parents(self, batch: Dict[str, Tensor], node: str) -> Dict[str, Tensor]:
        return {p: batch[p] for p in self.vbn.parents.get(node, [])}

    def nll(self, batch: Dict[str, Tensor]) -> Tensor:
        return torch.stack(
            [
                cpd.training_loss(batch[n], self._parents(batch, n))
                for n, cpd in self.cpds.items()
            ]
        ).sum()

    @torch.no_grad()
    def fit(self, data: Dict[str, Tensor]) -> None:
        for n, cpd in self.cpds.items():
            cpd.fit(self._parents(data, n), data[n])

    @torch.no_grad()
    def partial_fit(self, data: Dict[str, Tensor]) -> None:
        for n, cpd in self.cpds.items():
            cpd.update(self._parents(data, n), data[n])

    def train_minibatch(
        self, iterator, steps: int = 1000, lr: float = 1e-3, clip_grad: float = 1.0
    ) -> None:
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        it = iter(iterator)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = self.nll(batch)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
            opt.step()


class DictDataLoader:
    def __init__(
        self, data: Dict[str, Tensor], batch_size: int = 2048, shuffle: bool = True
    ):
        N = next(iter(data.values())).shape[0]
        self.data, self.N, self.batch_size, self.shuffle = data, N, batch_size, shuffle

    def __iter__(self):
        idx = torch.randperm(self.N) if self.shuffle else torch.arange(self.N)
        for s in range(0, self.N, self.batch_size):
            sel = idx[s : s + self.batch_size]
            yield {k: v[sel] for k, v in self.data.items()}
