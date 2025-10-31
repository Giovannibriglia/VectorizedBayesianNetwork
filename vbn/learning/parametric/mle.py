from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from vbn.learning.base import BaseCPD, Tensor


class MLECategoricalCPD(BaseCPD):
    # ─────────────────────────────────────────────────────────────────────────────
    # Discrete MLE CPD — Categorical with context-dependent logits (tables)
    # Supports: no parents → global categorical; with parents (discrete) → tabular CPT
    # For simplicity: assumes *discrete* parents already encoded as integer indices.
    # ─────────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        card_y: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        # build parameter table with shape [prod(parent_cards), card_y]
        self.parent_cards: List[int] = [int(c) for c in parents.values()]
        self.card_y = int(card_y)
        table_size = 1
        for c in self.parent_cards:
            table_size *= c
        # store logits as parameter (initialized uniform)
        self.logits = nn.Parameter(
            torch.zeros(table_size, self.card_y, device=self.device)
        )
        self.register_buffer(
            "counts", torch.ones_like(self.logits)
        )  # for online updates

    def _flat_index(self, parents: Dict[str, Tensor]) -> Tensor:
        """Map discrete parent assignments to a flat index in [0, table_size).
        parents: dict of integer tensors (N,)
        returns idx: (N,)
        """
        if len(self.parent_cards) == 0:
            return torch.zeros(
                next(iter(parents.values())).shape[0] if parents else 1,
                dtype=torch.long,
                device=self.device,
            )
        keys = list(self.parents.keys())
        N = parents[keys[0]].shape[0]
        idx = torch.zeros(N, dtype=torch.long, device=self.device)
        mult = 1
        for k, card in reversed(list(zip(keys[::-1], self.parent_cards[::-1]))):
            # (Note: keys reversed twice → original order); assume parents[k] in [0, card)
            v = parents[k].to(self.device).long()
            idx += v * mult
            mult *= card
        return idx

    def forward(self, parents: Dict[str, Tensor]) -> Tensor:
        idx = self._flat_index(parents)
        return self.logits[idx]  # (N, card_y)

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        logits = self.forward(parents)
        return -F.cross_entropy(logits, y.long().to(self.device), reduction="none")

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        # compute counts per parent-config & class
        if len(self.parent_cards) == 0:
            # global categorical
            y_oh = F.one_hot(y.long(), num_classes=self.card_y).float().sum(0)
            counts = y_oh.to(self.device).unsqueeze(0)
            self.counts.copy_(counts + 1.0)
            self.logits.copy_((self.counts / self.counts.sum(-1, keepdim=True)).log())
            return
        idx = self._flat_index(parents)
        K = self.logits.shape[0]
        counts = torch.zeros(K, self.card_y, device=self.device)
        counts.index_add_(0, idx, F.one_hot(y.long(), num_classes=self.card_y).float())
        counts += 1.0  # Laplace
        self.counts.copy_(counts)
        probs = counts / counts.sum(-1, keepdim=True)
        self.logits.copy_(probs.clamp_min(1e-8).log())

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 0.1) -> None:
        # exponential moving average of counts → logits
        idx = (
            self._flat_index(parents)
            if len(self.parent_cards)
            else torch.zeros_like(y, device=self.device)
        )
        K = self.logits.shape[0]
        fresh = torch.zeros(K, self.card_y, device=self.device)
        fresh.index_add_(0, idx, F.one_hot(y.long(), num_classes=self.card_y).float())
        fresh += 1.0
        self.counts.mul_(1 - alpha).add_(alpha * fresh)
        probs = self.counts / self.counts.sum(-1, keepdim=True)
        self.logits.copy_(probs.clamp_min(1e-8).log())

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        logits = self.forward(parents)  # (N, C)
        probs = logits.softmax(-1)
        N, C = probs.shape
        cat = torch.distributions.Categorical(probs=probs)
        draws = cat.sample((n_samples,)).transpose(0, 1)  # (N, n_samples)
        return draws.unsqueeze(-1).float()
