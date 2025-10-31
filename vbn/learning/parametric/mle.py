from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from ..base import BaseCPD, Tensor


class MLECategoricalCPD(BaseCPD):
    """Discrete parents (as integer codes) → Categorical child."""

    def __init__(
        self,
        name: str,
        parents: Dict[str, int],
        card_y: int,
        device: str | torch.device = "cpu",
    ):
        super().__init__(name, parents, device)
        self.parent_cards: List[int] = [int(c) for c in parents.values()]
        self.card_y = int(card_y)
        size = 1
        for c in self.parent_cards:
            size *= c
        self.logits = nn.Parameter(
            torch.zeros(max(size, 1), self.card_y, device=self.device)
        )
        self.register_buffer("counts", torch.ones_like(self.logits))

    def _flat_index(self, parents: Dict[str, Tensor]) -> Tensor:
        if not self.parent_cards:
            N = next(iter(parents.values())).shape[0] if parents else 1
            return torch.zeros(N, dtype=torch.long, device=self.device)
        keys = list(self.parents.keys())
        first = parents[keys[0]].to(self.device).long().view(-1)  # (N,)
        idx = first
        for k, card in zip(keys[1:], self.parent_cards[1:]):
            pk = parents[k].to(self.device).long().view(-1)  # (N,)
            idx = idx * card + pk
        return idx.view(-1)  # ensure 1-D

    def forward(self, parents: Dict[str, Tensor]) -> Tensor:
        return self.logits[self._flat_index(parents)]

    def log_prob(self, y: Tensor, parents: Dict[str, Tensor]) -> Tensor:
        y = y.view(-1)
        logits = self.forward(parents)
        # If root (no parents), forward() returns shape (1, K).
        # Expand to match the target batch size.

        if not self.parent_cards and logits.shape[0] == 1:
            logits = logits.expand(y.shape[0], -1)

        return -F.cross_entropy(logits, y.long().to(self.device), reduction="none")

    @torch.no_grad()
    def fit(self, parents: Dict[str, Tensor], y: Tensor) -> None:
        y = y.view(-1)
        # safety: clamp target labels to [0, K-1]
        y = y.clamp_(min=0, max=self.card_y - 1)
        if not self.parent_cards:
            counts = (
                F.one_hot(y.long(), num_classes=self.card_y)
                .float()
                .sum(0, keepdim=True)
                .to(self.device)
            )
        else:
            idx = self._flat_index(parents)
            # safety: ensure parent codes are within declared cardinalities
            K = self.logits.shape[0]
            idx = idx.clamp_(min=0, max=K - 1)
            counts = torch.zeros(K, self.card_y, device=self.device)
            counts.index_add_(
                0, idx, F.one_hot(y.long(), num_classes=self.card_y).float()
            )
        self.counts.copy_(counts + 1.0)
        probs = (self.counts / self.counts.sum(-1, keepdim=True)).clamp_min(1e-8)
        self.logits.copy_(probs.log())

    @torch.no_grad()
    def update(self, parents: Dict[str, Tensor], y: Tensor, alpha: float = 0.1) -> None:
        y = y.view(-1).clamp_(min=0, max=self.card_y - 1)
        if not self.parent_cards:
            idx = torch.zeros_like(y, device=self.device)
        else:
            idx = self._flat_index(parents)
            idx = idx.clamp_(min=0, max=self.logits.shape[0] - 1)
        K = self.logits.shape[0]
        fresh = torch.zeros(K, self.card_y, device=self.device)
        fresh.index_add_(0, idx, F.one_hot(y.long(), num_classes=self.card_y).float())
        fresh += 1.0
        self.counts.mul_(1 - alpha).add_(alpha * fresh)
        probs = (self.counts / self.counts.sum(-1, keepdim=True)).clamp_min(1e-8)
        self.logits.copy_(probs.log())

    @torch.no_grad()
    def sample(self, parents: Dict[str, Tensor], n_samples: int) -> Tensor:
        logits = self.forward(parents)
        if not self.parent_cards and logits.shape[0] == 1:
            # If you can infer an intended batch size N here, expand:
            # logits = logits.expand(N, -1)
            pass
        probs = logits.softmax(-1)
        cat = torch.distributions.Categorical(probs=probs)
        draws = cat.sample((n_samples,)).transpose(0, 1)
        return draws.unsqueeze(-1).float()
