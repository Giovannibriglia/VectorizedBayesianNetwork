from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch
from torch import nn


@dataclass
class CPDOutput:
    samples: torch.Tensor  # [B, S, Dx]
    log_prob: torch.Tensor  # [B, S]
    pdf: torch.Tensor  # [B, S]


@dataclass
class Query:
    target: str
    evidence: Dict[str, torch.Tensor]


class BaseCPD(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        device: torch.device,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def sample(self, parents: Optional[torch.Tensor], n_samples: int) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(
        self, x: torch.Tensor, parents: Optional[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, parents: Optional[torch.Tensor], n_samples: int) -> CPDOutput:
        samples = self.sample(parents, n_samples)
        log_prob = self.log_prob(samples, parents)
        pdf = torch.exp(log_prob)
        return CPDOutput(samples=samples, log_prob=log_prob, pdf=pdf)

    def fit(self, parents: Optional[torch.Tensor], x: torch.Tensor, **kwargs) -> None:
        raise NotImplementedError

    def update(
        self, parents: Optional[torch.Tensor], x: torch.Tensor, **kwargs
    ) -> None:
        raise NotImplementedError("Incremental update not implemented for this CPD.")

    def get_init_kwargs(self) -> Dict[str, object]:
        """Return CPD-specific init kwargs needed for reconstruction."""
        return {}

    def get_extra_state(self) -> Optional[Dict[str, object]]:
        """Return extra state not captured by state_dict()."""
        return None

    def set_extra_state(self, state: Optional[Dict[str, object]]) -> None:
        """Restore extra state captured by get_extra_state()."""
        return None


class BaseLearning(Protocol):
    def fit(
        self,
        vbn,
        data: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, BaseCPD]:
        raise NotImplementedError


class BaseInference(Protocol):
    def infer_posterior(self, vbn, query: Query, **kwargs):
        raise NotImplementedError


class BaseSampling(Protocol):
    def sample(self, vbn, query: Query, n_samples: int, **kwargs):
        raise NotImplementedError


class BaseUpdatePolicy(Protocol):
    def update(
        self, vbn, data: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, BaseCPD]:
        raise NotImplementedError
