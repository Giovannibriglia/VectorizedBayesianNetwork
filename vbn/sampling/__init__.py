# vbn/sampling/__init__.py
from __future__ import annotations

from typing import Dict, Optional

import torch

from .ancestral import AncestralSampler
from .rao_blackwellized import RaoBlackwellizedSampler
from .smc_conditional import ConditionalSMCSampler

Tensor = torch.Tensor


def sample(
    bn,
    n: int = 1024,
    do: Optional[Dict[str, Tensor]] = None,
    method: str = "ancestral",
    **kw,
):
    device = kw.pop("device", getattr(bn, "device", "cpu"))
    qmc = kw.pop("qmc", False)
    qmc_seed = kw.pop("qmc_seed", 42)

    if method == "ancestral":
        # constructor kwargs (init) vs call kwargs (none here)
        sampler = AncestralSampler(device=device, qmc=qmc, qmc_seed=qmc_seed)
        return sampler.sample(bn, n=n, do=do)

    elif method == "rb":
        # pull per-call flags
        return_gaussian_params = kw.pop("return_gaussian_params", False)
        sampler = RaoBlackwellizedSampler(device=device, qmc=qmc, qmc_seed=qmc_seed)
        return sampler.sample(
            bn, n=n, do=do, return_gaussian_params=return_gaussian_params, **kw
        )

    elif method == "smc":
        # constructor kwargs
        ess_threshold = kw.pop("ess_threshold", 0.5)
        evidence = kw.pop("evidence", None)
        sampler = ConditionalSMCSampler(
            device=device, qmc=qmc, qmc_seed=qmc_seed, ess_threshold=ess_threshold
        )
        return sampler.sample(bn, n=n, evidence=evidence, do=do, **kw)

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def sample_conditional(
    bn,
    evidence: Dict[str, Tensor],
    n: int = 1024,
    do: Optional[Dict[str, Tensor]] = None,
    **kw,
):
    # forward to SMC with clean constructor vs call kwargs split
    return sample(bn, n=n, do=do, method="smc", evidence=evidence, **kw)
