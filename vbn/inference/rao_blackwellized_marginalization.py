from __future__ import annotations

import math
from typing import Optional

import networkx as nx
import torch

from vbn.core.base import Query
from vbn.core.registry import INFERENCE_REGISTRY, register_inference
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.utils import infer_batch_size


@register_inference("rao_blackwellized_marginalization")
class RaoBlackwellizedMarginalization:
    """
    Rao-Blackwellized posterior inference for a single target node.

    When possible, it samples latent non-target nodes and analytically marginalizes
    the target conditional given sampled parents, instead of sampling the target.
    """

    def __init__(
        self,
        n_samples: int = 200,
        n_particles: Optional[int] = None,
        stddevs: float = 4.0,
        min_scale: float = 1e-6,
        fallback: str = "likelihood_weighting",
        **kwargs,
    ) -> None:
        self.n_samples = int(n_samples)
        self.n_particles = (
            int(n_particles) if n_particles is not None else self.n_samples
        )
        self.stddevs = float(stddevs)
        self.min_scale = float(min_scale)
        self.fallback = (
            str(fallback).strip().lower() if fallback is not None else "none"
        )
        self._cache = {}
        self._fallback = None
        self._last_fallback = False
        self._last_reason = None
        if self.fallback != "none":
            if self.fallback not in INFERENCE_REGISTRY:
                raise ValueError(
                    f"Unknown fallback inference '{fallback}'. Available: {list(INFERENCE_REGISTRY.keys())}"
                )
            if self.fallback == "rao_blackwellized_marginalization":
                raise ValueError(
                    "fallback cannot be 'rao_blackwellized_marginalization'"
                )
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.setdefault("n_samples", self.n_samples)
            self._fallback = INFERENCE_REGISTRY[self.fallback](**fallback_kwargs)

    def _fallback_infer(self, vbn, query: Query, *, reason: str, **kwargs):
        self._last_fallback = True
        self._last_reason = reason
        if self._fallback is None:
            raise RuntimeError(
                "rao_blackwellized_marginalization cannot handle this query and has no fallback"
            )
        return self._fallback.infer_posterior(vbn, query, **kwargs)

    def _normalized_weights(self, log_weights: torch.Tensor, eps: float = 1e-12):
        log_weights = torch.nan_to_num(
            log_weights, nan=-1e30, posinf=1e30, neginf=-1e30
        )
        log_weights = log_weights - log_weights.max(dim=1, keepdim=True).values
        weights = torch.exp(log_weights)
        denom = weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / max(1, weights.shape[1]))
        return torch.where(denom > eps, weights / denom.clamp_min(eps), uniform)

    def _to_3d(self, tensor: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
        if tensor.dim() == 1:
            tensor = tensor.view(1, 1, -1)
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(1)
        elif tensor.dim() != 3:
            return None
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1, -1)
        if tensor.shape[0] != batch_size:
            return None
        return tensor

    def _target_gaussian_params(
        self,
        cpd,
        parents_tensor: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(cpd, "n_classes") or hasattr(cpd, "n_components"):
            return None
        try:
            if (
                hasattr(cpd, "_weight")
                and hasattr(cpd, "_bias")
                and hasattr(cpd, "_var")
            ):
                if parents_tensor is None:
                    loc = cpd._bias.view(1, 1, -1)
                    if hasattr(cpd, "_scale") and callable(cpd._scale):
                        scale = cpd._scale().view(1, 1, -1)
                    else:
                        scale = torch.sqrt(
                            cpd._var.clamp_min(self.min_scale**2)
                        ).view(1, 1, -1)
                else:
                    parents = parents_tensor
                    if parents.dim() == 2:
                        parents = parents.unsqueeze(1)
                    loc = parents @ cpd._weight + cpd._bias
                    if hasattr(cpd, "_scale") and callable(cpd._scale):
                        scale = cpd._scale().view(1, 1, -1).expand_as(loc)
                    else:
                        scale = torch.sqrt(
                            cpd._var.clamp_min(self.min_scale**2)
                        ).view(1, 1, -1)
                        scale = scale.expand_as(loc)
            elif hasattr(cpd, "_params") and callable(cpd._params):
                loc, scale = (
                    cpd._params(None)
                    if parents_tensor is None
                    else cpd._params(parents_tensor)
                )
            else:
                return None
        except Exception:
            return None

        loc = self._to_3d(loc, batch_size)
        scale = self._to_3d(scale, batch_size)
        if loc is None or scale is None:
            return None
        if loc.shape[-1] != 1 or scale.shape[-1] != 1:
            return None
        if scale.shape[1] == 1 and loc.shape[1] > 1:
            scale = scale.expand(-1, loc.shape[1], -1)
        elif loc.shape[1] == 1 and scale.shape[1] > 1:
            loc = loc.expand(-1, scale.shape[1], -1)
        elif loc.shape[1] != scale.shape[1]:
            return None
        scale = torch.nan_to_num(
            scale, nan=self.min_scale, posinf=self.min_scale, neginf=self.min_scale
        ).abs()
        scale = scale.clamp_min(self.min_scale)
        return loc, scale

    def _target_categorical_probs(
        self,
        cpd,
        parents_tensor: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(cpd, "n_classes"):
            return None
        try:
            if parents_tensor is None:
                if hasattr(cpd, "_root_ready") and bool(
                    getattr(cpd, "_root_ready").item()
                ):
                    logits = cpd._root_log_probs
                    logits = torch.log_softmax(logits, dim=-1)
                elif hasattr(cpd, "_params") and callable(cpd._params):
                    logits = cpd._params(None)
                else:
                    logits = cpd._logits
                logits = logits.view(1, 1, cpd.output_dim, cpd.n_classes)
                logits = logits.expand(batch_size, 1, -1, -1)
            else:
                logits = cpd._logits_from_parents(parents_tensor)
            probs = torch.softmax(logits, dim=-1)
            if probs.dim() == 4 and probs.shape[2] == 1:
                probs = probs[:, :, 0, :]
            if probs.dim() != 3:
                return None
            if probs.shape[-1] != int(cpd.n_classes):
                return None
            if hasattr(cpd, "_sample_values"):
                support = cpd._sample_values[0].to(
                    device=probs.device, dtype=probs.dtype
                )
            else:
                support = torch.arange(
                    int(cpd.n_classes), device=probs.device, dtype=probs.dtype
                )
            return probs, support
        except Exception:
            return None

    def infer_posterior(self, vbn, query: Query, **kwargs):
        self._last_fallback = False
        self._last_reason = None

        n_samples = max(1, int(kwargs.get("n_samples", self.n_samples)))
        n_particles = max(1, int(kwargs.get("n_particles", self.n_particles)))
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        target_idx = state.target_idx
        target_node = state.topo_order[target_idx]
        descendants = nx.descendants(vbn.dag.graph, target_node)
        descendants_idx = {state.node_to_idx[n] for n in descendants}
        if any(state.evidence_mask[i] or state.do_mask[i] for i in descendants_idx):
            return self._fallback_infer(
                vbn,
                query,
                reason="target has observed/intervened descendants",
                **kwargs,
            )

        fixed_values = prepare_fixed_values(query, state, device, dtype, clamp_obs=True)
        target_fixed = fixed_values[target_idx]
        if target_fixed is not None:
            target_value = target_fixed.unsqueeze(1).expand(b, 1, -1)
            weights = torch.ones(b, 1, device=device, dtype=dtype)
            return weights.detach(), target_value.detach()

        skip_idx = set(descendants_idx)
        skip_idx.add(target_idx)
        samples = torch.zeros(
            b, n_particles, state.total_dim, device=device, dtype=dtype
        )
        log_weights = torch.zeros(b, n_particles, device=device, dtype=dtype)
        cpds = [vbn.nodes[node] for node in state.topo_order]

        for idx, node in enumerate(state.topo_order):
            if idx in skip_idx:
                continue
            node_slice = state.node_slices[idx]
            parent_slices = state.parent_slices[idx]
            if parent_slices:
                parent_tensor = torch.cat(
                    [samples[..., sl] for sl in parent_slices], dim=-1
                )
            else:
                parent_tensor = None

            fixed = fixed_values[idx]
            if fixed is not None:
                value = fixed.unsqueeze(1).expand(b, n_particles, -1)
                samples[..., node_slice] = value
                if state.evidence_mask[idx]:
                    log_weights = log_weights + cpds[idx].log_prob(value, parent_tensor)
                continue
            samples[..., node_slice] = cpds[idx].sample(parent_tensor, n_particles)

        weights = self._normalized_weights(log_weights)
        target_parent_slices = state.parent_slices[target_idx]
        if target_parent_slices:
            parent_tensor = torch.cat(
                [samples[..., sl] for sl in target_parent_slices], dim=-1
            )
        else:
            parent_tensor = None
        target_cpd = cpds[target_idx]

        cat = self._target_categorical_probs(target_cpd, parent_tensor, b)
        if cat is not None:
            probs_cond, support = cat
            if probs_cond.shape[1] != n_particles:
                if probs_cond.shape[1] == 1:
                    probs_cond = probs_cond.expand(-1, n_particles, -1)
                else:
                    return self._fallback_infer(
                        vbn,
                        query,
                        reason="categorical conditional shape mismatch",
                        **kwargs,
                    )
            marginal_probs = (weights.unsqueeze(-1) * probs_cond).sum(dim=1)
            support_samples = support.view(1, -1, 1).expand(b, -1, 1)
            return marginal_probs.detach(), support_samples.detach()

        gaussian = self._target_gaussian_params(target_cpd, parent_tensor, b)
        if gaussian is not None:
            loc, scale = gaussian
            if loc.shape[1] != n_particles:
                if loc.shape[1] == 1:
                    loc = loc.expand(-1, n_particles, -1)
                    scale = scale.expand(-1, n_particles, -1)
                else:
                    return self._fallback_infer(
                        vbn,
                        query,
                        reason="gaussian conditional shape mismatch",
                        **kwargs,
                    )

            comp_var = scale.squeeze(-1) ** 2
            comp_mean = loc.squeeze(-1)
            mix_mean = (weights * comp_mean).sum(dim=1)
            second = (weights * (comp_var + comp_mean**2)).sum(dim=1)
            mix_var = (second - mix_mean**2).clamp_min(self.min_scale**2)
            mix_std = mix_var.sqrt()

            z = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype).view(
                1, n_samples, 1
            )
            lo = (mix_mean - self.stddevs * mix_std).view(b, 1, 1)
            hi = (mix_mean + self.stddevs * mix_std).view(b, 1, 1)
            samples_grid = lo + (hi - lo) * z

            x = samples_grid.squeeze(-1).unsqueeze(1)  # [B,1,S_out]
            mu = loc.squeeze(-1).unsqueeze(-1)  # [B,S_part,1]
            sigma = scale.squeeze(-1).unsqueeze(-1).clamp_min(self.min_scale)
            z_norm = (x - mu) / sigma
            comp_pdf = torch.exp(-0.5 * z_norm**2) / (math.sqrt(2.0 * math.pi) * sigma)
            pdf = (weights.unsqueeze(-1) * comp_pdf).sum(dim=1)
            return pdf.detach(), samples_grid.detach()

        return self._fallback_infer(
            vbn,
            query,
            reason="unsupported target CPD for RB marginalization",
            **kwargs,
        )
