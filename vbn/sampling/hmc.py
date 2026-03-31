from __future__ import annotations

import torch

from vbn.core.base import Query
from vbn.core.registry import register_sampling
from vbn.inference._core import get_inference_state, prepare_fixed_values, resolve_dtype
from vbn.sampling.ancestral import _ancestral_sample_tensor
from vbn.utils import infer_batch_size


def _is_continuous_cpd(cpd) -> bool:
    name = cpd.__class__.__name__.lower()
    if "softmax" in name:
        return False
    return True


@register_sampling("hmc")
class HMCSampler:
    """Hamiltonian Monte Carlo sampler for continuous-only networks."""

    def __init__(self, n_samples: int = 200, **kwargs) -> None:
        self.n_samples = int(n_samples)
        self._cache = {}

    def sample(self, vbn, query: Query, n_samples: int | None = None, **kwargs):
        n_samples = int(n_samples or self.n_samples)
        if not all(_is_continuous_cpd(cpd) for cpd in vbn.nodes.values()):
            samples = _ancestral_sample_tensor(vbn, query, n_samples)
            state = get_inference_state(vbn, query, self._cache)
            if query.target:
                return samples[..., state.node_slices[state.target_idx]]
            return {
                node: samples[..., state.node_slices[idx]]
                for idx, node in enumerate(state.topo_order)
            }

        step_size = float(kwargs.get("step_size", 0.05))
        n_leapfrog = int(kwargs.get("n_leapfrog", 8))
        burn_in = int(kwargs.get("burn_in", 10))
        b = infer_batch_size(query.evidence, query.do)
        state = get_inference_state(vbn, query, self._cache)
        device = vbn.device
        dtype = resolve_dtype(vbn, query)

        fixed_values = prepare_fixed_values(query, state, device, dtype)
        latent_nodes = [
            idx for idx in range(len(state.topo_order)) if fixed_values[idx] is None
        ]
        cpds = [vbn.nodes[node] for node in state.topo_order]
        if not latent_nodes:
            samples = _ancestral_sample_tensor(vbn, query, n_samples)
            if query.target:
                return samples[..., state.node_slices[state.target_idx]]
            return {
                node: samples[..., state.node_slices[idx]]
                for idx, node in enumerate(state.topo_order)
            }

        latent_slices = []
        latent_dim = 0
        for idx in latent_nodes:
            dim = state.node_slices[idx].stop - state.node_slices[idx].start
            latent_slices.append(slice(latent_dim, latent_dim + dim))
            latent_dim += dim
        latent_pos = {idx: pos for pos, idx in enumerate(latent_nodes)}

        def build_samples(z: torch.Tensor) -> torch.Tensor:
            samples = torch.zeros(b, 1, state.total_dim, device=device, dtype=dtype)
            for idx, node in enumerate(state.topo_order):
                node_slice = state.node_slices[idx]
                if fixed_values[idx] is not None:
                    samples[..., node_slice] = fixed_values[idx].unsqueeze(1)
                    continue
                z_slice = latent_slices[latent_pos[idx]]
                samples[..., node_slice] = z[:, z_slice].unsqueeze(1)
            return samples

        def joint_log_prob(z: torch.Tensor) -> torch.Tensor:
            samples = build_samples(z)
            total = torch.zeros(b, device=device, dtype=dtype)
            for idx, node in enumerate(state.topo_order):
                node_slice = state.node_slices[idx]
                value = samples[..., node_slice]
                parent_slices = state.parent_slices[idx]
                if parent_slices:
                    parent_tensor = torch.cat(
                        [samples[..., sl] for sl in parent_slices], dim=-1
                    )
                else:
                    parent_tensor = None
                total = total + cpds[idx].log_prob(value, parent_tensor).squeeze(1)
            return total

        init_samples = _ancestral_sample_tensor(vbn, query, n_samples=1)
        z0 = []
        for idx, z_slice in zip(latent_nodes, latent_slices):
            node_slice = state.node_slices[idx]
            z0.append(init_samples[..., node_slice].reshape(b, -1))
        z = torch.cat(z0, dim=-1).detach()

        def hmc_step(z: torch.Tensor) -> torch.Tensor:
            z = z.detach().requires_grad_(True)
            logp = joint_log_prob(z)
            grad = torch.autograd.grad(logp.sum(), z)[0]
            momentum = torch.randn_like(z)
            current_h = -logp + 0.5 * (momentum**2).sum(dim=1)
            p = momentum + 0.5 * step_size * grad
            q = z
            for _ in range(max(n_leapfrog, 1)):
                q = (q + step_size * p).detach().requires_grad_(True)
                logp = joint_log_prob(q)
                grad = torch.autograd.grad(logp.sum(), q)[0]
                p = p + step_size * grad
            p = p - 0.5 * step_size * grad
            p = -p
            new_h = -logp + 0.5 * (p**2).sum(dim=1)
            accept_prob = torch.exp(current_h - new_h).clamp(max=1.0)
            accept = torch.rand_like(accept_prob) < accept_prob
            z_next = torch.where(accept.unsqueeze(1), q.detach(), z.detach())
            return z_next

        collected = []
        total_steps = burn_in + n_samples
        for step in range(total_steps):
            z = hmc_step(z)
            if step >= burn_in:
                samples = build_samples(z)
                if query.target:
                    collected.append(samples[..., state.node_slices[state.target_idx]])
                else:
                    collected.append(samples)

        if query.target:
            return torch.cat(collected, dim=1)
        full = torch.cat(collected, dim=1)
        return {
            node: full[..., state.node_slices[idx]]
            for idx, node in enumerate(state.topo_order)
        }
