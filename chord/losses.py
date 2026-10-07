"""Training objectives."""

from __future__ import annotations

import torch


def frobenius_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """E ||prediction - target||_F^2 per sample: squared error summed over entities and state dimensions."""
    return (prediction - target).pow(2).sum(dim=(-2, -1)).mean()


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``frobenius_mse`` over the entities in ``mask`` (B, N), rescaled so that every sample counts once."""
    m = mask.to(prediction.dtype)
    per_entity = (prediction - target).pow(2).sum(-1) * m
    return (per_entity.sum(-1) * m.shape[-1] / m.sum(-1).clamp_min(1.0)).mean()


def membership_entropy(z: torch.Tensor) -> torch.Tensor:
    """L_ent: mean row entropy of the memberships, with 0 log 0 = 0."""
    return -(z * torch.log(z.clamp_min(1e-12))).sum(dim=-1).mean()


def influence_volume(psi: torch.Tensor, jitter: float = 1e-4) -> torch.Tensor:
    """L_vol = log det(Psi Psi^T / (d_o n_b) + jitter I) for influences psi of shape (n_b, M, d_o)."""
    n_b, n_collectives, d_o = psi.shape
    stacked = psi.permute(1, 0, 2).reshape(n_collectives, n_b * d_o)
    gram = stacked @ stacked.T / (n_b * d_o)
    return torch.logdet(gram + jitter * torch.eye(n_collectives, dtype=psi.dtype, device=psi.device))


def regularisers(outputs: dict[str, torch.Tensor], lambda_ent: float, lambda_vol: float, lambda_size: float = 0.0,
                 jitter: float = 1e-4) -> torch.Tensor:
    """lambda_ent L_ent + lambda_vol L_vol + lambda_size L_size."""
    total = outputs["velocity"].new_zeros(())
    if lambda_ent > 0:
        total = total + lambda_ent * membership_entropy(outputs["memberships"])
    if lambda_vol > 0:
        total = total + lambda_vol * influence_volume(outputs["influences"], jitter)
    if lambda_size > 0 and "size_penalty" in outputs:
        total = total + lambda_size * outputs["size_penalty"]
    return total


def energy_score(prediction: torch.Tensor, target: torch.Tensor, observed: torch.Tensor, beta: float = 1.0,
                 dims: int = 2) -> torch.Tensor:
    """Unbiased energy score of every window, (W,), for prediction (W, K, h, N, d), target (W, h, N, d) and
    observed entity-steps (W, h, N). Norms over the observed positions, divided by sqrt(observed entity-steps)."""
    mask = observed.unsqueeze(-1).to(prediction.dtype)
    scale = observed.flatten(1).sum(-1).clamp_min(1.0).sqrt()
    y = (target[..., :dims] * mask).flatten(1)
    samples = (prediction[..., :dims] * mask.unsqueeze(1)).flatten(2)
    k = samples.shape[1]
    fit = ((samples - y.unsqueeze(1)).norm(dim=-1) / scale[:, None]).pow(beta).mean(1)
    spread = (torch.cdist(samples, samples) / scale[:, None, None]).pow(beta).sum((1, 2)) / (2 * k * (k - 1))
    return fit - spread
