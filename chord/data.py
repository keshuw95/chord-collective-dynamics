"""Data containers and velocity estimates from observed positions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryData:
    """Observed trajectories.

    times         (T,) observation times on a regular grid
    observations  (B, T, N, d) observed states
    mask          (B, T) or (B, T, N): which observations exist (entity observed and present)
    start_mask    (B, T, N) optional: states that can start a rollout
    events        (B, T) optional: categorical event per frame, an entity index (e.g. the ball possessor)
    """

    times: torch.Tensor
    observations: torch.Tensor
    mask: torch.Tensor
    start_mask: torch.Tensor | None = None
    events: torch.Tensor | None = None


@dataclass
class VelocityData:
    """States with velocity targets, for derivative matching (L_vel)."""

    states: torch.Tensor  # (n, N, d)
    velocities: torch.Tensor  # (n, N, d)
    mask: torch.Tensor | None = None  # (n, N) present entities
    target_mask: torch.Tensor | None = None  # (n, N) entities with a target; defaults to ``mask``
    events: torch.Tensor | None = None  # (n,)


def add_velocity_estimates(data: TrajectoryData, window: int = 5, degree: int = 2) -> TrajectoryData:
    """Append to each position observation a causal velocity estimate, the slope at t_k of a least-squares
    polynomial of degree ``degree`` fitted to the last ``window`` observations (a causal Savitzky-Golay filter),
    so that second-order rollouts can start from (p, v_hat). Where fewer consecutive observations exist, the
    longest available run is used, down to the backward difference. ``start_mask`` marks the estimated states."""
    p, h = data.observations, float(data.times[1] - data.times[0])
    mask = data.mask if data.mask.dim() == 3 else data.mask.unsqueeze(-1).expand(*p.shape[:3])
    velocity = torch.zeros_like(p)
    start = torch.zeros_like(mask)
    run = torch.zeros_like(mask, dtype=torch.long)
    for k in range(p.shape[1]):
        run[:, k] = torch.where(mask[:, k], (run[:, k - 1] if k else 0) + 1, 0)
    for length in range(2, window + 1):  # longer runs overwrite shorter ones
        offsets = torch.arange(-length + 1, 1, dtype=p.dtype) * h
        basis = torch.stack([offsets**j for j in range(min(degree, length - 1) + 1)], dim=1)
        weights = torch.linalg.pinv(basis)[1]
        usable = run >= length
        usable[:, : length - 1] = False
        estimate = torch.zeros_like(p)
        for j, w in enumerate(weights):
            estimate[:, length - 1 :] += w * p[:, j : p.shape[1] - length + 1 + j]
        velocity = torch.where(usable.unsqueeze(-1), estimate, velocity)
        start |= usable
    return TrajectoryData(data.times, torch.cat([p, velocity * start.unsqueeze(-1)], dim=-1), mask,
                          start_mask=start, events=data.events)


def derivative_targets(data: TrajectoryData, window: int = 7, degree: int = 2) -> VelocityData:
    """Velocity-level data from positions only, for the derivative-matching warm start of second-order models.
    At every observation with ``window`` consecutive observations centred on it, a least-squares polynomial gives
    a smoothed position, velocity and acceleration; the state is (p_hat, v_hat) and the target (v_hat, a_hat).
    Entities without a centred window enter with their causal estimate and act on the others, without a target."""
    causal = add_velocity_estimates(data)
    p, h = data.observations, float(data.times[1] - data.times[0])
    mask = causal.mask
    half = window // 2
    offsets = torch.arange(-half, half + 1, dtype=p.dtype) * h
    weights = torch.linalg.pinv(torch.stack([offsets**j for j in range(degree + 1)], dim=1))
    valid = torch.zeros_like(mask)
    coefficients = torch.zeros(*p.shape[:3], degree + 1, p.shape[-1], dtype=p.dtype)
    for k in range(half, p.shape[1] - half):
        frames = slice(k - half, k + half + 1)
        valid[:, k] = mask[:, frames].all(dim=1)
        coefficients[:, k] = torch.einsum("cw,bwnd->bncd", weights, p[:, frames])
    smoothed = torch.cat([coefficients[..., 0, :], coefficients[..., 1, :]], dim=-1)
    states = torch.where(valid.unsqueeze(-1), smoothed, causal.observations)
    targets = torch.cat([coefficients[..., 1, :], 2.0 * coefficients[..., 2, :]], dim=-1) * valid.unsqueeze(-1)
    keep = valid.any(-1)
    events = data.events[keep] if data.events is not None else None
    return VelocityData(states[keep], targets[keep], (causal.start_mask | valid)[keep], valid[keep], events)
