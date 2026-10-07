"""CHORD-Stochastic: latent collective intentions, training on K sampled futures, and sampling.

Every collective slot m of a model with ``latent_dim > 0`` carries an intention eps_m ~ N(0, I), drawn once per
forecast and held fixed along it; given the intentions, the model is an ordinary CHORD vector field. Training
starts from a trained deterministic model (the decoder's weights on the intentions start at zero) and minimises
either the best-of-K objective, E_w[min_k e_k + lambda_avg mean_k e_k], or the energy score of the K futures.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import torch
from torch.nn.functional import cross_entropy

from chord.data import TrajectoryData
from chord.losses import energy_score
from chord.train import ModelField, TrainResult, entity_masks, rollout, sample_windows, window_errors, window_targets


@dataclass
class StochasticTrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    samples: int = 6  # K during training
    objective: str = "variety"  # "variety" (best-of-K) or "energy" (energy score)
    avg_weight: float = 0.1  # lambda_avg of the best-of-K objective
    beta: float = 1.0  # exponent of the energy score, in (0, 2)
    lambda_event: float = 0.0
    windows_per_epoch: int = 1024
    batch_size: int = 32  # windows per step, each rolled out ``samples`` times
    micro_batch: int = 0  # windows per backward pass (gradients accumulated over the step); 0: the whole step
    grad_clip: float | None = None
    horizon: int = 20
    substeps: int = 2
    loss_dims: int | None = 2
    centre: bool = True
    val_samples: int = 20
    val_scenes: int = 500
    val_first: int = 0
    patience: int = 4
    seed: int = 0


def draw_latents(model, n: int, generator: torch.Generator | None = None) -> None:
    """Independent intentions for ``n`` forecasts, shape (n, M, latent_dim)."""
    device = next(model.parameters()).device
    shape = (n, model.config.n_collectives, model.config.latent_dim)
    model.latent = torch.randn(shape, generator=generator).to(device)


def sample_futures(model, observations, start_mask, target_mask, b, k, horizon, interval, substeps, loss_dims,
                   centre, samples, generator=None):
    """Roll out every window (b, k) ``samples`` times with independent intentions. Returns the per-step error of
    every future (W, K), the futures (W, K, h, N, d), the observed futures and the observed target mask."""
    field = ModelField(model)
    bb, kk = b.repeat_interleave(samples), k.repeat_interleave(samples)
    draw_latents(model, len(bb), generator)
    error, steps, prediction = window_errors(field, observations, start_mask, target_mask, bb, kk, horizon, interval,
                                             substeps, loss_dims, centre)
    errors = (error / steps.clamp_min(1.0)).view(len(b), samples)
    _, target, observed = window_targets(observations, start_mask, target_mask, b, k, horizon, loss_dims, centre)
    return errors, prediction.view(len(b), samples, *prediction.shape[1:]), target, observed


@torch.no_grad()
def sample(model, x0: torch.Tensor, n_samples: int, n_steps: int, interval: float, substeps: int = 2,
           mask: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """``n_samples`` futures from states x0 (B, N, d), shape (B, n_samples, n_steps, N, d)."""
    model.eval()
    x = x0.repeat_interleave(n_samples, dim=0)
    m = None if mask is None else mask.repeat_interleave(n_samples, dim=0)
    draw_latents(model, len(x), generator)
    futures = rollout(ModelField(model), x, n_steps, interval, substeps, m)
    model.latent = None
    return futures.view(len(x0), n_samples, *futures.shape[1:])


@torch.no_grad()
def validation_criterion(model, data: TrajectoryData, config: StochasticTrainConfig, generator=None) -> float:
    """Mean best-of-K error (objective "variety") or energy score (objective "energy") over the first
    ``val_scenes`` scenes, with rollouts from frame ``val_first``."""
    model.eval()
    device = next(model.parameters()).device
    start_mask, target_mask = entity_masks(data)
    b = torch.arange(min(config.val_scenes, len(data.observations)))
    b = b[start_mask[b, config.val_first].any(-1)]
    k = torch.full_like(b, config.val_first)
    interval = float(data.times[1] - data.times[0])
    observations, start_dev, target_dev = data.observations.to(device), start_mask.to(device), target_mask.to(device)
    total = 0.0
    for s in range(0, len(b), 64):
        errors, prediction, target, observed = sample_futures(
            model, observations, start_dev, target_dev, b[s : s + 64].to(device), k[s : s + 64].to(device),
            config.horizon, interval, config.substeps, config.loss_dims, config.centre, config.val_samples, generator)
        if config.objective == "energy":
            total += energy_score(prediction, target, observed, config.beta).sum().item()
        else:
            total += errors.min(dim=1).values.sum().item()
    model.latent = None
    return total / max(len(b), 1)


def train_stochastic(model, train: TrajectoryData, val: TrajectoryData, config: StochasticTrainConfig) -> TrainResult:
    """Fine-tune a model with intentions on K sampled futures per window; keep the best validation state."""
    if config.objective not in ("variety", "energy"):
        raise ValueError(f"unknown objective {config.objective!r}")
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = next(model.parameters()).device
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    steps = math.ceil(config.windows_per_epoch / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs * steps)
    interval = float(train.times[1] - train.times[0])
    start_mask, target_mask = entity_masks(train)
    observations, start_dev, target_dev = train.observations.to(device), start_mask.to(device), target_mask.to(device)
    events = train.events.to(device) if config.lambda_event > 0 and train.events is not None else None
    best_val = validation_criterion(model, val, config, torch.Generator().manual_seed(config.seed + 1))
    best_epoch, best_state = -1, copy.deepcopy(model.state_dict())
    history, start = [{"epoch": -1, "val": best_val}], time.time()
    for epoch in range(config.epochs):
        model.train()
        b, k = sample_windows(start_mask, config.horizon, config.windows_per_epoch, generator)
        running = 0.0
        for step in range(steps):
            window = slice(step * config.batch_size, (step + 1) * config.batch_size)
            b_step, k_step = b[window].to(device), k[window].to(device)
            chunk = config.micro_batch or len(b_step)
            optimiser.zero_grad(set_to_none=True)
            for c in range(0, len(b_step), chunk):  # every term is a mean over windows: chunks weigh by size
                bw, kw = b_step[c : c + chunk], k_step[c : c + chunk]
                errors, prediction, target, observed = sample_futures(
                    model, observations, start_dev, target_dev, bw, kw, config.horizon, interval, config.substeps,
                    config.loss_dims, config.centre, config.samples, generator)
                if config.objective == "energy":
                    loss = energy_score(prediction, target, observed, config.beta).mean()
                else:
                    loss = errors.min(dim=1).values.mean() + config.avg_weight * errors.mean()
                if events is not None:
                    frames = kw[:, None] + torch.arange(1, config.horizon + 1, device=device)
                    truth = events[bw[:, None], frames].repeat_interleave(config.samples, dim=0)
                    logits = model.event_logits(prediction.flatten(0, 1))
                    loss = loss + config.lambda_event * cross_entropy(logits.flatten(0, 1), truth.flatten())
                weight = len(bw) / len(b_step)
                (loss * weight).backward()
                running += loss.item() * weight
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            scheduler.step()
        val_error = validation_criterion(model, val, config, torch.Generator().manual_seed(config.seed + 1))
        history.append({"epoch": epoch, "train": running / steps, "val": val_error})
        if val_error < best_val:
            best_val, best_epoch, best_state = val_error, epoch, copy.deepcopy(model.state_dict())
        if epoch - max(best_epoch, 0) >= config.patience:
            break
    model.load_state_dict(best_state)
    model.latent = None
    return TrainResult(best_val, best_epoch, len(history) - 1, time.time() - start, history)
