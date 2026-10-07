"""Deterministic training: derivative matching (L_vel) and rollouts through the ODE solver (L_pred)."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch
from torch.nn.functional import cross_entropy
from torchdiffeq import odeint

from chord.data import TrajectoryData, VelocityData
from chord.losses import frobenius_mse, masked_mse, regularisers


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    lambda_ent: float = 0.0
    lambda_vol: float = 0.0
    reg_start_epoch: int = 0  # L_ent and L_vol are switched on from this epoch, after the fit converges
    lambda_size: float = 0.0  # weight of L_size (multiscale models)
    lambda_event: float = 0.0  # weight of the event cross-entropy (models with joint events)
    patience: int = 40
    seed: int = 0
    eval_batch_size: int = 1024


@dataclass
class TrainResult:
    best_val: float
    best_epoch: int
    epochs_run: int
    seconds: float
    history: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- derivative matching


def _velocity_loss(model, states, velocities, mask, targets):
    if mask is None:
        outputs = model(states)
        return outputs, frobenius_mse(outputs["velocity"], velocities)
    outputs = model(states, mask=mask)
    return outputs, masked_mse(outputs["velocity"], velocities, targets)


@torch.no_grad()
def evaluate_velocity(model: torch.nn.Module, data: VelocityData, batch_size: int = 1024) -> float:
    """Per-sample Frobenius error of the predicted velocities."""
    model.eval()
    device = next(model.parameters()).device
    targets = data.mask if data.target_mask is None else data.target_mask
    total = 0.0
    for s in range(0, len(data.states), batch_size):
        part = slice(s, s + batch_size)
        mask = None if data.mask is None else data.mask[part].to(device)
        target = None if targets is None else targets[part].to(device)
        _, loss = _velocity_loss(model, data.states[part].to(device), data.velocities[part].to(device), mask, target)
        total += loss.item() * len(data.states[part])
    return total / len(data.states)


def train_velocity(model: torch.nn.Module, train: VelocityData, val: VelocityData, config: TrainConfig) -> TrainResult:
    """Minimise L_vel (+ regularisers) with Adam and cosine decay; keep the best validation state."""
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = next(model.parameters()).device
    states, velocities = train.states.to(device), train.velocities.to(device)
    mask = train.mask.to(device) if train.mask is not None else None
    targets = mask if train.target_mask is None else train.target_mask.to(device)
    events = train.events.to(device) if train.events is not None else None
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    steps = math.ceil(len(states) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs * steps)

    best_val, best_epoch, best_state = float("inf"), -1, copy.deepcopy(model.state_dict())
    history, start = [], time.time()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(states), generator=generator).to(device)
        running = 0.0
        for step in range(steps):
            idx = order[step * config.batch_size : (step + 1) * config.batch_size]
            outputs, fit = _velocity_loss(model, states[idx], velocities[idx], None if mask is None else mask[idx],
                                          None if targets is None else targets[idx])
            loss = fit
            if config.lambda_event > 0 and events is not None:
                loss = loss + config.lambda_event * cross_entropy(outputs["event_logits"], events[idx])
            if epoch >= config.reg_start_epoch:
                loss = loss + regularisers(outputs, config.lambda_ent, config.lambda_vol, config.lambda_size)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            scheduler.step()
            running += fit.item()
        val_error = evaluate_velocity(model, val, config.eval_batch_size)
        history.append({"epoch": epoch, "train": running / steps, "val": val_error})
        if epoch >= config.reg_start_epoch and val_error < best_val:  # keep the regularisers' effect
            best_val, best_epoch, best_state = val_error, epoch, copy.deepcopy(model.state_dict())
        if epoch >= config.reg_start_epoch and epoch - max(best_epoch, config.reg_start_epoch) >= config.patience:
            break
    model.load_state_dict(best_state)
    return TrainResult(best_val, best_epoch, len(history), time.time() - start, history)


# --------------------------------------------------------------------------- rollouts


class ModelField(torch.nn.Module):
    """CHORD as a vector field over (X, S): d/dt (X, S) = (F_theta(X, S), (S~(X) - S) / tau_s). ``mask`` (B, N)
    marks the entities present at the start of the window; the others are frozen and excluded."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.mask = None

    def initial_state(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        self.mask = mask
        return x, self.model.initial_logits(x, mask)

    def forward(self, t, state):
        x, logits = state
        outputs = self.model(x, logits, self.mask)
        return outputs["velocity"], (outputs["compatibility"] - logits) / self.model.config.membership_time


def rollout(field: ModelField, x0: torch.Tensor, n_steps: int, interval: float, substeps: int = 4,
            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Predicted states at the next ``n_steps`` observation times (RK4), shape (B, n_steps, N, d)."""
    times = torch.arange(n_steps + 1, dtype=x0.dtype, device=x0.device) * interval
    states, _ = odeint(field, field.initial_state(x0, mask), times, method="rk4",
                       options={"step_size": interval / substeps})
    return states[1:].movedim(0, 1)


@dataclass
class TrajectoryTrainConfig(TrainConfig):
    horizon_start: int = 1  # observation intervals predicted at the start of the curriculum
    horizon_end: int = 10
    curriculum_epochs: int = 50  # epochs over which the horizon grows linearly
    windows_per_epoch: int = 4096
    batch_size: int = 128
    substeps: int = 4  # RK4 steps per observation interval
    val_horizon: int = 10
    val_first: int = 0  # first start frame of the validation rollouts
    loss_dims: int | None = None  # leading state dimensions in the loss (positions for order 2); None: all
    centre: bool = False  # window-centred coordinates
    grad_clip: float | None = None


def entity_masks(data: TrajectoryData) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-entity masks (B, T, N): which states can start a rollout, and which are observed targets."""
    mask = data.mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1).expand(*mask.shape, data.observations.shape[2])
    return (mask if data.start_mask is None else data.start_mask), mask


def sample_windows(start_mask: torch.Tensor, horizon: int, count: int,
                   generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (trajectory, start) pairs whose window fits and where at least one entity can start."""
    valid = start_mask[:, : start_mask.shape[1] - horizon].any(dim=-1).nonzero()
    pick = torch.randint(len(valid), (count,), generator=generator)
    return valid[pick, 0], valid[pick, 1]


def window_targets(observations, start_mask, target_mask, b, k, horizon, loss_dims, centre):
    """Start states (W, N, d), observed futures (W, h, N, d) and observed target mask (W, h, N) of the windows
    (b, k); with ``centre``, positions are relative to the mean start position of the present entities."""
    steps = k[:, None] + torch.arange(1, horizon + 1, device=k.device)
    present = start_mask[b, k]
    x0, target = observations[b, k], observations[b[:, None], steps]
    if centre:
        weights = present.to(x0.dtype).unsqueeze(-1)
        offset = torch.zeros_like(x0[:, :1])
        offset[..., :loss_dims] = ((x0[..., :loss_dims] * weights).sum(1, keepdim=True)
                                   / weights.sum(1, keepdim=True).clamp_min(1.0))
        x0, target = x0 - offset, target - offset.unsqueeze(1)
    return x0, target, target_mask[b[:, None], steps] & present[:, None, :]


def window_errors(field, observations, start_mask, target_mask, b, k, horizon, interval, substeps, loss_dims,
                  centre):
    """Squared error of every window over its observed targets, the number of observed steps (a step counts as
    the share of the present entities observed in it), and the predictions."""
    x0, target, observed = window_targets(observations, start_mask, target_mask, b, k, horizon, loss_dims, centre)
    present = start_mask[b, k]
    prediction = rollout(field, x0, horizon, interval, substeps, present)
    dims = slice(None) if loss_dims is None else slice(0, loss_dims)
    observed = observed.to(observations.dtype)
    errors = (prediction[..., dims] - target[..., dims]).pow(2).sum(-1)
    share = observed.sum(-1) / present.sum(-1, keepdim=True).to(observations.dtype).clamp_min(1.0)
    return (errors * observed).sum((1, 2)), share.sum(1), prediction


@torch.no_grad()
def trajectory_error(model: torch.nn.Module, data: TrajectoryData, horizon: int, stride: int = 10,
                     substeps: int = 4, batch_size: int = 256, loss_dims: int | None = None, centre: bool = False,
                     first: int = 0) -> float:
    """Mean per-step error of rollouts of ``horizon`` intervals from frames ``first``, ``first + stride``, ..."""
    model.eval()
    device = next(model.parameters()).device
    field = ModelField(model)
    interval = float(data.times[1] - data.times[0])
    start_mask, target_mask = (m.to(device) for m in entity_masks(data))
    observations = data.observations.to(device)
    any_start = start_mask.any(-1).cpu()
    starts = [(b, k) for b in range(len(observations))
              for k in range(first, observations.shape[1] - horizon, stride) if any_start[b, k]]
    total, count = 0.0, 0.0
    for s in range(0, len(starts), batch_size):
        b, k = (torch.tensor(v, device=device) for v in zip(*starts[s : s + batch_size]))
        error, steps, _ = window_errors(field, observations, start_mask, target_mask, b, k, horizon, interval,
                                        substeps, loss_dims, centre)
        total += error.sum().item()
        count += steps.sum().item()
    return total / max(count, 1.0)


def train_trajectories(model: torch.nn.Module, train: TrajectoryData, val: TrajectoryData,
                       config: TrajectoryTrainConfig) -> TrainResult:
    """Minimise L_pred on random windows with a horizon curriculum; keep the best validation state."""
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = next(model.parameters()).device
    field = ModelField(model)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    steps = math.ceil(config.windows_per_epoch / config.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs * steps)
    interval = float(train.times[1] - train.times[0])
    start_mask, target_mask = entity_masks(train)
    observations, start_dev, target_dev = train.observations.to(device), start_mask.to(device), target_mask.to(device)
    events = train.events.to(device) if config.lambda_event > 0 and train.events is not None else None

    best_val, best_epoch, best_state = float("inf"), -1, copy.deepcopy(model.state_dict())
    history, start = [], time.time()
    for epoch in range(config.epochs):
        model.train()
        progress = min(1.0, epoch / max(1, config.curriculum_epochs))
        horizon = round(config.horizon_start + progress * (config.horizon_end - config.horizon_start))
        b, k = sample_windows(start_mask, horizon, config.windows_per_epoch, generator)
        running = 0.0
        for step in range(steps):
            window = slice(step * config.batch_size, (step + 1) * config.batch_size)
            bw, kw = b[window].to(device), k[window].to(device)
            error, count, prediction = window_errors(field, observations, start_dev, target_dev, bw, kw, horizon,
                                                     interval, config.substeps, config.loss_dims, config.centre)
            fit = error.sum() / count.sum().clamp_min(1.0)
            loss = fit
            if events is not None:  # the event at every predicted step, from the predicted state
                frames = kw[:, None] + torch.arange(1, horizon + 1, device=device)
                logits = model.event_logits(prediction)
                loss = loss + config.lambda_event * cross_entropy(logits.flatten(0, 1),
                                                                  events[bw[:, None], frames].flatten())
            if config.lambda_size > 0 and model.config.scales is not None:  # L_size on the start states
                present = start_dev[bw, kw]
                z = model.memberships(model.initial_logits(observations[bw, kw], present), present)
                loss = loss + config.lambda_size * model.size_penalty(z, present)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            scheduler.step()
            running += fit.item()
        val_error = trajectory_error(model, val, config.val_horizon, substeps=config.substeps,
                                     loss_dims=config.loss_dims, centre=config.centre, first=config.val_first)
        history.append({"epoch": epoch, "horizon": horizon, "train": running / steps, "val": val_error})
        if epoch >= config.curriculum_epochs and val_error < best_val:
            best_val, best_epoch, best_state = val_error, epoch, copy.deepcopy(model.state_dict())
        if epoch >= config.curriculum_epochs and epoch - max(best_epoch, config.curriculum_epochs) >= config.patience:
            break
    model.load_state_dict(best_state)
    return TrainResult(best_val, best_epoch, len(history), time.time() - start, history)
