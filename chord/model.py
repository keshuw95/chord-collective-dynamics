"""CHORD vector field.

States have shape (B, N, d). ``forward`` returns the velocity together with the intermediate quantities used by
the losses. The default configuration is the closed model: static collectives indexed by entity, every entity
always present, first-order dynamics. Open systems switch on the relational compatibility, the null collective,
presence masks, the kinematic readout and the pairwise channel; each reduces to the closed model when switched off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from entmax import entmax15
from torch import nn
from torch.nn.utils import parametrizations

EMPTY_PENALTY = 1e3  # relational distance assigned to collectives without members
PADDING_LOGIT = -1e4  # (scaled) logit of the padding columns of the multiscale normalisation


@dataclass
class CHORDConfig:
    n_nodes: int
    state_dim: int = 1
    n_collectives: int = 32  # M
    query_dim: int = 32  # d_q
    embed_dim: int = 16  # d_e; 0 disables entity embeddings
    member_dim: int = 64
    collective_dim: int = 64
    coupling_dim: int | None = None  # d_o; defaults to state_dim (order 1) or state_dim / 2 (order 2)
    hidden: int = 128
    temperature: float = 0.1  # tau
    membership_time: float = 1.0  # tau_s of the membership dynamics along trajectories
    identity_first: bool = True  # zero-initialise the state part of W_q, so memberships start from identity
    # open systems
    order: int = 1  # 1: dx/dt = g(x) + B r; 2: x = (p, v), dp/dt = v, dv/dt = g(x) + B r
    relational: bool = False  # add the affinity -||W_r (x_i - xbar_m)||^2 / 2 to the collective's centroid
    relational_dim: int = 8  # d_r
    relational_iters: int = 2  # refinement passes of the instantaneous memberships
    null_collective: bool = False  # extra collective with zero influence, for entities outside every collective
    null_init: float = -2.0
    centroid_feedback: bool = False  # add W_c (xbar_m - x_i) to each collective's influence on member i
    pairwise_channel: bool = False  # add a pairwise background term to the collective coupling
    pair_aggregation: str = "mean"  # "mean" or "sum" over the other present entities
    pair_neighbours: int | None = None  # k: only the k nearest present entities send pair messages (None: all)
    translation_invariant: bool = False  # order 2: positions enter only as differences
    # full model
    receiver_specific: bool = False  # r_i = sum_m z_im psi(h_m, x_i - xbar_m) instead of sum_m z_im psi(h_m)
    scales: tuple[int, ...] | None = None  # multiscale collectives, e.g. (1, 2, 4, 8); sum must equal M
    scale_slack: float = 1.5  # L_size bounds a collective of scale s at scale_slack * N_present / M_s members
    event_anchor: int | None = None  # joint discrete events over the entities; the anchor entity means "none"
    latent_dim: int = 0  # dimension of the latent intention of each collective (CHORD-Stochastic); 0: deterministic


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, n_hidden: int = 2):
        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, out_dim))
        super().__init__(*layers)


class AffineReadout(nn.Module):
    """dx_i/dt = g(x_i) + B r_i with orthonormal columns of B."""

    def __init__(self, state_dim: int, coupling_dim: int, hidden: int):
        super().__init__()
        if coupling_dim > state_dim:
            raise ValueError("the affine readout needs coupling_dim <= state_dim")
        self.intrinsic = MLP(state_dim, state_dim, hidden)
        self.coupling_map = parametrizations.orthogonal(nn.Linear(coupling_dim, state_dim, bias=False))

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.intrinsic(x) + self.coupling_map(r)


class KinematicReadout(nn.Module):
    """Second-order readout for x = (p, v): dp/dt = v and dv/dt = g(x) + B r with orthonormal B."""

    def __init__(self, state_dim: int, coupling_dim: int, hidden: int, intrinsic_positions: bool = True):
        super().__init__()
        half = state_dim // 2
        if coupling_dim > half:
            raise ValueError("the kinematic readout needs coupling_dim <= state_dim / 2")
        self.intrinsic_positions = intrinsic_positions
        self.intrinsic = MLP(state_dim if intrinsic_positions else half, half, hidden)
        self.coupling_map = parametrizations.orthogonal(nn.Linear(coupling_dim, half, bias=False))

    def forward(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        acceleration = self.intrinsic(x if self.intrinsic_positions else x[..., half:]) + self.coupling_map(r)
        return torch.cat([x[..., half:], acceleration], dim=-1)


class CHORD(nn.Module):
    def __init__(self, config: CHORDConfig):
        super().__init__()
        c = self.config = config
        d, d_q = c.state_dim, c.query_dim
        if c.order not in (1, 2) or (c.order == 2 and d % 2):
            raise ValueError("order must be 1, or 2 with an even state dimension (positions and velocities)")
        if c.translation_invariant and c.order != 2:
            raise ValueError("translation invariance needs positions, i.e. order 2")
        if c.scales is not None and sum(c.scales) != c.n_collectives:
            raise ValueError("n_collectives must equal the sum of the scales")
        self.coupling_dim = c.coupling_dim or (d if c.order == 1 else d // 2)
        self.embeddings = nn.Parameter(0.1 * torch.randn(c.n_nodes, c.embed_dim)) if c.embed_dim > 0 else None
        n_event = int(c.event_anchor is not None)

        # latent incidence
        self.query = nn.Linear(d + c.embed_dim + n_event, d_q, bias=False)
        if c.identity_first:
            with torch.no_grad():
                self.query.weight[:, :d].zero_()
        self.prototypes = nn.Parameter(torch.randn(c.n_collectives, d_q) / math.sqrt(d_q))
        if c.relational:
            self.relational_map = nn.Linear(d, c.relational_dim, bias=False)
            with torch.no_grad():
                self.relational_map.weight.mul_(math.sqrt(3.0))  # unit-variance rows
        self.scales = tuple(c.scales) if c.scales is not None else (c.n_collectives,)
        starts = [sum(self.scales[:k]) for k in range(len(self.scales))]
        self.blocks = [(a, a + m) for a, m in zip(starts, self.scales)]  # slot range of each scale
        n_null = len(self.scales)
        self.null_logit = (nn.Parameter(torch.full((n_null,), float(c.null_init)) if c.scales is not None
                                        else torch.tensor(float(c.null_init))) if c.null_collective else None)
        self.register_buffer("slot_scale_size", torch.tensor([float(m) for m in self.scales for _ in range(m)]))
        self.register_buffer("slot_rank", torch.tensor([j for m in self.scales for j in range(m)]))
        # column layout of the multiscale normalisation: scale_index[s] lists the columns of scale s (its
        # collectives, then its null column), padded with an extra padding column; scale_inverse maps the
        # flattened (S, width) result back to the columns [M collectives, S null columns]
        n_columns = c.n_collectives + (n_null if c.null_collective else 0)
        width = max(self.scales) + int(c.null_collective)
        index, inverse = [], [0] * n_columns
        for k, (a, b) in enumerate(self.blocks):
            columns = list(range(a, b)) + ([c.n_collectives + k] if c.null_collective else [])
            for j, column in enumerate(columns):
                inverse[column] = k * width + j
            index.append(columns + [n_columns] * (width - len(columns)))
        self.register_buffer("scale_index", torch.tensor(index))
        self.register_buffer("scale_inverse", torch.tensor(inverse))
        if c.event_anchor is not None:
            self.event_head = MLP(d + c.embed_dim, 1, c.hidden)

        # hyperedge encoding
        self.member = MLP(d + 1 + n_event, c.member_dim, c.hidden)
        self.decoder = MLP(c.member_dim + 1 + d_q + c.latent_dim, c.collective_dim, c.hidden)
        self.latent = None  # (B, M, latent_dim) intentions of the current forecast, set by the sampler
        if c.latent_dim > 0:  # zero weights on the intentions: training starts from the deterministic model
            with torch.no_grad():
                self.decoder[0].weight[:, -c.latent_dim:].zero_()
        self.interaction = nn.Linear(c.collective_dim, c.collective_dim, bias=False)

        # collective feedback
        self.output = nn.Linear(c.collective_dim, self.coupling_dim, bias=False)
        if c.receiver_specific:
            self.receiver = MLP(c.collective_dim + d, self.coupling_dim, c.hidden)
        if c.centroid_feedback:
            self.feedback = nn.Linear(d, self.coupling_dim, bias=False)
            nn.init.zeros_(self.feedback.weight)
        if c.pairwise_channel:
            self.pair_term = MLP(2 * (d + c.embed_dim), self.coupling_dim, c.hidden)
            with torch.no_grad():  # the background starts at zero
                self.pair_term[-1].weight.zero_()
                self.pair_term[-1].bias.zero_()
        self.readout = (KinematicReadout(d, self.coupling_dim, c.hidden, not c.translation_invariant) if c.order == 2
                        else AffineReadout(d, self.coupling_dim, c.hidden))

    # ------------------------------------------------------------------ helpers

    def without_positions(self, x: torch.Tensor) -> torch.Tensor:
        """x with the position block zeroed when the model is translation invariant."""
        if not self.config.translation_invariant:
            return x
        half = x.shape[-1] // 2
        return torch.cat([torch.zeros_like(x[..., :half]), x[..., half:]], dim=-1)

    def relative_to(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """States with positions relative to ``reference`` when translation invariant; otherwise x."""
        if not self.config.translation_invariant:
            return x.expand(*torch.broadcast_shapes(x.shape, reference.shape))
        half = x.shape[-1] // 2
        position = x[..., :half] - reference[..., :half]
        return torch.cat([position, x[..., half:].expand_as(position)], dim=-1)

    @staticmethod
    def presence(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Presence indicators (B, N) as floats; all ones without a mask."""
        if mask is None:
            return x.new_ones(x.shape[:-1])
        return mask.to(x.dtype).expand(x.shape[:-1])

    def event_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Scores (..., N) of the categorical event, from each entity's state relative to the anchor."""
        a = self.config.event_anchor
        features = x - x[..., a : a + 1, :]
        if self.embeddings is not None:
            features = torch.cat([features, self.embeddings.expand(*x.shape[:-1], -1)], dim=-1)
        return self.event_head(features).squeeze(-1)

    def event_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.event_logits(x), dim=-1)

    # ------------------------------------------------------------------ latent incidence

    def entity_features(self, x: torch.Tensor) -> torch.Tensor:
        parts = [self.without_positions(x)]
        if self.embeddings is not None:
            parts.append(self.embeddings.expand(*x.shape[:-1], -1))
        if self.config.event_anchor is not None:
            parts.append(self.event_probabilities(x).unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def _with_null(self, logits: torch.Tensor) -> torch.Tensor:
        """Append the null column(s), one per scale, after the M collectives."""
        if self.null_logit is None:
            return logits
        null = self.null_logit.to(logits.dtype).reshape(-1)
        return torch.cat([logits, null.expand(*logits.shape[:-1], len(null))], dim=-1)

    def _prototype_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.query(self.entity_features(x)) @ self.prototypes.T / math.sqrt(self.config.query_dim)

    def relational_distance(self, x: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """||W_r x_i - W_r xbar_m||^2 to the membership-weighted centroid of each collective, (B, N, M)."""
        y = self.relational_map(x)
        weights = z[..., : self.config.n_collectives] * self.presence(x, mask).unsqueeze(-1)
        size = weights.sum(dim=1)
        centre = weights.transpose(1, 2) @ y / size.clamp_min(1e-6).unsqueeze(-1)
        distance = (y.unsqueeze(2) - centre.unsqueeze(1)).pow(2).sum(-1)
        return torch.where((size > 1e-3).unsqueeze(1), distance, torch.full_like(distance, EMPTY_PENALTY))

    def compatibility(self, x: torch.Tensor, z: torch.Tensor | None = None,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
        """S~ = Q K^T / sqrt(d_q), minus half the relational distance under memberships ``z`` when the
        relational term is on; shape (B, N, M), plus the null columns."""
        logits = self._prototype_logits(x)
        if self.config.relational:
            if z is None:
                raise ValueError("the relational compatibility needs the current memberships")
            logits = logits - 0.5 * self.relational_distance(x, z, mask)
        return self._with_null(logits)

    def seed_distance(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Relational distance to anchor seeds chosen by farthest-point sampling in W_r x among the present
        entities; slots beyond the number of present entities stay empty. Shape (B, N, M)."""
        y = self.relational_map(x)
        present = self.presence(x, mask) > 0
        batch = torch.arange(len(y), device=y.device)
        with torch.no_grad():
            spread = (y - (y * present.unsqueeze(-1)).sum(1, keepdim=True)
                      / present.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)).pow(2).sum(-1)
            score = torch.where(present, spread, torch.full_like(spread, -1.0))
            seeds, nearest = [], torch.full_like(spread, float("inf"))
            for _ in range(max(self.scales)):  # seeds are nested: scale s uses the first M_s
                index = score.argmax(dim=1)
                seeds.append(index)
                nearest = torch.minimum(nearest, (y - y[batch, index].unsqueeze(1)).pow(2).sum(-1))
                score = torch.where(present, nearest, torch.full_like(nearest, -1.0))
            seeds = torch.stack(seeds, dim=1)[:, self.slot_rank]
            valid = self.slot_rank < present.sum(1, keepdim=True)
        centre = y[batch.unsqueeze(1), seeds]
        distance = (y.unsqueeze(2) - centre.unsqueeze(1)).pow(2).sum(-1)
        return torch.where(valid.unsqueeze(1), distance, torch.full_like(distance, EMPTY_PENALTY))

    def initial_logits(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Membership logits at the current state. With the relational term, memberships start from anchor
        seeds and are refined by ``relational_iters`` centroid passes."""
        if not self.config.relational:
            return self.compatibility(x, mask=mask)
        logits = self._with_null(self._prototype_logits(x) - 0.5 * self.seed_distance(x, mask))
        for _ in range(self.config.relational_iters):
            logits = self.compatibility(x, self.memberships(logits, mask), mask)
        return logits

    def memberships(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Z = entmax(S / tau), per scale and weighted 1 / S; rows of absent entities are zero."""
        scaled = logits / self.config.temperature
        if self.config.scales is None:
            z = entmax15(scaled, dim=-1)
        else:
            padded = torch.cat([scaled, scaled.new_full((*scaled.shape[:-1], 1), PADDING_LOGIT)], dim=-1)
            z = entmax15(padded[..., self.scale_index], dim=-1) / len(self.scales)
            z = z.flatten(-2)[..., self.scale_inverse]
        return z if mask is None else z * self.presence(logits, mask).unsqueeze(-1)

    def size_penalty(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """L_size: a collective of a scale with M_s collectives may hold at most scale_slack * N_present / M_s
        members; mean over the batch of sum_m relu(n_m - bound_m)^2."""
        size = z[..., : self.config.n_collectives].sum(dim=-2) * len(self.scales)
        present = (mask.to(z.dtype) if mask is not None else z.new_ones(z.shape[:-1])).sum(-1, keepdim=True)
        bound = self.config.scale_slack * present / self.slot_scale_size.to(z.dtype)
        return torch.relu(size - bound).pow(2).sum(dim=-1).mean()

    # ------------------------------------------------------------------ hyperedge encoding

    def collective_states(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Size- and membership-aware set pooling, then the decoder rho; shape (B, M, d_h)."""
        batch, _, n_collectives = z.shape
        members_x = x.unsqueeze(2).expand(-1, -1, n_collectives, -1)
        if self.config.translation_invariant:
            centre = z.transpose(1, 2) @ x / z.sum(1).clamp_min(1e-6).unsqueeze(-1)
            members_x = self.relative_to(members_x, centre.unsqueeze(1))
        pairs = [members_x, z.unsqueeze(-1)]
        if self.config.event_anchor is not None:
            pairs.append(self.event_probabilities(x).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_collectives, 1))
        members = self.member(torch.cat(pairs, dim=-1))
        size = z.sum(dim=1)
        pooled = (z.unsqueeze(-1) * members).sum(dim=1) / (size.unsqueeze(-1) + 1e-6)
        inputs = [pooled, size.unsqueeze(-1) * len(self.scales), self.prototypes.expand(batch, -1, -1)]
        if self.config.latent_dim > 0:
            latent = self.latent if self.latent is not None else z.new_zeros(batch, n_collectives, self.config.latent_dim)
            inputs.append(latent.to(z.dtype))
        return self.decoder(torch.cat(inputs, dim=-1))

    def evolve(self, h: torch.Tensor) -> torch.Tensor:
        """Interaction between collectives, as a residual attention update."""
        attention = torch.softmax(h @ h.transpose(-1, -2) / math.sqrt(h.shape[-1]), dim=-1)
        return h + attention @ self.interaction(h)

    # ------------------------------------------------------------------ collective feedback

    def receiver_coupling(self, x: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """sum_m z_im psi(h_m, x_i - xbar_m), shape (B, N, d_o)."""
        centre = z.transpose(1, 2) @ x / z.sum(1).clamp_min(1e-6).unsqueeze(-1)
        if self.config.translation_invariant:
            own = self.relative_to(x.unsqueeze(2), centre.unsqueeze(1))
        else:
            own = x.unsqueeze(2).expand(-1, -1, centre.shape[1], -1)
        decoded = self.receiver(torch.cat([h.unsqueeze(1).expand(-1, x.shape[1], -1, -1), own], dim=-1))
        return (z.unsqueeze(-1) * decoded).sum(dim=2)

    def centroid_coupling(self, x: torch.Tensor, z: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """sum_m z_im W_c (xbar_m - x_i): a restoring term towards the centroids of an entity's collectives."""
        weights = z * self.presence(x, mask).unsqueeze(-1)
        centre = weights.transpose(1, 2) @ x / weights.sum(1).clamp_min(1e-6).unsqueeze(-1)
        return self.feedback(z @ centre - z.sum(-1, keepdim=True) * x)

    def pairwise_coupling(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Pairwise background: aggregate of phi(x_i, x_j, iota_i, iota_j) over the other present entities,
        or over the k nearest of them when ``pair_neighbours`` is set."""
        n_nodes = x.shape[-2]
        present = self.presence(x, mask)
        if self.config.pair_neighbours is None:
            receiver = self.without_positions(x).unsqueeze(-2).expand(*x.shape[:-2], n_nodes, n_nodes, -1)
            sender = self.relative_to(x.unsqueeze(-3), x.unsqueeze(-2))
            if self.embeddings is not None:
                iota = self.embeddings.expand(*x.shape[:-1], -1)
                receiver = torch.cat([receiver, iota.unsqueeze(-2).expand(*receiver.shape[:-1], -1)], dim=-1)
                sender = torch.cat([sender, iota.unsqueeze(-3).expand(*sender.shape[:-1], -1)], dim=-1)
            weights = present.unsqueeze(-2) * (1.0 - torch.eye(n_nodes, dtype=x.dtype, device=x.device))
        else:
            k = min(self.config.pair_neighbours, n_nodes - 1)
            position = x[..., : x.shape[-1] // 2] if self.config.order == 2 else x
            with torch.no_grad():
                excluded = torch.eye(n_nodes, dtype=torch.bool, device=x.device) | ~(present > 0).unsqueeze(-2)
                nearest, index = torch.cdist(position, position).masked_fill(excluded, float("inf")).topk(
                    k, dim=-1, largest=False)
            batch = torch.arange(x.shape[0], device=x.device).view(-1, 1, 1)
            receiver = self.without_positions(x).unsqueeze(-2).expand(*index.shape, -1)
            sender = self.relative_to(x[batch, index], x.unsqueeze(-2))
            if self.embeddings is not None:
                iota = self.embeddings.expand(*x.shape[:-1], -1)
                receiver = torch.cat([receiver, iota.unsqueeze(-2).expand(*index.shape, -1)], dim=-1)
                sender = torch.cat([sender, iota[batch, index]], dim=-1)
            weights = torch.isfinite(nearest).to(x.dtype)
        if self.config.pair_aggregation == "mean":
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
        return (self.pair_term(torch.cat([receiver, sender], dim=-1)) * weights.unsqueeze(-1)).sum(dim=-2)

    # ------------------------------------------------------------------ vector field

    def forward(self, x: torch.Tensor, logits: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Velocity at states ``x``. ``logits`` are the membership logits S (default: the instantaneous
        memberships); ``mask`` (B, N) marks the present entities."""
        if logits is None:
            logits = self.initial_logits(x, mask)
        z_full = self.memberships(logits, mask)
        z = z_full[..., : self.config.n_collectives]  # the null collective has no influence
        h = self.evolve(self.collective_states(x, z))
        psi = self.output(h)
        coupling = self.receiver_coupling(x, z, h) if self.config.receiver_specific else z @ psi
        if self.config.centroid_feedback:
            coupling = coupling + self.centroid_coupling(x, z, mask)
        if self.config.pairwise_channel:
            coupling = coupling + self.pairwise_coupling(x, mask)
        velocity = self.readout(x, coupling)
        if mask is not None:
            velocity = velocity * self.presence(x, mask).unsqueeze(-1)  # absent entities are frozen
        outputs = {
            "velocity": velocity,
            "coupling": coupling,
            "memberships": z,
            "influences": psi,
            # target of the membership dynamics dS/dt = (S~(X, Z) - S) / tau_s
            "compatibility": self.compatibility(x, z_full, mask) if self.config.relational else self.compatibility(x),
        }
        if self.config.scales is not None:
            outputs["size_penalty"] = self.size_penalty(z_full, mask)
        if self.config.event_anchor is not None:
            outputs["event_logits"] = self.event_logits(x)
        return outputs
