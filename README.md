# CHORD: Collective Higher-Order Relational Dynamics

Core implementation of CHORD, a neural ODE that infers sparse latent collectives from trajectories and lets them act on their members.

## Installation

```bash
pip install -e ".[dev]"
pytest
```

## Contents

| File | Contents |
|---|---|
| `chord/model.py` | The CHORD vector field: latent incidence (entmax memberships), hyperedge encoding (membership-weighted pooling, collective interaction) and collective feedback with an orthonormal readout. It also contains the open-system components (relational memberships, null collective, centroid feedback, pairwise background, kinematic readout), multiscale collectives, receiver-specific influence, joint events and latent intentions. |
| `chord/losses.py` | Prediction losses, the regularisers L_ent, L_vol and L_size, and the energy score |
| `chord/data.py` | Trajectory containers and velocity estimates from observed positions |
| `chord/train.py` | Derivative matching (L_vel) and rollout training through the ODE solver (L_pred) |
| `chord/stochastic.py` | CHORD-Stochastic: training on K sampled futures (best-of-K or energy score) and sampling |

## Usage

```python
from chord import CHORD, CHORDConfig
from chord.data import add_velocity_estimates, derivative_targets
from chord.stochastic import StochasticTrainConfig, sample, train_stochastic
from chord.train import TrainConfig, TrajectoryTrainConfig, train_trajectories, train_velocity

# full model for planar agents observed through positions only; the last entity is the ball
config = CHORDConfig(
    n_nodes=11, state_dim=4, order=2, embed_dim=8, identity_first=False, membership_time=0.3,
    relational=True, null_collective=True, centroid_feedback=True, pairwise_channel=True,
    pair_aggregation="sum", pair_neighbours=8, translation_invariant=True,
    n_collectives=15, scales=(1, 2, 4, 8), receiver_specific=True, event_anchor=10, latent_dim=8,
)
model = CHORD(config)

# train, val: chord.data.TrajectoryData with positions (B, T, N, 2)
train_velocity(model, derivative_targets(train), derivative_targets(val), TrainConfig(lambda_size=0.1))
train, val = add_velocity_estimates(train), add_velocity_estimates(val)
train_trajectories(model, train, val, TrajectoryTrainConfig(loss_dims=2, centre=True, lambda_size=0.1))
train_stochastic(model, train, val, StochasticTrainConfig(objective="variety"))
futures = sample(model, x0, n_samples=20, n_steps=20, interval=0.2, mask=present)
```
