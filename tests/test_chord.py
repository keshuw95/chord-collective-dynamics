import torch

from chord import CHORD, CHORDConfig
from chord.data import TrajectoryData, derivative_targets
from chord.stochastic import StochasticTrainConfig, sample, train_stochastic
from chord.train import TrainConfig, TrajectoryTrainConfig, train_trajectories, train_velocity

N = 6


def full_model(**overrides) -> CHORD:
    config = dict(n_nodes=N, state_dim=4, order=2, embed_dim=4, n_collectives=7, scales=(1, 2, 4), identity_first=False,
                  relational=True, null_collective=True, centroid_feedback=True, pairwise_channel=True,
                  pair_neighbours=3, pair_aggregation="sum", translation_invariant=True, receiver_specific=True,
                  event_anchor=N - 1, membership_time=0.3, hidden=16, member_dim=8, collective_dim=8)
    return CHORD(CHORDConfig(**{**config, **overrides}))


def trajectories(n: int = 8, t: int = 16) -> TrajectoryData:
    torch.manual_seed(0)
    positions = torch.cumsum(0.1 * torch.randn(n, t, N, 2), dim=1)
    mask = torch.ones(n, t, N, dtype=torch.bool)
    mask[:, :4, 0] = False  # an entity that arrives late
    events = torch.randint(N, (n, t))
    return TrajectoryData(torch.arange(t) * 0.2, positions, mask, events=events)


def test_closed_memberships_are_row_stochastic():
    model = CHORD(CHORDConfig(n_nodes=N, n_collectives=4))
    out = model(torch.rand(3, N, 1) * 2 - 1)
    z = out["memberships"]
    assert out["velocity"].shape == (3, N, 1)
    assert torch.allclose(z.sum(-1), torch.ones(3, N), atol=1e-5)


def test_full_model_masks_absent_entities():
    model = full_model()
    x = torch.randn(2, N, 4)
    mask = torch.ones(2, N, dtype=torch.bool)
    mask[0, 1] = False
    out = model(x, mask=mask)
    assert out["velocity"][0, 1].abs().max() == 0
    assert out["memberships"][0, 1].abs().max() == 0
    assert out["event_logits"].shape == (2, N)


def test_training_runs():
    data = trajectories()
    from chord.data import add_velocity_estimates

    model = full_model()
    train_velocity(model, derivative_targets(data), derivative_targets(data),
                   TrainConfig(epochs=1, batch_size=32, lambda_size=0.1, lambda_event=0.1))
    observed = add_velocity_estimates(data)
    train_trajectories(model, observed, observed,
                       TrajectoryTrainConfig(epochs=1, windows_per_epoch=8, batch_size=4, horizon_start=3,
                                             horizon_end=3, val_horizon=3, curriculum_epochs=0, loss_dims=2,
                                             centre=True, substeps=1, lambda_event=0.1, lambda_size=0.1))


def test_stochastic_training_and_sampling():
    from chord.data import add_velocity_estimates

    observed = add_velocity_estimates(trajectories())
    model = full_model(latent_dim=2)
    for objective in ("variety", "energy"):
        train_stochastic(model, observed, observed,
                         StochasticTrainConfig(epochs=1, windows_per_epoch=4, batch_size=2, micro_batch=1, samples=3,
                                               horizon=3, val_samples=3, val_scenes=2, val_first=5,
                                               objective=objective, lambda_event=0.1))
    futures = sample(model, observed.observations[:2, 6], 5, 3, 0.2, mask=observed.start_mask[:2, 6])
    assert futures.shape == (2, 5, 3, N, 4)
    assert not torch.allclose(futures[:, 0], futures[:, 1])
