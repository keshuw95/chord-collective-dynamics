# CHORD: Collective Higher-Order Relational Dynamics

Code for *Learning Collective Dynamics Beyond Pairwise Relations via Latent Higher-Order Interactions*.

## Overview

Flocks, crowds and sports teams act in groups, yet learned models of interacting dynamics represent interactions as pairs. A group of three and its three pairwise interactions induce the same interaction graph, so no model that sees only pairs can tell them apart, and the groups themselves are never observed.

We measure what pairs cannot express by the *higher-order energy* and show that:

- every pairwise model errs by at least a constant times the higher-order energy;
- models given only the pairwise graph face a minimax error floor;
- latent collectives are identifiable from trajectories under anchor and excitation conditions;
- the best collective model gains exactly the higher-order energy over pairwise models under an isometric readout.

These results dictate CHORD, a neural ODE with three modules:

1. **Latent incidence:** sparse entmax memberships assign entities to latent collectives.
2. **Hyperedge encoding:** each collective pools its members' states and decodes them nonlinearly; collectives interact through attention.
3. **Collective feedback:** collectives act on their members through an orthonormal readout.

The full model adds multiscale collectives, receiver-specific influence, joint events and components for open systems, in which groups form, split and merge. CHORD-Stochastic samples futures through a latent intention per collective.

## Repository Structure

```
chord/
├── model.py        # CHORD vector field and its full-model components
├── losses.py       # prediction losses, regularisers (entropy, minimum volume, size), energy score
├── data.py         # trajectory containers, velocity estimates from observed positions
├── train.py        # derivative matching and rollout training through the ODE solver
└── stochastic.py   # CHORD-Stochastic: training on K sampled futures, sampling
tests/
└── test_chord.py   # forward pass, presence masks, deterministic and stochastic training
```

## Installation

```bash
pip install -e ".[dev]"
pytest
```
