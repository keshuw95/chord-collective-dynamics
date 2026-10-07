# CHORD: Collective Higher-Order Relational Dynamics

Code for *Learning Collective Dynamics Beyond Pairwise Relations via Latent Higher-Order Interactions*.

## Overview

Flocks, crowds and sports teams act in groups, yet learned models of interacting dynamics represent interactions as pairs. We measure what pairs cannot express by the *higher-order energy*, show that it bounds the error of every pairwise model, and give conditions under which latent groups are identifiable from trajectories.

These results dictate CHORD, a neural ODE that infers sparse latent collectives (entmax memberships), pools each collective's members into a nonlinear state, and lets collectives act on their members through an orthonormal readout. The full model adds multiscale collectives, receiver-specific influence, joint events and components for open systems. CHORD-Stochastic samples futures through a latent intention per collective.

## Repository Structure

```
chord/
├── model.py        # CHORD vector field
├── losses.py       # prediction losses, regularisers, energy score
├── data.py         # trajectory containers, velocity estimates
├── train.py        # derivative matching and rollout training
└── stochastic.py   # CHORD-Stochastic: training on sampled futures, sampling
tests/
└── test_chord.py
```

## Installation

```bash
pip install -e ".[dev]"
pytest
```
