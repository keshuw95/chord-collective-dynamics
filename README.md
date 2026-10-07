# CHORD: Collective Higher-Order Relational Dynamics

Code for *Learning Collective Dynamics Beyond Pairwise Relations via Latent Higher-Order Interactions*.

## Overview

Learned models of interacting dynamics usually represent interactions as pairs, yet flocks, crowds and sports teams act in groups. A group of three and the three pairwise interactions it contains induce the same interaction graph, so no model that sees only pairs can tell them apart. The groups themselves are never observed: trajectories record where agents go, not which groups they act in.

**Theory.** We measure what pairs cannot express by the *higher-order energy* $E_{HO}$, the part of the dynamics that no sum of pairwise terms can represent, and show that:

- every pairwise model errs by at least a constant times $E_{HO}$;
- models given only the pairwise graph face a minimax error floor, because different hypergraphs share one graph;
- latent collectives are identifiable from trajectories when each collective has an anchor member and the dynamics excite it;
- the best collective model improves on the best pairwise model by exactly $E_{HO}$ under an isometric readout.

**Model.** These results dictate CHORD, a neural ODE whose vector field is a latent collective model:

1. **Latent incidence.** Entities are softly assigned to latent collectives through sparse entmax memberships $Z$, whose rows lie on the simplex and can be exactly one-hot (anchors).
2. **Hyperedge encoding.** Each collective pools its members' states, weighted by membership, and decodes them nonlinearly, which creates the higher-order terms; collectives then interact through attention.
3. **Collective feedback.** Collectives act on their members through $Z$, and an orthonormal readout turns the coupling into the time derivative.
4. **Neural ODE fit.** The states and the membership logits are integrated jointly and fitted to trajectories; entropy and minimum-volume regularisers select the identifiable factorisation.

The full model adds multiscale collectives, receiver-specific influence, joint discrete events (for example, ball possession), and components for *open* systems, in which groups form, split and merge and agents enter and leave. **CHORD-Stochastic** draws a latent intention for each collective at the start of a forecast, so that sampling yields multiple futures. It is trained with the best-of-K objective or with the energy score.

## Repository Structure

```
chord-collective-dynamics/
├── chord/
│   ├── model.py        # CHORD vector field: latent incidence, hyperedge encoding, collective feedback,
│   │                   #   open-system components, multiscale collectives, receiver-specific influence,
│   │                   #   joint events and latent intentions
│   ├── losses.py       # prediction losses, regularisers (entropy, minimum volume, size), energy score
│   ├── data.py         # trajectory containers, velocity estimates from observed positions
│   ├── train.py        # derivative matching and rollout training through the ODE solver
│   └── stochastic.py   # CHORD-Stochastic: training on K sampled futures, sampling
├── tests/
│   └── test_chord.py   # forward pass, presence masks, deterministic and stochastic training
└── pyproject.toml
```

## Installation

```bash
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+, PyTorch, torchdiffeq and entmax.
