"""Mamba-Branching 2026 — relpscost-expert trajectory collection.

Port of ``get_samples.py`` + ``SCIPCollectBrancher`` from the official
repository (github.com/doctor-watson626/Mamba-Branching). The paper's
collection: a *random explorer* branches for the first ``k`` decisions, then
the expert (SCIP's ``relpscost``) takes over; a sample is recorded at every
expert decision, and the solver's variable permutation is seeded with ``s``
— the (s, k) Zarpellon grid. Unlike the strong-branching families, one
episode here is not a bag of decisions but an ordered *trajectory*: the
sequence model's training signal is the solve's decision order itself, so
the collection keeps each episode's samples grouped (and so does its cache).

The augmentation is the harness's native ``ZarpellonAugmentation``, wired
through ``BranchingDynamics``: the dynamics applies the permutation on reset
and takes the k explorer decisions itself, so every decision point the
environment exposes is an expert decision.

The state is the ``BranchingState`` observation — the 25 + 8 + 53 triple
Zarpellon parameterized first (TreeGate, T-BranT and TGPPO share it), slots
unavailable on stock PySCIPOpt zeroed.

Deviations from the official collection protocol, documented (the TreeGate
and T-BranT ports' stance):

- the official 'sandbox' setting disables all primal heuristics and provides
  a per-instance cutoff (a precomputed optimum); the pilot keeps default
  SCIP — the collection fidelity that matters here is the expert and the
  state, not the solver side effects;
- one cell of the official s ∈ {0..4} × k ∈ {0, 1, 5, 10, 15} grid is drawn
  per campaign (``MAMBA_S``, ``MAMBA_K``, the repository's defaults — the
  same s = 0, k = 10 cell the TreeGate port draws); the full grid is a
  collection-setup choice for the frozen protocol;
- the official drops nodes with a single branching candidate (the decision
  is forced, nothing to learn); so does the collection here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from gyozas.collection import CollectionMetadata
from gyozas.dynamics.branching import ExtraBranchingActions
from gyozas.environment import Environment
from gyozas.experts import RelpscostExpert

# Default SCIP (the official 'sandbox' deviation is documented above).
MAMBA_SCIP_PARAMS: dict[str, Any] = {}

MAMBA_TIME_LIMIT = 3600.0  # the paper's per-instance collection budget

# One cell of the official augmentation grid (the repository's defaults; the
# same cell as the TreeGate port). The (s, k) pair is part of the collection
# protocol, recorded in the campaign's metadata.
MAMBA_S = 0
MAMBA_K = 10


@dataclass
class MambaSample:
    """One collected expert decision.

    ``candidate_features`` holds one 25-dim row per branching candidate, in
    ``action_set`` order (the LP column positions the policy may branch on);
    ``action`` is the expert's choice among them.
    """

    candidate_features: np.ndarray  # (n_candidates, 25)
    action_set: np.ndarray  # (n_candidates,) LP column positions
    action: int  # the expert's choice (an LP column position)
    node_features: np.ndarray  # (8,)
    mip_features: np.ndarray  # (53,)


def collect_episode(
    env: Environment,
    metadata: CollectionMetadata | None = None,
) -> tuple[list[MambaSample], int, str]:
    """Run one relpscost-expert episode; return its samples in decision order.

    The order is the training signal — the sequence model reads a trajectory
    as the chronological list of the solve's decisions. The environment must
    be configured with the ``BranchingStateObservation`` and the (s, k)
    augmentation; nodes with a single candidate are skipped (the official's
    ``> 1 candidate`` filter — the decision is forced, nothing to learn).
    """
    expert = RelpscostExpert()
    samples: list[MambaSample] = []
    n_nodes = 0

    obs, action_set, _reward, done, info = env.reset()
    expert.reset(env.model)
    status = info.get("status")

    while not done:
        n_nodes += 1
        # Branching candidates only — SKIP is not a branching decision.
        candidates = np.asarray(action_set)
        candidates = candidates[candidates >= 0]

        action = None
        if len(candidates) > 1:  # the official's forced-decision filter
            action = expert.select(env.model, candidates)

        if action is not None:
            samples.append(
                MambaSample(
                    candidate_features=np.asarray(obs.candidate_features, dtype=np.float32),
                    action_set=candidates.copy(),
                    action=action,
                    node_features=np.asarray(obs.node_features, dtype=np.float32),
                    mip_features=np.asarray(obs.mip_features, dtype=np.float32),
                )
            )

        step_action = action if action is not None else int(ExtraBranchingActions.SKIP)
        obs, action_set, _reward, done, info = env.step(step_action)
        status = info.get("status")

    return samples, n_nodes, str(status)
