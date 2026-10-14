"""TreeGate (Zarpellon 2021) — expert data collection on solver statistics.

Port of ``collect.py`` + ``SCIPCollectBrancher`` from the official repository
(github.com/ds4dm/branch-search-trees). The paper's collection: a *random
explorer* branches for the first ``k`` decisions, then the expert (SCIP's
``relpscost``) takes over; a sample is recorded at every expert decision, and
the solver's variable permutation is seeded with ``s`` — the (s, k)
augmentation grid of the paper.

The augmentation is the harness's native ``ZarpellonAugmentation`` (R4.2),
wired through ``BranchingDynamics``: the dynamics applies the permutation on
reset and takes the k explorer decisions itself, so every decision point the
environment exposes is an expert decision.

The state is the ``BranchingState`` observation — Zarpellon's original 25 + 8
+ 53 parameterization (T-BranT and TGPPO reused it verbatim), slots
unavailable on stock PySCIPOpt zeroed (see
``gyozas.observations.branching_state``).

Deviations from the official collection protocol, documented:

- the official 'sandbox' setting disables all primal heuristics and provides a
  per-instance cutoff (a precomputed dict of optimal values); the pilot keeps
  default SCIP, like the tbrant port — the collection fidelity that matters
  here is the expert and the state, not the solver side effects;
- one cell of the official s ∈ {0..4} × k ∈ {0, 1, 5, 10, 15} grid is drawn
  per campaign (``TREEGATE_S``, ``TREEGATE_K``, the repository's collect.py
  defaults: s = 0, k = 10); the full grid is a collection-setup choice for
  the frozen protocol.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gyozas.augmentation import ZarpellonAugmentation
from gyozas.collection import CollectionMetadata
from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
from gyozas.environment import Environment
from gyozas.experts import RelpscostExpert
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import BranchingStateObservation
from gyozas.seeds import SeedTriplet

# Default SCIP (the paper's 'default' setting: heuristics, cuts, restarts on).
TREEGATE_SCIP_PARAMS: dict[str, Any] = {}

TREEGATE_TIME_LIMIT = 3600.0  # the paper's per-instance collection budget

# One cell of the official augmentation grid (the repository's collect.py
# defaults); the (s, k) pair is part of the collection protocol, recorded in
# the campaign's metadata.
TREEGATE_S = 0
TREEGATE_K = 10


@dataclass
class TreeGateSample:
    """One collected (state, action) pair.

    ``candidate_features`` holds one 25-dim row per branching candidate, in
    ``action_set`` order (the LP column positions the policy may branch on);
    ``action`` is the expert's choice among them.
    """

    candidate_features: np.ndarray  # (n_candidates, 25)
    action_set: np.ndarray  # (n_candidates,) LP column positions
    action: int  # the expert's choice (an LP column position)
    node_features: np.ndarray  # (8,)
    mip_features: np.ndarray  # (53,)


@dataclass
class CollectionResult:
    """Everything a collection campaign produces (samples + protocol metadata)."""

    samples: list[TreeGateSample] = field(default_factory=list)
    metadata: CollectionMetadata = field(default_factory=CollectionMetadata)
    n_episodes: int = 0
    n_nodes: int = 0
    statuses: list[str] = field(default_factory=list)


def collect_episode(
    env: Environment,
    metadata: CollectionMetadata | None = None,
) -> tuple[list[TreeGateSample], int, str]:
    """Run one relpscost-expert episode; return its solver-stats samples.

    The environment must be configured with the ``BranchingStateObservation``
    and the (s, k) augmentation (see :func:`collect`); the dynamics hides the
    k explorer decisions, so every exposed decision is the expert's.
    """
    expert = RelpscostExpert()
    samples: list[TreeGateSample] = []
    n_nodes = 0

    obs, action_set, _reward, done, info = env.reset()
    expert.reset(env.model)
    status = info.get("status")

    while not done:
        n_nodes += 1
        # Branching candidates only — SKIP is not a branching decision.
        candidates = np.asarray(action_set)
        candidates = candidates[candidates >= 0]

        action = expert.select(env.model, candidates) if len(candidates) else None

        if action is not None:
            samples.append(
                TreeGateSample(
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


def collect(
    instance_generator: Any,
    n_episodes: int,
    seeds: list[int],
    time_limit: float = TREEGATE_TIME_LIMIT,
    s: int = TREEGATE_S,
    k: int = TREEGATE_K,
    on_episode: Any = None,
) -> CollectionResult:
    """Collect relpscost-expert samples over ``n_episodes`` episodes.

    Parameters
    ----------
    seeds:
        One base seed per episode — the solver seed of the episode's
        :class:`~gyozas.seeds.SeedTriplet` (R2.4).
    time_limit:
        Per-episode SCIP time limit (the paper: one hour).
    s, k:
        The Zarpellon augmentation cell: variable-permutation seed and number
        of initial random-explorer decisions.
    on_episode:
        Optional callback ``on_episode(done, status, n_nodes, n_samples)``
        invoked after each episode — campaign progress reporting.
    """
    if len(seeds) < n_episodes:
        raise ValueError(f"{n_episodes} episodes need {n_episodes} seeds, got {len(seeds)}")

    scip_params = dict(TREEGATE_SCIP_PARAMS)
    if time_limit is not None:
        scip_params["limits/time"] = time_limit
    # Cosmetic: keep SCIP's log quiet (not part of the protocol).
    scip_params.setdefault("display/verblevel", 0)
    env = Environment(
        instance_generator=instance_generator,
        observation_function=BranchingStateObservation(),
        information_function=SolveStatus(),
        scip_params=scip_params,
        dynamics=BranchingDynamics(
            augmentation=ZarpellonAugmentation(seed=s, k=k),
            with_extra_actions=[ExtraBranchingActions.SKIP],
        ),
    )
    result = CollectionResult()
    try:
        for episode in range(n_episodes):
            triplet = SeedTriplet.from_base_seed(int(seeds[episode]))
            env.seed(triplet.solver)
            triplet.seed_python()
            samples, n_nodes, status = collect_episode(env, metadata=result.metadata)
            result.samples.extend(samples)
            result.n_episodes += 1
            result.n_nodes += n_nodes
            result.statuses.append(status)
            if on_episode is not None:
                on_episode(result.n_episodes, status, n_nodes, len(samples))
    finally:
        env.close()
    return result
