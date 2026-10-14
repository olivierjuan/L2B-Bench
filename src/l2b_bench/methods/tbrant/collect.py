"""T-BranT (Lin 2022) — expert data collection on solver statistics.

Port of ``collect.py`` + ``SCIPCollectBrancher`` from the official repository
(github.com/linjc16/TBranT). The paper's collection: an explorer rule runs
for the first *k* branching decisions, then the expert (SCIP's ``relpscost``)
takes over; a sample is recorded at every expert decision. The benchmark's
tbrant fidelity profile uses k=0 — the expert from the root — so this port
records every node's state and the expert's choice.

The official repo queries the expert through ``executeBranchRule`` on a
patched PySCIPOpt; stock PySCIPOpt exposes the same selection through the
harness's ``RelpscostExpert`` (SCIP's pseudocost score per LP candidate).

The state is the ``BranchingState`` observation (25 per-candidate + 8 node +
53 tree features of Lin 2022, slots unavailable on stock PySCIPOpt zeroed —
see ``gyozas.observations.branching_state``).

Deviations from the official collection protocol, documented:

- the official 'sandbox' setting disables all primal heuristics and provides
  a per-instance cutoff (a precomputed dict of optimal values); the pilot
  keeps default SCIP — the collection fidelity that matters here is the
  expert and the state, not the solver side effects of strong branching;
- when the expert returns no score (empty action set), the decision falls
  back to SCIP's default rule via the SKIP action.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gyozas.collection import CollectionMetadata
from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
from gyozas.environment import Environment
from gyozas.experts import RelpscostExpert
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import BranchingStateObservation
from gyozas.seeds import SeedTriplet

# Default SCIP (the paper's 'default' setting: heuristics, cuts, restarts on).
TBRANT_SCIP_PARAMS: dict[str, Any] = {}

TBRANT_TIME_LIMIT = 3600.0  # the paper's per-instance collection budget


@dataclass
class SolverStatsSample:
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

    samples: list[SolverStatsSample] = field(default_factory=list)
    metadata: CollectionMetadata = field(default_factory=CollectionMetadata)
    n_episodes: int = 0
    n_nodes: int = 0
    statuses: list[str] = field(default_factory=list)


def collect_episode(
    env: Environment,
    metadata: CollectionMetadata | None = None,
) -> tuple[list[SolverStatsSample], int, str]:
    """Run one relpscost-expert episode; return its solver-stats samples.

    The environment must be configured with the ``BranchingStateObservation``
    (see :func:`collect`).
    """
    expert = RelpscostExpert()
    samples: list[SolverStatsSample] = []
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
                SolverStatsSample(
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
    time_limit: float = TBRANT_TIME_LIMIT,
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
    on_episode:
        Optional callback ``on_episode(done, status, n_nodes, n_samples)``
        invoked after each episode — campaign progress reporting.
    """
    if len(seeds) < n_episodes:
        raise ValueError(f"{n_episodes} episodes need {n_episodes} seeds, got {len(seeds)}")

    scip_params = dict(TBRANT_SCIP_PARAMS)
    if time_limit is not None:
        scip_params["limits/time"] = time_limit
    # Cosmetic: keep SCIP's log quiet (not part of the protocol).
    scip_params.setdefault("display/verblevel", 0)
    env = Environment(
        instance_generator=instance_generator,
        observation_function=BranchingStateObservation(),
        information_function=SolveStatus(),
        scip_params=scip_params,
        dynamics=BranchingDynamics(with_extra_actions=[ExtraBranchingActions.SKIP]),
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
