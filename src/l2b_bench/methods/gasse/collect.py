"""Gasse 2019 — expert data collection: explore-then-strong-branch.

Faithful port of ``02_generate_dataset.py`` from the official repository
(github.com/ds4dm/learn2branch-ecole). At every node, a coin with probability
``expert_probability`` decides the brancher of the step:

- **expert** — full strong branching scores (vanilla fullstrong, idempotent):
  the scores are recorded as a sample and the best candidate branches;
- **exploration** — Ecole-style pseudocosts accumulated over the episode:
  nothing is recorded, the best pseudocost candidate branches.

This is the Zarpellon-style exploration of the paper: the expert data covers
states the pure-expert trajectory would never visit.

SCIP parameters are the paper's protocol, verified on SCIP 10
(``separating/maxrounds``, ``presolving/maxrestarts``, ``timing/clocktype``
all exist with the same semantics):

- ``separating/maxrounds = 0`` — no cut separation rounds,
- ``presolving/maxrestarts = 0`` — no restarts,
- ``timing/clocktype = 2`` — wall clock timing,
- ``limits/time`` — the episode budget (45 min in the paper).

One deviation from the official script, for safety: when neither scorer
returns a finite score for any candidate (empty action set aside), the
decision falls back to SCIP's default rule via the SKIP action instead of
argmaxing over NaNs.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gyozas.collection import CollectionMetadata
from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import NodeBipartite
from gyozas.observations.pseudo_cost import Pseudocosts
from gyozas.observations.strong_branching_scores import StrongBranchingScores
from gyozas.observations.structs import BipartiteGraph
from gyozas.seeds import SeedTriplet

# The paper's collection protocol (SCIP 6.0.1 names, unchanged in SCIP 10).
GASSE_SCIP_PARAMS = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "timing/clocktype": 2,
}

# The paper's episode budget.
GASSE_TIME_LIMIT = 45 * 60.0


@dataclass
class BranchingSample:
    """One collected (state, action) pair with the expert's full scores.

    ``scores`` is the strong-branching score vector indexed by LP position
    (NaN outside the candidate set) — the training target's raw material, and
    what the multi-optimal distribution (R4.3) is measured on.
    """

    observation: BipartiteGraph
    action_set: np.ndarray
    action: int
    scores: np.ndarray


@dataclass
class CollectionResult:
    """Everything a collection campaign produces (samples + protocol metadata)."""

    samples: list[BranchingSample] = field(default_factory=list)
    metadata: CollectionMetadata = field(default_factory=CollectionMetadata)
    n_episodes: int = 0
    n_nodes: int = 0
    statuses: list[str] = field(default_factory=list)


def _argmax_finite(scores: np.ndarray, action_set: np.ndarray) -> int | None:
    """Best candidate by score; None when no candidate has a finite score."""
    best, best_score = None, -np.inf
    for action in action_set:
        score = scores[int(action)]
        if np.isfinite(score) and score > best_score:
            best, best_score = int(action), score
    return best


def collect_episode(
    env: Environment,
    rng: np.random.Generator,
    expert_probability: float = 0.5,
    metadata: CollectionMetadata | None = None,
) -> tuple[list[BranchingSample], int, str]:
    """Run one episode under explore-then-strong-branch; return its samples.

    The environment must be configured with the ``NodeBipartite`` observation
    and the gasse SCIP parameters (see :func:`collect`). The multi-optimal
    record (R4.3) accumulates into ``metadata`` when given.
    """
    pseudocosts = Pseudocosts()
    strong_branching = StrongBranchingScores()
    samples: list[BranchingSample] = []
    n_nodes = 0

    obs, action_set, _reward, done, info = env.reset()
    pseudocosts.reset(env.model)
    strong_branching.reset(env.model)
    status = info.get("status")

    while not done:
        n_nodes += 1
        # Branching candidates only — the SKIP action (handed back to SCIP's
        # default rule when no scorer decides) is not a branching decision and
        # never enters a sample.
        candidates = np.asarray(action_set)
        candidates = candidates[candidates >= 0]

        query_expert = rng.random() < expert_probability
        if query_expert:
            scores = strong_branching.extract(env.model, done=False)
        else:
            scores = pseudocosts.extract(env.model, done=False)

        action = None
        if scores is not None and len(scores):
            action = _argmax_finite(scores, candidates)

        if query_expert and action is not None and len(scores):
            sample = BranchingSample(
                observation=obs,
                action_set=candidates.copy(),
                action=action,
                scores=np.asarray(scores, dtype=np.float64).copy(),
            )
            samples.append(sample)
            if metadata is not None:
                metadata.record(sample.scores, sample.action_set)

        # No finite score anywhere (or an empty score vector): hand the
        # decision back to SCIP's default rule.
        step_action = action if action is not None else int(ExtraBranchingActions.SKIP)
        obs, action_set, _reward, done, info = env.step(step_action)
        status = info.get("status")

    return samples, n_nodes, str(status)


def collect(
    instance_generator: Any,
    n_episodes: int,
    seeds: list[int],
    expert_probability: float = 0.5,
    time_limit: float = GASSE_TIME_LIMIT,
    on_episode: Any = None,
) -> CollectionResult:
    """Collect expert samples over ``n_episodes`` episodes.

    Parameters
    ----------
    instance_generator:
        A gyozas ``InstanceGenerator`` (e.g. ``SetCoverGenerator``); each
        episode pulls the next instance.
    seeds:
        One base seed per episode — the solver seed of the episode's
        :class:`~gyozas.seeds.SeedTriplet` (R2.4).
    expert_probability:
        Probability of querying (and recording) the strong-branching expert at
        each node; otherwise the episode explores on pseudocosts.
    time_limit:
        Per-episode SCIP time limit (the paper: 45 minutes).
    on_episode:
        Optional callback ``on_episode(done, status, n_nodes, n_samples)``
        invoked after each episode — campaign progress reporting.
    """
    if len(seeds) < n_episodes:
        raise ValueError(f"{n_episodes} episodes need {n_episodes} seeds, got {len(seeds)}")

    scip_params = dict(GASSE_SCIP_PARAMS)
    if time_limit is not None:
        scip_params["limits/time"] = time_limit
    # Cosmetic: keep SCIP's log quiet (not part of the protocol).
    scip_params.setdefault("display/verblevel", 0)
    env = Environment(
        instance_generator=instance_generator,
        observation_function=NodeBipartite(),
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
            rng = np.random.default_rng(triplet.augmentation)
            samples, n_nodes, status = collect_episode(
                env, rng, expert_probability, metadata=result.metadata
            )
            result.samples.extend(samples)
            result.n_episodes += 1
            result.n_nodes += n_nodes
            result.statuses.append(status)
            if on_episode is not None:
                on_episode(result.n_episodes, status, n_nodes, len(samples))
    finally:
        env.close()
    return result
