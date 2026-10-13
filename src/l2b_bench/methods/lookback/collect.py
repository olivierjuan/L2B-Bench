"""Lookback 2022 — expert data collection with parent-child pairs.

Port of ``generate_dataset.py`` from the official repository
(github.com/pg2455/lookback-learn2branch), adapted to the gyozas decision
loop the same way the Gasse port is. The Lookback protocol differs from
Gasse's explore-then-strong-branch in two ways:

- the **root node is always** queried to the expert;
- a node whose parent was queried is **also** queried (``child_collection``)
  — these are the parent-child pairs the PAT loss trains on (the paper's
  observation: a child's best branch is often the parent's second-best).

Elsewhere a coin with probability ``node_record_prob`` (the official 0.05)
decides; the exploration brancher is pseudocosts, as in Gasse's.

Samples additionally record the SCIP node numbers (``node_number``,
``parent_number``), the ``child_collection`` flag, and the node depth, so
training can pair each child with its collected parent.

Deviations from the official script, mirroring the Gasse port's adaptation:

- the official ``randomization/permutevars`` randomization (inherited from
  Hybrid-learn2branch) is not set — solver randomization is the harness's
  ``env.seed`` (R2.4);
- the Khalil variable features appended by the official collector serve
  their hybrid model only; the GCNN consumes the bipartite graph alone;
- when no scorer returns a finite score for any candidate, the decision
  falls back to SCIP's default rule via the SKIP action instead of
  argmaxing over NaNs.
"""

from dataclasses import dataclass

import numpy as np

from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
from gyozas.environment import Environment
from gyozas.observations.pseudo_cost import Pseudocosts
from gyozas.observations.strong_branching_scores import StrongBranchingScores

from l2b_bench.methods.gasse.collect import BranchingSample, _argmax_finite

# The official collection protocol (see init_scip_params in the official
# repository's utilities.py — same SCIP 6.0.1 names as Gasse's, minus the
# variable permutation, see the module docstring).
LOOKBACK_SCIP_PARAMS = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "timing/clocktype": 2,
}

# The official node_record_prob (generate_dataset.py), not the class default.
LOOKBACK_NODE_RECORD_PROB = 0.05


@dataclass
class LookbackSample(BranchingSample):
    """One collected expert decision, with the node's place in the tree.

    ``node_number``/``parent_number`` are SCIP node numbers (the root's
    parent is 0); ``child_collection`` marks the lookback nodes — nodes
    collected *because* their parent was collected.
    """

    node_number: int = 0
    parent_number: int = 0
    child_collection: bool = False
    depth: int = 0


def collect_episode(
    env: Environment,
    rng: np.random.Generator,
    node_record_prob: float = LOOKBACK_NODE_RECORD_PROB,
) -> tuple[list[LookbackSample], int, str]:
    """Run one episode under the Lookback protocol; return its samples.

    The environment must be configured with the ``NodeBipartite`` observation
    and the lookback SCIP parameters.
    """
    pseudocosts = Pseudocosts()
    strong_branching = StrongBranchingScores()
    samples: list[LookbackSample] = []
    n_nodes = 0
    parent_collected: set[int] = set()

    obs, action_set, _reward, done, info = env.reset()
    pseudocosts.reset(env.model)
    strong_branching.reset(env.model)
    status = info.get("status")

    while not done:
        n_nodes += 1
        # Branching candidates only — the SKIP action never enters a sample.
        candidates = np.asarray(action_set)
        candidates = candidates[candidates >= 0]

        node = env.model.getCurrentNode()
        if node is None:
            node_number, parent_number, depth = 0, 0, 0
        else:
            node_number = node.getNumber()
            parent = node.getParent()
            parent_number = parent.getNumber() if parent is not None else 0
            depth = node.getDepth()
        is_root = env.model.getNNodes() == 1

        child_collection = (not is_root) and parent_number in parent_collected
        query_expert = is_root or child_collection or rng.random() < node_record_prob

        if query_expert:
            scores = strong_branching.extract(env.model, done=False)
        else:
            scores = pseudocosts.extract(env.model, done=False)

        action = None
        if scores is not None and len(scores):
            action = _argmax_finite(scores, candidates)

        if query_expert and action is not None and len(scores):
            samples.append(
                LookbackSample(
                    observation=obs,
                    action_set=candidates.copy(),
                    action=action,
                    scores=np.asarray(scores, dtype=np.float64).copy(),
                    node_number=node_number,
                    parent_number=parent_number,
                    child_collection=child_collection,
                    depth=depth,
                )
            )
            # The official rule: a node enters the parent set only when it was
            # queried on its own coin (or is the root) — lookback children do
            # not propagate the collection further.
            if not child_collection:
                parent_collected.add(node_number)

        step_action = action if action is not None else int(ExtraBranchingActions.SKIP)
        obs, action_set, _reward, done, info = env.step(step_action)
        status = info.get("status")

    return samples, n_nodes, str(status)


def collect(
    instance_generator,
    n_episodes: int,
    seeds: list[int],
    node_record_prob: float = LOOKBACK_NODE_RECORD_PROB,
    time_limit: float | None = None,
    on_episode=None,
):
    """Collect Lookback expert samples over ``n_episodes`` episodes.

    Same contract as :func:`l2b_bench.methods.gasse.collect.collect`: one
    base seed per episode (the solver seed of the episode's SeedTriplet,
    R2.4), an optional per-episode SCIP time limit, and an optional
    ``on_episode(done, status, n_nodes, n_samples)`` progress callback.
    """
    from gyozas.informations.solve_status import SolveStatus
    from gyozas.observations import NodeBipartite
    from gyozas.seeds import SeedTriplet

    from l2b_bench.methods.gasse.collect import CollectionResult

    if len(seeds) < n_episodes:
        raise ValueError(f"{n_episodes} episodes need {n_episodes} seeds, got {len(seeds)}")

    scip_params = dict(LOOKBACK_SCIP_PARAMS)
    if time_limit is not None:
        scip_params["limits/time"] = time_limit
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
            samples, n_nodes, status = collect_episode(env, rng, node_record_prob)
            result.samples.extend(samples)
            result.n_episodes += 1
            result.n_nodes += n_nodes
            result.statuses.append(status)
            if on_episode is not None:
                on_episode(result.n_episodes, status, n_nodes, len(samples))
    finally:
        env.close()
    return result


def pair_parent_children(samples: list[LookbackSample]) -> dict[int, int]:
    """Map each child sample's index to its collected parent's index.

    A pair exists when the child's ``parent_number`` matches a recorded
    sample's ``node_number`` — the parent-child pairs the PAT loss consumes.
    """
    by_node = {sample.node_number: i for i, sample in enumerate(samples)}
    return {
        i: by_node[sample.parent_number]
        for i, sample in enumerate(samples)
        if sample.parent_number in by_node and by_node[sample.parent_number] != i
    }


def lookback_condition(parent: LookbackSample, child: LookbackSample) -> bool:
    """The paper's lookback indicator (Eq. 4): the child's expert choice is
    among the parent's second-best strong-branching decisions."""
    scores = np.asarray(parent.scores, dtype=np.float64).copy()
    scores[parent.action] = -np.inf
    finite = np.isfinite(scores)
    if not finite.any():
        return False
    second_best = scores[finite].max()
    return bool(child.action in set(np.where(scores == second_best)[0].tolist()))
