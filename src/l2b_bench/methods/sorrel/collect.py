"""SORREL 2024 (Feng et al., AAAI 2025) — VHB demonstration collection.

Faithful port of the official repository (github.com/Shengyu-Feng/SORREL,
``02_generate_dataset.py`` + ``common/{env,observation,reward,search_tree}.py``),
adapted to the harness's ``Environment``:

- **Demonstrator** — vanilla hybrid branching (VHB): at every node a coin with
  probability ``FSB_probability`` (0.05) queries full strong branching scores,
  otherwise Ecole-style pseudocosts (the official ``ExploreThenStrongBranch``).
  The action is the plain argmax of the scores over the node's *pseudo*
  branching candidates — no finite-score guard, exactly like the official
  ``action_set[scores[action_set].argmax()]``.
- **Tree MDP** — the official ``Children`` information function: a
  :class:`TreeTracker` (the official ``SearchTree``, reduced to what the tree
  MDP needs) records the visitation order, every node's parent and lower
  bound; at the end of the episode each visited decision exposes its children
  — their position among the visited decisions (``None`` when unvisited) and
  their reward, the child's lower bound minus the parent's.
- **Episode filtering** — episodes shorter than ``sampling_threshold``
  decisions are dropped entirely; longer ones keep each decision with
  probability ``log(len, 1.01) / len`` (≥ 1 past ~100 decisions, i.e. all of
  them at the official scale). Episodes are length-limited
  (``length_limit`` = 200 on set covering).

SCIP protocol (official ``BranchingDynamics.reset_dynamics`` + ``create_env``):
heuristics off, restarts off (``restartpolicy 'n'``), no cut separation
rounds, no presolving restarts, pseudo branching candidates (the harness's
``BranchingDynamics(pseudo_candidates=True)``) — training-side only, like
every other RL method here; the benchmark evaluation keeps the runner's
default solver profile.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import NodeBipartite
from gyozas.observations.pseudo_cost import Pseudocosts
from gyozas.observations.strong_branching_scores import StrongBranchingScores
from pyscipopt import Model

from ..tgppo.train import _HEURISTICS_OFF

# The official demonstrator (02_generate_dataset.py --FSB_probability).
FSB_PROBABILITY = 0.05

# The official set-covering episode knobs (scripts/generate_dataset.sh).
LENGTH_LIMIT = 200
SAMPLING_THRESHOLD = 100

#: The official SCIP protocol: no cut rounds, no restarts of either kind, all
#: primal heuristics off (the official calls ``setHeuristics(OFF)`` in the
#: dynamics' reset — the freq=-1 trick is its parameter-dict equivalent, as in
#: the tgppo and mamba sandbox protocols).
SORREL_SCIP_PARAMS: dict[str, Any] = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "estimation/restarts/restartpolicy": "n",
    **{f"heuristics/{name}/freq": -1 for name in _HEURISTICS_OFF},
}


@dataclass
class SorrelSample:
    """One tree-MDP transition of the VHB demonstrator.

    The two children carry the observation and action set of the visited
    child (or ``None`` when unvisited — the child's reward still counts, with
    no bootstrap), and the reward ``child lower bound - parent lower bound``.
    ``returns`` and ``log_prob`` are filled by the online stage's rollouts.
    """

    observation: Any  # the NodeBipartite observation at the decision
    action_set: np.ndarray  # the node's pseudo branching candidates
    action: int  # the chosen variable's index
    candidate_scores: np.ndarray  # the demonstrator's scores over the candidates
    is_fsb: bool  # whether the decision's scores were full strong branching
    child1_obs: Any
    child1_action_set: np.ndarray | None
    reward1: float
    child2_obs: Any
    child2_action_set: np.ndarray | None
    reward2: float
    returns: float | None = None
    log_prob: float | None = None


class TreeTracker:
    """The official ``SearchTree``, reduced to the tree MDP's needs.

    Records, per update: the current focus node (visitation order — one entry
    per branching decision) and every node seen (open leaves/children/siblings
    included) with its parent and the lower bound read at first discovery.
    Nodes pruned, infeasible, or cut by the bound are never seen — that is
    why ``getNNodes()`` can exceed the tracked tree, exactly as the official
    docstring warns. Like the official ``update_tree``, the tracker is *not*
    updated on the episode's final extraction: the last decision's children
    never enter the tree.
    """

    def __init__(self, model: Model) -> None:
        self._nodes: dict[int, tuple[int | None, float]] = {}  # id -> (parent, lb)
        self._visited: set[int] = set()
        self.visited: list[int] = []  # visitation order, one entry per decision
        self.update(model)

    def update(self, model: Model) -> None:
        current = model.getCurrentNode()
        if current is not None:
            node_id = current.getNumber()
            if node_id not in self._nodes:
                self._add(current)
            if node_id not in self._visited:
                self._visited.add(node_id)
                self.visited.append(node_id)
        # Open nodes are added in the official's discovery order
        # (leaves, then children, then siblings) — the order the children
        # lists below are built from.
        open_leaves, open_children, open_siblings = model.getOpenNodes()
        for node in (*open_leaves, *open_children, *open_siblings):
            if node.getNumber() not in self._nodes:
                self._add(node)

    def _add(self, node: Any) -> None:
        parent = node.getParent()
        self._nodes[node.getNumber()] = (
            parent.getNumber() if parent is not None else None,
            node.getLowerbound(),
        )

    def children_of(self) -> list[tuple[list[int | None], list[float]]]:
        """Per visited decision (visitation order): its children's positions
        among the visited decisions (``None`` when unvisited) and rewards."""
        position = {node_id: i for i, node_id in enumerate(self.visited)}
        children: dict[int, list[int]] = {}
        for node_id, (parent, _lb) in self._nodes.items():
            if parent is not None:
                children.setdefault(parent, []).append(node_id)
        out = []
        for node_id in self.visited:
            parent_lb = self._nodes[node_id][1]
            ids = children.get(node_id, [])
            out.append(
                (
                    [position.get(child) for child in ids],
                    [self._nodes[child][1] - parent_lb for child in ids],
                )
            )
        return out


def tree_returns(
    children: list[tuple[list[int | None], list[float]]],
    gamma: float,
    kappa: float,
) -> dict[int, float]:
    """The official ``get_return``: the kappa-weighted return of every visited
    decision, accumulated top-down from the root (position 0).

    A decision with no tracked children returns 0; an unvisited child
    contributes its reward with no bootstrap; among two children, the
    smaller-reward one carries weight ``kappa``.
    """
    returns: dict[int, float] = {}

    def visit(node: int) -> None:
        positions, rewards = children[node]
        if len(positions) == 0:
            returns[node] = 0.0
        elif len(positions) == 1:
            child = positions[0]
            if child is None:
                returns[node] = rewards[0]
            else:
                visit(child)
                returns[node] = rewards[0] + gamma * returns[child]
        else:
            left, right = positions[0], positions[1]
            if left is None:
                left_return = 0.0
            else:
                visit(left)
                left_return = returns[left]
            if right is None:
                right_return = 0.0
            else:
                visit(right)
                right_return = returns[right]
            coeff = kappa if rewards[0] < rewards[1] else 1 - kappa
            returns[node] = coeff * (rewards[0] + gamma * left_return) + (
                1 - coeff
            ) * (rewards[1] + gamma * right_return)

    if children:
        visit(0)
    return returns


def make_env(
    instances: Iterator[Model],
    time_limit: float | None = None,
    seed: int = 0,
) -> Environment:
    """One SORREL environment: NodeBipartite observation, pseudo branching
    candidates, the official no-heuristics/no-restarts SCIP protocol."""
    params = dict(SORREL_SCIP_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    params["display/verblevel"] = 0
    env = Environment(
        instances,
        observation_function=NodeBipartite(),
        information_function=SolveStatus(),
        scip_params=params,
        dynamics=BranchingDynamics(pseudo_candidates=True),
    )
    env.dynamics.seed(seed)
    return env


def collect_episode(
    env: Environment,
    rng: np.random.Generator,
    fsb_probability: float = FSB_PROBABILITY,
    length_limit: float = LENGTH_LIMIT,
    sampling_threshold: int = SAMPLING_THRESHOLD,
) -> tuple[list[SorrelSample], int, str]:
    """One VHB episode over the tree MDP; returns its kept samples, the
    episode's decision count, and the solve status."""
    pseudocosts = Pseudocosts()
    strong_branching = StrongBranchingScores()
    obs, action_set, _reward, done, info = env.reset()
    pseudocosts.reset(env.model)
    strong_branching.reset(env.model)
    status = info.get("status")
    tracker = None if done else TreeTracker(env.model)

    all_observations: list[Any] = []
    all_action_sets: list[np.ndarray] = []
    all_actions: list[int] = []
    all_scores: list[np.ndarray] = []
    is_fsbs: list[bool] = []
    n_iter = 0
    while not done:
        candidates = np.asarray(action_set)
        query_expert = rng.random() < fsb_probability
        scores = (
            strong_branching.extract(env.model, done=False)
            if query_expert
            else pseudocosts.extract(env.model, done=False)
        )
        # The official argmax: no finite-score guard (a NaN score wins, as in
        # torch's argmax over the official score tensors).
        cand_scores = np.asarray(scores, dtype=np.float64)[candidates]
        action_idx = int(np.argmax(cand_scores))
        action = int(candidates[action_idx])
        all_observations.append(obs)
        all_action_sets.append(candidates.copy())
        all_actions.append(action)
        all_scores.append(cand_scores.copy())
        is_fsbs.append(query_expert)
        obs, action_set, _reward, done, info = env.step(action)
        n_iter += 1
        if n_iter >= length_limit:
            # The official LengthLimitEnvironment cuts the episode here, and
            # its Children extraction at done=True does not update the tree —
            # the length-limited decision's children never enter it either.
            done = True
        elif not done:
            # The official Children extraction does not update the tree on the
            # final step — the last decision's children never enter it.
            tracker.update(env.model)
        status = info.get("status")
    if tracker is None:
        return [], n_iter, str(status)

    samples: list[SorrelSample] = []
    if n_iter >= sampling_threshold:
        sample_rate = math.log(n_iter, 1.01) / n_iter
        children = tracker.children_of()
        for i, (positions, rewards) in enumerate(children):
            if rng.random() >= sample_rate:
                continue
            # The official next_observation: the two children in discovery
            # order (unvisited -> (None, None, reward)), padded to two.
            child_obs = []
            for child, reward in zip(positions, rewards, strict=True):
                if child is None:
                    child_obs.append((None, None, reward))
                else:
                    child_obs.append((all_observations[child], all_action_sets[child], reward))
            child_obs += [(None, None, 0.0)] * (2 - len(child_obs))
            samples.append(
                SorrelSample(
                    observation=all_observations[i],
                    action_set=all_action_sets[i],
                    action=all_actions[i],
                    candidate_scores=all_scores[i],
                    is_fsb=is_fsbs[i],
                    child1_obs=child_obs[0][0],
                    child1_action_set=child_obs[0][1],
                    reward1=float(child_obs[0][2]),
                    child2_obs=child_obs[1][0],
                    child2_action_set=child_obs[1][1],
                    reward2=float(child_obs[1][2]),
                )
            )
    return samples, n_iter, str(status)
