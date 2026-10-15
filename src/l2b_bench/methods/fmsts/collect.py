"""FMSTS 2020 (Etheve, Ales, Bissuel, Juan & Kedad-Sidhoum, arXiv 2005.10026)
— the bespoke observation and the episode collection with observed Q-values.

Ported from the paper (no official repository):

- **Observable Q-values** — the MDP is deterministic, so the temporal-difference
  loss reduces to the plain MSE against the observed Q-values (Eq 3). The
  value function is the *subtree size* (Prop 2: minimising every subtree is
  optimal for the whole tree under DFS; Prop 1 shows it fails under other node
  selections), and Q(s, a) = 1 + V(child0) + V(child1) is observed by counting
  the nodes once the tree is fully expanded. Only solved episodes carry
  targets — a time-limited tree is not fully expanded, so those episodes are
  rejected.
- **The observation** — static: a PCA of the concatenated (A, b, c) instance
  vector (the fitted PCA is part of the checkpoint; the component count is not
  given in the paper — a pilot choice); dynamic: the node depth, the primal LP
  solution's distance to its bounds, and the 3|J| one-hot branching state (set
  to 0 / set to 1 / free along the ancestry, read from the local domains, so
  propagation fixings count — a documented interpretation).
- **Canonical padding** — the paper's problems are fixed-size (A = J with a
  flat architecture), but the benchmark's evaluation family is bigger than the
  training one, so the instance vector is zero-padded to a canonical
  (rows, cols) chosen at PCA-fit time: the training instances are padded up to
  the evaluation shape and nothing is truncated.
- **The full tree** — the subtree sizes need every node, not just the visited
  ones: :class:`EpisodeTree` snapshots the open nodes after every step (as
  SORREL's tracker does) *and* harvests the dynamics' node-feasibility events,
  which catch the nodes processed between two decisions — a dive-pruned child
  is closed before any open-nodes snapshot can see it. The last decision's
  children, created by the final branching after which no snapshot runs, are
  unvisited leaves by construction: its subtree is 3 nodes.

SCIP protocol: CPLEX 12.7.1 under DFS with all presolving and cutting off in
the paper — the harness runs SCIP with presolve rounds, cut rounds, and
restarts off and DFS node selection, training-side only like every RL port;
the benchmark evaluation keeps the runner's default solver profile.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from pyscipopt import Model

#: The paper's protocol: all presolving and cutting off, DFS — expressed with
#: the harness's parameter dictionary (heuristics stay at their defaults; the
#: paper only rules out presolving and cutting). Cutting needs both limits
#: zeroed: ``maxrounds`` covers the nodes, ``maxroundsroot`` the root's own
#: separation rounds.
FMSTS_SCIP_PARAMS: dict[str, Any] = {
    "presolving/maxrounds": 0,
    "separating/maxrounds": 0,
    "separating/maxroundsroot": 0,
    "estimation/restarts/restartpolicy": "n",
    "nodeselection/dfs/stdpriority": 300000,
    "nodeselection/dfs/memsavepriority": 300000,
}

#: The dynamic features per variable: depth (broadcast), the primal LP
#: solution's distance to its bounds, and the 3-slot one-hot branching state.
N_DYNAMIC_FEATURES = 5


def instance_vector(model: Model, canonical: tuple[int, int]) -> np.ndarray:
    """The concatenated (A, b, c) instance vector, zero-padded to the
    canonical (rows, cols) — a row's side is its rhs, or its lhs when the rhs
    is infinite. Read from the ORIGINAL problem (original variables and
    original linear constraints), so the vector is the same instance
    fingerprint under any protocol — the training regime reads it with
    presolving and cutting off, the benchmark evaluation with the default
    solver, and both yield the same static vector. Rows follow the
    constraint order, columns and the c vector the original variable order."""
    n_rows_canon, n_cols_canon = canonical
    variables = model.getVars(transformed=False)
    index = {variable.name: j for j, variable in enumerate(variables)}
    n_cols = len(variables)
    objective = np.array(
        [variable.getObj() for variable in variables], dtype=np.float64
    )

    conss = []
    for cons in model.getConss(transformed=False):
        if not cons.isLinear():
            continue
        rhs = model.getRhs(cons)
        side = rhs if not model.isInfinity(abs(rhs)) else model.getLhs(cons)
        conss.append((model.getValsLinear(cons), side))
    n_rows = len(conss)

    matrix = np.zeros((n_rows, n_cols))
    sides = np.zeros(n_rows)
    for i, (coefficients, side) in enumerate(conss):
        sides[i] = side
        for name, value in coefficients.items():
            column = index.get(name)
            if column is not None:
                matrix[i, column] = value

    # Zero-pad up to the canonical shape (bigger instances would be
    # truncated — the canonical shape is chosen to cover the evaluation
    # family, so this never happens in the benchmark).
    padded_matrix = np.zeros((n_rows_canon, n_cols_canon))
    padded_matrix[: min(n_rows, n_rows_canon), : min(n_cols, n_cols_canon)] = matrix[
        :n_rows_canon, :n_cols_canon
    ]
    padded_sides = np.zeros(n_rows_canon)
    padded_sides[: min(n_rows, n_rows_canon)] = sides[:n_rows_canon]
    padded_objective = np.zeros(n_cols_canon)
    padded_objective[: min(n_cols, n_cols_canon)] = objective[:n_cols_canon]
    return np.concatenate([padded_matrix.ravel(), padded_sides, padded_objective])


@dataclass
class FMSTSNodeObservation:
    """One node's observation: the static PCA vector (per instance) and the
    per-variable dynamic features, one row per LP column position."""

    static: np.ndarray  # [n_pca]
    dynamic: np.ndarray  # [n_vars, N_DYNAMIC_FEATURES]


class FMSTSObservation:
    """The bespoke observation, carrying the fitted PCA: the static vector is
    ``(instance_vector - mean) @ components.T``; the dynamic features are read
    from the current node's local domains and LP solution."""

    def __init__(
        self, mean: np.ndarray, components: np.ndarray, canonical: tuple[int, int]
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.components = np.asarray(components, dtype=np.float64)
        self.canonical = canonical
        self._static: np.ndarray | None = None

    def reset(self, _model: Model) -> None:
        self._static = None

    def extract(self, model: Model, done: bool) -> FMSTSNodeObservation:
        if self._static is None:
            vector = instance_vector(model, self.canonical)
            self._static = (vector - self.mean) @ self.components.T
        dynamic = dynamic_features(model)
        # Zero-pad the variable axis to the canonical width, like the static
        # vector — so batches stack across instances of different sizes and
        # the network always scores the canonical variable layout.
        n_cols = self.canonical[1]
        if dynamic.shape[0] < n_cols:
            padding = np.zeros((n_cols - dynamic.shape[0], N_DYNAMIC_FEATURES))
            dynamic = np.concatenate([dynamic, padding])
        return FMSTSNodeObservation(static=self._static, dynamic=dynamic)


class RawInstanceObservation:
    """The PCA-fit pass's observation: the raw padded instance vector,
    extracted once per instance at the root."""

    def __init__(self, canonical: tuple[int, int]) -> None:
        self.canonical = canonical

    def reset(self, _model: Model) -> None:
        pass

    def extract(self, model: Model, done: bool) -> np.ndarray:
        return instance_vector(model, self.canonical)


def dynamic_features(model: Model) -> np.ndarray:
    """The per-variable dynamic features, one row per LP column position: the
    node depth, the primal LP solution's distance to its (local) bounds, and
    the one-hot branching state — set to 1 / set to 0 / free along the
    ancestry, read from the local domains (propagation fixings included)."""
    variables = model.getVars(transformed=True)
    depth = float(model.getDepth())
    n_vars = 0
    rows = []
    for variable in variables:
        column = variable.getCol()
        position = column.getLPPos() if column is not None else -1
        if position < 0:
            continue
        n_vars = max(n_vars, position + 1)
        lb, ub = column.getLb(), column.getUb()
        solution = column.getPrimsol()
        if model.isInfinity(abs(lb)) and model.isInfinity(abs(ub)):
            distance = 0.0
        elif model.isInfinity(abs(lb)):
            distance = ub - solution
        elif model.isInfinity(abs(ub)):
            distance = solution - lb
        else:
            distance = min(solution - lb, ub - solution)
        if ub < 0.5:
            one_hot = (1.0, 0.0, 0.0)  # set to 0 along the ancestry
        elif lb > 0.5:
            one_hot = (0.0, 1.0, 0.0)  # set to 1 along the ancestry
        else:
            one_hot = (0.0, 0.0, 1.0)  # free
        rows.append((position, depth, distance, *one_hot))
    features = np.zeros((n_vars, N_DYNAMIC_FEATURES))
    for position, *values in rows:
        features[position] = values
    return features


class EpisodeTree:
    """Every node the solve exposes, so the subtree sizes count the whole
    tree — SORREL's :class:`TreeTracker` alone undercounts: a node processed
    and closed between two decisions (a dive-pruned child under DFS) never
    appears in an open-nodes snapshot.

    ``update`` runs after every step (the current node is the next decision,
    the open nodes its yet-unprocessed siblings and leaves); ``absorb``
    harvests, at the end of the episode, the dynamics' node-feasibility
    events — one entry per *processed* node, whatever snapshot missed it. The
    only nodes neither source can see are the last decision's children,
    created by the final branching (handled by :func:`subtree_sizes`).
    """

    def __init__(self) -> None:
        self.parent: dict[int, int | None] = {}  # node number -> parent number
        self.visited: list[int] = []  # visitation order, one entry per decision

    def update(self, model: Model) -> None:
        current = model.getCurrentNode()
        if current is not None:
            self._add(current)
            if not self.visited or self.visited[-1] != current.getNumber():
                self.visited.append(current.getNumber())
        open_leaves, open_children, open_siblings = model.getOpenNodes()
        for node in (*open_leaves, *open_children, *open_siblings):
            if node.getNumber() not in self.parent:
                self._add(node)

    def _add(self, node: Any) -> None:
        parent = node.getParent()
        self.parent[node.getNumber()] = (
            parent.getNumber() if parent is not None else None
        )

    def absorb(self, dynamics: BranchingDynamics) -> None:
        """Merge the processed nodes recorded by the harness's node events
        (feasible and infeasible leaves alike)."""
        for entries in (dynamics.infeasible_nodes, dynamics.feasible_nodes):
            for _node, number, _parent, parent_number, _lower_bound in entries:
                if parent_number is not None and number not in self.parent:
                    self.parent[number] = parent_number


def subtree_sizes(tree: EpisodeTree) -> list[int]:
    """V per visited decision, in visitation order: V(i) = 1 + the sum over
    the decision's children (V of a visited child, 1 per unvisited child — an
    unvisited child is a leaf: pruned, infeasible, or cut by the bound).
    Children are visited after their parents, so one reverse pass solves the
    recursion. The last decision's children never entered the tree: both are
    unvisited leaves by construction, so its subtree is 3 nodes."""
    position = {number: i for i, number in enumerate(tree.visited)}
    children: dict[int, list[int]] = {number: [] for number in tree.visited}
    for number, parent in tree.parent.items():
        if parent is not None:
            children.setdefault(parent, []).append(number)
    n = len(tree.visited)
    sizes = [0] * n
    for i in range(n - 1, -1, -1):
        ids = children[tree.visited[i]]
        if not ids and i == n - 1:
            sizes[i] = 3
        else:
            sizes[i] = 1 + sum(
                sizes[position[child]] if child in position else 1 for child in ids
            )
    return sizes


@dataclass
class FMSTSExperience:
    """One stored decision of a solved episode: the observation, the taken
    action (an LP column position, the rank within the action set), the
    observed Q-value (the decision's subtree size), and the Eq 5 instance
    weight 1/V(tree)."""

    observation: FMSTSNodeObservation
    action_set: np.ndarray
    action: int
    action_idx: int
    depth: int
    q_target: float
    instance_weight: float


def make_env(
    instances: Iterator[Model],
    observation: Any,
    time_limit: float | None = None,
    seed: int = 0,
) -> Environment:
    """One FMSTS environment over the bespoke observation, LP branching
    candidates, and the paper's no-presolving/no-cutting DFS protocol."""
    params = dict(FMSTS_SCIP_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    params["display/verblevel"] = 0
    env = Environment(
        instances,
        observation_function=observation,
        information_function=SolveStatus(),
        scip_params=params,
        dynamics=BranchingDynamics(),
    )
    env.dynamics.seed(seed)
    return env


def run_episode(
    env: Environment,
    select_action: Callable[[Any, np.ndarray], tuple[int, int]],
) -> tuple[list[FMSTSExperience] | None, int, str]:
    """One episode under ``select_action`` (returning the chosen variable and
    its rank in the action set). The tree is updated after every step while
    SCIP is solving; a rejected episode returns ``None`` — the solve did not
    terminate on its own (no fully-expanded tree, no observable Q-values),
    the visitation order disagrees with the decisions, or the solve re-focused
    a node it had already branched on (a double root)."""
    obs, action_set, _reward, done, info = env.reset()
    if done or obs is None:
        return None, 0, str(info.get("status"))
    tree = EpisodeTree()
    tree.update(env.model)

    observations: list[Any] = []
    action_sets: list[np.ndarray] = []
    actions: list[int] = []
    action_idxs: list[int] = []
    depths: list[int] = []
    while not done:
        candidates = np.asarray(action_set)
        action, action_idx = select_action(obs, candidates)
        observations.append(obs)
        action_sets.append(candidates.copy())
        actions.append(int(action))
        action_idxs.append(int(action_idx))
        depths.append(int(env.model.getDepth()))
        obs, action_set, _reward, done, info = env.step(int(action))
        if not done:
            tree.update(env.model)
    n_decisions = len(observations)
    status = str(info.get("status"))
    if not info.get("solved"):
        return None, n_decisions, status
    if len(tree.visited) != n_decisions:
        return None, n_decisions, status
    if n_decisions > 1 and depths[0] == 0 and depths[1] == 0:
        return None, n_decisions, status
    tree.absorb(env.dynamics)
    sizes = subtree_sizes(tree)
    tree_size = sizes[0]
    experiences = [
        FMSTSExperience(
            observation=observation,
            action_set=action_set_,
            action=action,
            action_idx=action_idx,
            depth=depth,
            q_target=float(size),
            instance_weight=1.0 / tree_size,
        )
        for observation, action_set_, action, action_idx, depth, size in zip(
            observations, action_sets, actions, action_idxs, depths, sizes, strict=True
        )
    ]
    return experiences, n_decisions, status
