"""Seq-BnB 2026 — expert data collection over the joint regime.

Paper: "Seq-BnB: Sequence Model for Branch-and-Bound Node Selection and
Branching" (arXiv 2610.14835), Liu, Liu, Yan, Shi, Zhao. No official
repository — the paper promises the code upon publication — so this port is
reconstructed from the paper's description, with every reconstruction choice
documented here.

The method controls BOTH solver decisions through one policy (the harness's
``BranchingNodeSelectionDynamics``): at each pause the collector builds the
typed token sequence the paper conditions on —

- **node tokens** (one per known open node, 8 features): lower bound
  ℓ(u), estimate c(u), depth d(u), visited / fathomed / incumbent flags,
  creation order s(u), and the current gap g_t;
- **action tokens** (one per past decision): [g_t, |O_t|, t] plus the chosen
  node's d/ℓ/c (node selection) — the graph-derived slots of the paper's
  action tokens are zeroed in the stored history (a documented deviation:
  they would need every past decision's LP graph).

Expert labels: strong branching (argmax over the LP candidates) for the
branching decisions; the paper's offline-oracle node selection for the
node-selection decisions — among the open nodes compatible with a reference
optimum x* (solved offline per instance), the deepest first.

Empirically pinned deviations (probes under this exact regime):

- ``getBipartiteGraphRepresentation`` SEGFAULTS at node-selection pauses —
  from the main thread and from inside the nodesel plugin, before and after
  the first branching. The current-LP bipartite graph is therefore extracted
  at branching pauses only, and every node-selection decision reuses the
  graph cached from the last branching (zero-pooled before the first
  branching). Node getters, the ancestor-branchings walk, and ``getGap`` are
  all safe at node-selection pauses.
- The token table is refreshed at node-selection decisions only (the open
  set is exposed there); at a branching decision the sequence is the one
  from the last node selection.
- The paper's window (L_MAX) keeps the most recent tokens of the sequence;
  here the node-token table and the action history are each windowed to the
  L_MAX most recent entries (the open nodes are always retained). An
  unbounded action history made the encoder's attention quadratic in the
  episode length — the first full campaign OOM'd on a 13 GiB attention
  tensor from a ~3,600-decision episode.
- ``visited`` counts the node numbers this episode has already selected;
  ``fathomed`` / ``incumbent`` come from the dynamics' NODEINFEASIBLE /
  NODEFEASIBLE bookkeeping (a processed node whose LP was infeasible by
  bounds / a processed node that yielded a feasible solution — the closest
  SCIP-event reading of the paper's flags).

SCIP parameters (the paper's protocol, on SCIP 10): cuts at the root only
(``separating/maxrounds = 0``), no restarts
(``estimation/restarts/restartpolicy = "n"``).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gyozas.dynamics import (
    BranchingDynamics,
    BranchingNodeSelectionDynamics,
    Decision,
    DecisionKind,
)
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import NodeBipartite
from gyozas.observations.strong_branching_scores import StrongBranchingScores
from gyozas.observations.structs import BipartiteGraph
from numpy.typing import NDArray
from pyscipopt import Model

# The paper's protocol (verified on SCIP 10): cut generation at the root
# only, no restarts.
SEQBNB_SCIP_PARAMS = {
    "separating/maxrounds": 0,
    "estimation/restarts/restartpolicy": "n",
}

# The token window: the paper's L_max — the most recent tokens are kept,
# and the currently open nodes' tokens are always retained.
L_MAX = 500

# Node-token feature count: [l, c, d, visited, fathomed, incumbent, s, g_t].
NODE_TOKEN_DIM = 8

# Action-token history scalar count: [g_t, |O_t|, t, d, l, c] — the
# graph-derived slots are zeroed (documented deviation).
ACTION_TOKEN_DIM = 6

# Before the first incumbent, SCIP's gap comes back as its 1e20 convention,
# and unprocessed nodes' bounds/estimates can be ±1e20 (SCIP's infinities) —
# all clamped to a finite ceiling so the tokens stay on a trainable scale (a
# pilot choice; the paper does not specify its feature preprocessing).
GAP_CEILING = 1e6

# Bound types from pyscipopt's getParentBranchings (verified: 0 = lower,
# 1 = upper).
_BOUNDTYPE_LOWER = 0
_BOUNDTYPE_UPPER = 1


@dataclass
class SeqBnBState:
    """What the policy (or the expert) sees at one decision of the episode.

    ``tokens`` is the windowed node-token table and ``token_ids`` the node
    number of each row, so ``open_ids`` / ``candidates`` index into both.
    """

    kind: DecisionKind
    tokens: NDArray[np.float64]
    token_ids: NDArray[np.int64]
    open_ids: NDArray[np.int64] | None  # node numbers (node selection)
    candidates: NDArray[np.int64] | None  # LP column positions (branching)
    graph: BipartiteGraph | None  # last-branching cache (None before root branching)
    history: NDArray[np.float64]  # (t, 6) past action tokens (graph slots zeroed)
    history_kinds: NDArray[np.int64]  # (t,) DecisionKind of each past decision
    t: int
    n_open: int
    gap: float


@dataclass
class SeqBnBSample:
    """One collected decision: the sequence, the action, and the expert's."""

    kind: DecisionKind
    tokens: NDArray[np.float64]
    token_ids: NDArray[np.int64]
    open_ids: NDArray[np.int64] | None
    candidates: NDArray[np.int64] | None
    history: NDArray[np.float64]  # (t, 6) past action tokens (graph slots zeroed)
    history_kinds: NDArray[np.int64]  # (t,) DecisionKind of each past decision
    action: int
    expert_action: int
    t: int
    n_open: int
    gap: float
    graph: BipartiteGraph | None = None


@dataclass
class CollectionResult:
    """Everything a collection campaign produces."""

    samples: list[SeqBnBSample] = field(default_factory=list)
    n_episodes: int = 0
    n_nodes: int = 0
    statuses: list[str] = field(default_factory=list)


# -- node tokens ------------------------------------------------------------


def node_token(
    node, visited: set[int], fathomed: set[int], incumbent: set[int], gap: float
) -> np.ndarray:
    """The 8-feature node token: [l, c, d, visited, fathomed, incumbent, s, g_t]."""

    def clamp(value: float) -> float:
        # SCIP's infinities (±1e20) on unprocessed nodes stay finite.
        return float(np.clip(value, -GAP_CEILING, GAP_CEILING))

    return np.array(
        [
            clamp(node.getLowerbound()),
            clamp(node.getEstimate()),
            node.getDepth(),
            1.0 if node.getNumber() in visited else 0.0,
            1.0 if node.getNumber() in fathomed else 0.0,
            1.0 if node.getNumber() in incumbent else 0.0,
            node.getNumber(),
            clamp(gap),
        ],
        dtype=np.float64,
    )


def _window(
    token_table: dict[int, np.ndarray], open_ids: NDArray[np.int64], order: list[int]
):
    """The paper's window: the L_MAX most recently created tokens, with the
    currently open nodes' tokens always retained."""
    keep = {int(n) for n in open_ids}
    recent = [n for n in order if n in token_table]
    for n in reversed(recent):
        if len(keep) >= L_MAX:
            break
        keep.add(n)
    ids = np.array(sorted(keep), dtype=np.int64)
    if not len(ids):  # no node has a token yet (the BR-only regime never makes one)
        return np.zeros((0, NODE_TOKEN_DIM)), ids
    tokens = np.stack([token_table[int(n)] for n in ids])
    return tokens, ids


# -- the node-selection oracle ----------------------------------------------


def ancestor_bounds(node) -> list[tuple[str, float, int]]:
    """The branching bounds on the path root -> node: (variable name, value,
    bound type) triples, accumulated over the parent chain (probe-verified
    safe at node-selection pauses)."""
    bounds = []
    current = node
    while current is not None:
        branching = current.getParentBranchings()
        if branching is not None:
            vars_, vals, btypes = branching
            for var, val, btype in zip(vars_, vals, btypes, strict=True):
                bounds.append((var.name, float(val), int(btype)))
        current = current.getParent()
    return bounds


def compatible_with(node, x_star: dict[str, float], tolerance: float = 1e-9) -> bool:
    """Whether the reference optimum x* survives every bound on the node's
    path (the paper's compatibility): a lower bound (btype 0) needs
    x*_j >= v, an upper bound (btype 1) needs x*_j <= v. A bound on a
    variable absent from x* (presolve renaming) counts as incompatible."""
    for name, value, btype in ancestor_bounds(node):
        reference = x_star.get(name)
        if reference is None:
            return False
        if btype == _BOUNDTYPE_LOWER and reference < value - tolerance:
            return False
        if btype == _BOUNDTYPE_UPPER and reference > value + tolerance:
            return False
    return True


def oracle_node_selection(nodes, x_star: dict[str, float]) -> int:
    """The paper's node-selection oracle: among the open nodes compatible
    with x*, the deepest (ties by lower bound, estimate, creation order);
    when none is compatible, the best by (lower bound, estimate, deepest,
    order)."""
    compatible = [node for node in nodes if compatible_with(node, x_star)]
    if compatible:
        return min(
            compatible,
            key=lambda n: (
                -n.getDepth(),
                n.getLowerbound(),
                n.getEstimate(),
                n.getNumber(),
            ),
        ).getNumber()
    return min(
        nodes,
        key=lambda n: (
            n.getLowerbound(),
            n.getEstimate(),
            -n.getDepth(),
            n.getNumber(),
        ),
    ).getNumber()


# -- the offline reference optimum ------------------------------------------


def solve_reference(path: str | Path) -> dict[str, float]:
    """Solve an instance offline (default settings) for its reference
    optimum x*, keyed by original variable name — the compatibility oracle's
    ground truth."""
    model = Model()
    model.readProblem(filename=str(path))
    model.hideOutput()
    model.optimize()
    if model.getNSols() == 0:
        raise RuntimeError(f"reference solve of {path} found no solution")
    solution = model.getBestSol()
    return {
        var.name: round(float(model.getSolVal(solution, var)))
        for var in model.getVars(transformed=False)
    }


class ReferenceCache:
    """Memoizes the per-instance x* over a collection campaign."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, float]] = {}

    def get(self, path: str | Path) -> dict[str, float]:
        key = str(Path(path))
        if key not in self._cache:
            self._cache[key] = solve_reference(key)
        return self._cache[key]


# -- the episode -------------------------------------------------------------


def _argmax_finite(scores: np.ndarray, candidates: NDArray[np.int64]) -> int | None:
    """Best candidate by score; None when no candidate has a finite score."""
    best, best_score = None, -np.inf
    for action in candidates:
        score = scores[int(action)]
        if np.isfinite(score) and score > best_score:
            best, best_score = int(action), score
    return best


class EpisodeTracker:
    """The per-episode token bookkeeping, shared by the collector and the
    benchmark policy: the node-token table (refreshed at node-selection
    decisions, where the open set is exposed), the action-token history,
    and the last-branching graph cache. The collector adds the expert
    labels on top; a deployed policy uses the same tracker to build its
    inputs, so training and inference see identical states.
    """

    def __init__(self) -> None:
        self.token_table: dict[int, np.ndarray] = {}
        self.order: list[int] = []  # node numbers, first-seen order
        self.visited: set[int] = set()
        self.history: list[np.ndarray] = []  # past action tokens
        self.history_kinds: list[int] = []  # DecisionKind of each past decision
        self.cached_graph: BipartiteGraph | None = None
        self._bipartite: NodeBipartite | None = None
        self._model = None

    def reset(self, model) -> None:
        """Bind to a new episode's model (the graph extraction needs it)."""
        self.token_table = {}
        self.order = []
        self.visited = set()
        self.history = []
        self.history_kinds = []
        self.cached_graph = None
        self._bipartite = NodeBipartite()
        self._bipartite.reset(model)
        self._model = model

    def _snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        # The paper's window keeps the L_MAX most recent tokens of the
        # sequence — the most recent actions carry the episode's context.
        # An unbounded history would make the encoder's attention quadratic
        # in the episode length (a deep search → thousands of action tokens
        # → the campaign OOM'd at a 13 GiB attention tensor).
        recent = self.history[-L_MAX:]
        past = np.stack(recent) if recent else np.zeros((0, ACTION_TOKEN_DIM))
        return past, np.array(self.history_kinds[-L_MAX:], dtype=np.int64)

    def observe(self, env: Environment, decision: Decision) -> SeqBnBState:
        """Build the decision's state, refreshing whatever it exposes."""
        if self._model is not env.model:
            self.reset(env.model)
        dynamics: BranchingNodeSelectionDynamics = env.dynamics
        if decision.kind == DecisionKind.NODE_SELECTION:
            nodes = dynamics.nodesel_rule.nodes
            assert nodes is not None
            gap = min(float(env.model.getGap()), GAP_CEILING)
            fathomed = {entry[1] for entry in dynamics.infeasible_nodes}
            incumbent = {entry[1] for entry in dynamics.feasible_nodes}
            for node in nodes:
                number = int(node.getNumber())
                if number not in self.token_table:
                    self.order.append(number)
                self.token_table[number] = node_token(
                    node, self.visited, fathomed, incumbent, gap
                )
            open_ids = np.array(
                [int(node.getNumber()) for node in nodes], dtype=np.int64
            )
            tokens, token_ids = _window(self.token_table, open_ids, self.order)
        else:
            graph = self._bipartite.extract(env.model, done=False)
            if graph is not None and graph.variable_features is not None:
                self.cached_graph = graph
            open_ids = None
            tokens, token_ids = _window(
                self.token_table, np.array([], dtype=np.int64), self.order
            )
            gap = min(float(env.model.getGap()), GAP_CEILING)
        past, past_kinds = self._snapshot()
        return SeqBnBState(
            kind=decision.kind,
            tokens=tokens,
            token_ids=token_ids,
            open_ids=open_ids,
            candidates=None
            if open_ids is not None
            else np.asarray(decision.actions, dtype=np.int64),
            graph=self.cached_graph,
            history=past,
            history_kinds=past_kinds,
            t=len(self.history),
            n_open=len(open_ids) if open_ids is not None else len(self.token_table),
            gap=gap,
        )

    def record(self, state: SeqBnBState, action: int) -> None:
        """Fold the taken action into the history (after the decision)."""
        if state.kind == DecisionKind.NODE_SELECTION:
            self.visited.add(action)
            chosen = self.token_table[action][[2, 0, 1]].tolist()  # d, l, c
            self.history.append(
                np.array([state.gap, state.n_open, state.t] + chosen, dtype=np.float64)
            )
        else:
            # The branching action's graph-derived slots are zeroed in the
            # stored history (documented deviation).
            self.history.append(
                np.array(
                    [state.gap, state.n_open, state.t, 0.0, 0.0, 0.0], dtype=np.float64
                )
            )
        self.history_kinds.append(int(state.kind))


def _sample_from_state(
    state: SeqBnBState, action: int, expert: int | None
) -> SeqBnBSample:
    """The training sample of one decided state (labels optional: a deployed
    policy builds unlabeled samples through the same path)."""
    return SeqBnBSample(
        kind=state.kind,
        tokens=state.tokens.copy(),
        token_ids=state.token_ids.copy(),
        open_ids=None if state.open_ids is None else state.open_ids.copy(),
        candidates=None if state.candidates is None else state.candidates.copy(),
        history=state.history,
        history_kinds=state.history_kinds,
        action=action,
        expert_action=-1 if expert is None else int(expert),
        t=state.t,
        n_open=state.n_open,
        gap=state.gap,
        graph=state.graph,
    )


def run_episode(
    env: Environment,
    select_action: Callable[[SeqBnBState, Decision, int | None], int],
    reference: dict[str, float] | Callable[[Environment], dict[str, float]],
) -> tuple[list[SeqBnBSample], int, str]:
    """One joint-regime episode; every decision becomes a sample.

    ``select_action(state, decision, expert_action)`` decides — the current
    policy during DAgger rounds, ``lambda s, d, e: e`` at round 0 (the
    expert). ``expert_action`` is the label recorded on the sample: the
    strong-branching argmax at branching decisions (None when no candidate
    has a finite score), the x*-compatibility oracle at node-selection
    decisions. ``reference`` is either the episode's x* directly (a
    single-instance pool) or a callable resolved after the reset — the
    instance a pool generator yields is only known once it has been drawn.
    """
    strong_branching = StrongBranchingScores()
    tracker = EpisodeTracker()
    samples: list[SeqBnBSample] = []
    n_nodes = 0

    _obs, decision, _reward, done, info = env.reset()
    if callable(reference):
        reference = reference(env)
    strong_branching.reset(env.model)
    tracker.reset(env.model)
    status = info.get("status")

    while not done:
        assert decision is not None
        state = tracker.observe(env, decision)
        if decision.kind == DecisionKind.NODE_SELECTION:
            nodes = env.dynamics.nodesel_rule.nodes
            assert nodes is not None
            expert: int | None = oracle_node_selection(nodes, reference)
        else:
            n_nodes += 1
            scores = strong_branching.extract(env.model, done=False)
            expert = None
            if scores is not None and len(scores):
                expert = _argmax_finite(np.asarray(scores), state.candidates)
        action = int(select_action(state, decision, expert))
        if expert is not None:
            samples.append(_sample_from_state(state, action, expert))
        tracker.record(state, action)
        _obs, decision, _reward, done, info = env.step(action)
        status = info.get("status")

    return samples, n_nodes, str(status)


class PoolGenerator:
    """Instance generator over a file pool that remembers the current file —
    the collector needs to know which instance an episode solved (for its
    reference optimum x*), which the stock ``FileGenerator`` does not expose."""

    def __init__(
        self, instance_dir: str | Path, pattern: str = "*.mps", rng: Any = None
    ) -> None:
        self.files = sorted(Path(instance_dir).glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no {pattern} files under {instance_dir}")
        self.rng = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        self.current_path: Path | None = None

    def __iter__(self):
        return self

    def __next__(self) -> Model:
        self.current_path = self.files[int(self.rng.integers(len(self.files)))]
        model = Model()
        model.readProblem(filename=str(self.current_path))
        return model


class BranchingOnlyDynamics(BranchingDynamics):
    """Learned branching under SCIP's default node selection, speaking the
    joint regime's :class:`Decision` protocol (every decision is a branching
    one), so the tracker and the policy need no second code path."""

    def reset(self, model):
        done, actions = super().reset(model)
        return done, _branching_decision(actions)

    def step(self, action):
        done, actions = super().step(action)
        return done, _branching_decision(actions)


def _branching_decision(actions) -> Decision | None:
    if actions is None:
        return None
    return Decision(kind=DecisionKind.BRANCHING, actions=actions)


def make_env(
    generator: PoolGenerator,
    scip_params: dict | None = None,
    time_limit: float | None = None,
    node_selection: bool = True,
) -> Environment:
    """The collection environment: the joint dynamics, no observation
    function (the collector extracts the bipartite graph itself, at the
    branching pauses where it is safe). ``node_selection=False`` is the
    paper's BR-only ablation: SCIP's default (Estimate) selects the nodes."""
    params = dict(scip_params or SEQBNB_SCIP_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    params.setdefault("display/verblevel", 0)
    return Environment(
        instance_generator=generator,
        information_function=SolveStatus(),
        scip_params=params,
        dynamics=(
            BranchingNodeSelectionDynamics() if node_selection else BranchingOnlyDynamics()
        ),
    )


def collect(
    instance_dir: str | Path,
    n_episodes: int,
    seeds: list[int],
    select_action: Callable[[SeqBnBState, Decision, int | None], int],
    pattern: str = "*.mps",
    time_limit: float | None = None,
    on_episode: Any = None,
    node_selection: bool = True,
) -> CollectionResult:
    """Collect samples over ``n_episodes`` joint-regime episodes.

    One base seed per episode (the solver seed of its
    :class:`~gyozas.seeds.SeedTriplet`); ``select_action`` is the acting
    policy — ``lambda s, d, e: e`` (the expert) at DAgger round 0, the
    current model afterwards (the expert's label is always recorded).
    """
    if len(seeds) < n_episodes:
        raise ValueError(
            f"{n_episodes} episodes need {n_episodes} seeds, got {len(seeds)}"
        )
    from gyozas.seeds import SeedTriplet

    references = ReferenceCache()
    result = CollectionResult()
    generator = PoolGenerator(instance_dir, pattern=pattern, rng=seeds[0])
    env = make_env(generator, time_limit=time_limit, node_selection=node_selection)

    def reference_for(_env: Environment) -> dict[str, float] | None:
        # The x* oracle labels node selection only; the BR-only regime has none.
        if not node_selection:
            return None
        return references.get(generator.current_path)

    try:
        for episode in range(n_episodes):
            triplet = SeedTriplet.from_base_seed(int(seeds[episode]))
            env.seed(triplet.solver)
            samples, n_nodes, status = run_episode(env, select_action, reference_for)
            result.samples.extend(samples)
            result.n_episodes += 1
            result.n_nodes += n_nodes
            result.statuses.append(status)
            if on_episode is not None:
                on_episode(result.n_episodes, status, n_nodes, len(samples))
    finally:
        env.close()
    return result
