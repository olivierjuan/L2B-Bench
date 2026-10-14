"""BBMDP 2025 (Forel et al., NeurIPS 2025) — episode collection over the
corrected vanilla MDP.

Faithful port of the official repository (github.com/abfariah/bbmdp,
``rl/environments/{ecole,search_tree}.py`` + ``rl/rewards/reward_agent.py``
+ ``process_episode_into_bbmdp``), adapted to the harness's ``Environment``:

- **Vanilla MDP (Prop. 3.1)** — the state is the current node's bipartite
  observation and the action a variable among its LP branching candidates;
  after the episode, each visited decision's *reward* and *next states* are
  recomputed retrospectively over the tracked B&B tree.
- **k-step returns (RewardAgent)** — per visited node, a BFS over its visited
  descendants only, the frontier kept sorted by global visit order: the
  reward is ``-gamma^step`` per dequeued node, the 1-step next states are the
  node's visited children and the n-step next states the frontier after
  ``n_step`` dequeues (``n_step`` 3, ``gamma`` 1 on set covering).
- **Episode post-processing** — a double root (two consecutive decisions at
  depth 0) drops the first transition; a visited node without a transition
  (the episode cut by the time limit) is dropped from the tree; any other
  memory/tree mismatch rejects the episode.

SCIP protocol (official ``set_params`` + DFS): heuristics stay ON, cuts at
the root only (``maxrounds`` 0), no restarts (``restartpolicy 'n'``), and
depth-first node selection (``nodeselection/dfs`` priorities 300000) —
training-side only, like every other RL method here; the benchmark
evaluation keeps the runner's default solver profile.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import NodeBipartiteRetro
from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE

from ..sorrel.collect import TreeTracker

#: The official n-step and discount (experiments/params/learner.py).
N_STEP = 3
GAMMA = 1.0

#: The official SCIP protocol: root cuts only, no restarts, DFS node
#: selection (rl/environments/ecole.py set_params). Heuristics stay ON.
BBMDP_SCIP_PARAMS: dict[str, Any] = {
    "separating/maxrounds": 0,
    "estimation/restarts/restartpolicy": "n",
    "nodeselection/dfs/stdpriority": 300000,
    "nodeselection/dfs/memsavepriority": 300000,
}


@dataclass
class BBMDPTransition:
    """One vanilla-MDP transition of the corrected tree MDP.

    ``next_states``/``n_step_next_states`` are lists of the children's states
    — ``(observation, action_set)`` references into the episode, no copies —
    and ``done`` means *this node has no visited child* (a fathomed node),
    not the end of the SCIP episode.
    """

    observation: Any  # the RetroNodeBipartite observation at the decision
    action_set: np.ndarray  # the node's LP branching candidates
    action: int  # the chosen variable's index
    action_idx: int  # the chosen candidate's rank within action_set
    depth: int  # the focus node's depth
    reward: float  # the 1-step reward (minus the node itself)
    next_states: list  # the visited children's states
    done: bool
    n_step_reward: float
    n_step_next_states: list
    n_step_done: bool

    @property
    def state(self) -> tuple:
        return (self.observation, self.action_set)


def make_env(
    instances: Iterator[Model],
    time_limit: float | None = None,
    seed: int = 0,
) -> Environment:
    """One BBMDP environment: the 43-feature RetroNodeBipartite observation,
    LP branching candidates, the official DFS/no-restarts SCIP protocol."""
    params = dict(BBMDP_SCIP_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    params["display/verblevel"] = 0
    env = Environment(
        instances,
        observation_function=NodeBipartiteRetro(),
        information_function=SolveStatus(),
        scip_params=params,
        dynamics=BranchingDynamics(),
    )
    env.dynamics.seed(seed)
    return env


def bbmdp_returns(
    children_positions: list[list[int]],
    n_step: int = N_STEP,
    gamma: float = GAMMA,
) -> list[dict[str, Any]]:
    """The official ``RewardAgent.extract`` BFS, per visited decision.

    ``children_positions[i]`` lists the step indices of decision ``i``'s
    visited children (the tree's successors filtered to visited nodes). The
    BFS dequeues nodes one at a time — the reward is ``-gamma^step`` per
    dequeue, the frontier re-sorted by visit order after every dequeue — and
    captures the 1-step reward/frontier after one dequeue and the n-step
    ones after ``n_step`` dequeues (or when the frontier empties, keeping the
    last captured values).
    """
    out = []
    for node in range(len(children_positions)):
        reward = 0.0
        step = 0
        one_step_reward = one_step_children = None
        n_step_reward = n_step_children = None
        open_nodes = [node]
        while open_nodes:
            current = open_nodes.pop(0)
            children = children_positions[current]
            reward -= gamma**step
            step += 1
            open_nodes = sorted(open_nodes + children)
            if step == 1:
                one_step_reward = reward
                one_step_children = open_nodes[:]
                if n_step == 1:
                    break
            if step <= n_step:
                n_step_reward = reward
                n_step_children = open_nodes[:]
            if step == n_step:
                break
        out.append(
            {
                "next_states": one_step_children,
                "reward": one_step_reward,
                "n_step_next_states": n_step_children,
                "n_step_reward": n_step_reward,
            }
        )
    return out


def _visited_children(tracker: TreeTracker) -> list[list[int]]:
    """Per visited decision, its children's positions among the visited
    decisions (the official successors-in-visited filter)."""
    return [
        [position for position in positions if position is not None]
        for positions, _rewards in tracker.children_of()
    ]


def process_episode(
    observations: list[Any],
    action_sets: list[np.ndarray],
    actions: list[int],
    action_idxs: list[int],
    depths: list[int],
    tree_returns: list[dict[str, Any]],
    n_nodes: int,
) -> list[BBMDPTransition] | None:
    """The official ``process_episode_into_bbmdp``: build the transitions,
    applying the double-root and memory/tree-gap sanity checks. Returns
    ``None`` (a rejected episode) when the checks fail."""
    n_memory = len(observations)
    n_tree = len(tree_returns)
    if n_memory <= 1 or not n_nodes:
        return None
    if depths[0] == 0 and depths[1] == 0:
        # Evacuate a redundant first root (the official drops the first
        # transition; the tree transitions keep their original indices).
        observations = observations[1:]
        action_sets = action_sets[1:]
        actions = actions[1:]
        action_idxs = action_idxs[1:]
        depths = depths[1:]
        n_memory -= 1
    gap = n_tree - n_memory
    if gap != 0:
        if gap == 1:
            # Runtime limit: the tree's last visited node has no transition.
            tree_returns = tree_returns[:n_memory]
        else:
            return None
    transitions = []
    for idx, tree_transition in enumerate(tree_returns):
        def resolve(indices):
            # Step indices beyond the episode's memory (the runtime-limit
            # node) never become next states.
            return [
                (observations[i], action_sets[i]) for i in indices if 0 <= i < n_memory
            ]

        next_states = resolve(tree_transition["next_states"])
        n_step_next_states = resolve(tree_transition["n_step_next_states"])
        transitions.append(
            BBMDPTransition(
                observation=observations[idx],
                action_set=action_sets[idx],
                action=actions[idx],
                action_idx=action_idxs[idx],
                depth=depths[idx],
                reward=float(tree_transition["reward"]),
                next_states=next_states,
                done=len(next_states) == 0,
                n_step_reward=float(tree_transition["n_step_reward"]),
                n_step_next_states=n_step_next_states,
                n_step_done=len(n_step_next_states) == 0,
            )
        )
    return transitions


def run_episode(
    env: Environment,
    select_action: Callable[[Any, np.ndarray], tuple[int, int]],
    n_step: int = N_STEP,
    gamma: float = GAMMA,
) -> tuple[list[BBMDPTransition] | None, int, str]:
    """One episode under ``select_action`` (returning the chosen variable and
    its rank in the action set); the tracker is updated after every step,
    the final one included while SCIP still solves — a runtime-limited
    episode thus carries one extra visited node (gap 1), a solved one
    doesn't, exactly like the official ``update_tree`` at done."""
    obs, action_set, _reward, done, info = env.reset()
    if done or obs is None:
        return None, 0, str(info.get("status"))
    tracker = TreeTracker(env.model)

    observations: list[Any] = []
    action_sets: list[np.ndarray] = []
    actions: list[int] = []
    action_idxs: list[int] = []
    depths: list[int] = []
    status = info.get("status")
    while not done:
        candidates = np.asarray(action_set)
        action, action_idx = select_action(obs, candidates)
        observations.append(obs)
        action_sets.append(candidates.copy())
        actions.append(int(action))
        action_idxs.append(int(action_idx))
        depths.append(int(env.model.getDepth()))
        obs, action_set, _reward, done, info = env.step(int(action))
        status = info.get("status")
        # The official update_tree runs on every extract, done included. When
        # the episode ended by solving, SCIP has left the tree stage: the
        # official's release-SCIP update there is a no-op (no current node —
        # no visited entry, so gap 0), while pyscipopt raises on getOpenNodes
        # in the solved stage — so the update runs only while SCIP is still
        # solving. A time-limited episode stays in the solving stage, and its
        # final update records the node in process — the extra visited node
        # that makes the memory/tree gap of 1.
        if not done or env.model.getStage() == PY_SCIP_STAGE.SOLVING:
            tracker.update(env.model)
    n_nodes = len(observations)
    tree_returns = bbmdp_returns(_visited_children(tracker), n_step=n_step, gamma=gamma)
    transitions = process_episode(
        observations,
        action_sets,
        actions,
        action_idxs,
        depths,
        tree_returns,
        n_nodes,
    )
    return transitions, n_nodes, str(status)
