"""PlanB&B — self-play episode collection over the B&B tree MDP.

Faithful port of the official repository (github.com/abfariah/planbb,
``core/{milp,selfplay_worker}.py`` + ``config/milp/env_wrapper.py`` +
``core/search_tree.py``), adapted to the harness's ``Environment``:

- **Self-play** — at every visited node the policy runs a Gumbel MCTS from
  the current observation (``initial_inference`` at the root, then
  ``recurrent_inference`` along the latent children of the open-node
  frontier); the action is sampled from the improved root policy at the
  visit-count temperature, except *before training starts*, where the action
  is uniform over the candidates (the official ``np.random.choice`` — the
  MCTS statistics are still stored). The reward is the official constant -1
  per decision.
- **The trajectory object** (``MILPTrajectory``) — records the observations,
  actions, depths, the MCTS improved-policy distributions and root values;
  at episode end it validates the episode (double root dropped, memory/tree
  gap 0 or 1, node-count limit) and precomputes, per visited decision, the
  unrolled *state* windows the MuZero losses need: the batch of real and
  ghost nodes (``is_real`` 1 / 0.3 left ghost / 0.7 right ghost), the
  open-node stacks per step, the td-step reward sums, and the n-step
  bootstrap frontier indices.
- **Ghost nodes** — a visited node's unvisited children keep their slot in
  the unroll as a copy of the parent observation with ``branchability``
  zeroed and ``is_best_sibling`` set to the ghost side (the official
  ``zero_obs``); padding past the trajectory end repeats the terminal node
  three times per step with ``is_real`` [0.3, 0.7, 1].

Deviations from the official, all documented here:

- the official's ``target_rewards`` / left / right rewards are dropped —
  they are dead code under the released configuration (``reward_module =
  False``, ``strong_reward = False``: ``update_weights`` unpacks them as
  ``_``);
- the official's gap-1 branch references ``node_step_idx_to_child_nodes``,
  which is never defined anywhere in the repository — the port implements
  its evident intent (drop the time-limited extra visited node from the
  step maps), exactly like the bbmdp port's gap-1 handling;
- a pre-solved instance (no tracked root) is rejected instead of the
  official's truthy ``[{0: 0}]`` return — in the harness such episodes
  never start (``reset`` is done) so the branch is unreachable either way.

SCIP protocol = the official ``set_params`` with the released ``--dfs``:
heuristics ON, cuts at the root only (``maxrounds`` 0), no restarts,
depth-first node selection, training-side only, like every other RL method
here; the benchmark evaluation keeps the runner's default solver profile.
"""

import copy
from collections.abc import Iterator
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import torch
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.observations import NodeBipartitePlanBB
from pyscipopt import Model
from pyscipopt.scip import PY_SCIP_STAGE
from scipy.stats import entropy

from ..bbmdp.collect import BBMDP_SCIP_PARAMS
from ..sorrel.collect import TreeTracker
from .model import PlanBBObs, geometric_batch
from .search import MCTS

#: The released hyperparameters (config/milp/__init__.py).
UNROLL_STEPS = 3
TD_STEPS = 3
DISCOUNT = 1.0
NODE_LIMIT = 10_000
#: The official reward: ecole Constant(-1) per branching decision.
STEP_REWARD = -1.0
#: The official MCTS branchability threshold (cy_mcts search default).
BREWARD_THRESHOLD = 0.5

#: The official SCIP protocol = BBMDP's exactly (set_params + released --dfs).
PLANBB_SCIP_PARAMS: dict[str, Any] = BBMDP_SCIP_PARAMS


def make_env(
    instances: Iterator[Model],
    node_limit: int = NODE_LIMIT,
    time_limit: float | None = None,
    seed: int = 0,
) -> Environment:
    """One PlanB&B environment: the 43-feature NodeBipartitePlanBB
    observation, LP branching candidates, the official DFS/no-restarts SCIP
    protocol."""
    params = dict(PLANBB_SCIP_PARAMS)
    params["limits/nodes"] = node_limit
    if time_limit is not None:
        params["limits/time"] = time_limit
    params["display/verblevel"] = 0
    env = Environment(
        instances,
        observation_function=NodeBipartitePlanBB(),
        information_function=SolveStatus(),
        scip_params=params,
        dynamics=BranchingDynamics(),
    )
    env.dynamics.seed(seed)
    return env


class SearchTreeAdapter:
    """The official ``SearchTree`` (core/search_tree.py), reduced to what
    the trajectory needs: the visitation order and every node's successors
    in discovery order (built from the harness's ``TreeTracker`` — both
    record the current node first, then the open leaves/children/siblings)."""

    def __init__(self, tracker: TreeTracker) -> None:
        self._tracker = tracker
        self._children: dict[int, list[int]] = {}
        for node_id, (parent, _lb) in tracker._nodes.items():
            if parent is not None:
                self._children.setdefault(parent, []).append(node_id)

    @property
    def visited_node_ids(self) -> list[int]:
        return list(self._tracker.visited)

    def successors(self, node: int) -> list[int]:
        return self._children.get(node, [])


class BatchObs(NamedTuple):
    """The official ``batch_obs`` output: everything one sampled position
    needs to build its MuZero training window."""

    batch_observations: list  # the (ghost-modified) observations of the window
    batch_nodes_idx: list  # step index of each window node
    is_real_nodes: list  # 1 real / 0.3 left ghost / 0.7 right ghost
    state_open_nodes_idx: list  # per step, the window slots of the open nodes
    pad_len: int
    reward_sums: Any  # per-root td-capped reward sums (non-tree-MDP target)
    traj_index: list
    td_value_open_nodes: list
    td_reward_sums: list
    n_steps_discount_powers: list
    n_steps_td_observations_idx: list
    n_steps_td_observations: list
    value_loss_obs_weight: list
    num_obs: int


def zero_obs(in_obs: PlanBBObs, is_real: float) -> PlanBBObs:
    """The official ghost observation: ``branchability`` zeroed,
    ``is_best_sibling`` set on the left ghost (features -1 / -2 of the
    planbb block)."""
    assert is_real < 1
    is_best_sibling = 1 if is_real == 0.3 else 0
    obs = copy.deepcopy(in_obs)
    obs.variable_features[:, -1] = 0
    obs.variable_features[:, -2] = is_best_sibling
    return obs


class PlanBBTrajectory:
    """The official ``MILPTrajectory``: one episode's history plus the
    per-decision state windows (``build_milp_states``)."""

    def __init__(
        self,
        unroll_steps: int = UNROLL_STEPS,
        td_steps: int = TD_STEPS,
        discount: float = DISCOUNT,
        node_limit: int = NODE_LIMIT,
    ) -> None:
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        self.discount = discount
        self.node_limit = node_limit
        self.traj_horizon = 1 + unroll_steps
        self.horizon = unroll_steps + td_steps

        self.child_visits: list[list[float]] = []
        self.root_values: list[float] = []
        self.actions: list[int] = []
        self.observations: list[PlanBBObs] = []
        self.rewards: list[float] = []
        self.depths: list[int] = []
        self.search_tree: Optional[SearchTreeAdapter] = None
        self.store_milp = False

    # -- recording ----------------------------------------------------------

    def init(self, obs: PlanBBObs, depth: int = 0) -> None:
        """Initialize the history block (one root observation, its depth)."""
        self.child_visits = []
        self.root_values = []
        self.actions = []
        self.observations = []
        self.rewards = []
        self.depths = []
        self.search_tree = None
        self.action_num = obs.variable_features.shape[0]
        self.observations.append(obs)
        self.depths.append(depth)

    def step_obs(self) -> PlanBBObs:
        """The observation of the current (last recorded) decision."""
        return self.observations[-1]

    def store_search_stats(self, visit_counts: list[float], root_value: float, idx: int | None = None) -> None:
        """Store the MCTS improved-policy distribution and root value."""
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visits.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_values.append(root_value)
        else:
            self.child_visits[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_values[idx] = root_value

    def append(self, action: int, obs: PlanBBObs | None, depth: int) -> None:
        """Append one transition (the constant -1 reward). The next
        observation and its depth are recorded together — the official
        skips both when the episode is done."""
        self.actions.append(action)
        self.rewards.append(STEP_REWARD)
        if obs is not None:
            self.observations.append(obs)
            self.depths.append(depth)

    def __len__(self) -> int:
        return len(self.actions)

    # -- tree indexing ------------------------------------------------------

    def sort_exploration(self, open_nodes: list[int]) -> list[int]:
        """The official frontier order: deepest first, ties by visit order."""
        return sorted(
            open_nodes,
            key=lambda x: (
                -self.depths[self.visited_nodes_to_step_idx[x]],
                self.visited_nodes_to_step_idx[x],
            ),
        )

    def get_children(self, current_node: int, true_children: list[int]) -> Tuple[list, list]:
        """The official ghost-child completion of a node with <2 visited
        children (its unvisited tree children count towards the shape)."""
        ghost_children = list(self.search_tree.successors(current_node))
        if len(true_children) == 0:
            left_node, right_node = current_node, current_node
            is_real = [0.3, 0.7]
        elif len(true_children) == 1 and len(ghost_children) == 1:
            left_node, right_node = current_node, true_children[0]
            is_real = [0.3, 1]
        elif len(true_children) == 1 and len(ghost_children) == 2:
            left_node, right_node = true_children[0], current_node
            is_real = [1, 0.7]
        else:
            return [None, None], [-1, -1]
        return [left_node, right_node], is_real

    # -- episode validation --------------------------------------------------

    def is_milp_solved(self) -> bool:
        """The official ``is_milp_solved``: drop a redundant first root,
        reject over-long episodes and memory/tree gaps other than 0/1."""
        if len(self.visited_node_ids) == 0:
            # Pre-solved instance: nothing to store.
            return False
        self.transition_gap = 0

        self.visited_nodes_to_step_idx = {node: idx for idx, node in enumerate(self.visited_node_ids)}
        self.step_idx_to_visited_nodes = dict(enumerate(self.visited_node_ids))
        self.reward_sums: list[Optional[float]] = [None] * len(self.observations)

        if self.depths[:2] == [0, 0]:
            # The official drops the first transition of a double root; the
            # tree keeps its original indices.
            self.depths = self.depths[1:]
            self.reward_sums = self.reward_sums[1:]
            self.observations = self.observations[1:]
            self.rewards = self.rewards[1:]
            self.actions = self.actions[1:]
            self.child_visits = self.child_visits[1:]
            self.root_values = self.root_values[1:]
            self.action_num = self.observations[0].variable_features.shape[0]

        # Do not store unsolved trajectories.
        if len(self.observations) >= self.node_limit // 2 - 10:
            return False

        if len(self.observations) != len(self.visited_nodes_to_step_idx):
            # A time-limited episode carries one extra visited node; the
            # official drops it (its ``node_step_idx_to_child_nodes`` filter
            # — undefined there — implemented by its intent).
            self.transition_gap = len(self.visited_nodes_to_step_idx) - len(self.observations)
            if self.transition_gap == 1:
                idx_to_remove = len(self.observations)
                self.visited_nodes_to_step_idx = {
                    node: step_idx
                    for node, step_idx in self.visited_nodes_to_step_idx.items()
                    if step_idx != idx_to_remove
                }
                self.step_idx_to_visited_nodes = {
                    step_idx: node
                    for step_idx, node in self.step_idx_to_visited_nodes.items()
                    if step_idx != idx_to_remove
                }
            else:
                return False
        return True

    # -- the state windows ---------------------------------------------------

    def state_obs(self, current_node_idx: int, padding: bool = False):
        """The official ``state_obs``: the observation window rooted at the
        visited decision ``current_node_idx`` — up to ``horizon`` decisions,
        each contributing its (ghost-completed) children — plus the open-node
        stacks and td-step frontier indices the targets need."""
        trajectory_idx, step, step_counter = 0, 0, 0
        visited_nodes, batch_nodes, is_real_nodes = [], [], []
        traj_indices, state_open_nodes = [], []
        pad_len = max(0, current_node_idx + self.horizon - len(self.observations))
        ep_len = len(self.step_idx_to_visited_nodes)
        horizon = self.horizon - pad_len
        doublons: list[int] = []
        open_nodes: list[int] = []
        while step < horizon and current_node_idx + step_counter < ep_len:
            step_counter = 0
            current_node = self.step_idx_to_visited_nodes[current_node_idx + step_counter]
            while current_node in doublons and current_node_idx + step_counter < ep_len:
                step_counter += 1
                current_node = self.step_idx_to_visited_nodes[current_node_idx + step_counter]
            open_nodes = [current_node]
            state_open_nodes += [[current_node]]
            traj_indices.append(trajectory_idx)
            while len(open_nodes) > 0 and current_node_idx + step_counter < ep_len and step < horizon:
                step += 1
                step_counter += 1
                current_node = open_nodes.pop(0)
                doublons.append(current_node)
                visited_nodes.append(current_node)

                children = [
                    child
                    for child in self.search_tree.successors(current_node)
                    if child in self.visited_nodes_to_step_idx
                ]

                for child in children:
                    open_nodes.append(child)

                if len(children) < 2:
                    added_nodes, is_real = self.get_children(current_node, children)
                else:
                    added_nodes = children[:]
                    is_real = [1, 1]

                # sort children according to their step visit
                open_nodes = self.sort_exploration(open_nodes)

                is_real_nodes += [1] + is_real
                batch_nodes += [current_node] + added_nodes[:]
                if len(open_nodes) > 0:
                    state_open_nodes += [open_nodes[:]]
                    traj_indices.append(trajectory_idx)
                else:
                    trajectory_idx += 1

        if len(open_nodes) > 0:
            batch_nodes.append(open_nodes[0])
            is_real_nodes.append(1)
        else:
            step_counter = 0
            while current_node in doublons and current_node_idx + step_counter < ep_len:
                current_node = self.step_idx_to_visited_nodes[current_node_idx + step_counter]
                step_counter += 1
            if current_node_idx + step_counter < ep_len:
                final_node = self.step_idx_to_visited_nodes[current_node_idx + step_counter]
                state_open_nodes += [[final_node]]
                traj_indices.append(trajectory_idx)
                batch_nodes.append(final_node)
                is_real_nodes.append(1)
            else:
                batch_nodes.append(visited_nodes[-1])
                is_real_nodes.append(1)

        batch_nodes_idx = [self.visited_nodes_to_step_idx[batch_node] for batch_node in batch_nodes]

        # Ghost slots get a distinct key: node + 0.3 (left) / + 0.7 (right).
        batch_node_obs = [obs_num + is_real * (is_real < 1) for obs_num, is_real in zip(batch_nodes, is_real_nodes)]
        state_open_nodes_idx = [
            [batch_node_obs.index(node) for node in open_node] for open_node in state_open_nodes
        ]

        if padding:
            if pad_len > 0:
                assert len(open_nodes) == 0
                terminal_node = visited_nodes[-1]
                terminal_node_idx = self.visited_nodes_to_step_idx[visited_nodes[-1]]
                padding_nodes = [terminal_node] * 3 * pad_len
                padding_nodes_idx = [terminal_node_idx] * 3 * pad_len
                batch_nodes += padding_nodes
                batch_nodes_idx += padding_nodes_idx
                is_real_nodes += [0.3, 0.7, 1] * pad_len
            while len(state_open_nodes_idx) < self.horizon + 1:
                state_open_nodes_idx += [state_open_nodes_idx[-1]]

        td_value_open_nodes = []
        for step_i in range(self.unroll_steps + 1):
            for td_step in range(1, self.td_steps + 1):
                if step_i + td_step < len(traj_indices):
                    if traj_indices[step_i] == traj_indices[step_i + td_step]:
                        # Traj continues
                        td_step_state = state_open_nodes_idx[step_i + td_step][:]
                    else:
                        # Traj is over, no observation
                        td_step_state = []
                        break
                else:
                    # Traj is over, no observation
                    td_step_state = []
                    break
            td_value_open_nodes.append(td_step_state)

        assert len(traj_indices) == len(state_open_nodes)

        return (
            batch_nodes,
            batch_nodes_idx,
            is_real_nodes,
            state_open_nodes_idx,
            pad_len,
            traj_indices,
            td_value_open_nodes,
        )

    def post_process_search_tree(self) -> None:
        """The official td-capped BFS reward sums, per visited node (the
        frontier kept in exploration order)."""
        for node in self.visited_nodes_to_step_idx:
            reward_sum, step = 0, 0
            open_nodes = [node]
            root_node_step_idx = self.visited_nodes_to_step_idx[node]
            root_node_depth = self.depths[root_node_step_idx]
            while len(open_nodes) > 0:
                current_node = open_nodes.pop(0)
                current_step_idx = self.visited_nodes_to_step_idx[current_node]
                current_node_depth = self.depths[current_step_idx]

                children = [
                    child
                    for child in self.search_tree.successors(current_node)
                    if child in self.visited_nodes_to_step_idx
                ]

                r = self.rewards[current_step_idx]
                reward_sum += r * (self.discount ** (current_node_depth - root_node_depth))
                self.reward_sums[root_node_step_idx] = reward_sum
                step += 1

                for child in children:
                    open_nodes.append(child)

                open_nodes = self.sort_exploration(open_nodes)

                if step == self.td_steps:
                    break

        assert not any(rs is None for rs in self.reward_sums)
        assert len(self.reward_sums) == len(self.visited_nodes_to_step_idx)

    def build_milp_states(self) -> None:
        """The official per-decision precomputation: the state windows plus
        the tree-MDP n-step data (td reward sums, bootstrap frontiers, value
        loss weights)."""
        self.batch_nodes: list[list] = [None] * len(self.observations)
        self.batch_nodes_idx: list[list] = [None] * len(self.observations)
        self.state_obs_mask: list[list] = [None] * len(self.observations)
        self.state_open_nodes: list[list] = [None] * len(self.observations)
        self.padding_lengths: list[int] = [0] * len(self.observations)
        self.traj_indices: list[list] = [None] * len(self.observations)
        self.td_value_open_nodes: list[list] = [None] * len(self.observations)

        for step_idx in self.step_idx_to_visited_nodes:
            (
                batch_nodes,
                batch_nodes_idx,
                obs_mask,
                state_open_nodes_obs_idx,
                pad_len,
                traj_index,
                td_value_open_nodes,
            ) = self.state_obs(step_idx, padding=True)

            self.batch_nodes[step_idx] = batch_nodes
            self.batch_nodes_idx[step_idx] = batch_nodes_idx
            self.state_obs_mask[step_idx] = obs_mask
            self.state_open_nodes[step_idx] = state_open_nodes_obs_idx
            self.padding_lengths[step_idx] = max(0, pad_len - self.td_steps)
            self.traj_indices[step_idx] = traj_index
            self.td_value_open_nodes[step_idx] = td_value_open_nodes

        # The tree-MDP n-step data (the official build_milp_states branch).
        traj_horizon = self.unroll_steps + 1
        traj_obs = 3 * self.unroll_steps + 1
        self.td_reward_sums: list[list] = [None] * len(self.observations)
        self.n_steps_td_observations_idx: list[list] = [None] * len(self.observations)
        self.value_loss_obs_weight: list[list] = [None] * len(self.observations)
        for milp_pos in self.step_idx_to_visited_nodes:
            doublons = []
            td_reward_sums = []
            n_steps_td_observations_idx = []
            value_loss_obs_weight = []
            rest_is_padded = False

            for batch_idx, milp_step_idx in enumerate(self.batch_nodes_idx[milp_pos][:traj_obs]):
                value_obs_weight = 0
                if self.state_obs_mask[milp_pos][batch_idx] == 1:
                    # Real node.
                    if self.batch_nodes_idx[milp_pos][batch_idx] in doublons:
                        td_reward_sums.append(0)
                        n_steps_td_observations_idx.append([])
                        value_loss_obs_weight.append(value_obs_weight)
                        continue
                    doublons.append(self.batch_nodes_idx[milp_pos][batch_idx])

                    # The node's weight in the value loss: its number of
                    # appearances in the window's open-node stacks.
                    for open_nodes in self.state_open_nodes[milp_pos][:traj_horizon]:
                        if batch_idx in open_nodes:
                            value_obs_weight += 1

                    td_reward_sum = self.reward_sums[milp_step_idx]

                    # Does the trajectory continue td_steps ahead?
                    traj_idx = self.traj_indices[milp_step_idx]
                    traj_over = len(traj_idx) <= self.td_steps
                    future_obs_exist = False if traj_over else traj_idx[0] == traj_idx[self.td_steps]
                    if future_obs_exist:
                        td_state_open_nodes = self.state_open_nodes[milp_step_idx][self.td_steps]
                        td_state_open_nodes_step_idx = [
                            self.batch_nodes_idx[milp_step_idx][node] for node in td_state_open_nodes
                        ]
                    else:
                        td_state_open_nodes_step_idx = []
                else:
                    # Ghost node: value 0.
                    if batch_idx % 3 == 0 and not rest_is_padded:
                        rest_is_padded = True
                    td_state_open_nodes_step_idx = []
                    td_reward_sum = 0
                    value_obs_weight = 1 if not rest_is_padded else 0

                td_reward_sums.append(td_reward_sum)
                n_steps_td_observations_idx.append(td_state_open_nodes_step_idx)
                value_loss_obs_weight.append(value_obs_weight)

            self.td_reward_sums[milp_pos] = td_reward_sums
            self.n_steps_td_observations_idx[milp_pos] = n_steps_td_observations_idx
            self.value_loss_obs_weight[milp_pos] = value_loss_obs_weight

    def milp_solved(self, search_tree: SearchTreeAdapter) -> bool:
        """The official ``milp_solved``: validate the episode and precompute
        everything the replay sample needs. Returns whether the trajectory
        is stored."""
        assert len(self.rewards) == len(self.observations) == len(self.actions)
        self.search_tree = search_tree
        self.visited_node_ids = search_tree.visited_node_ids
        self.store_milp = self.is_milp_solved()
        if self.store_milp:
            self.post_process_search_tree()
            self.build_milp_states()
        return self.store_milp

    # -- batch construction ---------------------------------------------------

    def get_n_steps_discount_powers(self, batch_nodes_idx, n_steps_td_observations_idx):
        """The official depth differences between each bootstrap frontier
        node and its window root."""
        flat_indices = [idx for sublist in n_steps_td_observations_idx for idx in sublist]
        batch_nodes_depth = [self.depths[idx] for idx in batch_nodes_idx]
        fetched_depths = [self.depths[idx] for idx in flat_indices]
        it = iter(fetched_depths)
        return [
            [next(it) - root_depth for _ in td_observations_idx]
            for root_depth, td_observations_idx in zip(batch_nodes_depth, n_steps_td_observations_idx)
        ]

    def batch_obs(self, milp_pos: int) -> BatchObs:
        """The official ``batch_obs`` (without reanalyze): the ghost-completed
        observation window at position ``milp_pos`` and its n-step data. The
        frontier observations are plain references into the episode — the
        official fetches them lazily from ray memory at train time."""
        batch_nodes_idx = self.batch_nodes_idx[milp_pos]
        is_real_nodes = self.state_obs_mask[milp_pos]
        state_open_nodes_idx = self.state_open_nodes[milp_pos]
        pad_len = self.padding_lengths[milp_pos]
        reward_sums = self.reward_sums[milp_pos]
        traj_index = self.traj_indices[milp_pos]
        td_value_open_nodes = self.td_value_open_nodes[milp_pos]

        batch_observations = [
            obs if is_real == 1 else zero_obs(obs, is_real)
            for obs, is_real in zip(
                (self.observations[idx] for idx in batch_nodes_idx), is_real_nodes
            )
        ]

        td_reward_sums = self.td_reward_sums[milp_pos]
        value_loss_obs_weight = self.value_loss_obs_weight[milp_pos]
        n_steps_td_observations_idx = self.n_steps_td_observations_idx[milp_pos]
        n_steps_discount_powers = self.get_n_steps_discount_powers(batch_nodes_idx, n_steps_td_observations_idx)
        num_obs = sum(len(td_observations_idx) for td_observations_idx in n_steps_td_observations_idx)
        n_steps_td_observations = [
            [self.observations[idx] for idx in td_observations_idx]
            for td_observations_idx in n_steps_td_observations_idx
        ]
        return BatchObs(
            batch_observations,
            batch_nodes_idx,
            is_real_nodes,
            state_open_nodes_idx,
            pad_len,
            reward_sums,
            traj_index,
            td_value_open_nodes,
            td_reward_sums,
            n_steps_discount_powers,
            n_steps_td_observations_idx,
            n_steps_td_observations,
            value_loss_obs_weight,
            num_obs,
        )


def select_action(visit_counts, temperature: float = 1.0, deterministic: bool = True) -> Tuple[int, float]:
    """The official ``core/utils.select_action``: sample the visit-count
    distribution at the temperature (or argmax of the raw counts); returns
    the action position and the sampling entropy."""
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = int(np.argmax(list(visit_counts)))
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)
    count_entropy = entropy(action_probs, base=2)
    return int(action_pos), float(count_entropy)


def run_selfplay_episode(
    env: Environment,
    model,
    device: torch.device,
    temperature: float = 1.0,
    start_training: bool = True,
    rng: np.random.Generator | None = None,
    mcts: MCTS | None = None,
    threshold: float = BREWARD_THRESHOLD,
) -> Tuple[Optional[PlanBBTrajectory], int, str]:
    """One self-play episode (the official selfplay_worker loop, sequential —
    ``p_mcts_num`` only parallelizes independent episodes there). Before
    training starts the action is uniform over the candidates but the MCTS
    statistics are still stored, exactly like the official."""
    obs, action_set, _reward, done, info = env.reset()
    if done or obs is None:
        return None, 0, str(info.get("status"))
    tracker = TreeTracker(env.model)
    trajectory = PlanBBTrajectory()
    trajectory.init(PlanBBObs(obs, np.asarray(action_set)), depth=int(env.model.getDepth()))
    status = info.get("status")
    if mcts is None:
        mcts = MCTS()
    model.mcts_mode()
    n_decisions = 0

    with torch.no_grad():
        while not done:
            step_obs = trajectory.step_obs()
            action_nums = [step_obs.variable_features.shape[0]]
            batched = geometric_batch([step_obs]).to(device)
            network_output = model.initial_inference(batched, root=True)

            root_states = [[0]]
            value_pool = list(network_output.value.tolist())
            root_state_obs = [[node_obs] for node_obs in network_output.hidden_state.to_data_list()]
            policy_logits_pool = [[policy_logits] for policy_logits in network_output.policy_logits.tolist()]
            breward_pool = list(network_output.breward.tolist())

            roots_values, roots_distributions, _, _ = mcts.search(
                model,
                1,
                action_nums,
                [step_obs.masked_action_set],
                root_states,
                root_state_obs,
                policy_logits_pool,
                value_pool,
                breward_pool,
                device,
                threshold=threshold,
            )

            distribution = roots_distributions[0][: action_nums[0]]
            action, _visit_entropy = select_action(distribution, temperature=temperature, deterministic=False)
            if not start_training:
                # Before starting training, use random policy.
                action = int(rng.choice(step_obs.action_set)) if rng is not None else int(np.random.choice(step_obs.action_set))

            obs, action_set, _reward, done, info = env.step(action)
            status = info.get("status")

            trajectory.store_search_stats(distribution, float(roots_values[0]))
            next_obs = None if done or obs is None else PlanBBObs(obs, np.asarray(action_set))
            # The recorded depth is the *next* decision's (the official
            # appends the post-step info's depth with the next obs).
            trajectory.append(int(action), next_obs, int(env.model.getDepth()))
            n_decisions += 1

            # The official update_tree runs on every extract, done included;
            # pyscipopt raises on getOpenNodes once solved — update only
            # while SCIP is still solving (the bbmdp port's adaptation).
            if not done or env.model.getStage() == PY_SCIP_STAGE.SOLVING:
                tracker.update(env.model)

    stored = trajectory.milp_solved(SearchTreeAdapter(tracker))
    return (trajectory if stored else None), n_decisions, str(status)
