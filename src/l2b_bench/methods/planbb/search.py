"""PlanB&B — the Gumbel MCTS over B&B open-node states, in pure Python.

Faithful port of ``core/mcts/ctree_v1/{cnode,cminimax}.cpp`` (via
``cytree.pyx``) plus ``core/mcts/cy_mcts.py``'s ``CyMCTS.search`` — the
C++ only accelerates; every branch below mirrors the official control flow:

- ``MinMaxStats`` — the cminimax normalize (delta vs ``value_delta_max``,
  clamped [0, 1]) with the official ±1e6 sentinels.
- ``Node`` — one search-tree node: priors are the *raw* stored logits, the
  value is the running mean over backups, ``get_v_mix`` mixes the parent
  value with the expanded children's π-weighted Q.
- ``Roots`` — one root per episode with its candidate mask; ``prepare``
  expands with the root policy and backs up the root value once.
- ``batch_traverse`` — Gumbel-Top-k at the first simulation (only
  candidates can be picked), then equal visits among the selected; below
  the root the improved-policy minus a visit penalty. Sequential halving
  never fires with the released constants (num_top_actions ==
  num_simulations = 50 → 50 visits before the phase would change), but the
  logic is ported anyway.
- ``back_propagate`` — r = -1 per depth, the MinMaxStats updated with the
  *shifted* (discounted) bootstrap like the C++.
- ``MCTS.search`` — the open-node stacks: each simulation pops the front
  open node, runs ``recurrent_inference`` on the batched hidden states,
  re-inserts the children whose predicted branchability clears the
  threshold (DFS order), and backs up the state-value update
  ``v + r + v_left + v_right - v_current``.

The released search constants (``config/milp/__init__.py``) are the class
defaults: 50 simulations, 50 top actions, c_visit 50, c_scale 0.1,
value_minmax_delta 0.01, discount 1.0, threshold 0.5.
"""

from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from .model import amp_autocast, format_batch_actions

FLOAT_MAX = 1e6
FLOAT_MIN = -FLOAT_MAX
EPSILON = 1e-6
STEP_REWARD = -1.0


def softmax(logits: List[float]) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


class MinMaxStats:
    """The official running min/max normalizer (cminimax.cpp)."""

    def __init__(self) -> None:
        self.maximum = FLOAT_MIN
        self.minimum = FLOAT_MAX
        self.value_delta_max = 0.0
        self.c_visit = 0
        self.c_scale = 0.0

    def set_static_val(self, value_delta_max: float, c_visit: int, c_scale: float) -> None:
        self.value_delta_max = value_delta_max
        self.c_visit = c_visit
        self.c_scale = c_scale

    def update(self, value: float) -> None:
        if value > self.maximum:
            self.maximum = value
        if value < self.minimum:
            self.minimum = value

    def normalize(self, value: float) -> float:
        norm_value = value
        delta = self.maximum - self.minimum
        if delta > 0:
            if delta < self.value_delta_max:
                norm_value = (norm_value - self.minimum) / self.value_delta_max
            else:
                norm_value = (norm_value - self.minimum) / delta
        return min(1.0, max(0.0, norm_value))


class Node:
    """One search-tree node (cnode.cpp CNode). Children are allocated in
    the root's pool at expansion time, like the C++ node pool."""

    def __init__(self, prior: float, action: int, parent: Optional["Node"], discount: float, num_actions: int):
        self.num_actions = num_actions
        self.action = action
        self.best_action = -1
        self.depth = 0 if parent is None else parent.depth + 1
        self.visit_count = 0
        self.hidden_state_index_x = 0
        self.hidden_state_index_y = 0
        self.prior = prior
        self.discount = discount
        self.parent = parent
        self.children_idx: List[int] = []
        self.selected_children_idx: List[int] = []
        self.estimated_value_lst: List[float] = []
        self.root_action_set: Optional[List[int]] = None  # set on the roots only
        # The C++ node pool: children are appended to the owning root's pool.
        self.pool: Optional[List["Node"]] = None
        self.root_ref: Optional["Roots"] = None
        if parent is not None:
            self.pool = parent.pool
            self.root_ref = parent.root_ref

    # -- tree bookkeeping ---------------------------------------------------

    def set_root_action_set(self, root_action_set: List[int]) -> None:
        self.root_action_set = root_action_set

    def expand(self, hidden_state_index_x: int, hidden_state_index_y: int, policy_logits: List[float], leaf_action_num: int) -> None:
        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        assert len(policy_logits) == self.num_actions
        pool = self.pool
        for a, logit in enumerate(policy_logits):
            self.children_idx.append(len(pool))
            pool.append(Node(logit, a, self, self.discount, leaf_action_num))

    def get_child(self, action: int) -> "Node":
        return self.pool[self.children_idx[action]]

    def is_root(self) -> bool:
        return self.parent is None

    def is_expanded(self) -> bool:
        return len(self.children_idx) > 0

    # -- statistics ---------------------------------------------------------

    def get_children_priors(self) -> List[float]:
        return [self.get_child(a).prior for a in range(self.num_actions)]

    def get_children_visits(self) -> List[int]:
        return [self.get_child(a).visit_count for a in range(self.num_actions)]

    def get_policy(self) -> np.ndarray:
        return softmax(self.get_children_priors())

    def get_value(self) -> float:
        if self.is_expanded():
            return sum(self.estimated_value_lst) / len(self.estimated_value_lst)
        return self.parent.get_v_mix()

    def get_qsa(self, action: int) -> float:
        child = self.get_child(action)
        return -1 + self.discount * child.get_value()

    def get_v_mix(self) -> float:
        pi_lst = self.get_policy()
        pi_sum = 0.0
        pi_qsa_sum = 0.0
        for action in range(self.num_actions):
            child = self.get_child(action)
            if child.is_expanded():
                pi_sum += pi_lst[action]
                pi_qsa_sum += pi_lst[action] * self.get_qsa(action)
        if pi_sum < EPSILON:
            return self.get_value()
        return (1.0 / (1.0 + self.visit_count)) * (
            self.get_value() + self.visit_count * pi_qsa_sum / pi_sum
        )

    def get_completed_Q(self, min_max_stats: MinMaxStats, to_normalize: int = 1) -> List[float]:
        v_mix = self.get_v_mix()
        completed_Qs = []
        for action in range(self.num_actions):
            child = self.get_child(action)
            Q = self.get_qsa(action) if child.is_expanded() else v_mix
            if to_normalize == 1:
                Q = min_max_stats.normalize(Q)
                Q = min(1.0, max(0.0, Q))
            completed_Qs.append(Q)
        return completed_Qs

    def get_improved_policy(self, transformed_completed_Qs: List[float]) -> np.ndarray:
        logits = [
            self.get_child(a).prior + transformed_completed_Qs[a] for a in range(self.num_actions)
        ]
        return softmax(logits)

    def do_equal_visit(self, num_simulations: int) -> int:
        min_visit_count = num_simulations + 1
        action = -1
        for selected_child_idx in self.selected_children_idx:
            visit_count = self.get_child(selected_child_idx).visit_count
            if visit_count < min_visit_count:
                action = selected_child_idx
                min_visit_count = visit_count
        return action


class Roots:
    """One search tree per episode in the batch (cnode.cpp CRoots)."""

    def __init__(self, num_roots: int, num_actionss: List[int], root_action_sets: List[List[int]], discount: float):
        self.num_roots = num_roots
        self.num_actionss = num_actionss
        self.discount = discount
        self.pools: List[List[Node]] = []
        self.roots: List[Node] = []
        for i in range(num_roots):
            pool: List[Node] = []
            root = Node(1.0, -1, None, discount, num_actionss[i])
            root.pool = pool
            root.root_ref = self
            self.pools.append(pool)
            self.roots.append(root)
            self.roots[i].set_root_action_set(root_action_sets[i])

    def prepare(self, values: List[float], policies: List[List[float]], leaf_action_nums: List[int]) -> None:
        for i in range(self.num_roots):
            self.roots[i].expand(0, i, policies[i], leaf_action_nums[i])
            self.roots[i].estimated_value_lst.append(values[i])
            self.roots[i].visit_count += 1

    def get_values(self) -> List[float]:
        return [root.get_value() for root in self.roots]

    def get_root_policies(self, min_max_stats_lst: List[MinMaxStats]) -> List[List[float]]:
        policies = []
        for i in range(self.num_roots):
            tcqs = get_transformed_completed_Qs(self.roots[i], min_max_stats_lst[i])
            policies.append(self.roots[i].get_improved_policy(tcqs).tolist())
        return policies

    def get_best_actions(self) -> List[int]:
        return [root.selected_children_idx[0] for root in self.roots]

    def get_distributions(self) -> List[List[int]]:
        return [root.get_children_visits() for root in self.roots]


def get_transformed_completed_Qs(node: Node, min_max_stats: MinMaxStats, final: bool = False) -> List[float]:
    """Completed Q (v_mix for unvisited) normalized to [0, 1], then scaled by
    the official (c_visit + max child visits) * c_scale."""
    to_normalize = 2 if final else 1
    completed_Qs = node.get_completed_Q(min_max_stats, to_normalize)
    max_child_visit_count = max(node.get_children_visits())
    return [
        (min_max_stats.c_visit + max_child_visit_count) * min_max_stats.c_scale * q for q in completed_Qs
    ]


def sequential_halving(root: Node, gumble_noise: List[float], min_max_stats: MinMaxStats, current_num_top_actions: int) -> int:
    children_prior = root.get_children_priors()
    transformed_completed_Qs = get_transformed_completed_Qs(root, min_max_stats)
    selected_children_idx = root.selected_children_idx
    children_scores = [
        gumble_noise[action] + children_prior[action] + transformed_completed_Qs[action]
        for action in selected_children_idx
    ]
    order = np.argsort(children_scores)[::-1]  # descending
    root.selected_children_idx = [selected_children_idx[i] for i in order[:current_num_top_actions]]
    return root.selected_children_idx[0]


def select_action(
    node: Node,
    min_max_stats: MinMaxStats,
    num_simulations: int,
    simulation_idx: int,
    gumble_noise: List[float],
    current_num_top_actions: int,
) -> int:
    if node.is_root():
        if simulation_idx == 0:
            # First phase: g + logits over the candidates only.
            children_prior = node.get_children_priors()
            children_scores = []
            for action in range(node.num_actions):
                if node.root_action_set[action] == 1:
                    children_scores.append(gumble_noise[action] + children_prior[action])
                else:
                    children_scores.append(-10000.0)
            order = np.argsort(children_scores)[::-1]
            node.selected_children_idx = [int(i) for i in order[:current_num_top_actions]]
        return node.do_equal_visit(num_simulations)
    transformed_completed_Qs = get_transformed_completed_Qs(node, min_max_stats)
    improved_policy = node.get_improved_policy(transformed_completed_Qs)
    children_visits = node.get_children_visits()
    children_scores = [
        improved_policy[a] - children_visits[a] / (1.0 + node.visit_count) for a in range(node.num_actions)
    ]
    return int(np.argmax(children_scores))


def batch_traverse(
    roots: Roots,
    min_max_stats_lst: List[MinMaxStats],
    num_simulations: int,
    simulation_idx: int,
    gumble_noises: List[List[float]],
    current_num_top_actions: int,
):
    hidden_state_index_x_lst = []
    hidden_state_index_y_lst = []
    last_actions = []
    search_paths = []
    for i in range(roots.num_roots):
        node = roots.roots[i]
        path = [node]
        last_action = -1
        while node.is_expanded():
            action = select_action(
                node,
                min_max_stats_lst[i],
                num_simulations,
                simulation_idx,
                gumble_noises[i],
                current_num_top_actions,
            )
            node.best_action = action
            node = node.get_child(action)
            last_action = action
            path.append(node)
        parent = path[-2]
        hidden_state_index_x_lst.append(parent.hidden_state_index_x)
        hidden_state_index_y_lst.append(parent.hidden_state_index_y)
        last_actions.append(last_action)
        search_paths.append(path)
    return hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_paths


def back_propagate(search_path: List[Node], min_max_stats: MinMaxStats, value: float) -> None:
    bootstrap_value = value
    for node in reversed(search_path):
        node.estimated_value_lst.append(bootstrap_value)
        node.visit_count += 1
        bootstrap_value = STEP_REWARD + node.discount * bootstrap_value
        min_max_stats.update(bootstrap_value)


def batch_back_propagate(
    hidden_state_index_x: int,
    values: List[float],
    policies: List[List[float]],
    min_max_stats_lst: List[MinMaxStats],
    search_paths: List[List[Node]],
    leaf_action_nums: List[int],
) -> None:
    for i, path in enumerate(search_paths):
        path[-1].expand(hidden_state_index_x, i, policies[i], leaf_action_nums[i])
        back_propagate(path, min_max_stats_lst[i], values[i])


class MCTS:
    """The pure-Python ``CyMCTS`` — same search() protocol, numpy-friendly
    outputs, with the released constants as defaults."""

    def __init__(
        self,
        num_simulations: int = 50,
        num_top_actions: int = 50,
        c_visit: int = 50,
        c_scale: float = 0.1,
        value_minmax_delta: float = 0.01,
        discount: float = 1.0,
    ) -> None:
        self.num_simulations = num_simulations
        self.num_top_actions = num_top_actions
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.discount = discount
        self.value_minmax_delta = value_minmax_delta
        self.current_num_top_actions = self.num_top_actions
        self.current_phase = 0
        self.visit_num_for_next_phase = int(
            max(np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.num_top_actions)), 1)
            * self.num_top_actions
        )
        self.used_visit_num = 0

    def test_mode(self, n_simulations: int, num_top_actions: int) -> "MCTS":
        self.num_simulations = n_simulations
        self.num_top_actions = min(num_top_actions, n_simulations)
        self.current_num_top_actions = self.num_top_actions
        return self

    def ready_for_next_gumble_phase(self, simulation_idx: int) -> bool:
        ready = (simulation_idx + 1) >= self.visit_num_for_next_phase
        if ready:
            self.current_phase += 1
            self.current_num_top_actions = self.current_num_top_actions // 2
            n = self.num_simulations
            m = self.num_top_actions
            current_m = self.current_num_top_actions
            if current_m > 2:
                extra_visit = max(np.floor(n / (np.log2(m) * current_m)), 1) * current_m
            else:
                extra_visit = n - self.used_visit_num
            self.used_visit_num += extra_visit
            self.visit_num_for_next_phase = min(
                self.visit_num_for_next_phase + extra_visit, self.num_simulations
            )
        return ready

    def _reset_phase_state(self) -> None:
        """The official builds a fresh ``CyMCTS`` per search (selfplay_worker
        line 424); the phase counters are per-search state, so a reused MCTS
        object must reset them — otherwise ``current_num_top_actions``
        halves on every search until it starves the root selection."""
        self.current_num_top_actions = self.num_top_actions
        self.current_phase = 0
        self.used_visit_num = 0
        self.visit_num_for_next_phase = int(
            max(np.floor(self.num_simulations / (np.log2(self.num_top_actions) * self.num_top_actions)), 1)
            * self.num_top_actions
        )

    def search(
        self,
        model,
        batch_size: int,
        action_nums: List[int],
        root_action_sets: List[List[int]],
        root_states: List[List[int]],
        root_state_obs: List[List],
        root_policy_logits: List[List[List[float]]],
        root_state_node_values: List[List[float]],
        root_brewards: List[List[float]],
        device: torch.device,
        use_gumble_noise: bool = True,
        threshold: float = 0.5,
    ):
        self._reset_phase_state()
        state_pool = [root_states]
        root_values = [sum(open_node_value) for open_node_value in root_state_node_values]
        hidden_state_value_pool = [root_values]
        hidden_state_open_nodes_obs_pool = root_state_obs
        hidden_state_nodes_value_pool = root_state_node_values
        hidden_state_nodes_policy_pool = root_policy_logits
        hidden_state_nodes_breward_pool = root_brewards

        roots = Roots(batch_size, action_nums, root_action_sets, self.discount)
        prepare_policy_logits = [root_policy_logits[i][0][: action_nums[i]] for i in range(batch_size)]
        roots.prepare(root_values, prepare_policy_logits, action_nums)
        min_max_stats_lst = [MinMaxStats() for _ in range(batch_size)]
        for stats in min_max_stats_lst:
            stats.set_static_val(self.value_minmax_delta, self.c_visit, self.c_scale)

        mcts_info = {}  # per-simulation diagnostics (the official's 4th return)
        hidden_state_index_x = 0
        if use_gumble_noise:
            gumble_noises = np.random.gumbel(0, 1, (batch_size, max(action_nums)))
        else:
            gumble_noises = np.zeros((batch_size, max(action_nums)))
        gumble_noises = gumble_noises.tolist()

        for simulation_idx in range(self.num_simulations):
            current_state_value = []
            current_state_open_nodes = []
            currently_visited_nodes_obs = []
            currently_visited_nodes_value = []
            currently_visited_nodes_policy = []
            currently_visited_nodes_breward = []
            next_visited_nodes_policy = []

            (
                hidden_state_index_x_lst,
                hidden_state_index_y_lst,
                last_actions,
                search_paths,
            ) = batch_traverse(
                roots,
                min_max_stats_lst,
                self.num_simulations,
                simulation_idx,
                gumble_noises,
                self.current_num_top_actions,
            )

            for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                current_open_nodes_obs_idx = state_pool[ix][iy]
                current_state_value.append(hidden_state_value_pool[ix][iy])
                assert len(current_open_nodes_obs_idx) > 0
                current_node_idx = current_open_nodes_obs_idx[0]
                current_state_open_nodes.append(current_open_nodes_obs_idx)
                currently_visited_nodes_obs.append(hidden_state_open_nodes_obs_pool[iy][current_node_idx])
                currently_visited_nodes_value.append(hidden_state_nodes_value_pool[iy][current_node_idx])
                currently_visited_nodes_policy.append(hidden_state_nodes_policy_pool[iy][current_node_idx])
                currently_visited_nodes_breward.append(hidden_state_nodes_breward_pool[iy][current_node_idx])
                next_node_idx = current_open_nodes_obs_idx[1] if len(current_open_nodes_obs_idx) > 1 else None
                if next_node_idx is not None:
                    next_visited_nodes_policy.append(hidden_state_nodes_policy_pool[iy][next_node_idx])
                else:
                    next_visited_nodes_policy.append(None)

            last_actions_t = torch.from_numpy(np.asarray(last_actions)).to(device).long()
            hidden_states = Batch.from_data_list(currently_visited_nodes_obs).to(device)
            batch_actions = format_batch_actions(last_actions_t, hidden_states.nb_variables)

            with amp_autocast():
                left_output, right_output = model.recurrent_inference(hidden_states, batch_actions)

            left_output_v = left_output.value.reshape(-1).tolist()
            left_output_b = left_output.breward.reshape(-1).tolist()
            left_output_p = left_output.policy_logits.tolist()
            left_output_o = left_output.hidden_state.to_data_list()
            right_output_v = right_output.value.reshape(-1).tolist()
            right_output_b = right_output.breward.reshape(-1).tolist()
            right_output_p = right_output.policy_logits.tolist()
            right_output_o = right_output.hidden_state.to_data_list()

            next_values = []
            next_logits = []
            next_states = []

            for iy in range(batch_size):
                new_open_nodes_obs = []
                new_open_nodes_value = []
                new_open_nodes_policy = []
                new_open_nodes_breward = []
                dfs = False

                left_node_v, right_node_v = left_output_v[iy], right_output_v[iy]
                left_node_b, right_node_b = left_output_b[iy], right_output_b[iy]
                left_node_p, right_node_p = left_output_p[iy], right_output_p[iy]
                left_node_o, right_node_o = left_output_o[iy], right_output_o[iy]

                next_state_open_nodes_idx = current_state_open_nodes[iy][:]
                next_state_open_nodes_idx.pop(0)

                if right_node_b > threshold:
                    policy_logits = right_node_p
                    right_node_idx = len(hidden_state_open_nodes_obs_pool[iy])
                    next_state_open_nodes_idx.insert(0, right_node_idx)
                    new_open_nodes_obs.append(right_node_o)
                    new_open_nodes_value.append(right_node_v)
                    new_open_nodes_policy.append(right_node_p)
                    new_open_nodes_breward.append(right_node_b)
                    dfs = True

                if left_node_b > threshold:
                    policy_logits = left_node_p
                    left_node_idx = len(hidden_state_open_nodes_obs_pool[iy]) + int(dfs)
                    next_state_open_nodes_idx.insert(0, left_node_idx)
                    new_open_nodes_obs.append(left_node_o)
                    new_open_nodes_value.append(left_node_v)
                    new_open_nodes_policy.append(left_node_p)
                    new_open_nodes_breward.append(left_node_b)
                    dfs = True

                if not dfs:
                    policy_logits = next_visited_nodes_policy[iy]
                    # Terminal states --> absorbing states, pretend nothing happened.
                    if len(next_state_open_nodes_idx) == 0:
                        policy_logits = left_node_p
                        left_node_idx = len(hidden_state_open_nodes_obs_pool[iy])
                        next_state_open_nodes_idx.insert(0, left_node_idx)
                        new_open_nodes_obs.append(left_node_o)
                        new_open_nodes_value.append(left_node_v)
                        new_open_nodes_policy.append(left_node_p)
                        new_open_nodes_breward.append(left_node_b)

                r = -1 if currently_visited_nodes_breward[iy] > threshold else 0
                state_v = current_state_value[iy]
                current_node_v = currently_visited_nodes_value[iy]
                update_state_v = state_v + r + left_node_v + right_node_v - current_node_v
                next_values.append(update_state_v)
                next_logits.append(policy_logits[: action_nums[iy]])

                if len(new_open_nodes_obs) > 0:
                    new_open_nodes_obs.reverse()
                    new_open_nodes_value.reverse()
                    new_open_nodes_policy.reverse()
                    new_open_nodes_breward.reverse()
                    hidden_state_open_nodes_obs_pool[iy] += new_open_nodes_obs
                    hidden_state_nodes_value_pool[iy] += new_open_nodes_value
                    hidden_state_nodes_policy_pool[iy] += new_open_nodes_policy
                    hidden_state_nodes_breward_pool[iy] += new_open_nodes_breward

                next_states.append(next_state_open_nodes_idx)

            hidden_state_index_x += 1
            state_pool.append(next_states)
            hidden_state_value_pool.append(next_values)
            mcts_info[simulation_idx] = {
                "next_states": next_states,
                "next_values": next_values,
                "next_logits": next_logits,
            }

            batch_back_propagate(
                hidden_state_index_x,
                next_values,
                next_logits,
                min_max_stats_lst,
                search_paths,
                action_nums,
            )

            if self.ready_for_next_gumble_phase(simulation_idx):
                for i in range(batch_size):
                    sequential_halving(roots.roots[i], gumble_noises[i], min_max_stats_lst[i], self.current_num_top_actions)

        search_root_values = np.asarray(roots.get_values())
        search_root_policies = roots.get_root_policies(min_max_stats_lst)
        search_root_policies = np.asarray(
            [
                policy + [0.0] * (max(action_nums) - action_num)
                for policy, action_num in zip(search_root_policies, action_nums)
            ]
        )
        search_best_actions = np.asarray(roots.get_best_actions())
        return search_root_values, search_root_policies, search_best_actions, mcts_info
