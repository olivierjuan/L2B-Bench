"""BBMDP 2025 — the prioritized replay buffer.

Faithful port of ``rl/agents/{segment_tree,replay_buffer}.py`` from the
official repository (the OpenAI-baselines segment trees, verbatim). A
transition stores, in addition to the state/action/reward triple, its *tree*
next states: the lists of children states of the 1-step and the n-step
frontier (references into the episode's observations — no copies), so a
mini-batch pads the children into rectangular child slots with a mask.

One deliberate deviation from the official code: ``initialize_fill_state``
there zero-ises the constraint and variable features of ``state_buf[0]``
*in place* — the first stored transition is silently corrupted for the rest
of the run. Here the fill state is a derived copy (same edge structure and
action set, zero features); the stored transition keeps its real features.
The fill state itself is byte-for-byte what the official network sees.
"""

import operator
import random
from collections.abc import Callable

import numpy as np


class SegmentTree:
    """OpenAI baselines' segment tree (point update, range query)."""

    def __init__(self, capacity: int, operation: Callable, init_value: float) -> None:
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        if mid + 1 <= start:
            return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
        return self.operation(
            self._operate_helper(start, mid, 2 * node, node_start, mid),
            self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
        )

    def operate(self, start: int = 0, end: int = 0) -> float:
        if end <= 0:
            end += self.capacity
        end -= 1
        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        assert 0 <= idx < self.capacity
        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, operation=operator.add, init_value=0.0)

    def sum(self, start: int = 0, end: int = 0) -> float:
        return self.operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        assert 0 <= upperbound <= self.sum() + 1e-5, f"upperbound: {upperbound}"
        idx = 1
        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, operation=min, init_value=float("inf"))

    def min(self, start: int = 0, end: int = 0) -> float:
        return self.operate(start, end)


class ReplayBuffer:
    """The plain ring buffer; BBMDP transitions are stored as objects."""

    def __init__(self, size: int) -> None:
        self.buf = np.empty(size, dtype=object)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, transition) -> None:
        self.buf[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def __len__(self) -> int:
        return self.size


class _ZeroFeatureObservation:
    """The fill state's observation: the first stored state's edge structure
    and action set with zeroed constraint/variable features."""

    def __init__(self, observation) -> None:
        self.row_features = np.zeros_like(observation.row_features)
        self.edge_features = observation.edge_features
        self.variable_features = np.zeros_like(observation.variable_features)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportionally prioritized replay (OpenAI baselines' PER)."""

    def __init__(self, size: int, batch_size: int = 32, alpha: float = 0.6) -> None:
        super().__init__(size)
        assert alpha >= 0
        self.batch_size = batch_size
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        self.fill_state = None
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, transition) -> None:
        super().store(transition)
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4, rng: random.Random | None = None) -> dict:
        """One prioritized batch, with the tree next states padded into
        rectangular child slots (per 1-step and n-step)."""
        assert len(self) >= self.batch_size
        assert beta > 0
        rng = rng or random

        indices = self._sample_proportional(rng)
        transitions = self.buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        next_states, children_mask = self._preprocess_next_states([t.next_states for t in transitions])
        n_step_next_states, n_step_children_mask = self._preprocess_next_states(
            [t.n_step_next_states for t in transitions]
        )
        return {
            "transitions": transitions,
            "next_states": next_states,
            "children_mask": children_mask,
            "n_step_next_states": n_step_next_states,
            "n_step_children_mask": n_step_children_mask,
            "weights": weights,
            "indices": indices,
        }

    def update_priorities(self, indices, priorities) -> None:
        assert len(indices) == len(priorities)
        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, rng: random.Random) -> list[int]:
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        for i in range(self.batch_size):
            upperbound = rng.uniform(segment * i, segment * (i + 1))
            indices.append(self.sum_tree.retrieve(upperbound))
        return indices

    def _calculate_weight(self, idx: int, beta: float) -> float:
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        return weight / max_weight

    def _preprocess_next_states(self, next_states: list[list]):
        """Pad the children lists to one rectangle of child slots; returns the
        per-slot state lists and the [max_children, batch] mask."""
        batch_size = len(next_states)
        max_children = max(len(children) for children in next_states)
        if max_children == 0:
            return None, None
        if self.fill_state is None:
            # The official fill state: the first stored state's graph with
            # zeroed features (derived as a copy here, not mutated in place).
            first_observation, first_action_set = self.buf[0].state
            self.fill_state = (_ZeroFeatureObservation(first_observation), first_action_set)
        children_mask = np.ones((max_children, batch_size))
        padded = []
        for i, children in enumerate(next_states):
            padded.append(children + [self.fill_state] * (max_children - len(children)))
            children_mask[len(children) :, i] = 0
        # [child slot][batch] -> per-slot batch of states.
        per_slot = [[padded[j][i] for j in range(batch_size)] for i in range(max_children)]
        return per_slot, children_mask
