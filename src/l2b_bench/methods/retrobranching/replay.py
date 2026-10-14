"""Uniform experience replay for the RetroBranching DQN.

Port of the official ``ReplayBuffer`` (``src/learners/dqn_learner.py``) under
its ``dqn_learner.yaml`` configuration: uniform sampling, no prioritization,
no combined experience replay — a fixed-capacity ring over agent transitions.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GraphState:
    """One observation as tensors, plus its branching candidates.

    The DQN's state: the bipartite graph of the node being branched on and
    the LP-fractional variables one may branch on.
    """

    constraint_features: torch.Tensor  # (n_cons, 5)
    edge_index: torch.Tensor  # (2, n_edges) — constraint-major
    edge_attr: torch.Tensor  # (n_edges, 1)
    variable_features: torch.Tensor  # (n_vars, 43)
    candidates: torch.Tensor  # (n_cands,) — variable ids


def graph_state(observation, action_set) -> GraphState:
    """Extract the tensors of one gyozas ``NodeBipartiteRetro`` observation."""
    return GraphState(
        constraint_features=torch.as_tensor(observation.row_features, dtype=torch.float32),
        edge_index=torch.as_tensor(observation.edge_features.indices, dtype=torch.long),
        edge_attr=torch.as_tensor(observation.edge_features.values, dtype=torch.float32).reshape(-1, 1),
        variable_features=torch.as_tensor(observation.variable_features, dtype=torch.float32),
        candidates=torch.as_tensor(np.asarray(action_set, dtype=np.int64)),
    )


@dataclass(frozen=True)
class Transition:
    """One DQN transition."""

    state: GraphState
    action: int  # the branched variable id (raw, not the candidate index)
    reward: float
    done: bool
    next_state: GraphState


class ReplayBuffer:
    """A fixed-capacity ring buffer with uniform sampling (the official one)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[Transition | None] = [None] * capacity
        self._write_idx = 0
        self.available = 0

    def append(self, transition: Transition) -> None:
        self.buffer[self._write_idx] = transition
        self._write_idx = (self._write_idx + 1) % self.capacity
        self.available = min(self.available + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        indices = rng.choice(self.available, size=batch_size, replace=False)
        return [self.buffer[int(i)] for i in indices]

    def __len__(self) -> int:
        return self.available
