"""SORREL 2024 — the actor and the critic are Gasse's GCNN, verbatim.

The official ``brancher/networks.py`` reuses Gasse's ``GNNPolicy`` for both
networks; the only addition is ``action_logits`` — per-variable scores split
per sample over its candidates and padded to one rectangular block with
``-inf`` (a softmax over the block is then a distribution over the node's
candidates alone). Here the same padding is a helper over the manual
collation, and every sample additionally carries its two children's graphs:
an unvisited child becomes the official dummy graph — one constraint, one
variable, no edges, one candidate — so a whole mini-batch collates into
three disconnected bipartite graphs (state, child 1, child 2).
"""

import copy

import numpy as np
import torch

from ..gasse.model import GNNPolicy  # noqa: F401 — the official re-export
from .collect import SorrelSample

N_ROW_FEATURES = 5
N_VAR_FEATURES = 19


def _dummy_graph() -> dict[str, torch.Tensor]:
    """The official ``build_graph(None, None)``: one constraint, one variable,
    no edges, one candidate (the dummy variable itself)."""
    return {
        "constraint_features": torch.zeros(1, N_ROW_FEATURES),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "edge_attr": torch.empty(0, 1),
        "variable_features": torch.zeros(1, N_VAR_FEATURES),
        "candidates": [torch.zeros(1, dtype=torch.long)],
    }


def _graph_tensors(observation, action_set) -> dict[str, torch.Tensor]:
    return {
        "constraint_features": torch.as_tensor(observation.row_features, dtype=torch.float32),
        "edge_index": torch.as_tensor(observation.edge_features.indices, dtype=torch.long),
        "edge_attr": torch.as_tensor(observation.edge_features.values, dtype=torch.float32).reshape(-1, 1),
        "variable_features": torch.as_tensor(observation.variable_features, dtype=torch.float32),
        "candidates": [torch.as_tensor(np.asarray(action_set), dtype=torch.long)],
    }


class _GraphGroup:
    """A mini-batch of bipartite graphs collated into one disconnected graph
    (official torch-geometric batching, by hand like the other methods)."""

    def __init__(self, graphs: list[dict[str, torch.Tensor]]) -> None:
        n_vars = [g["variable_features"].shape[0] for g in graphs]
        edge_index = []
        cons_offset = 0
        var_offset = 0
        for graph, n_v in zip(graphs, n_vars, strict=True):
            edge_index.append(graph["edge_index"] + torch.tensor([[cons_offset], [var_offset]], dtype=torch.long))
            cons_offset += graph["constraint_features"].shape[0]
            var_offset += n_v
        self.constraint_features = torch.cat([g["constraint_features"] for g in graphs])
        self.edge_index = torch.cat(edge_index, dim=1)
        self.edge_attr = torch.cat([g["edge_attr"] for g in graphs])
        self.variable_features = torch.cat([g["variable_features"] for g in graphs])
        # Each sample's candidates carry its variable offset in the collated graph.
        self.candidates = [graph["candidates"][0] + sum(n_vars[:i]) for i, graph in enumerate(graphs)]

    def forward_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.constraint_features,
            self.edge_index,
            self.edge_attr,
            self.variable_features,
        )

    def to(self, device: torch.device) -> "_GraphGroup":
        """A moved copy — the original stays where it is. The training
        loops keep CPU-resident batches and materialize one device batch
        at a time; a mutating move would pin the whole list on the GPU
        (the campaign OOM)."""
        moved = copy.copy(self)
        moved.constraint_features = self.constraint_features.to(device)
        moved.edge_index = self.edge_index.to(device)
        moved.edge_attr = self.edge_attr.to(device)
        moved.variable_features = self.variable_features.to(device)
        moved.candidates = [c.to(device) for c in self.candidates]
        return moved


class SorrelBatch:
    """A mini-batch of tree-MDP transitions: the states' collated graph plus
    the two children's (unvisited children as the official dummy graph)."""

    def __init__(self, samples: list[SorrelSample]) -> None:
        states = [
            _graph_tensors(s.observation, s.action_set) if s.observation is not None else _dummy_graph()
            for s in samples
        ]
        children1 = [
            _graph_tensors(s.child1_obs, s.child1_action_set)
            if s.child1_obs is not None
            else _dummy_graph()
            for s in samples
        ]
        children2 = [
            _graph_tensors(s.child2_obs, s.child2_action_set)
            if s.child2_obs is not None
            else _dummy_graph()
            for s in samples
        ]
        self.states = _GraphGroup(states)
        self.children1 = _GraphGroup(children1)
        self.children2 = _GraphGroup(children2)
        self.actions = torch.tensor(
            [int(np.where(np.asarray(s.action_set) == s.action)[0][0]) for s in samples], dtype=torch.long
        )
        self.rewards1 = torch.tensor([s.reward1 for s in samples], dtype=torch.float32)
        self.rewards2 = torch.tensor([s.reward2 for s in samples], dtype=torch.float32)
        # A child is done when it was never visited.
        self.dones1 = torch.tensor([s.child1_obs is None for s in samples], dtype=torch.long)
        self.dones2 = torch.tensor([s.child2_obs is None for s in samples], dtype=torch.long)
        self.returns = torch.tensor(
            [s.returns if s.returns is not None else 0.0 for s in samples], dtype=torch.float32
        )
        self.log_probs = torch.tensor(
            [s.log_prob if s.log_prob is not None else 0.0 for s in samples], dtype=torch.float32
        )

    def to(self, device: torch.device) -> "SorrelBatch":
        """A moved copy (see ``_GraphGroup.to``) — the original stays on the
        CPU so a whole epoch's batches never sit on the GPU at once."""
        moved = copy.copy(self)
        moved.states = self.states.to(device)
        moved.children1 = self.children1.to(device)
        moved.children2 = self.children2.to(device)
        moved.actions = self.actions.to(device)
        moved.rewards1 = self.rewards1.to(device)
        moved.rewards2 = self.rewards2.to(device)
        moved.dones1 = self.dones1.to(device)
        moved.dones2 = self.dones2.to(device)
        moved.returns = self.returns.to(device)
        moved.log_probs = self.log_probs.to(device)
        return moved


def padded_logits(logits: torch.Tensor, candidates: list[torch.Tensor]) -> torch.Tensor:
    """The official ``action_logits``: per-variable scores split per sample
    over its candidate slice, padded to one block with ``-inf``."""
    blocks = [logits[c] for c in candidates]
    return torch.nn.utils.rnn.pad_sequence(blocks, batch_first=True, padding_value=-float("inf"))
