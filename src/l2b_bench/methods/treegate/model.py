"""TreeGate (Zarpellon 2021) — the feature-gated feedforward policies.

Port of ``models/feedforward.py`` + ``models/modules.py`` from the official
repository (github.com/ds4dm/branch-search-trees):

- ``NoTreePolicy``: a per-candidate MLP whose output layer is averaged over
  the feature axis into one score per candidate — no tree state input;
- ``TreeGatePolicy``: the same pyramid, but every unit of every layer is
  modulated by a sigmoid gate computed from the 61-dim tree state (node + MIP
  features) — the paper's *feature gating*.

Both consume **one node at a time** (no batch dimension, like the official
forward): ``cands_state_mat`` (S, 25), ``node_state`` (8,), ``mip_state``
(53,) → candidate scores (S, 1). The official loss transposes to (1, S) and
applies cross-entropy against the expert's chosen candidate.

The defaults are the released ``TreeGatePolicy_hidden64_depth5_lr0.01``
checkpoint's hyperparameters: hidden 64, gating depth 5, dimension-reduce
factor 2, no normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CANDIDATE_FEATURES = 25
N_NODE_FEATURES = 8
N_MIP_FEATURES = 53


def _get_norm_layer(norm: str):
    if norm == "batch":
        return nn.BatchNorm1d
    if norm == "instance":
        return lambda dim: nn.InstanceNorm1d(dim, affine=False, track_running_stats=False)
    if norm == "layer":
        return nn.LayerNorm
    if norm == "none":
        return nn.Identity
    raise NotImplementedError(f"normalization layer [{norm}] is not found")


class TreeGateBranchingNet(nn.Module):
    """The paper's feature-gated branching network.

    A pyramid of layers shrinking ``hidden_size`` by ``dim_reduce_factor``
    down to ``infimum`` units; a sigmoid ``GatingNet`` maps the tree state to
    one gate per unit of every layer, and each layer's input is multiplied by
    its slice of gates before the linear map.
    """

    def __init__(
        self,
        branch_size: int,
        tree_state_size: int,
        dim_reduce_factor: int,
        infimum: int = 8,
        norm: str = "none",
        depth: int = 5,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm)
        self.dim_reduce_factor = dim_reduce_factor
        self.depth = depth

        # The shrinking pyramid: hidden -> hidden/2 -> ... down to infimum.
        self.n_layers = 0
        unit_count = infimum
        while unit_count < branch_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1
        self.n_units = dict.fromkeys(range(self.n_layers))
        self.branching_net = nn.ModuleList()
        input_dim = hidden_size
        for i in range(self.n_layers):
            output_dim = int(input_dim / dim_reduce_factor)
            self.n_units[i] = input_dim
            if i < self.n_layers - 1:
                layer = [nn.Linear(input_dim, output_dim), norm_layer(output_dim), nn.ReLU(True)]
            else:
                layer = [nn.Linear(input_dim, output_dim)]  # dense output
            input_dim = output_dim
            self.branching_net.append(nn.Sequential(*layer))

        # The gating network: tree state -> one sigmoid gate per unit.
        n_gated_units = sum(self.n_units.values())
        if depth == 1:
            self.gating_net = nn.Sequential(
                nn.Linear(tree_state_size, n_gated_units), nn.Sigmoid()
            )
        else:
            layers = [nn.Linear(tree_state_size, hidden_size), nn.ReLU(True)]
            for _ in range(depth - 2):
                layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(True)]
            layers += [nn.Linear(hidden_size, n_gated_units), nn.Sigmoid()]
            self.gating_net = nn.Sequential(*layers)

    def forward(self, cands_state_mat: torch.Tensor, tree_state: torch.Tensor) -> torch.Tensor:
        """``cands_state_mat`` (S, H); ``tree_state`` (node_dim + mip_dim,) → (S, 1)."""
        gates = self.gating_net(tree_state)
        start = 0
        for index, layer in enumerate(self.branching_net):
            end = start + self.n_units[index]
            cands_state_mat = cands_state_mat * gates[start:end]  # no in-place ops
            cands_state_mat = layer(cands_state_mat)
            start = end
        return cands_state_mat.mean(dim=1, keepdim=True)  # keep the axis


class NoTreePolicy(nn.Module):
    """The ablation: the same pyramid without any tree-state input."""

    def __init__(
        self,
        var_dim: int = N_CANDIDATE_FEATURES,
        node_dim: int = N_NODE_FEATURES,
        mip_dim: int = N_MIP_FEATURES,
        hidden_size: int = 64,
        dim_reduce_factor: int = 2,
        infimum: int = 8,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.candidate_embedding_net = nn.Sequential(nn.Linear(var_dim, hidden_size))

        self.n_layers = 0
        unit_count = infimum
        while unit_count < hidden_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1
        layers = []
        input_dim = hidden_size
        for i in range(self.n_layers):
            output_dim = int(input_dim / dim_reduce_factor)
            if i < self.n_layers - 1:
                layers += [nn.Linear(input_dim, output_dim), _get_norm_layer(norm)(output_dim), nn.ReLU(True)]
            else:
                layers += [nn.Linear(input_dim, output_dim)]
            input_dim = output_dim
        self.branching_net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        cands_state_mat: torch.Tensor,
        node_state: torch.Tensor | None = None,
        mip_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``cands_state_mat`` (S, 25) → candidate scores (S, 1)."""
        cands_state_mat = self.candidate_embedding_net(cands_state_mat)
        cands_state_mat = self.branching_net(cands_state_mat)
        return cands_state_mat.mean(dim=1, keepdim=True)  # keep the axis


class TreeGatePolicy(nn.Module):
    """TreeGate: the pyramid modulated by sigmoid gates from the tree state."""

    def __init__(
        self,
        var_dim: int = N_CANDIDATE_FEATURES,
        node_dim: int = N_NODE_FEATURES,
        mip_dim: int = N_MIP_FEATURES,
        hidden_size: int = 64,
        depth: int = 5,
        dim_reduce_factor: int = 2,
        infimum: int = 8,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.candidate_embedding_net = nn.Sequential(nn.Linear(var_dim, hidden_size))
        self.tree_gate_branching_net = TreeGateBranchingNet(
            hidden_size,
            node_dim + mip_dim,
            dim_reduce_factor,
            infimum,
            norm,
            depth,
            hidden_size,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        cands_state_mat: torch.Tensor,
        node_state: torch.Tensor,
        mip_state: torch.Tensor,
    ) -> torch.Tensor:
        """``cands_state_mat`` (S, 25), ``node_state`` (8,), ``mip_state`` (53,) → (S, 1)."""
        cands_state_mat = self.candidate_embedding_net(cands_state_mat)
        return self.tree_gate_branching_net(
            cands_state_mat, torch.cat((node_state, mip_state))
        )
