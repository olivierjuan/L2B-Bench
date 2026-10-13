"""Gasse 2019 — the GCNN on the bipartite LP graph.

Faithful port of ``model/model.py`` from the official repository
(github.com/ds4dm/learn2branch-ecole): the same architecture, layer for
layer, with the same names.

- ``PreNormLayer`` — data-dependent normalization: statistics are collected
  over the first training batches, then frozen. The pre-training loop in
  :mod:`l2b_bench.methods.gasse.train` drives it exactly like the official
  ``03_train_gnn.py``.
- ``BipartiteGraphConvolution`` — one message-passing round between the two
  node types (variables, constraints) through the constraint matrix's
  coefficients.
- ``GNNPolicy`` — variable embedding → conv v→c → conv c→v → per-variable
  score. Input feature dimensions are the paper's (19 variable, 5 constraint,
  1 edge), matching gyozas' ``NodeBipartite`` observation.
"""

import torch
import torch_geometric


class PreNormException(Exception):
    """Raised by a :class:`PreNormLayer` still collecting its statistics."""


class PreNormLayer(torch.nn.Module):
    """Data-dependent normalization layer (shift and/or scale frozen after pre-training)."""

    def __init__(self, n_units: int, shift: bool = True, scale: bool = True, name=None) -> None:
        super().__init__()
        assert shift or scale
        self.register_buffer("shift", torch.zeros(n_units) if shift else None)
        self.register_buffer("scale", torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self) -> None:
        """Begin (or resume) collecting statistics for the shift and scale."""
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_) -> None:
        """Online mean and variance estimation (Chan et al., 1979)."""
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, (
            f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."
        )

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = input_.numel() / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta**2 * self.count * sample_count / (
            self.count + sample_count
        )

        self.count += sample_count
        self.avg = self.avg + delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self) -> None:
        """End pre-training for this layer and freeze its shift and scale."""
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.received_updates = False


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """One bipartite message-passing round: left features → right features."""

    def __init__(self) -> None:
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False),
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False),
        )
        self.feature_module_final = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(
            PreNormLayer(1, shift=False),
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class BaseModel(torch.nn.Module):
    """Base model class implementing the PreNorm pre-training protocol."""

    def pre_train_init(self) -> None:
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs) -> bool:
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True


class GNNPolicy(BaseModel):
    """The paper's branching policy network (Gasse et al., NeurIPS 2019).

    Input feature dimensions are the paper's and match gyozas'
    ``NodeBipartite`` observation: 19 variable features, 5 constraint
    features, 1 edge feature per nonzero.
    """

    def __init__(self) -> None:
        super().__init__()
        emb_size = 64
        cons_nfeats = 5
        edge_nfeats = 1
        var_nfeats = 19

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        output = self.output_module(variable_features).squeeze(-1)
        return output
