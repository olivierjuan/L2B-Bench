"""RetroBranching (Parsonson, Laterza & Barrett 2022) — the DQN value network.

Faithful port of ``src/networks/bipartite_gcn.py`` and
``bipartite_graph_convolution.py`` from the official repository
(github.com/cwfparsonson/retro_branching), with the ``dqn_value_network.yaml``
configuration baked in: 64-dim embeddings, one message-passing round, add
aggregation, edge features included, one head of depth 2, and the
``inverse_leaky_relu`` output activation (a LeakyReLU whose output is then
negated — non-positive whenever the head output is, a small positive
otherwise). Initialization is the official's:
``normal(0, 0.01)`` on every Linear weight, zeros on every bias.

Unlike Gasse's GCNN there is no data-dependent PreNorm pre-training — this
network trains from scratch under RL, with plain ``torch.nn.LayerNorm``.
"""

import torch
import torch_geometric

#: Input feature dimensions of the ``43_var_features`` observation.
CON_NFEATS = 5
EDGE_NFEATS = 1
VAR_NFEATS = 43

EMB_SIZE = 64


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """One bipartite message-passing round, the official layer for layer."""

    def __init__(self, emb_size: int = EMB_SIZE) -> None:
        super().__init__("add")

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(EDGE_NFEATS, emb_size, bias=False),
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False),
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(
                self.feature_module_left(left_features),
                self.feature_module_right(right_features),
            ),
            edge_features=self.feature_module_edge(edge_features),
        )
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        return self.feature_module_final(node_features_i + node_features_j + edge_features)


class RetroGCN(torch.nn.Module):
    """The paper's value network: one Q value per variable.

    Consumes the ``NodeBipartiteRetro`` observation (5 constraint features,
    43 variable features, 1 edge feature per nonzero); only the Q values of
    the node's branching candidates are meaningful.
    """

    def __init__(self, emb_size: int = EMB_SIZE) -> None:
        super().__init__()

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(CON_NFEATS),
            torch.nn.Linear(CON_NFEATS, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(EDGE_NFEATS),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(VAR_NFEATS),
            torch.nn.Linear(VAR_NFEATS, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        # HEAD (num_heads=1, head_depth=2)
        self.heads_module = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, emb_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(emb_size, 1, bias=True),
                )
            ]
        )

        # linear_weight_init='normal' (std 0.01), linear_bias_init='zeros'
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # num_rounds=1
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        head_output = [head(variable_features).squeeze(-1) for head in self.heads_module]
        # head_aggregator='add' over one head; activation='inverse_leaky_relu'
        head_output = torch.stack(head_output, dim=0).sum(dim=0)
        return -torch.nn.functional.leaky_relu(head_output)
