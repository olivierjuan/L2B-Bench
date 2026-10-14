"""T-BranT (Lin 2022) — the branching transformer.

Port of ``BranchFormer`` (graph=False) from the official repository
(github.com/linjc16/TBranT, ``models/transformer.py`` + ``models/modules.py``)
— the LTBranT variant of the paper: a transformer encoder over the
candidate-variable features, gated by the tree state (the node + MIP feature
blocks), with the paper's Bidirectional-Matching head combining the tree
state with each candidate embedding. The full T-BranT adds a GAT over the
whole branching tree; that variant is future work (it needs the per-tree-node
history collected by the official repo).

Inputs (batch dimension N, max candidate count S):

- ``cands_state_mat``: (N, S, 25) per-candidate solver statistics, padded;
- ``padding_mask``: (N, S) — 1 where the candidate slot is padding;
- ``node_state``: (N, 8) current-node features;
- ``mip_state``: (N, 53) search-tree features.

Returns the candidate logits (N, S), ``-inf`` on padded slots.

The defaults are the official ``train_LTBranT.sh`` hyperparameters: hidden
size 80, 8 heads, 2 encoder layers, tree gating on, dropout 0.1,
feed-forward 256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CANDIDATE_FEATURES = 25
N_NODE_FEATURES = 8
N_MIP_FEATURES = 53


class BiMatchingNet(nn.Module):
    """The paper's Bidirectional-Matching head.

    Matches the tree state against each candidate embedding in both
    directions (tree→candidates attention and candidates→tree attention),
    then blends the two aligned representations through a sigmoid gate.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1_1 = nn.Linear(hidden_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, tree_feat: torch.Tensor, var_feat: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """``tree_feat`` (N, E); ``var_feat`` (N, S, E); ``padding_mask`` (N, S)."""
        tree_feat = tree_feat.unsqueeze(1)

        g_tc = torch.bmm(self.linear1_1(tree_feat), var_feat.transpose(1, 2)).squeeze(1)
        g_tc = g_tc.masked_fill(padding_mask, float("-inf"))
        g_tc = F.softmax(g_tc, dim=1).unsqueeze(1)  # (N, 1, S)

        g_ct = torch.bmm(self.linear1_2(var_feat), tree_feat.transpose(1, 2)).squeeze(2)
        g_ct = g_ct.masked_fill(padding_mask, float("-inf"))
        g_ct = F.softmax(g_ct, dim=1).unsqueeze(2)  # (N, S, 1)

        e_t = torch.bmm(g_tc, var_feat)  # (N, 1, E)
        e_c = torch.bmm(g_ct, tree_feat)  # (N, S, E)

        s_tc = F.relu(self.linear2_1(e_t))
        s_ct = F.relu(self.linear2_2(e_c))

        attn_weight = torch.sigmoid(self.linear3_1(s_tc) + self.linear3_2(s_ct))  # (N, S, E)
        return attn_weight * s_tc + (1 - attn_weight) * s_ct


class BranchFormer(nn.Module):
    """The LTBranT branching transformer (Lin et al., 2022)."""

    def __init__(
        self,
        var_dim: int = N_CANDIDATE_FEATURES,
        node_dim: int = N_NODE_FEATURES,
        mip_dim: int = N_MIP_FEATURES,
        hidden_size: int = 80,
        dropout_rate: float = 0.1,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        tree_gate: bool = True,
    ) -> None:
        super().__init__()
        self.tree_gate = tree_gate

        self.candidate_embedding_net = nn.Sequential(
            nn.LayerNorm(var_dim), nn.Linear(var_dim, hidden_size)
        )
        self.tree_embedding_net = nn.Sequential(
            nn.LayerNorm(node_dim + mip_dim), nn.Linear(node_dim + mip_dim, hidden_size)
        )
        self.global_embedding_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(hidden_size))

        self.bi_matching_net = BiMatchingNet(hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_size, 1)

        # Xavier initialization for the linear layers, as the official code.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        cands_state_mat: torch.Tensor,
        padding_mask: torch.Tensor,
        node_state: torch.Tensor,
        mip_state: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate logits (N, S), ``-inf`` on padded slots."""
        cands = self.candidate_embedding_net(cands_state_mat)  # (N, S, H)
        tree_state = self.tree_embedding_net(torch.cat((node_state, mip_state), dim=1))  # (N, H)

        if self.tree_gate:
            cands = torch.cat(
                (cands, tree_state.unsqueeze(1).repeat(1, cands.size(1), 1)), dim=2
            )
            cands = self.global_embedding_net(cands)

        cands = cands.transpose(0, 1)  # (S, N, H)
        padding = padding_mask == 1
        cands_embedding = self.transformer(cands, src_key_padding_mask=padding)
        cands_embedding = cands_embedding.transpose(0, 1)  # (N, S, H)

        output = self.bi_matching_net(tree_state, cands_embedding, padding)
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.linear2(output).squeeze(-1)  # (N, S)
        return output.masked_fill(padding, float("-inf"))
