"""TGPPO 2025 (Tree-Gate PPO) — the actor-critic transformer.

Port of ``project/policy/{actor,critic,modules}.py`` from the official
repository (github.com/benmhamed-a/tgppo-aaai-26):

- both nets embed the 25-dim candidate statistics (LayerNorm + Linear) and
  the concatenated 8+53 node/tree state, and encode the candidate set with a
  ``TransformerEncoder`` (GELU, feedforward 4H, candidate padding masked);
- a ``BiMatchingNet`` co-attends the tree state and the encoded candidates
  (bidirectional soft attention followed by a sigmoid-gated mixture);
- the output head is TreeGate's own gated pyramid (``TreeGateBranchingNet``,
  depth 2, layer norm, infimum 1, dimension-reduce factor 2): the actor reads
  one score per candidate, the critic one state value — its head mean-pools
  the refined candidate features over the valid (non-padded) candidates.

The actor returns action *probabilities* (softmax over candidates, pads
masked out); the official PPO loss consumes probabilities, not logits.
"""

import torch
import torch.nn.functional as F
from torch import nn

N_CANDIDATE_FEATURES = 25
N_NODE_FEATURES = 8
N_MIP_FEATURES = 53


def _get_norm_layer(norm: str):
    if norm == "batch":
        return nn.BatchNorm1d
    if norm == "instance":
        return lambda dim: nn.InstanceNorm1d(
            dim, affine=False, track_running_stats=False
        )
    if norm == "layer":
        return nn.LayerNorm
    if norm == "none":
        return nn.Identity
    raise NotImplementedError(f"normalization layer [{norm}] is not found")


class TreeGateBranchingNet(nn.Module):
    """TreeGate's feature-gated pyramid, in TGPPO's batched variant.

    A stack of layers shrinking ``hidden_size`` by ``dim_reduce_factor``
    toward ``infimum`` units; a sigmoid ``GatingNet`` maps the tree state to
    one gate per unit of every layer, and each layer's input is multiplied by
    its slice of gates before the linear map. With a 3D input the output is
    mean-pooled over the candidate axis (the critic's value head); with a 2D
    input (one node) the last layer outputs one score per candidate.
    """

    def __init__(
        self,
        branch_size: int,
        tree_state_size: int,
        dim_reduce_factor: int,
        infimum: int = 1,
        norm: str = "layer",
        depth: int = 2,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        norm_layer = _get_norm_layer(norm)
        self.branch_size = branch_size
        self.dim_reduce_factor = dim_reduce_factor

        self.n_layers = 0
        unit_count = infimum
        while unit_count < branch_size:
            unit_count *= dim_reduce_factor
            self.n_layers += 1
        self.n_units: dict[int, int] = {}

        self.branching_net = nn.ModuleList()
        input_dim = hidden_size
        for i in range(self.n_layers):
            output_dim = max(int(input_dim / dim_reduce_factor), 1)
            self.n_units[i] = input_dim
            if i < self.n_layers - 1:
                layer = [
                    nn.Linear(input_dim, output_dim),
                    norm_layer(output_dim),
                    nn.ReLU(True),
                ]
            else:
                layer = [nn.Linear(input_dim, output_dim)]
            input_dim = output_dim
            self.branching_net.append(nn.Sequential(*layer))

        self.n_attentional_units = sum(self.n_units.values())
        gating: list[nn.Module] = []
        if depth == 1:
            gating.append(nn.Linear(tree_state_size, self.n_attentional_units))
        else:
            gating.append(nn.Linear(tree_state_size, hidden_size))
            gating.append(nn.ReLU(True))
            for i in range(depth - 2):
                gating.append(nn.Linear(hidden_size, hidden_size))
                gating.append(nn.ReLU(True))
            gating.append(nn.Linear(hidden_size, self.n_attentional_units))
        gating.append(nn.Sigmoid())
        self.gating_net = nn.Sequential(*gating)

    def forward(
        self, cands_state_mat: torch.Tensor, tree_state: torch.Tensor
    ) -> torch.Tensor:
        attn_weights = self.gating_net(tree_state)
        start = 0
        for index, layer in enumerate(self.branching_net):
            end = start + self.n_units[index]
            attn_slice = attn_weights[:, start:end]
            if cands_state_mat.dim() == 3:
                cands_state_mat = cands_state_mat * attn_slice.unsqueeze(1)
            else:
                cands_state_mat = cands_state_mat * attn_slice
            cands_state_mat = layer(cands_state_mat)
            start = end
        if cands_state_mat.size(-1) == 1:
            return cands_state_mat.squeeze(-1)
        # 3D -> mean over candidates (the critic's value); 2D -> one score
        # per candidate (the actor's head before softmax).
        dim = 1 if cands_state_mat.dim() == 3 else -1
        return cands_state_mat.mean(dim=dim)


class BiMatchingNet(nn.Module):
    """Bidirectional tree/candidate matching (the official six-linear-block)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1_1 = nn.Linear(hidden_size, hidden_size)
        self.linear1_2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_1 = nn.Linear(hidden_size, hidden_size)
        self.linear2_2 = nn.Linear(hidden_size, hidden_size)
        self.linear3_1 = nn.Linear(hidden_size, hidden_size)
        self.linear3_2 = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        tree_feat: torch.Tensor,
        var_feat: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """tree_feat (B, E); var_feat (B, L, E); padding_mask (B, L), True=pad."""
        tree_feat = tree_feat.unsqueeze(1)  # (B, 1, E)
        g_tc = torch.bmm(self.linear1_1(tree_feat), var_feat.transpose(1, 2)).squeeze(1)
        g_tc = F.softmax(
            g_tc.masked_fill(padding_mask, float("-inf")), dim=1
        ).unsqueeze(1)
        g_ct = torch.bmm(self.linear1_2(var_feat), tree_feat.transpose(1, 2)).squeeze(2)
        g_ct = F.softmax(
            g_ct.masked_fill(padding_mask, float("-inf")), dim=1
        ).unsqueeze(2)

        e_t = torch.bmm(g_tc, var_feat)  # (B, 1, E)
        e_c = torch.bmm(g_ct, tree_feat)  # (B, L, E)
        s_t = F.relu(self.linear2_1(e_t))
        s_c = F.relu(self.linear2_2(e_c))
        attn = torch.sigmoid(self.linear3_1(s_t) + self.linear3_2(s_c))
        return attn * s_t + (1 - attn) * s_c


class Actor(nn.Module):
    """PPO actor: one probability per branching candidate."""

    def __init__(
        self,
        var_dim: int = N_CANDIDATE_FEATURES,
        node_dim: int = N_NODE_FEATURES,
        mip_dim: int = N_MIP_FEATURES,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Recorded in the checkpoint so load_checkpoint can rebuild the net
        # (the official build_models_from_ckpt reads the same four values).
        self.hparams = {
            "var_dim": var_dim,
            "node_dim": node_dim,
            "mip_dim": mip_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_dim), nn.Linear(var_dim, hidden_dim)
        )
        self.tree_embedding = nn.Sequential(
            nn.LayerNorm(node_dim + mip_dim), nn.Linear(node_dim + mip_dim, hidden_dim)
        )
        self.global_embedding = nn.Linear(hidden_dim * 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.tree_refinement = BiMatchingNet(hidden_dim)
        self.output_layer = TreeGateBranchingNet(
            branch_size=hidden_dim,
            tree_state_size=hidden_dim,
            dim_reduce_factor=2,
            infimum=1,
            norm="layer",
            depth=2,
            hidden_size=hidden_dim,
        )

    def forward(
        self,
        cands_state_mat: torch.Tensor,
        node_state: torch.Tensor,
        mip_state: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """cands (B, L, 25), node (B, 8), mip (B, 53) -> probs (B, L)."""
        batch_size, num_candidates, _ = cands_state_mat.shape
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, num_candidates),
                dtype=torch.bool,
                device=cands_state_mat.device,
            )

        var_features = self.var_embedding(cands_state_mat)
        tree_features = self.tree_embedding(torch.cat([node_state, mip_state], dim=-1))
        expanded = tree_features.unsqueeze(1).expand(-1, num_candidates, -1)
        var_features = self.global_embedding(
            torch.cat([var_features, expanded], dim=-1)
        )

        # The official transformer runs batch-first=False: (L, B, H).
        transformed = self.transformer(
            var_features.transpose(0, 1), src_key_padding_mask=padding_mask
        ).transpose(0, 1)

        refined = self.tree_refinement(tree_features, transformed, padding_mask)
        action_logits = self.output_layer(refined, tree_features)
        action_logits = action_logits.masked_fill(padding_mask, float("-inf"))
        return F.softmax(action_logits, dim=-1)


class Critic(nn.Module):
    """PPO critic: one state value, mean-pooled over the valid candidates."""

    def __init__(
        self,
        var_dim: int = N_CANDIDATE_FEATURES,
        node_dim: int = N_NODE_FEATURES,
        mip_dim: int = N_MIP_FEATURES,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hparams = {
            "var_dim": var_dim,
            "node_dim": node_dim,
            "mip_dim": mip_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_dim), nn.Linear(var_dim, hidden_dim)
        )
        self.tree_embedding = nn.Sequential(
            nn.LayerNorm(node_dim + mip_dim), nn.Linear(node_dim + mip_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.tree_refinement = BiMatchingNet(hidden_dim)
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value_head = TreeGateBranchingNet(
            branch_size=hidden_dim,
            tree_state_size=hidden_dim,
            dim_reduce_factor=2,
            infimum=1,
            norm="layer",
            depth=2,
            hidden_size=hidden_dim,
        )

    def forward(
        self,
        cands_state_mat: torch.Tensor,
        node_state: torch.Tensor,
        mip_state: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """cands (B, L, 25), node (B, 8), mip (B, 53) -> value (B, 1)."""
        batch_size, num_candidates, _ = cands_state_mat.shape
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, num_candidates),
                dtype=torch.bool,
                device=cands_state_mat.device,
            )

        var_features = self.var_embedding(cands_state_mat)
        transformed = self.transformer(
            var_features.transpose(0, 1), src_key_padding_mask=padding_mask
        ).transpose(0, 1)
        tree_features = self.tree_embedding(torch.cat([node_state, mip_state], dim=-1))
        refined = self.tree_refinement(tree_features, transformed, padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1)
        pooled = (refined * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        aggregated = self.aggregation(torch.cat([pooled, tree_features], dim=-1))
        return self.value_head(aggregated, tree_features).unsqueeze(-1)
