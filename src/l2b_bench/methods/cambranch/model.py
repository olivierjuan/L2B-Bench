"""CAMBranch 2024 — Gasse's GCNN plus the contrastive projection head.

Paper: "Contrastive Learning with Augmented MILPs for Learning to Branch"
(Lin et al., ICLR 2024 poster). The backbone is Gasse's GCNN verbatim (the
official ``GNNPolicy`` port, inherited unchanged — PreNorm embeddings, the
two message-passing passes, the per-variable output head); CAMBranch adds the
projection head of its Sec. 3.3:

- per node type, the post-convolution embeddings of one graph are summarized
  by MaxPool + MeanPool, both pooled vectors concatenated;
- each node type's summary goes through its own MLP (``var_projection`` /
  ``cons_projection``);
- the two are concatenated and mapped to the graph embedding ``g``, which is
  L2-normalized for the contrastive losses.

The projection-head sizes are not specified in the paper — two-layer MLPs
(2*64 -> 64 -> 64, ReLU) are a declared pilot choice.

``forward`` keeps the gasse contract (per-variable logits alone) so the
PreNorm pre-training and the policy bridge work unchanged; when per-sample
segment sizes are given it additionally returns the graph embeddings, one
per sample of the concatenated batch.
"""

import torch
import torch.nn.functional as F

from ..gasse.model import GNNPolicy


class CAMBranchGNN(GNNPolicy):
    """Gasse's GCNN with the AMILP contrastive projection head."""

    def __init__(self) -> None:
        super().__init__()
        emb_size = 64
        self.cons_projection = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.var_projection = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.graph_projection = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    @staticmethod
    def _pool(features: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        """Per-graph [MaxPool; MeanPool] summaries: (sum N, emb) -> (B, 2*emb)."""
        pooled = []
        offset = 0
        for n in sizes:
            block = features[offset : offset + n]
            pooled.append(torch.cat([block.max(dim=0).values, block.mean(dim=0)]))
            offset += n
        return torch.stack(pooled)

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        n_vars_per_sample: list[int] | None = None,
        n_cons_per_sample: list[int] | None = None,
    ):
        # Gasse's two passes, spelled out (the intermediate post-conv
        # embeddings are what the projection head pools).
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
        if n_vars_per_sample is None:
            return output

        assert n_cons_per_sample is not None and len(n_vars_per_sample) == len(
            n_cons_per_sample
        )
        v_pools = self._pool(variable_features, n_vars_per_sample)
        c_pools = self._pool(constraint_features, n_cons_per_sample)
        g = self.graph_projection(
            torch.cat(
                [self.var_projection(v_pools), self.cons_projection(c_pools)], dim=-1
            )
        )
        return output, F.normalize(g, dim=-1)
