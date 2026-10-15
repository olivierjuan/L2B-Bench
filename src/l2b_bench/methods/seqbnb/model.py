"""Seq-BnB 2026 — the sequence model over the joint regime's tokens.

Reconstructed from the paper (arXiv 2610.14835, no repository): a typed
token sequence — node tokens (the open nodes' features), one graph token
(the pooled current-LP embedding), and one action token per past decision —
encoded by a Transformer; two heads read it back:

- **branching head** — cross-attention: each branching candidate's variable
  embedding (from a Gasse-style bipartite GCNN over the current LP) queries
  the sequence, and an MLP over [h_var; h_contextual] scores it;
- **node-selection head** — an MLP over the open nodes' contextual token
  representations.

The bipartite GCNN reuses the Gasse port's ``BipartiteGraphConvolution``
layer for layer; its input embeddings are plain ``Linear + ReLU`` here (the
paper conditions on its own GCNN, unspecified layer for layer), so its
PreNorm sub-layers stay identity — they are never pre-trained. Head counts
and depths are pilot choices: the paper gives the window (L_max = 500) and
the general shape, not the encoder's layer budget.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..gasse.model import BipartiteGraphConvolution
from .collect import ACTION_TOKEN_DIM, NODE_TOKEN_DIM, SeqBnBSample

# Token types in the sequence: node, pooled-graph, past node-selection
# action, past branching action.
TYPE_NODE = 0
TYPE_GRAPH = 1
TYPE_ACTION_NS = 2
TYPE_ACTION_BR = 3
N_TOKEN_TYPES = 4

# Feature dimensions of the NodeBipartite observation (as in Gasse 2019).
CONS_NFEATS = 5
EDGE_NFEATS = 1
VAR_NFEATS = 19


@dataclass
class SeqBnBBatch:
    """A mini-batch of decision samples: one big disconnected bipartite
    graph (as in the Gasse port) plus per-sample padded token sequences."""

    constraint_features: torch.Tensor  # (sum R, 5)
    edge_index: torch.Tensor  # (2, sum E)
    edge_attr: torch.Tensor  # (sum E, 1)
    variable_features: torch.Tensor  # (sum V, 19)
    var_offsets: torch.Tensor  # (B + 1,) — variable row offsets per sample
    graph_rows: list[torch.Tensor]  # per sample: variable rows of its graph
    tokens: torch.Tensor  # (B, T_max, 8) — node tokens (padded)
    token_mask: torch.Tensor  # (B, T_max) — True where padded
    history: torch.Tensor  # (B, H_max, 6) — past action tokens (padded)
    history_mask: torch.Tensor  # (B, H_max) — True where padded
    history_kinds: torch.Tensor  # (B, H_max) — DecisionKind per history row
    open_positions: list[torch.Tensor]  # per NS sample: token rows of open nodes
    candidate_rows: list[torch.Tensor]  # per BR sample: variable rows of candidates
    labels: torch.Tensor  # (B,) — expert's rank in its action set
    kinds: torch.Tensor  # (B,) — DecisionKind per sample


def sample_to_tensors(sample: SeqBnBSample) -> dict[str, object]:
    """Convert one collected sample to the model's tensor inputs."""
    obs = sample.graph
    if obs is not None:
        constraint_features = torch.as_tensor(obs.row_features, dtype=torch.float32)
        edge_index = torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
        edge_attr = torch.as_tensor(
            obs.edge_features.values, dtype=torch.float32
        ).reshape(-1, 1)
        variable_features = torch.as_tensor(obs.variable_features, dtype=torch.float32)
    else:
        # Before the first branching there is no cached graph: a single
        # zero-feature row stands in (its pooled embedding is a constant —
        # the same stand-in for every pre-root sample — and no branching
        # sample ever takes this path).
        constraint_features = torch.zeros((1, CONS_NFEATS))
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, EDGE_NFEATS))
        variable_features = torch.zeros((1, VAR_NFEATS))

    if sample.kind.value == 0:  # node selection: label ranks the open nodes
        action_ids = np.asarray(sample.open_ids)
    else:  # branching: label ranks the LP candidates
        action_ids = np.asarray(sample.candidates)
    label = int(np.where(action_ids == sample.expert_action)[0][0])
    open_positions = None
    candidate_rows = None
    if sample.open_ids is not None:
        lookup = {int(n): i for i, n in enumerate(sample.token_ids.tolist())}
        open_positions = torch.tensor(
            [lookup[int(n)] for n in sample.open_ids.tolist()], dtype=torch.long
        )
    if sample.candidates is not None:
        candidate_rows = torch.as_tensor(
            np.asarray(sample.candidates), dtype=torch.long
        )
    return {
        "constraint_features": constraint_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "variable_features": variable_features,
        "tokens": torch.as_tensor(sample.tokens, dtype=torch.float32),
        "history": torch.as_tensor(sample.history, dtype=torch.float32),
        "history_kinds": torch.as_tensor(sample.history_kinds, dtype=torch.long),
        "open_positions": open_positions,
        "candidate_rows": candidate_rows,
        "label": label,
        "kind": int(sample.kind.value),
    }


def collate(samples: list[SeqBnBSample]) -> SeqBnBBatch:
    """Collate decision samples: concatenated graph + padded sequences."""
    tensors = [sample_to_tensors(s) for s in samples]
    n_vars = [t["variable_features"].shape[0] for t in tensors]
    var_offsets = [0]
    for n in n_vars:
        var_offsets.append(var_offsets[-1] + n)
    edge_index = []
    cons_offset = 0
    var_offset = 0
    for t, n_v in zip(tensors, n_vars, strict=True):
        edge_index.append(t["edge_index"] + torch.tensor([[cons_offset], [var_offset]]))
        cons_offset += t["constraint_features"].shape[0]
        var_offset += n_v
    max_tokens = max(t["tokens"].shape[0] for t in tensors)
    max_history = max(t["history"].shape[0] for t in tensors)

    def pad(stack, width, value):
        rows = []
        for t in tensors:
            piece = t[stack]
            pad_len = width - piece.shape[0]
            if pad_len:
                piece = torch.cat(
                    [
                        piece,
                        torch.full(
                            (pad_len,) + piece.shape[1:], value, dtype=piece.dtype
                        ),
                    ]
                )
            rows.append(piece)
        return torch.stack(rows)

    mask_value = True
    token_rows = pad("tokens", max_tokens, 0.0)
    token_mask = torch.zeros(len(tensors), max_tokens, dtype=torch.bool)
    history_rows = pad("history", max_history, 0.0)
    history_mask = torch.zeros(len(tensors), max_history, dtype=torch.bool)
    history_kinds = pad("history_kinds", max_history, 0)
    for i, t in enumerate(tensors):
        token_mask[i, t["tokens"].shape[0] :] = mask_value
        history_mask[i, t["history"].shape[0] :] = mask_value
    return SeqBnBBatch(
        constraint_features=torch.cat([t["constraint_features"] for t in tensors]),
        edge_index=torch.cat(edge_index, dim=1),
        edge_attr=torch.cat([t["edge_attr"] for t in tensors]),
        variable_features=torch.cat([t["variable_features"] for t in tensors]),
        var_offsets=torch.tensor(var_offsets, dtype=torch.long),
        graph_rows=[
            torch.arange(var_offsets[i], var_offsets[i + 1])
            for i in range(len(tensors))
        ],
        tokens=token_rows,
        token_mask=token_mask,
        history=history_rows,
        history_mask=history_mask,
        history_kinds=history_kinds,
        open_positions=[t["open_positions"] for t in tensors],
        # Candidate LP positions shifted into the concatenated graph, so the
        # branching head queries the sample's own variable rows.
        candidate_rows=[
            t["candidate_rows"] + var_offsets[i]
            if t["candidate_rows"] is not None
            else None
            for i, t in enumerate(tensors)
        ],
        labels=torch.tensor([t["label"] for t in tensors], dtype=torch.long),
        kinds=torch.tensor([t["kind"] for t in tensors], dtype=torch.long),
    )


def batched_graph(batch: SeqBnBBatch):
    """The GCNN's forward arguments for a batch."""
    return (
        batch.constraint_features,
        batch.edge_index,
        batch.edge_attr,
        batch.variable_features,
    )


class SeqBnBModel(torch.nn.Module):
    """The sequence model: GCNN-encoded LP + Transformer over typed tokens."""

    def __init__(self, d_model: int = 64, nhead: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model

        # LP encoder (Gasse's convolutions, plain input embeddings).
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.Linear(CONS_NFEATS, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
        )
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(VAR_NFEATS, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
        )
        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        # Token embeddings + type embeddings.
        self.node_token_embedding = torch.nn.Sequential(
            torch.nn.Linear(NODE_TOKEN_DIM, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.graph_token_embedding = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.action_token_embedding = torch.nn.Sequential(
            torch.nn.Linear(ACTION_TOKEN_DIM, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.type_embedding = torch.nn.Embedding(N_TOKEN_TYPES, d_model)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2 * d_model,
                batch_first=True,
            ),
            num_layers=layers,
        )

        # Heads.
        self.ns_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 1),
        )
        self.cross_attention = torch.nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.br_head = torch.nn.Sequential(
            torch.nn.Linear(2 * d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 1),
        )

    def _variable_embeddings(self, batch: SeqBnBBatch) -> torch.Tensor:
        """The GCNN: variable embeddings for the batch's big graph."""
        constraint_features, edge_index, edge_attr, variable_features = batched_graph(
            batch
        )
        reversed_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        constraint_features = self.cons_embedding(constraint_features)
        # Edge features stay 1-dimensional here: the Gasse convolutions
        # embed them internally (their own Linear(1, emb_size)).
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edges, edge_attr, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_index, edge_attr, variable_features
        )
        return variable_features

    def _sequence(self, batch: SeqBnBBatch, pooled: torch.Tensor) -> torch.Tensor:
        """Embed the typed token sequence: [nodes..., graph, actions...]."""
        node = self.node_token_embedding(batch.tokens) + self.type_embedding(
            torch.full(batch.tokens.shape[:2], TYPE_NODE, device=batch.tokens.device)
        )
        graph = self.graph_token_embedding(pooled).unsqueeze(1) + self.type_embedding(
            torch.full((pooled.shape[0], 1), TYPE_GRAPH, device=pooled.device)
        )
        action_types = torch.where(
            batch.history_kinds == 1, TYPE_ACTION_BR, TYPE_ACTION_NS
        )
        action = self.action_token_embedding(batch.history) + self.type_embedding(
            action_types
        )
        return torch.cat([node, graph, action], dim=1)

    def forward(self, batch: SeqBnBBatch) -> list[torch.Tensor]:
        """Logits over each sample's action set (variable length, by kind)."""
        variables = self._variable_embeddings(batch)
        pooled = torch.stack(
            [
                variables[rows].mean(dim=0)
                if len(rows)
                else torch.zeros(self.d_model, device=variables.device)
                for rows in batch.graph_rows
            ]
        )
        sequence = self._sequence(batch, pooled)
        # The sequence's own padding mask: node + graph tokens are real,
        # action tokens pad at the end.
        padding = torch.cat(
            [
                batch.token_mask,
                torch.zeros_like(batch.token_mask[:, :1]),
                batch.history_mask,
            ],
            dim=1,
        )
        contextual = self.encoder(sequence, src_key_padding_mask=padding)

        logits: list[torch.Tensor] = []
        for i in range(batch.kinds.shape[0]):
            if batch.kinds[i] == 0:  # node selection: score the open nodes
                positions = batch.open_positions[i].to(contextual.device)
                rows = contextual[i, positions]
                logits.append(self.ns_head(rows).squeeze(-1))
            else:  # branching: each candidate queries the sequence
                rows = batch.candidate_rows[i].to(variables.device)
                queries = variables[rows].unsqueeze(0)
                attended, _ = self.cross_attention(
                    queries,
                    contextual[i : i + 1],
                    contextual[i : i + 1],
                    key_padding_mask=padding[i : i + 1],
                )
                h = torch.cat([queries, attended], dim=-1)
                logits.append(self.br_head(h).squeeze(-1).squeeze(0))
        return logits
