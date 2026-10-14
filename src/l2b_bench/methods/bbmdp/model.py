"""BBMDP 2025 — the DQN GCNN with an HL-Gauss categorical value head.

Faithful port of ``rl/networks/{gcnn,loss}.py`` from the official repository
(github.com/abfariah/bbmdp, the ``DQNPolicy`` used by ``dqn_bbmdp``):

- ``DQNBipartiteGraphConvolution`` — Gasse's message passing with the DQN
  substitution: LayerNorm everywhere instead of the pre-trained PreNorm, and
  LeakyReLU instead of ReLU. The left/right feature modules are applied
  *before* the propagation (the official ``forward``), not per message — the
  same linear maps either way.
- ``DQNPolicy`` — 43 variable features (19 bipartite-LP + 24 retro), the two
  convolutions, and a per-variable head with one logit per HL-Gauss bin
  (18 bins). "Watchout: no pretrain!" — all weights ~ Normal(0, 0.01), zero
  biases, nothing frozen.
- ``TreeHLGaussLoss`` — the categorical value head over log2(-z): targets are
  distributions over the 18 bins (erf-CDF differences, sigma = 0.75 bin
  widths), predictions decode to a value through the bins' centers,
  ``Q = -2^E[centers]``. The official's entropy/KL bookkeeping (pure logging)
  is dropped.
- ``pad_per_sample`` — the official ``pad_tensor``: per-sample blocks padded
  to one rectangle with -1e9, so an ``argmax`` or a masked softmax stays
  inside the sample's own variables/candidates.
"""

import numpy as np
import torch
import torch_geometric

#: The RetroNodeBipartite feature dimensions.
N_ROW_FEATURES = 5
N_VAR_FEATURES = 43

#: The HL-Gauss head: 18 bins over log2(-z) in [-1.5, 16.5], sigma in
#: bin-width units (constants of the official ``TreeDQNAgent``).
MIN_VALUE = -1.5
MAX_VALUE = 16.5
NUM_BINS = 18
SIGMA = 0.75


class DQNBipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """One bipartite message-passing round: left features → right features."""

    def __init__(self, emb_size: int = 64) -> None:
        super().__init__("add")
        self.feature_module_left = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size))
        self.feature_module_edge = torch.nn.Sequential(torch.nn.Linear(1, emb_size, bias=False))
        self.feature_module_right = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size, bias=False))
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size)
        )
        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))
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


class DQNPolicy(torch.nn.Module):
    """The branching value network — "Watchout: no pretrain!".

    Returns, per variable, either one logit per HL-Gauss bin (classification,
    the BBMDP regime) or a single Q value (regression, unused here).
    """

    def __init__(self, emb_size: int = 64, classification: bool = True) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.classification = classification
        self.regression = not classification

        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(N_ROW_FEATURES),
            torch.nn.Linear(N_ROW_FEATURES, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.edge_embedding = torch.nn.Sequential(torch.nn.LayerNorm(1))
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(N_VAR_FEATURES),
            torch.nn.Linear(N_VAR_FEATURES, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )

        self.conv_v_to_c = DQNBipartiteGraphConvolution(emb_size=emb_size)
        self.conv_c_to_v = DQNBipartiteGraphConvolution(emb_size=emb_size)

        if self.regression:
            output_dim = 1
        else:
            self.num_bins = NUM_BINS
            output_dim = self.num_bins
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, output_dim),
        )
        self.init_model_parameters()

    def init_model_parameters(self) -> None:
        """The official init: every Linear ~ Normal(0, 0.01), zero biases."""

        def init_params(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_params)

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

        return self.output_module(variable_features).squeeze(-1)


class TreeHLGaussLoss(torch.nn.Module):
    """The categorical HL-Gauss loss over log2(-z) values.

    ``target`` holds values in [min_value, max_value] (the log2 of minus the
    Q target); ``forward`` returns the *element-wise* cross-entropy between
    the logits and the target's bin distribution. ``transform_from_probs``
    decodes a softmax over the logits back to a value (the bins' expectation)
    — the caller applies ``-2^value``.
    """

    def __init__(
        self,
        min_value: float = MIN_VALUE,
        max_value: float = MAX_VALUE,
        num_bins: int = NUM_BINS,
        sigma: float = SIGMA,
    ) -> None:
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        support = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer("support", support)
        self.register_buffer("centers", (support[:-1] + support[1:]) / 2)
        # sigma is expressed in bin-width units (the official multiplies it
        # by the bin width, which is 1 over [-1.5, 16.5] in 18 bins).
        self.sigma = sigma * (support[1] - support[0])

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits, self.transform_to_probs(target), reduction="none")

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (target.unsqueeze(-1) - self.support) / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        return torch.sum(probs * self.centers, dim=-1)


def pad_per_sample(values: torch.Tensor, nb_per_sample: torch.Tensor, pad_value: float = -1e9) -> torch.Tensor:
    """The official ``pad_tensor``: one rectangular block of per-sample rows,
    short samples padded with ``pad_value``. Handles both the per-variable
    logit blocks ([N, bins]) and the decoded 1-D value rows."""
    max_size = int(nb_per_sample.max())
    is_logits = values.dim() > 1
    blocks = []
    offset = 0
    for nb in nb_per_sample.tolist():
        block = values[offset : offset + nb]
        if is_logits:
            pad = (0, 0, 0, max_size - block.shape[0])
        else:
            pad = (0, max_size - block.shape[0])
        blocks.append(torch.nn.functional.pad(block, pad, "constant", pad_value))
        offset += nb
    return torch.stack(blocks, dim=0)


class StateBatch:
    """A mini-batch of (observation, action_set) states collated into one
    disconnected bipartite graph (by hand, like the other methods)."""

    def __init__(self, states) -> None:
        n_vars = [len(observation.variable_features) for observation, _ in states]
        edge_index = []
        cons_offset = 0
        var_offset = 0
        for (observation, _), n_v in zip(states, n_vars, strict=True):
            indices = np.asarray(observation.edge_features.indices, dtype=np.int64)
            edge_index.append(
                torch.as_tensor(indices, dtype=torch.long)
                + torch.tensor([[cons_offset], [var_offset]], dtype=torch.long)
            )
            cons_offset += len(observation.row_features)
            var_offset += n_v
        self.constraint_features = torch.cat(
            [torch.as_tensor(o.row_features, dtype=torch.float32) for o, _ in states]
        )
        self.edge_index = torch.cat(edge_index, dim=1)
        self.edge_attr = torch.cat(
            [
                torch.as_tensor(o.edge_features.values, dtype=torch.float32).reshape(-1, 1)
                for o, _ in states
            ]
        )
        self.variable_features = torch.cat(
            [torch.as_tensor(o.variable_features, dtype=torch.float32) for o, _ in states]
        )
        # Each sample's candidates carry its variable offset in the collated graph.
        self.candidates = [
            torch.as_tensor(np.asarray(action_set), dtype=torch.long) + sum(n_vars[:i])
            for i, (_, action_set) in enumerate(states)
        ]
        self.nb_variables = torch.tensor(n_vars, dtype=torch.long)
        self.nb_candidates = torch.tensor([len(c) for c in self.candidates], dtype=torch.long)

    def forward_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.constraint_features,
            self.edge_index,
            self.edge_attr,
            self.variable_features,
        )

    def to(self, device: torch.device) -> "StateBatch":
        self.constraint_features = self.constraint_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.variable_features = self.variable_features.to(device)
        self.candidates = [c.to(device) for c in self.candidates]
        self.nb_variables = self.nb_variables.to(device)
        self.nb_candidates = self.nb_candidates.to(device)
        return self
