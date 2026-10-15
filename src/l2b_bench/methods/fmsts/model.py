"""FMSTS 2020 — the MDA (Multiplicative Dueling Architecture) value network.

Ported from the paper (no official repository): Q(s, .) is the elementwise
product of

- a 1-D *linear-activation* output of a dense block fed with the static (PCA)
  features only — the per-instance scale, whose multiplicative form tracks the
  Q-function's exponential scale variation between depths and instances;
- a |J|-D output of a dense block fed with the static and dynamic features —
  one Q per variable. The paper's problems are fixed-size, so the faithful
  shape is a flat dense net; the per-variable weights are shared, which keeps
  the head applicable to any |J| (the benchmark evaluates on a bigger family
  than it trains on).

The hidden sizes and depths are not specified in the paper — pilot choices
(one hidden layer of 64 units per branch).
"""

import torch

from .collect import N_DYNAMIC_FEATURES

#: Pilot choice (not specified in the paper).
HIDDEN = 64


class MDANetwork(torch.nn.Module):
    """The multiplicative dueling value network over the bespoke observation."""

    def __init__(self, n_pca: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        # The static branch: the PCA vector -> one linear unit (the scale).
        self.static_branch = torch.nn.Sequential(
            torch.nn.Linear(n_pca, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        # The dynamic branch: per-variable (static broadcast + dynamic) -> one
        # linear unit per variable.
        self.dynamic_branch = torch.nn.Sequential(
            torch.nn.Linear(n_pca + N_DYNAMIC_FEATURES, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        """``static`` is [batch, n_pca]; ``dynamic`` is [batch, n_vars, 5].
        Returns the per-variable Q values, [batch, n_vars]."""
        scale = self.static_branch(static).squeeze(-1)  # [batch]
        n_vars = dynamic.shape[1]
        inputs = torch.cat(
            [static.unsqueeze(1).expand(-1, n_vars, -1), dynamic], dim=-1
        )  # [batch, n_vars, n_pca + 5]
        values = self.dynamic_branch(inputs).squeeze(-1)  # [batch, n_vars]
        return scale.unsqueeze(-1) * values
