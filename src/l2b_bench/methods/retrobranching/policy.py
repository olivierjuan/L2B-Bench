"""RetroBranching (Parsonson & Deisenroth 2022) — the trained DQN as a gyozas policy.

The bridge between the method and the harness: the same ``select_action``
contract every policy exposes (R1.1), consuming the ``NodeBipartiteRetro``
observation the method declares. At evaluation the agent is pure greedy —
epsilon is a training-time exploration device.
"""

from pathlib import Path
from typing import Any

import numpy as np
from gyozas.runner import Method
from numpy.typing import NDArray

from .model import RetroGCN
from .train import load_checkpoint


class RetroPolicy:
    """Branch on the candidate with the highest Q value."""

    def __init__(self, model: RetroGCN) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "RetroPolicy":
        from .train import get_device

        return cls(load_checkpoint(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        import torch

        obs = observation
        device = self.device
        edge_index = torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
        with torch.no_grad():
            q = self.model(
                torch.as_tensor(obs.row_features, dtype=torch.float32).to(device),
                edge_index.to(device),
                torch.as_tensor(obs.edge_features.values, dtype=torch.float32).reshape(-1, 1).to(device),
                torch.as_tensor(obs.variable_features, dtype=torch.float32).to(device),
            )
        candidates = torch.as_tensor(np.asarray(action_set), dtype=torch.long).to(device)
        return int(candidates[q[candidates].argmax()])


def retrobranching_method(checkpoint: str | Path) -> Method:
    """The retrobranching method, ready for the runner's method table.

    The fidelity profile ``retrobranching`` recharges the native regime
    (R5.4): the ``NodeBipartiteRetro`` observation (43 variable features)
    under the paper's RL training protocol; the policy is the trained DQN.
    """
    return Method(
        name="retrobranching",
        policy=lambda: RetroPolicy.from_checkpoint(checkpoint),
        config={"checkpoint": str(checkpoint), "architecture": "BipartiteGCN DQN (Parsonson & Deisenroth, 2022)"},
        fidelity="retrobranching",
    )
