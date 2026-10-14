"""TreeGate (Zarpellon 2021) — the trained policy as a gyozas policy.

The bridge between the method and the harness: the same ``select_action``
contract every policy exposes (R1.1), consuming the ``BranchingState``
observation the method declares. Like the official model, the policy sees one
node at a time — no batch dimension.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from gyozas.runner import Method

from .model import TreeGatePolicy as TreeGateModel
from .train import load_checkpoint


class TreeGatePolicyModel:
    """Branch on the candidate with the highest TreeGate score."""

    def __init__(self, model: TreeGateModel) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TreeGatePolicyModel":
        from .train import get_device

        return cls(load_checkpoint(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation
        device = self.device
        with torch.no_grad():
            scores = self.model(
                torch.as_tensor(obs.candidate_features, dtype=torch.float32).to(device),
                torch.as_tensor(obs.node_features, dtype=torch.float32).to(device),
                torch.as_tensor(obs.mip_features, dtype=torch.float32).to(device),
            )
        candidates = np.asarray(action_set)
        return int(candidates[int(scores[:, 0].argmax())])


def treegate_method(checkpoint: str | Path) -> Method:
    """The treegate method, ready for the runner's method table.

    The fidelity profile ``treegate`` recharges the native regime (R5.4): the
    ``BranchingStateObservation`` state (Zarpellon's original 25+8+53
    parameterization) and the relpscost expert under the (s, k) augmentation;
    the policy is the trained feature-gated checkpoint.
    """
    return Method(
        name="treegate",
        policy=lambda: TreeGatePolicyModel.from_checkpoint(checkpoint),
        config={"checkpoint": str(checkpoint), "architecture": "TreeGate (Zarpellon et al., 2021)"},
        fidelity="treegate",
    )
