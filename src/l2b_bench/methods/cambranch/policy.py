"""CAMBranch 2024 — the trained GCNN as a gyozas policy.

The bridge, exactly like the gasse one: the projection head is training-side
only (the contrastive losses shaped the encoder), so at inference the model is
its gasse backbone — the candidate with the highest per-variable score. The
root-cuts-only collection protocol is likewise training-side (the fidelity
profile ``cambranch`` records it); the benchmark evaluation runs the default
solver profile like every other method.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .model import CAMBranchGNN
from .train import get_device, load_checkpoint


class CAMBranchPolicy:
    """Branch on the candidate with the highest GCNN score."""

    def __init__(self, model: CAMBranchGNN) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "CAMBranchPolicy":
        return cls(load_checkpoint(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation
        device = self.device
        edge_index = torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(obs.row_features, dtype=torch.float32).to(device),
                edge_index.to(device),
                torch.as_tensor(obs.edge_features.values, dtype=torch.float32)
                .reshape(-1, 1)
                .to(device),
                torch.as_tensor(obs.variable_features, dtype=torch.float32).to(device),
            )
        candidates = torch.as_tensor(np.asarray(action_set), dtype=torch.long).to(
            device
        )
        return int(candidates[logits[candidates].argmax()])


def cambranch_method(checkpoint: str | Path) -> Method:
    """The cambranch method, ready for the runner's method table.

    The fidelity profile ``cambranch`` recharges the native regime (R5.4):
    Gasse's ``NodeBipartite`` observation and IL (FSB) regime, augmented by
    the variable-shifted AMILPs and the joint contrastive loss; the policy is
    the trained checkpoint's greedy head.
    """
    return Method(
        name="cambranch",
        policy=lambda: CAMBranchPolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "GCNN + contrastive projection head (Lin et al., ICLR 2024)",
        },
        fidelity="cambranch",
    )
