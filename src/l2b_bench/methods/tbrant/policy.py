"""T-BranT (Lin 2022) — the trained BranchFormer as a gyozas policy.

The bridge between the method and the harness: the same ``select_action``
contract every policy exposes (R1.1), consuming the ``BranchingState``
observation the method declares. At inference the policy sees one node at a
time — a batch of one, padded to the node's own candidate count.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from gyozas.runner import Method

from .model import BranchFormer
from .train import load_checkpoint


class TBranTPolicy:
    """Branch on the candidate with the highest BranchFormer logit."""

    def __init__(self, model: BranchFormer) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TBranTPolicy":
        from .train import get_device

        return cls(load_checkpoint(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation
        device = self.device
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(obs.candidate_features, dtype=torch.float32).unsqueeze(0).to(device),
                torch.zeros(1, len(action_set), dtype=torch.long, device=device),  # no padding
                torch.as_tensor(obs.node_features, dtype=torch.float32).unsqueeze(0).to(device),
                torch.as_tensor(obs.mip_features, dtype=torch.float32).unsqueeze(0).to(device),
            )
        candidates = np.asarray(action_set)
        return int(candidates[int(logits[0].argmax())])


def tbrant_method(checkpoint: str | Path) -> Method:
    """The tbrant method, ready for the runner's method table.

    The fidelity profile ``tbrant`` recharges the native regime (R5.4): the
    ``BranchingStateObservation`` state and the relpscost expert; the policy
    is the trained checkpoint (the LTBranT variant — tree gating and
    BiMatching, no tree GAT).
    """
    return Method(
        name="tbrant",
        policy=lambda: TBranTPolicy.from_checkpoint(checkpoint),
        config={"checkpoint": str(checkpoint), "architecture": "LTBranT (Lin et al., 2022)"},
        fidelity="tbrant",
    )
