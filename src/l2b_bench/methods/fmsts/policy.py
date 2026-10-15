"""FMSTS 2020 — the trained MDA as a gyozas policy.

The bridge between the method and the harness, exactly like the other RL
bridges: the trained :class:`MDANetwork` scores every variable and the policy
branches on the candidate with the *smallest* Q value (the value function is
the subtree size — the paper minimises it). The bespoke observation (the
fitted PCA carried in the checkpoint) is declared on the :class:`Method`, and
the DFS/no-presolving protocol and the epsilon exploration are training-side
only (the fidelity profile records them); the benchmark evaluation runs the
default solver profile like every other method.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .collect import FMSTSObservation
from .model import MDANetwork
from .train import get_device, load_checkpoint


class FMSTSPolicy:
    """Branch on the candidate with the smallest predicted subtree size."""

    def __init__(self, model: MDANetwork) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "FMSTSPolicy":
        model, _mean, _components, _canonical = load_checkpoint(
            path, device=get_device()
        )
        return cls(model)

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        static = torch.as_tensor(
            observation.static, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        dynamic = torch.as_tensor(
            observation.dynamic, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            values = self.model(static, dynamic).squeeze(0)
        candidates = torch.as_tensor(
            np.asarray(action_set), dtype=torch.long, device=self.device
        )
        return int(candidates[values[candidates].argmin()])


def fmsts_method(checkpoint: str | Path) -> Method:
    """The fmsts method, ready for the runner's method table.

    The fidelity profile ``fmsts`` records the native regime (R5.4): the
    bespoke PCA observation, the subtree-size Q-values, and the MDA. The
    observation is part of the method (the fitted PCA lives in the
    checkpoint), so the profile itself declares none.
    """
    model, mean, components, canonical = load_checkpoint(
        checkpoint, device=get_device()
    )

    def observation() -> FMSTSObservation:
        return FMSTSObservation(mean, components, canonical)

    return Method(
        name="fmsts",
        policy=lambda: FMSTSPolicy(model),
        observation=observation,
        config={
            "checkpoint": str(checkpoint),
            "architecture": "MDA (multiplicative dueling) value network (FMSTS, Etheve et al., 2020)",
        },
        fidelity="fmsts",
    )
