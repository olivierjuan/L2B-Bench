"""TGPPO 2025 — the trained actor as a gyozas policy.

The bridge between the method and the harness (the ``select_action``
contract of R1.1), consuming the ``BranchingState`` observation the method
declares. The official evaluation keeps the *sampled* action selection of
training (its brancher hardcodes ``deterministic=False``), seeding each
evaluation job; the bridge samples from the actor's softmax over the node's
candidates accordingly.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .train import get_device, load_checkpoint


class TGPPOPolicy:
    """Sample a branching candidate from the actor's softmax (the official
    evaluation-time behaviour; the ambient torch RNG provides the draws)."""

    def __init__(self, actor, critic) -> None:
        self.actor = actor
        self.device = next(actor.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TGPPOPolicy":
        actor, critic = load_checkpoint(path, device=get_device())
        return cls(actor, critic)

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation
        device = self.device
        with torch.no_grad():
            probs = self.actor(
                torch.as_tensor(obs.candidate_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                torch.as_tensor(obs.node_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                torch.as_tensor(obs.mip_features, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
            )
        action = int(torch.distributions.Categorical(probs=probs).sample())
        return int(np.asarray(action_set)[action])


def tgppo_method(checkpoint: str | Path, fixed: bool = False) -> Method:
    """The tgppo method, ready for the runner's method table.

    The fidelity profile ``tgppo`` recharges the native regime (R5.4): the
    ``BranchingStateObservation`` state and the sandbox protocol; the policy
    is the PPO-trained actor-critic checkpoint (sampled actions).
    ``fixed=True`` names the ``tgppo-fixed`` variant (same regime; the
    architecture flag itself travels in the checkpoint's hparams).
    """
    return Method(
        name="tgppo-fixed" if fixed else "tgppo",
        policy=lambda: TGPPOPolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "Tree-Gate PPO actor-critic transformer (benmhamed-a, 2025)"
            + (", fixed: BiMatchingNet residual + 8-unit score head" if fixed else ""),
        },
        fidelity="tgppo",
    )
