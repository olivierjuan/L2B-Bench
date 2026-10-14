"""BBMDP 2025 — the trained DQN as a gyozas policy.

The bridge between the method and the harness, exactly like the gasse and
sorrel bridges: the trained ``DQNPolicy`` decodes its per-variable HL-Gauss
head into Q values and branches on the candidate argmax — the official
``select_greedy_action``. The DFS/no-restarts protocol and the epsilon /
temperature exploration are training-side only (the fidelity profile records
them); the benchmark evaluation runs the default solver profile like every
other method.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .model import DQNPolicy, TreeHLGaussLoss
from .train import get_device, load_dqn


class BBMDPPolicy:
    """Branch on the candidate with the highest decoded Q value."""

    def __init__(self, model: DQNPolicy) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.loss_fn = TreeHLGaussLoss().to(self.device)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "BBMDPPolicy":
        return cls(load_dqn(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        device = self.device
        edge_index = torch.as_tensor(np.asarray(observation.edge_features.indices), dtype=torch.long)
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(observation.row_features, dtype=torch.float32).to(device),
                edge_index.to(device),
                torch.as_tensor(observation.edge_features.values, dtype=torch.float32)
                .reshape(-1, 1)
                .to(device),
                torch.as_tensor(observation.variable_features, dtype=torch.float32).to(device),
            )
            # Q = -2^E[centers] over the bins' softmax.
            q_values = -torch.pow(2.0, self.loss_fn.transform_from_probs(logits.softmax(dim=-1)))
        candidates = torch.as_tensor(np.asarray(action_set), dtype=torch.long).to(device)
        return int(candidates[q_values[candidates].argmax()])


def bbmdp_method(checkpoint: str | Path) -> Method:
    """The bbmdp method, ready for the runner's method table.

    The fidelity profile ``bbmdp`` recharges the native regime (R5.4): the
    43-feature ``NodeBipartiteRetro`` observation and the corrected vanilla
    MDP; the policy is the DQN's greedy head.
    """
    return Method(
        name="bbmdp",
        policy=lambda: BBMDPPolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "DQN GCNN with an HL-Gauss value head (BBMDP, Forel et al., 2025)",
        },
        fidelity="bbmdp",
    )
