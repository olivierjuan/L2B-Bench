"""Gasse 2019 — the trained GCNN as a gyozas policy.

The bridge between the method and the harness: the same ``select_action``
contract every policy exposes (R1.1), consuming the ``NodeBipartite``
observation the method declares. Once wrapped, the method runs through the
standard benchmark grid like any other — nothing harness-side knows it is a
GCNN.
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from gyozas.runner import Method

from .model import GNNPolicy
from .train import load_checkpoint


class GassePolicy:
    """Branch on the candidate with the highest GCNN score.

    Parameters
    ----------
    model:
        A ``GNNPolicy`` with its PreNorm layers frozen (i.e. after training).
    """

    def __init__(self, model: GNNPolicy) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "GassePolicy":
        from .train import get_device

        return cls(load_checkpoint(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation
        device = self.device
        edge_index = torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(
                torch.as_tensor(obs.row_features, dtype=torch.float32).to(device),
                edge_index.to(device),
                torch.as_tensor(obs.edge_features.values, dtype=torch.float32).reshape(-1, 1).to(device),
                torch.as_tensor(obs.variable_features, dtype=torch.float32).to(device),
            )
        candidates = torch.as_tensor(np.asarray(action_set), dtype=torch.long).to(device)
        return int(candidates[logits[candidates].argmax()])


def gasse_method(checkpoint: str | Path) -> Method:
    """The gasse method, ready for the runner's method table.

    The fidelity profile ``gasse`` recharges the native regime (R5.4): the
    ``NodeBipartite`` observation and the paper's collection protocol; the
    policy is the trained checkpoint.
    """
    return Method(
        name="gasse",
        policy=lambda: GassePolicy.from_checkpoint(checkpoint),
        config={"checkpoint": str(checkpoint), "architecture": "GCNN (Gasse et al., 2019)"},
        fidelity="gasse",
    )


def gasse_relpscost_method(checkpoint: str | Path) -> Method:
    """R6.4's control arm: the same GCNN, trained on relpscost labels.

    The ``gasse-relpscost`` fidelity profile recharges Gasse's regime with
    the scalable expert swapped in; run against the gasse method on the same
    synthetic strata, the pair prices the cost of the switch.
    """
    return Method(
        name="gasse-relpscost",
        policy=lambda: GassePolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "GCNN (Gasse et al., 2019), relpscost labels",
        },
        fidelity="gasse-relpscost",
    )
