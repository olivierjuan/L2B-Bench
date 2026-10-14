"""DSC-CL 2025 — the trained GCNN as a gyozas policy, with hybrid branching.

The paper's inference scheme (Sec. 4.5): at *upstream* nodes — those whose
branching-candidate count still exceeds rho times the root's (rho = 0.8; the
candidate sets only shrink with depth, so this identifies the shallow nodes
without tracking the tree) — the model proposes its top-k candidates (k = 5)
and *full strong branching decides among them* (the expert stays in the
inference loop); at downstream nodes the model branches directly.

The SB scores reach the policy through its observation: the fidelity profile
``dsccl`` declares the composite ``{bipartite, sb_scores}`` observation, so
the runner's environment extracts both and the policy receives the dict. The
runner builds a fresh policy per run, which is what lets the first decision
of an episode record the root candidate count.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .model import DSCCLGNN
from .train import get_device, load_checkpoint

#: A node is upstream while its candidate count exceeds rho times the root's.
RHO = 0.8

#: The model's shortlist at upstream nodes; full SB picks among them.
TOP_K_HYBRID = 5


class DSCCLPolicy:
    """Hybrid branching: top-5 + full SB upstream, the model downstream."""

    def __init__(self, model: DSCCLGNN) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self._root_count: int | None = None

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "DSCCLPolicy":
        return cls(load_checkpoint(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation["bipartite"]
        sb_scores = observation.get("sb_scores")
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
        candidates = np.asarray(action_set)

        if self._root_count is None:
            self._root_count = len(candidates)
        upstream = len(candidates) > RHO * self._root_count
        if not (upstream and sb_scores is not None):
            # Downstream node (or SB unavailable): the model decides alone.
            return int(candidates[np.argmax(logits.cpu().numpy()[candidates])])

        # Upstream node: the model's top-k shortlist, full SB among them.
        scores = np.asarray(sb_scores, dtype=float)
        order = np.argsort(-logits.cpu().numpy()[candidates])
        shortlist = candidates[order[:TOP_K_HYBRID]]
        shortlist_scores = scores[shortlist]
        if np.isnan(shortlist_scores).all():
            return int(shortlist[0])
        return int(shortlist[np.nanargmax(shortlist_scores)])


def dsccl_method(checkpoint: str | Path) -> Method:
    """The dsccl method, ready for the runner's method table.

    The fidelity profile ``dsccl`` recharges the native regime (R5.4): the
    composite bipartite + SB-scores observation, Gasse's IL (FSB) collection,
    and the hybrid branching of the paper's Sec. 4.5.
    """
    return Method(
        name="dsccl",
        policy=lambda: DSCCLPolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "GCNN + stratified contrastive learning "
            "(Lu, Ma & Tao, arXiv 2511.21107)",
            "hybrid": f"top-{TOP_K_HYBRID} + full SB upstream (rho = {RHO})",
        },
        fidelity="dsccl",
    )
