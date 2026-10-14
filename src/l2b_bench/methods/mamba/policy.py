"""Mamba-Branching 2026 — the trained policy as a gyozas policy.

Port of ``MambaEvalBrancher`` from the official repository
(github.com/doctor-watson626/Mamba-Branching): a streaming reader. At each
decision the policy embeds the current node's candidates (the contrastively
pretrained TreeGateNet), appends the block to its path, runs the GroupedMamba
over the last ``EVAL_MAX_SEQ_LEN`` blocks (49 — the official
``evaluate_all.py`` window), scores the current candidates from the Mamba
outputs at their positions (the mean of the 8 dims), branches on the argmax
(deterministic — the official sets no sampling at evaluation), and appends
the chosen action's embedding as the path's next block.

The runner builds a fresh policy per run (``method.policy()`` once per
``_run_one``), so the streaming path starts empty every episode.
"""

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .model import GroupedMamba, TreeGateNet
from .train import EVAL_MAX_SEQ_LEN, load_checkpoint


class MambaPolicyModel:
    """Branch on the candidate the Mamba path model scores highest."""

    def __init__(self, net: TreeGateNet, mamba: GroupedMamba, max_seq_len: int = EVAL_MAX_SEQ_LEN) -> None:
        self.net = net
        self.mamba = mamba
        self.device = next(mamba.parameters()).device
        # The streaming path: candidate blocks and action blocks, oldest
        # dropped beyond the window (a deque of (1, cands_i, 8) tensors).
        self.blocks: deque[torch.Tensor] = deque(maxlen=max_seq_len)
        self.max_seq_length = mamba.position_embed.num_embeddings

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "MambaPolicyModel":
        net, mamba = load_checkpoint(path)
        return cls(net, mamba)

    def _embed(self, candidate_features: Any, node_features: Any, mip_features: Any) -> torch.Tensor:
        """The current node's candidate rows -> (1, n_candidates, 8)."""
        with torch.no_grad():
            cands = torch.as_tensor(np.asarray(candidate_features), dtype=torch.float32).to(self.device)
            state = torch.cat(
                [
                    torch.as_tensor(np.asarray(node_features), dtype=torch.float32),
                    torch.as_tensor(np.asarray(mip_features), dtype=torch.float32),
                ]
            ).to(self.device)
            embedded, _ = self.net(cands.unsqueeze(0), state.unsqueeze(0))
        return embedded

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        obs = observation
        embedded = self._embed(obs.candidate_features, obs.node_features, obs.mip_features)
        n_candidates = embedded.size(1)
        self.blocks.append(embedded)  # the state block

        # The path so far: state and action blocks strictly alternate from
        # the deque's head — the official's i % 2 parities, position ids
        # shared by each state/action pair (the counter advances after an
        # action block).
        path = torch.cat(list(self.blocks), dim=1)  # (1, T, 8)
        position_ids = torch.zeros(1, path.size(1), dtype=torch.long, device=self.device)
        counter = 0
        position = 0
        for i, block in enumerate(self.blocks):
            block_len = block.size(1)
            # A state block (even i) and its action block (odd i) share the
            # counter's value; the counter advances after an action block.
            position_ids[0, position : position + block_len] = counter
            if i % 2 == 1:
                counter += 1
            position += block_len

        with torch.no_grad():
            output = self.mamba(path, position_ids)  # (1, T, 8)
        # The last n_candidates tokens are the current node's candidates.
        logits = output[0, -n_candidates:, :].mean(dim=-1)  # (n_candidates,)
        action_index = int(logits.argmax())

        # The chosen action's embedding becomes the path's next block.
        self.blocks.append(embedded[:, action_index : action_index + 1, :])

        candidates = np.asarray(action_set)
        candidates = candidates[candidates >= 0]  # SKIP is not a branching decision
        return int(candidates[action_index])


def mamba_method(checkpoint: str | Path) -> Method:
    """The mamba method, ready for the runner's method table.

    The fidelity profile ``mamba`` recharges the native regime (R5.4): the
    ``BranchingStateObservation`` state and the relpscost expert along whole
    solve trajectories; the policy is the two-stage (contrastive TreeGateNet
    + Mamba path) checkpoint, reading the solve as a stream.
    """
    return Method(
        name="mamba",
        policy=lambda: MambaPolicyModel.from_checkpoint(checkpoint),
        config={"checkpoint": str(checkpoint), "architecture": "Mamba-Branching (2026)"},
        fidelity="mamba",
    )
