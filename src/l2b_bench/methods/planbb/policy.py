"""PlanB&B — the trained MuZero network as a gyozas policy.

The evaluation bridge mirrors the official ``select_greedy_action``: the
observation is encoded once by the representation function ``h`` and the
argmax over the candidates of the policy head's logits wins — the headline
setting is 0 MCTS simulations at test time, so no search is run. The Gumbel
MCTS (self-play training, fidelity profile ``--p_mcts_num 16``) and the DFS /
no-restart protocol stay training-side only, exactly like the other
fidelity profiles.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.runner import Method
from numpy.typing import NDArray

from .model import PlanBBNet, PlanBBObs, geometric_batch
from .train import get_device


class PlanBBPolicy:
    """Branch on the candidate with the highest policy logit."""

    def __init__(self, model: PlanBBNet, device: torch.device | None = None) -> None:
        self.model = model
        self.model.eval()
        self.model.train_mode()  # non-MCTS inference: raw policy logits
        self.device = next(model.parameters()).device if device is None else device

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "PlanBBPolicy":
        return cls(load_planbb(path, device=get_device()))

    def select_action(self, observation: Any, action_set: NDArray) -> int:
        """The official ``select_greedy_action``: argmax of the policy head
        over the masked action set, single graph, no padding."""
        candidates = np.asarray(action_set)
        batched = geometric_batch([PlanBBObs(observation, candidates)]).to(self.device)
        with torch.no_grad():
            output = self.model.initial_inference(batched, root=True)
        # [1, n_variables] root-masked logits, non-MCTS mode (always padded).
        logits = torch.as_tensor(output.policy_logits).reshape(-1)
        cand = torch.as_tensor(candidates, dtype=torch.long, device=self.device)
        return int(cand[logits[cand].argmax()])


def load_planbb(path: str | Path, device: torch.device | None = None) -> PlanBBNet:
    """Load a ``train()`` checkpoint (a raw state_dict) back onto ``device``."""
    if device is None:
        device = get_device()
    state = torch.load(path, map_location=device, weights_only=True)
    model = PlanBBNet()
    model.load_state_dict(state)
    return model.to(device)


def planbb_method(checkpoint: str | Path) -> Method:
    """The planbb method, ready for the runner's method table.

    The fidelity profile ``planbb`` recharges the native regime (R5.4): the
    43-feature ``NodeBipartitePlanBB`` observation, the tree MDP over B&B
    open-node states, and the MuZero stack (Gumbel MCTS self-play, SimSiam
    consistency, HL-Gauss value bins) used to train it.
    """
    return Method(
        name="planbb",
        policy=lambda: PlanBBPolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": (
                "MuZero h/g/f GCNN with Gumbel MCTS self-play "
                "(PlanB&B, Strang et al., 2026)"
            ),
        },
        fidelity="planbb",
    )
