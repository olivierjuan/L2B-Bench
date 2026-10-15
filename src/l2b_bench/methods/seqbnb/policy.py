"""Seq-BnB 2026 — the trained sequence model as a gyozas policy.

The bridge between the method and the harness. The method runs on the joint
regime (``BranchingNodeSelectionDynamics``), so the policy receives the
dynamics' own ``Decision`` objects as action sets and reads the episode's
model through the ``attach(env)`` hook — the plain ``(obs, action_set)``
protocol carries neither. The token bookkeeping (the same
:class:`EpisodeTracker` the collector uses) lives inside the policy, so
benchmark episodes and training episodes see identical states.
"""

from pathlib import Path
from typing import Any

import torch
from gyozas.dynamics import BranchingNodeSelectionDynamics, Decision, DecisionKind
from gyozas.runner import Method

from .collect import EpisodeTracker, SeqBnBState, _sample_from_state
from .model import SeqBnBModel, collate, to_device


class SeqBnBPolicy:
    """Act on the joint regime with the trained sequence model.

    Parameters
    ----------
    model:
        The trained ``SeqBnBModel`` (eval mode, on its device).
    """

    def __init__(self, model: SeqBnBModel) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.tracker = EpisodeTracker()
        self.env: Any = None

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SeqBnBPolicy":
        from .train import get_device, load_checkpoint

        return cls(load_checkpoint(path, device=get_device()))

    def attach(self, env) -> None:
        """Receive the run's environment (the runner's hook): the policy
        reads the model and the dynamics' bookkeeping through it."""
        self.env = env

    def act(self, state: SeqBnBState) -> int:
        """The argmax action on one decision's state — no bookkeeping, so
        the DAgger rounds reuse it on the collector's own states."""
        sample = _sample_from_state(state, action=-1, expert=None)
        batch = to_device(collate([sample]), self.device)
        with torch.no_grad():
            logits = self.model(batch)[0]
        ids = (
            state.open_ids
            if state.kind == DecisionKind.NODE_SELECTION
            else state.candidates
        )
        return int(ids[int(logits.argmax())])

    def select_action(self, observation: Any, action_set: Decision) -> int:
        """The policy contract: track the episode, act, fold the action in."""
        assert self.env is not None, "attach(env) must run before the first decision"
        state = self.tracker.observe(self.env, action_set)
        action = self.act(state)
        self.tracker.record(state, action)
        return action


def seqbnb_method(checkpoint: str | Path) -> Method:
    """The Seq-BnB method, ready for the runner's method table.

    The method declares its own dynamics — the joint branch-and-node-
    selection regime — and the fidelity profile ``seqbnb`` recharges the
    native regime (R5.4): the paper's SCIP parameters.
    """
    return Method(
        name="seqbnb",
        policy=lambda: SeqBnBPolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "Seq-BnB (arXiv 2610.14835)",
        },
        fidelity="seqbnb",
        dynamics=lambda: BranchingNodeSelectionDynamics(),
    )
