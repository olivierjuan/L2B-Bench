"""SORREL 2024 — the trained actor as a gyozas policy.

The bridge between the method and the harness, exactly like the gasse and
scavuzzo bridges: the trained ``GNNPolicy`` argmaxes over whatever candidates
the runner's environment offers. At benchmark evaluation that is the default
solver profile and the LP branching candidates — the pseudo-candidates regime
and the no-heuristics protocol are training-side only (the fidelity profile
records them), so every method is compared under one solver setting.
"""

from pathlib import Path

from gyozas.runner import Method

from ..gasse.policy import GassePolicy
from .train import load_actor


def sorrel_method(checkpoint: str | Path) -> Method:
    """The sorrel method, ready for the runner's method table.

    The fidelity profile ``sorrel`` recharges the native regime (R5.4): the
    ``NodeBipartite`` observation and the VHB demonstration protocol; the
    policy is the online stage's trained actor.
    """
    return Method(
        name="sorrel",
        policy=lambda: GassePolicy(load_actor(checkpoint)),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "GCNN (Gasse et al., 2019) — TD3BC offline then PPO+SIL online (SORREL)",
        },
        fidelity="sorrel",
    )
