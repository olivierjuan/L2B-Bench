"""Scavuzzo et al. 2022 — the fine-tuned GNNPolicy as a gyozas policy.

At evaluation the policy is pure greedy — action sampling is a training-time
exploration device — over the same ``NodeBipartite`` observation the IL stage
consumed, so the bridge is Gasse's own :class:`GassePolicy`.
"""

from pathlib import Path

from gyozas.runner import Method

from ..gasse.policy import GassePolicy
from .train import VARIANTS, load_trained_checkpoint


def scavuzzo_method(checkpoint: str | Path, variant: str) -> Method:
    """One scavuzzo variant, ready for the runner's method table.

    The fidelity profile ``scavuzzo`` recharges the native regime (R5.4):
    the bipartite-LP observation under the paper's IL-then-REINFORCE
    protocol; the variant (one of ``VARIANTS``) selects the credit
    assignment the policy was fine-tuned with.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    name = f"scavuzzo-{variant.lower().replace('+', '-')}"
    return Method(
        name=name,
        policy=lambda: GassePolicy(load_trained_checkpoint(checkpoint)),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "GCNN (Gasse et al., 2019) fine-tuned with REINFORCE (Scavuzzo et al., 2022)",
            "variant": variant,
        },
        fidelity="scavuzzo",
    )
