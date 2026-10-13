"""Lookback 2022 — the trained GCNN as a gyozas policy.

Same architecture and inference as the Gasse port (the papers share the
bipartite GCNN backbone) — the Lookback difference lives in the collection
protocol and the loss. See :mod:`l2b_bench.methods.lookback.collect` and
:mod:`l2b_bench.methods.lookback.train`.
"""

from pathlib import Path

from gyozas.runner import Method

from l2b_bench.methods.gasse.policy import GassePolicy


def lookback_method(checkpoint: str | Path) -> Method:
    """The lookback method, ready for the runner's method table.

    The fidelity profile ``lookback`` is the documented regime (R5.4):
    ``NodeBipartite`` observation, strong-branching expert, and the
    parent-child collection of Gupta et al. (TMLR 2022).
    """
    return Method(
        name="lookback",
        policy=lambda: GassePolicy.from_checkpoint(checkpoint),
        config={
            "checkpoint": str(checkpoint),
            "architecture": "GCNN (Gasse et al., 2019) + Lookback loss (Gupta et al., TMLR 2022)",
        },
        fidelity="lookback",
    )
