"""The benchmark's method table.

Every entry maps to a :class:`gyozas.runner.Method`; the harness executes it,
this repository defines it. The classical baselines (R6.1) are wired through
small local fidelity profiles so the experts drive the solver exactly like a
documented method's native expert would — same machinery, no special case.
"""

from gyozas import (
    FidelityProfile,
    Method,
    PseudocostExpert,
    RelpscostExpert,
    register_fidelity_profile,
)

register_fidelity_profile(
    FidelityProfile(
        name="baseline/relpscost",
        expert=lambda: RelpscostExpert(),
        training={"regime": "baseline", "source": "SCIP's native pseudocost scores"},
    )
)
register_fidelity_profile(
    FidelityProfile(
        name="baseline/pscost",
        expert=lambda: PseudocostExpert(),
        training={"regime": "baseline", "source": "SCIP's native pseudocost branching"},
    )
)

# The default-solver baseline: no policy, no expert — every decision falls
# back to SCIP's own branching rule (the 1.000x unit of the benchmark).
BASELINE_METHODS = {
    "default": Method(name="default"),
    "pscost": Method(name="pscost", fidelity="baseline/pscost"),
    "relpscost": Method(name="relpscost", fidelity="baseline/relpscost"),
}
