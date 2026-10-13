# Protocol freeze (G2)

Everything before the freeze is negotiable; everything after it is not. The
freeze is a **signed snapshot** of the protocol's moving parts
(`gyozas.ProtocolSnapshot`):

- the strate manifests (content hashes of every instance, split layout),
- the solver profile in force,
- the SCIP / pyscipopt / gyozas versions,
- the provenance notes.

## Freezing (once, when the protocol is settled)

```python
from gyozas import ProtocolSnapshot

snapshot = ProtocolSnapshot.capture(
    name="treerope-g2",
    strate_dirs={"miplib_s": "data/miplib_s",
                 "milp_l": "data/milp_l"},
    provenance={"document": "protocol/G2.md", "decided_by": "Olivier JUAN"},
)
snapshot.save("protocol/treerope-g2.json")
```

## After the freeze

Every campaign checks itself against the snapshot before running
(`ProtocolSnapshot.check_run`): a drifted manifest, solver profile, or version
flags the run instead of silently producing non-comparable results. The
snapshot file is committed here and never edited; a protocol change is a new
snapshot with a new name, decided explicitly.

Still to freeze (decisions pending):

- the per-strate budgets of the baseline grid (`configs/baselines.json`
  currently uses a provisional 1 h time limit),
- the bounded-strong-branching baseline (R6.1's fourth baseline),
- the training/collection protocol of each method (native, per its fidelity
  profile's `training` metadata).
