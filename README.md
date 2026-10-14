# l2b-bench

https://github.com/olivierjuan/L2B-Bench

The unified learn2branch benchmark: the eight methods of the literature
(Gasse 2019, TreeGate 2021, T-BranT 2022, Lookback 2022, RetroBranching 2022,
Scavuzzo 2022, TGPPO 2025, Mamba-Branching 2026) compared on the same
instances, solver, seeds, and metrics.

**This repository is the benchmark; [GYOZAS](../gyozas) is the harness.** All
mechanisms — policy contract, experts, augmentation, runner, metrics,
reproducibility — live in the `gyozas` library; this repository holds what the
harness must not: the frozen protocol (G2), the strate data and their signed
manifests, the method implementations and their training, the experiment
grids, and the results.

## Layout

```
data/       The three strates of the universal trunk, with their frozen splits
            and signed manifests (R3.1/R3.2). The instance files are not in
            git; rebuild them with scripts/build_strate_data.py, then verify
            with Strate.verify().
configs/    Experiment grids (ExperimentConfig JSON): baselines, methods.
scripts/    Data-building and campaign scripts (e.g. pilot_gasse.py: the
            end-to-end pilot — collect, train, evaluate).
src/        Method implementations: one package per paper, each exposing a
            policy factory consumable by the gyozas runner. Ported from the
            official repositories (see GYOZAS docs/references.md).
protocol/   The G2 protocol freeze: the signed ProtocolSnapshot and the frozen
            protocol document. After the freeze, every campaign is checked
            against it (no renegotiation).
results/    Run records (results/<method>/<strate>/<split>/<seed>/<instance>.json).
            Not in git.
```

## Quickstart

```bash
uv sync   # installs gyozas (path dependency for now) and the method extras
```

```python
from gyozas import ExperimentConfig, Runner
from l2b_bench.methods import BASELINE_METHODS

config = ExperimentConfig.load("configs/baselines.json")
runner = Runner(
    config,
    methods=BASELINE_METHODS,
    strate_dirs={"miplib_s": "data/miplib_s", "milp_l": "data/milp_l"},
    results_dir="results/",
)
runner.run()
```

## Status

- [x] Strate data + signed manifests (moved from GYOZAS)
- [x] Baseline grid (default solver, pscost, relpscost)
- [x] Gasse 2019 pilot: collect SB trajectories → train GCNN → evaluate vs
      the baselines through the runner
      (`src/l2b_bench/methods/gasse/`, `scripts/pilot_gasse.py`,
      `tests/test_gasse_pilot.py`)
- [x] Lookback 2022 port: root+lookback collection, parent-child pairs, PAT
      distillation (`src/l2b_bench/methods/lookback/`,
      `tests/test_lookback_pilot.py`)
- [x] T-BranT 2022 port (LTBranT variant): relpscost expert on the
      BranchingState observation, BranchFormer with tree gating + BiMatching,
      Noam schedule (`src/l2b_bench/methods/tbrant/`,
      `tests/test_tbrant_pilot.py`; the tree GAT of the full T-BranT is
      future work)
- [ ] The five other methods
- [ ] G2 protocol freeze
- [ ] Full campaign (methods × strates × 5 seeds) on the lab cluster
