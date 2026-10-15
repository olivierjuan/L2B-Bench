# l2b-bench

https://github.com/olivierjuan/L2B-Bench

The unified learn2branch benchmark: the methods of the literature
(Gasse 2019, TreeGate 2021, T-BranT 2022, Lookback 2022, RetroBranching 2022,
Scavuzzo 2022, TGPPO 2025, Mamba-Branching 2026, Seq-BnB 2026, and more)
compared on the same instances, solver, seeds, and metrics.

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
- [x] All method ports (one package per paper under `src/l2b_bench/methods/`):
      gasse, lookback, tbrant, treegate, retrobranching, scavuzzo (three
      variants), tgppo, mamba, cambranch, dsccl, fmsts, sorrel, bbmdp, seqbnb,
      plus the `gasse-relpscost` control arm (same protocol as gasse with the
      scalable expert — measured cost of the expert switch: +25.7% nodes)
- [x] Pilot campaign (2026-10): every method trained and evaluated at pilot
      scale on set covers — results below
- [ ] Sorrel pilot campaign (job running; results land in `results/sorrel_gpu_v1`)
- [ ] G2 protocol freeze
- [ ] Full campaign (methods × strates × 5 seeds) on the lab cluster

## Pilot campaign results

Pilot scale, on the setcover strate (500×600×0.10): 6 instances × 5 seeds,
120 s budget, identical protocol/instances/seeds across all campaigns —
baseline columns are bit-identical between campaign summaries, so the
comparisons are paired. Metric: shifted geometric mean of per-instance
B&B-node ratios vs the default solver (lower is better; 1.0× = SCIP default).

| Method | SGM nodes vs default |
|---|---:|
| **seqbnb** (joint branching + node selection, DAgger) | **1.016×** 🏆 |
| lookback | 1.293× |
| scavuzzo tmdp+objlim | 1.529× |
| cambranch | 1.557× |
| dsccl | 1.616× |
| mamba | 1.621× |
| treegate | 1.671× |
| *relpscost* (baseline expert) | *1.677×* |
| gasse | 2.996× |
| retrobranching | 3.162× |
| *pscost* (baseline expert) | *3.436×* |
| tbrant | 3.481× |
| gasse-relpscost (control arm) | 3.766× |
| scavuzzo mdp | 5.074× |
| tgppo | 7.728× |
| scavuzzo tmdp+dfs | 8.620× |
| bbmdp | 9.009× |
| fmsts | 20.263× |

Reading notes, at pilot scale:

- **Seq-BnB reaches near-parity with SCIP's default** (1.016×, best of the
  benchmark; 3 of 30 runs below 1.0×). It is the only port that trains both
  branching and node selection jointly.
- The imitation-learning methods with a full-fidelity expert label
  (seqbnb, lookback) sit at the top; RL-from-scratch methods (tgppo, bbmdp,
  fmsts, retrobranching) are undertrained at this scale — their published
  regimes need far more episodes.
- **gasse vs gasse-relpscost (+25.7%)** prices the switch from strong
  branching to the scalable relpscost expert for labels: relpscost labels are
  much easier to imitate (acc@1 0.73 vs ~0.40) but imitate a weaker expert.
- Cross-family transfer does not hold (the first gasse campaign, trained on
  500×1000×0.05 and evaluated on 500×600×0.10, scored 27.0×); the numbers
  above are all train-family = eval-family.

Full per-run records: `results/<campaign>/` (each `summary.json` carries the
grid's mean node counts, statuses, and the SGM of paired ratios).
