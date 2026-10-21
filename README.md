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

## Papers & official repositories

Every method ported in `src/l2b_bench/methods/`, with its paper and the
implementation it was ported from (references verified October 2026).

| Method | Paper | Venue | Ported from |
|---|---|---|---|
| `gasse` | Gasse, Chételat, Ferroni, Ferron, Létourneau-Pinto, Lodi — *Exact Combinatorial Optimization with Graph Convolutional Neural Networks* ([arXiv 1906.01629](https://arxiv.org/abs/1906.01629)) | NeurIPS 2019 | [ds4dm/learn2branch-ecole](https://github.com/ds4dm/learn2branch-ecole) |
| `treegate` | Zarpellon, Jo, Lodi, Bengio — *Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies* ([arXiv 2002.05120](https://arxiv.org/abs/2002.05120)) | AAAI 2021 | [ds4dm/branch-search-trees](https://github.com/ds4dm/branch-search-trees) |
| `tbrant` | Lin et al. — *Learning to Branch with Tree-aware Branching Transformers* ([doi 10.1016/j.knosys.2022.109455](https://doi.org/10.1016/j.knosys.2022.109455)) | KBS 2022 | [linjc16/TBranT](https://github.com/linjc16/TBranT) |
| `lookback` | Gupta, Khalil, Chételat, Gasse, Bengio, Lodi, Kumar — *Lookback for learning to branch* ([arXiv 2206.14987](https://arxiv.org/abs/2206.14987)) | TMLR 2022 | [pg2455/lookback-learn2branch](https://github.com/pg2455/lookback-learn2branch) |
| `retrobranching` | Parsonson, Laterza, Barrett — *Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories* ([arXiv 2205.14345](https://arxiv.org/abs/2205.14345)) | AAAI 2023 | [cwfparsonson/retro_branching](https://github.com/cwfparsonson/retro_branching) |
| `scavuzzo` | Scavuzzo, Chen, Chételat, Gasse, Lodi, Yorke-Smith, Aardal — *Learning to branch with Tree MDPs* ([arXiv 2205.11107](https://arxiv.org/abs/2205.11107)) | AAAI 2022 | [lascavana/rl2branch](https://github.com/lascavana/rl2branch) |
| `cambranch` | Lin, Xiu, You, Chen, Cai — *Contrastive Learning with Augmented MILPs for Learning to Branch* | ICLR 2024 | paper-only port (no official repository) |
| `fmsts` | Etheve, Alès, Bissuel, Juan, Kedad-Sidhoum — *Reinforcement Learning for Variable Selection in a Branch and Bound Algorithm* ([arXiv 2005.10026](https://arxiv.org/abs/2005.10026)) | 2020 | paper-only port |
| `sorrel` | Feng, Yang — *SORREL: Suboptimal-Demonstration-Guided Reinforcement Learning for Learning to Branch* ([arXiv 2412.15534](https://arxiv.org/abs/2412.15534)) | AAAI 2025 | [Shengyu-Feng/SORREL](https://github.com/Shengyu-Feng/SORREL) |
| `tgppo` | Ben Mhamed, Kamal-Idrissi, El Fallah Seghrouchni — *Learning Branching Policies for MILPs with Proximal Policy Optimization* (the tree-gate PPO method; [arXiv 2511.12986](https://arxiv.org/abs/2511.12986)) | AAAI 2026 | [benmhamed-a/tgppo-aaai-26](https://github.com/benmhamed-a/tgppo-aaai-26) |
| `dsccl` | Lu, Ma, Tao — *Dynamic Stratified Contrastive Learning with Upstream Augmentation for MILP Branching* ([arXiv 2511.21107](https://arxiv.org/abs/2511.21107)) | preprint 2025 | paper-only port |
| `mamba` | Mamba-Branching (no paper record found; the repository — anonymous, "DoctorWatson", mirrored from Gitee — is the reference) | 2026 | [doctor-watson626/Mamba-Branching](https://github.com/doctor-watson626/Mamba-Branching) |
| `bbmdp` | Strang, Alès, Bissuel, Juan, Kedad-Sidhoum, Rachelson — *A Markov Decision Process for Variable Selection in Branch & Bound* ([arXiv 2510.19348](https://arxiv.org/abs/2510.19348)) | NeurIPS 2025 | [abfariah/bbmdp](https://github.com/abfariah/bbmdp) |
| `planbb` | Strang, Alès, Bissuel, Juan, Kedad-Sidhoum, Rachelson — *Planning in Branch-and-Bound: Model-Based Reinforcement Learning for Exact Combinatorial Optimization* ([arXiv 2511.09219](https://arxiv.org/abs/2511.09219)) | AAAI 2026 | [abfariah/planbb](https://github.com/abfariah/planbb) |
| `seqbnb` | Liu, Liu, Yan, Shi, Zhao — *Seq-BnB: Sequence Model for Branch-and-Bound Node Selection and Branching* ([arXiv 2610.14835](https://arxiv.org/abs/2610.14835)) | 2026 | paper-only port (code promised upon publication) |
| `gasse-relpscost` | control arm of the benchmark, not a published method: Gasse's protocol with the scalable expert (see the pilot results) | — | — |

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
      variants), tgppo, mamba, cambranch, dsccl, fmsts, sorrel, bbmdp, planbb,
      seqbnb,
      plus the `gasse-relpscost` control arm (same protocol as gasse with the
      scalable expert — measured cost of the expert switch: +25.7% nodes)
- [x] Pilot campaign (2026-10): every method trained and evaluated at pilot
      scale on set covers — results below
- [x] Sorrel pilot campaign (results in `results/sorrel_gpu_v1`)
- [x] PlanB&B pilot campaign (results in `results/planbb_gpu_v1`)
- [x] Seq-BnB BR-only ablation (results in `results/seqbnb_br_gpu_v1`)
- [ ] G2 protocol freeze
- [ ] Full campaign (methods × strates × 5 seeds) on the lab cluster

## Pilot campaign results

Pilot scale, on the setcover strate (500×600×0.10): 6 instances × 5 seeds,
120 s budget, identical protocol/instances/seeds across all campaigns —
baseline columns are bit-identical between campaign summaries, so the
comparisons are paired. Metric: shifted geometric mean of per-instance
B&B-node ratios vs the default solver (lower is better; 1.0× = SCIP default).

| Method | RL | SGM nodes vs default |
|---|:---:|---:|
| **seqbnb-br** (BR-only ablation: learned branching, SCIP default node selection) | no | **0.981×** 🏆 |
| **seqbnb** (joint branching + node selection, DAgger) | no | **1.016×** |
| lookback | no | 1.293× |
| scavuzzo tmdp+objlim | yes | 1.529× |
| cambranch | no | 1.557× |
| **planbb** ‡ (MuZero, Gumbel MCTS, 62 500 steps) | yes | **1.589×** |
| dsccl | no | 1.616× |
| mamba | no | 1.621× |
| treegate | no | 1.671× |
| *relpscost* (baseline expert) | — | *1.677×* |
| gasse | no | 2.996× |
| retrobranching ‡ | yes | 3.162× |
| *pscost* (baseline expert) | — | *3.436×* |
| tbrant | no | 3.481× |
| gasse-relpscost (control arm) | no | 3.766× |
| sorrel ‡ | yes | 3.924× |
| scavuzzo mdp ‡ | yes | 5.074× |
| tgppo § | yes | 7.728× |
| tgppo-fixed (documented deviation, see §) | yes | *pending* |
| scavuzzo tmdp+dfs ‡ | yes | 8.620× |
| bbmdp ‡ | yes | 9.009× |
| fmsts ‡ | yes | 20.263× |

Reading notes, at pilot scale:

- **Seq-BnB reaches near-parity with SCIP's default** (1.016×; 3 of 30 runs
  below 1.0×). It is the only port that trains both branching and node
  selection jointly.
- **The BR-only ablation beats the full joint model** (seqbnb-br 0.981× vs
  seqbnb 1.016×, same DAgger budget and instance pool): at pilot scale,
  SCIP's default node selection (Estimate) is enough — the learned node
  selection adds nothing here, where the paper's Table 9 found BR-only
  slightly *worse* than the full model (34.24 vs 28.62 nodes). The joint
  model spends capacity and x*-oracle supervision on a subproblem SCIP
  already solves well at this scale.
- **Generalization to 1000×1200×0.10** (same checkpoints, 900 s budget,
  results in `results/gen1000x1200_v2_*`): relpscost 1.531× (0 timelimit),
  gasse 2.676× (0), pscost 4.408× (0), planbb ≥ 5.242× (5/30 timelimit, all
  on the hardest instance — its SGM is right-censored). The learned
  policies degrade out of distribution while relpscost barely moves
  (1.677× → 1.531×); planbb, the best RL method in-distribution, suffers
  the most.
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
- ‡ **Undertrained at pilot scale** — the row measures the method at a small
  slice of its published training regime, not its quality. Effective budgets
  consumed: sorrel 1 PPO epoch after its offline stage, bbmdp 100 episodes /
  11.8k gradient steps, retrobranching 100 episodes / 7.7k gradient updates,
  fmsts 1k iterations, the scavuzzo mdp/tmdp+dfs variants likewise. planbb
  ran the official gradient-step count (62 500) but starved on experience:
  sequential self-play collected 65 episodes / 15.4k transitions — less than
  the official regime's 40k-transition starting buffer (sized for 11
  parallel actors) — and its loss plateaued from ~step 36k on the static
  replay. Convergence campaigns for the ‡ methods (planbb included, with a
  larger experience budget) are the next step before the G2 freeze.
- § **TGPPO as released outputs a uniform policy — need to contact the
  authors.** The tgppo actor gives every candidate the same probability
  (uniform to 1e-7) at initialization and throughout training: all 39
  snapshots of a 40-update convergence campaign (5 000 episodes), evaluated
  on real branching states. The official modules from
  [benmhamed-a/tgppo-aaai-26](https://github.com/benmhamed-a/tgppo-aaai-26),
  run as released, do the same for every size and seed tried, and on a
  trivial supervised ranking task their loss stays at exactly ln(n). The
  port is faithful; two collapse points sit in the released architecture:
  `BiMatchingNet` hands each candidate the tree vector weighted by a single
  softmax scalar (~1/L), so the encoded candidate features barely reach the
  head (spread across candidates 0.57 → 0.0035), and the gated output
  pyramid narrows to 2 units under `LayerNorm(2)`, which keeps at most one
  bit per candidate (logit spread 7e-8). The tgppo row is therefore uniform
  random branching, whatever the budget. The official hyperparameters come
  from an unpublished tuning file (`--best_params_json`), so the authors may
  have run a different revision — we have not yet contacted them. The
  **tgppo-fixed** variant (`--method tgppo-fixed`) adds a residual
  connection around `BiMatchingNet` and stops the pyramid at 8 units (the
  official module's own default `infimum`) under a `Linear(8, 1)` score; it
  learns the probe task, and its PPO campaign is running.
- Sorrel hit the 120 s time limit on 23 of 30 runs at pilot scale — the PPO
  stage did not learn within the pilot budget, so its node ratios are
  right-censored by the budget rather than converged (the SGM is an
  over-optimistic lower bound on its node count).

Full per-run records: `results/<campaign>/` (each `summary.json` carries the
grid's mean node counts, statuses, and the SGM of paired ratios).
