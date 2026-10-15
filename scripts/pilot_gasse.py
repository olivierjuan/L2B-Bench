"""The pilot campaign: collect → train → evaluate, end to end.

A small, fully reproducible demonstration that the benchmark loop works on
the harness — not a scientific result. Everything runs on CPU in minutes:

1. **Instances** — a handful of held-out set covers (the paper's synthetic
   family, scaled down) written as a pilot strate;
2. **Collect** — expert trajectories on training set covers (a different
   generator seed than the strate's), under the selected method's protocol;
3. **Train** — the GCNN under the official imitation-learning loop (with
   the Lookback loss for the lookback method);
4. **Evaluate** — the trained policy against the classical baselines
   through the runner: the same config-driven grid, records, and
   publication guard every method of the benchmark will use.

The pilot deliberately runs the *benchmark* path (the Runner), not a
hand-rolled evaluation loop: the point is to validate the harness and the
method-porting recipe, and to leave a template for the real campaigns.

Usage::

    uv run python scripts/pilot_gasse.py [--method gasse] [--episodes 20] [--epochs 20]

Results land in ``results/pilot_gasse/`` (gitignored): the instance strate,
the checkpoint, the run records, and a summary JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from gyozas.metrics import paired_per_instance_ratios, sgm_of_ratios
from gyozas.runner import Cell, ExperimentConfig, Method, Runner

from l2b_bench.methods.baselines import BASELINE_METHODS
from l2b_bench.methods.bbmdp.policy import bbmdp_method
from l2b_bench.methods.cambranch.policy import cambranch_method
from l2b_bench.methods.dsccl.policy import dsccl_method
from l2b_bench.methods.fmsts.policy import fmsts_method
from l2b_bench.methods.gasse.model import GNNPolicy
from l2b_bench.methods.gasse.policy import gasse_method
from l2b_bench.methods.gasse.train import (
    collate,
    get_device,
    pretrain,
    process,
    save_checkpoint,
    to_device,
)
from l2b_bench.methods.lookback.collect import (
    collect_episode as lookback_collect_episode,
)
from l2b_bench.methods.lookback.policy import lookback_method
from l2b_bench.methods.lookback.train import (
    LookbackBatch,
    collate_pairs,
    lb_to_device,
    process_lookback,
)
from l2b_bench.methods.mamba.collect import (
    MAMBA_K,
    MAMBA_S,
)
from l2b_bench.methods.mamba.collect import (
    collect_episode as mamba_collect_episode,
)
from l2b_bench.methods.mamba.policy import mamba_method
from l2b_bench.methods.retrobranching.policy import retrobranching_method
from l2b_bench.methods.scavuzzo.policy import scavuzzo_method
from l2b_bench.methods.seqbnb.policy import seqbnb_method
from l2b_bench.methods.sorrel.policy import sorrel_method
from l2b_bench.methods.tbrant.collect import collect_episode as tbrant_collect_episode
from l2b_bench.methods.tbrant.policy import tbrant_method
from l2b_bench.methods.tgppo.policy import tgppo_method
from l2b_bench.methods.treegate.collect import (
    TREEGATE_K,
    TREEGATE_S,
)
from l2b_bench.methods.treegate.collect import (
    collect_episode as treegate_collect_episode,
)
from l2b_bench.methods.treegate.policy import treegate_method

# The ported methods this pilot can drive, with their protocol knobs.
METHOD_SPECS = {
    "gasse": {
        "prob": 0.5,
        "method": gasse_method,
        "cache_name": "expert_samples.pt",
    },
    "lookback": {
        "prob": 0.05,  # the official node_record_prob
        "method": lookback_method,
        "cache_name": "expert_samples.pt",
    },
    # T-BranT sees a different observation than the bipartite-graph methods,
    # so its samples cannot share their cache. Batch size 20 is the official
    # train_LTBranT.sh (the bipartite-graph methods use the pilot's 32).
    "tbrant": {
        "prob": None,
        "method": tbrant_method,
        "cache_name": "tbrant_samples.pt",
        "batch_size": 20,
    },
    # TreeGate shares T-BranT's observation (Zarpellon's original triple) but
    # collects under the (s, k) augmentation — its own cache again.
    "treegate": {
        "prob": None,
        "method": treegate_method,
        "cache_name": "treegate_samples.pt",
    },
    # Mamba collects like TreeGate (the same (s, k) cell) but keeps each
    # episode's samples grouped — the decision order is the training signal
    # of its path model, so the cache holds trajectories, not flat samples.
    "mamba": {
        "prob": None,
        "method": mamba_method,
        "cache_name": "mamba_trajectories.pt",
    },
    # RetroBranching is the benchmark's first RL method: no expert to collect
    # from, the agent itself explores (episodes solve fresh training
    # instances), so it has no sample cache.
    "retrobranching": {
        "prob": None,
        "method": retrobranching_method,
        "cache_name": None,
    },
    # Scavuzzo's three variants share one IL pretraining stage (collected
    # exactly like gasse's data, expert_probability 0.05) and differ only in
    # the REINFORCE credit assignment — see l2b_bench.methods.scavuzzo.
    "scavuzzo-mdp": {
        "prob": 0.05,
        "method": lambda checkpoint: scavuzzo_method(checkpoint, "mdp"),
        "cache_name": "expert_samples.pt",
        "variant": "mdp",
    },
    "scavuzzo-tmdp-dfs": {
        "prob": 0.05,
        "method": lambda checkpoint: scavuzzo_method(checkpoint, "tmdp+DFS"),
        "cache_name": "expert_samples.pt",
        "variant": "tmdp+DFS",
    },
    "scavuzzo-tmdp-objlim": {
        "prob": 0.05,
        "method": lambda checkpoint: scavuzzo_method(checkpoint, "tmdp+ObjLim"),
        "cache_name": "expert_samples.pt",
        "variant": "tmdp+ObjLim",
    },
    # TGPPO is PPO from scratch: no expert and no IL stage, so no sample
    # cache either. The sandbox protocol (inline baseline solves, cutoff at
    # the optimum, heuristics off) lives inside the method's train module.
    "tgppo": {
        "prob": None,
        "method": tgppo_method,
        "cache_name": None,
    },
    # SORREL collects suboptimal VHB demonstrations (full strong branching
    # w.p. 0.05, else pseudocosts) over the pseudo branching candidates — its
    # own cache: the tree-MDP samples carry two children's graphs, nothing
    # like the gasse samples.
    "sorrel": {
        "prob": 0.05,  # the official FSB_probability
        "method": sorrel_method,
        "cache_name": "sorrel_transitions.pt",
    },
    # BBMDP is DQN from scratch on the corrected vanilla MDP: the agent
    # itself explores (fresh instances every episode), so no sample cache —
    # like retrobranching and tgppo.
    "bbmdp": {
        "prob": None,
        "method": bbmdp_method,
        "cache_name": None,
    },
    # CAMBranch collects exactly like gasse (the paper's SCIP 6.0.1
    # root-cuts-only, no-restart setting coincides with Gasse's collection
    # parameters) and diverges at training: 10% of the samples plus the free
    # variable-shifted AMILPs, under the joint contrastive loss.
    "cambranch": {
        "prob": 0.5,
        "method": cambranch_method,
        "cache_name": "expert_samples.pt",
    },
    # DSC-CL collects exactly like gasse too (the paper's regime is Gasse's:
    # the Easy level of the 4 Ecole families, FSB expert) and diverges at
    # training (stratified contrastive loss over K-means node groups) and at
    # inference (hybrid top-5 + full SB at upstream nodes).
    "dsccl": {
        "prob": 0.5,
        "method": dsccl_method,
        "cache_name": "expert_samples.pt",
    },
    # FMSTS is RL from scratch with observed subtree-size Q-values: the agent
    # explores a fixed instance pool (redrawn every iteration, like the
    # paper's 200 training instances), so no sample cache either.
    "fmsts": {
        "prob": None,
        "method": fmsts_method,
        "cache_name": None,
    },
    # Seq-BnB is the joint regime: one policy for branching AND node
    # selection, DAgger-trained (expert round, then its own episodes with
    # expert labels). The pool lives on disk — the node-selection oracle
    # needs each training instance's offline reference optimum x*.
    "seqbnb": {
        "prob": None,
        "method": seqbnb_method,
        "cache_name": None,
    },
}

# SORREL's episode filter, scaled to the pilot's instances: the official
# threshold (100 decisions) would drop every pilot episode outright.
SORREL_SAMPLING_THRESHOLD = 10


def parse_family(text: str) -> dict:
    """``"500x1000x0.05"`` → ``dict(n_rows=500, n_cols=1000, density=0.05)``."""
    rows, cols, density = text.split("x")
    return {"n_rows": int(rows), "n_cols": int(cols), "density": float(density)}


# The pilot's defaults — the paper's set covers, scaled down so a CPU run
# finishes in minutes. Training and evaluation draws come from different
# generator seeds (disjoint instances); the evaluation family is harder than
# the training one (default solver: ~600 B&B nodes per instance) so the node
# counts are not degenerate. Both families are CLI-overridable for the
# cluster campaigns.
TRAIN_FAMILY = {"n_rows": 200, "n_cols": 250, "density": 0.10}
TRAIN_RNG_SEED = 123
EVAL_FAMILY = {"n_rows": 500, "n_cols": 600, "density": 0.10}
EVAL_RNG_SEED = 2024
N_EVAL_INSTANCES = 6

# The pilot's protocol knobs (the paper's, scaled down).
EVAL_SEEDS = list(range(5))  # the protocol's minimum (MIN_SEEDS)
EVAL_BUDGET = {"limits/time": 120.0, "display/verblevel": 0}
BATCH_SIZE = 32


def write_eval_instances(strate_root: Path, eval_family: dict) -> None:
    """Write the pilot's held-out set covers as a one-split strate."""
    if strate_root.exists() and any((strate_root / "test").glob("*.mps")):
        return
    from gyozas.instances import SetCoverGenerator

    split = strate_root / "test"
    split.mkdir(parents=True, exist_ok=True)
    generator = SetCoverGenerator(
        rng=np.random.default_rng(EVAL_RNG_SEED), **eval_family
    )
    for i in range(N_EVAL_INSTANCES):
        model = next(generator)
        model.writeProblem(
            filename=str(split / f"sc_{i:02d}.mps"), genericnames=True, verbose=False
        )
    print(f"[instances] wrote {N_EVAL_INSTANCES} set covers to {split}")


def _collect_one_episode(job):
    """One collection episode in a worker process: its own env, its own SCIP.

    The instance file and base seed fully determine the episode, so parallel
    collection produces exactly the sequential campaign's data.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    path, seed, kind, probability, time_limit = job
    from gyozas.augmentation import ZarpellonAugmentation
    from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
    from gyozas.environment import Environment
    from gyozas.informations.solve_status import SolveStatus
    from gyozas.instances.files import FileGenerator
    from gyozas.observations import BranchingStateObservation, NodeBipartite
    from gyozas.seeds import SeedTriplet

    from l2b_bench.methods.gasse.collect import GASSE_SCIP_PARAMS, collect_episode

    if kind in ("tbrant", "treegate", "mamba"):
        # Default SCIP: the solver-stats protocols keep heuristics on (the
        # official 'sandbox' deviation is documented in each collect.py).
        scip_params = {}
    elif kind == "sorrel":
        # The official VHB protocol: no cuts, no restarts of either kind, all
        # primal heuristics off — and the pseudo branching candidates.
        from l2b_bench.methods.sorrel.collect import SORREL_SCIP_PARAMS

        scip_params = dict(SORREL_SCIP_PARAMS)
    else:
        scip_params = dict(GASSE_SCIP_PARAMS)
    scip_params["limits/time"] = time_limit
    scip_params["display/verblevel"] = 0
    observation = (
        BranchingStateObservation()
        if kind in ("tbrant", "treegate", "mamba")
        else NodeBipartite()
    )
    if kind == "sorrel":
        dynamics = BranchingDynamics(pseudo_candidates=True)
    else:
        dynamics = BranchingDynamics(with_extra_actions=[ExtraBranchingActions.SKIP])
    if kind == "treegate":
        # The (s, k) augmentation cell: the dynamics takes the k random-explorer
        # decisions itself and seeds the variable permutation.
        dynamics = BranchingDynamics(
            augmentation=ZarpellonAugmentation(seed=TREEGATE_S, k=TREEGATE_K),
            with_extra_actions=[ExtraBranchingActions.SKIP],
        )
    elif kind == "mamba":
        # The same (s, k) shape as treegate, on mamba's own cell (the same
        # s = 0, k = 10 defaults — see methods/mamba/collect.py).
        dynamics = BranchingDynamics(
            augmentation=ZarpellonAugmentation(seed=MAMBA_S, k=MAMBA_K),
            with_extra_actions=[ExtraBranchingActions.SKIP],
        )
    env = Environment(
        instance_generator=FileGenerator(
            directory=path.parent,
            pattern=path.name,
            sampling_mode="replace",
            rng=np.random.default_rng(0),
        ),
        observation_function=observation,
        information_function=SolveStatus(),
        scip_params=scip_params,
        dynamics=dynamics,
    )
    try:
        triplet = SeedTriplet.from_base_seed(seed)
        env.seed(triplet.solver)
        triplet.seed_python()
        if kind in ("tbrant", "treegate", "mamba"):
            # The solver-stats methods share the same collection shape: the
            # relpscost expert at every exposed decision (for treegate and
            # mamba, after the dynamics' k explorer decisions).
            if kind == "treegate":
                samples, n_nodes, status = treegate_collect_episode(env)
            elif kind == "mamba":
                samples, n_nodes, status = mamba_collect_episode(env)
            else:
                samples, n_nodes, status = tbrant_collect_episode(env)
        else:
            rng = np.random.default_rng(triplet.augmentation)
            if kind == "lookback":
                samples, n_nodes, status = lookback_collect_episode(
                    env, rng, probability
                )
            elif kind == "sorrel":
                from l2b_bench.methods.sorrel.collect import (
                    LENGTH_LIMIT,
                )
                from l2b_bench.methods.sorrel.collect import (
                    collect_episode as sorrel_collect_episode,
                )

                samples, n_nodes, status = sorrel_collect_episode(
                    env,
                    rng,
                    fsb_probability=probability,
                    length_limit=LENGTH_LIMIT,
                    sampling_threshold=SORREL_SAMPLING_THRESHOLD,
                )
            else:
                samples, n_nodes, status = collect_episode(env, rng, probability)
        return samples, n_nodes, status
    finally:
        env.close()


def collect_expert_data(
    n_episodes: int,
    train_family: dict,
    time_limit: float = 60.0,
    cache: Path | None = None,
    n_workers: int = 1,
    kind: str = "gasse",
) -> list:
    """Explore-then-strong-branch trajectories on the training family.

    Instances are drawn once from the generator and written to disk (episode i
    gets the i-th draw — identical to the sequential campaign), then episodes
    run in parallel worker processes, each with its own SCIP environment.
    With ``cache``, the collected samples are saved there and reloaded on the
    next call — a restarted campaign never re-collects.
    """
    import torch

    if cache is not None and cache.exists():
        samples = torch.load(cache, weights_only=False)
        print(
            f"[collect] reusing the cached collection from {cache} ({len(samples)} entries)",
            flush=True,
        )
        return samples

    from gyozas.instances import SetCoverGenerator

    # Deterministic instance pool: written once, reused across restarts.
    instance_dir = cache.parent / "train_instances" if cache is not None else None
    instance_dir = instance_dir or Path("train_instances")
    instance_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    for i in range(n_episodes):
        path = instance_dir / f"train_{i:03d}.mps"
        if not path.exists():
            next(generator).writeProblem(
                filename=str(path), genericnames=True, verbose=False
            )
        paths.append(path)
    print(f"[collect] {n_episodes} training instances in {instance_dir}", flush=True)

    jobs = [
        (path, i, kind, METHOD_SPECS[kind]["prob"], time_limit)
        for i, path in enumerate(paths)
    ]
    # Mamba trains on trajectories: each episode's samples stay grouped (the
    # decision order is its path model's input); every other method flattens.
    keep_trajectories = kind == "mamba"
    all_samples: list = []
    statuses: list[str] = []
    n_nodes_total = 0

    def _absorb(samples) -> None:
        if keep_trajectories:
            all_samples.append(samples)
        else:
            all_samples.extend(samples)

    if n_workers <= 1:
        results = map(_collect_one_episode, jobs)
        for done, (samples, n_nodes, status) in enumerate(results, 1):
            _absorb(samples)
            statuses.append(status)
            n_nodes_total += n_nodes
            print(
                f"[collect] episode {done}/{n_episodes}: {status}, {n_nodes} nodes",
                flush=True,
            )
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_collect_one_episode, job): job for job in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                samples, n_nodes, status = future.result()
                _absorb(samples)
                statuses.append(status)
                n_nodes_total += n_nodes
                print(
                    f"[collect] episode {done}/{n_episodes}: {status}, {n_nodes} nodes",
                    flush=True,
                )

    solved = sum(1 for s in statuses if s == "optimal")
    n_decisions = (
        sum(len(episode) for episode in all_samples)
        if keep_trajectories
        else len(all_samples)
    )
    print(
        f"[collect] {n_episodes} episodes ({solved} optimal), "
        f"{n_nodes_total} nodes, {len(all_samples)} "
        f"{'trajectories' if keep_trajectories else 'expert samples'} "
        f"({n_decisions} decisions)",
        flush=True,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_samples, cache)
        print(f"[collect] samples cached to {cache}", flush=True)
    return all_samples


def train_retrobranching(
    episodes: int,
    checkpoint: Path,
    train_family: dict,
    time_limit: float = 60.0,
    buffer_min_length: int | None = None,
) -> None:
    """The official DQN loop: the agent explores, no expert in the loop.

    Each episode solves a fresh instance drawn from the training family (a
    solved model cannot drive a second environment), under the method's own
    gasse_2019 solver parameters. ``episodes`` is the RL budget — the
    imitation methods' ``--epochs`` has no equivalent here.
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.retrobranching import train
    from l2b_bench.methods.retrobranching.train import BUFFER_MIN_LENGTH

    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    train(
        (next(generator) for _ in range(episodes)),
        checkpoint,
        episodes=episodes,
        buffer_min_length=buffer_min_length or BUFFER_MIN_LENGTH,
        time_limit=time_limit,
        seed=0,
    )


def train_scavuzzo(
    il_checkpoint: Path,
    checkpoint: Path,
    episodes: int,
    train_family: dict,
    variant: str,
    time_limit: float = 60.0,
) -> None:
    """The RL stage: REINFORCE fine-tuning of the IL-pretrained GNNPolicy.

    The IL stage is the pilot's own ``train_policy`` (gasse loss over the
    prob-0.05 expert samples) — run by ``main`` before this. Each episode
    solves a fresh instance drawn from the training family; the tmdp+ObjLim
    variant additionally solves each instance to optimality first (the
    objective limit needs it), so it costs roughly twice the solve time.
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.scavuzzo.train import train as train_rl

    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    train_rl(
        (next(generator) for _ in range(episodes)),
        il_checkpoint,
        checkpoint,
        episodes=episodes,
        variant=variant,
        time_limit=time_limit,
        seed=0,
    )


def train_tgppo(
    episodes: int,
    checkpoint: Path,
    train_family: dict,
    time_limit: float = 60.0,
) -> None:
    """The official PPO loop: from scratch, no imitation stage.

    Each episode solves a fresh instance drawn from the training family
    under the paper's 'sandbox' setting — the method's own train module
    solves each instance once for the reward baseline and sets the cutoff
    at that optimum, so an episode costs roughly two solves. One PPO
    update every EPISODES_PER_UPDATE (125) rollouts.
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.tgppo.train import train

    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    train(
        (next(generator) for _ in range(episodes)),
        checkpoint,
        episodes=episodes,
        time_limit=time_limit,
        seed=0,
    )


def train_sorrel(
    samples: list,
    epochs: int,
    checkpoint: Path,
    train_family: dict,
    episodes: int,
    time_limit: float = 60.0,
) -> None:
    """The two official stages: TD3BC offline over the VHB demonstrations,
    then PPO + SIL online on fresh training instances.

    Pilot-scale: the offline epochs cover one pass over the demonstrations
    each (the official draws 624 batches with replacement from 100k samples),
    and the online stage skips the official 10 buffer-filling pre-rollouts —
    the pilot's episode budget goes straight to learning updates.
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.sorrel.train import train_online, train_td3bc

    td3bc_checkpoint = checkpoint.parent / "sorrel_td3bc.pt"
    if td3bc_checkpoint.exists():
        # Idempotent restart: the offline stage is deterministic (seed 0) and
        # takes ~5h at pilot scale — a re-submission continues from it.
        print(f"[train] reusing the cached td3bc stage from {td3bc_checkpoint}")
    else:
        train_td3bc(
            samples,
            td3bc_checkpoint,
            epochs=epochs,
            epoch_size=max(1, len(samples) // 32),
            seed=0,
        )

    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    train_online(
        (next(generator) for _ in range(episodes)),
        td3bc_checkpoint,
        checkpoint,
        episodes=episodes,
        time_limit=time_limit,
        seed=0,
        pre_rollouts=0,
    )


def train_bbmdp(
    episodes: int,
    checkpoint: Path,
    train_family: dict,
    epochs: int,
    time_limit: float = 60.0,
) -> None:
    """The official DQN loop: from scratch on the corrected vanilla MDP — no
    expert data, the agent itself explores.

    Each episode solves a fresh instance drawn from the training family
    under the paper's DFS protocol (pre-solved draws are skipped, like the
    official redraw). ``--episodes`` bounds the episodes; ``--epochs`` bounds
    the gradient-step budget at the pilot's scale (the official counts 200k
    gradient steps over unbounded episodes — here 625 gradient steps per
    epoch, the other pilots' batch-per-epoch scale).
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.bbmdp.train import train

    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    train(
        generator,  # fresh instances forever; the episode budget stops the loop
        checkpoint,
        episodes=episodes,
        n_epochs=epochs * 625,
        time_limit=time_limit,
        seed=42,
    )


def train_fmsts(
    episodes: int,
    checkpoint: Path,
    train_family: dict,
    epochs: int,
    eval_family: dict,
    time_limit: float = 60.0,
) -> None:
    """The paper's Algorithm 1: RL from scratch with observed subtree-size
    Q-values — no expert, the agent itself explores a fixed pool.

    The paper draws each iteration's instance from 200 fixed training
    instances; the pilot writes/reuses ``episodes`` set covers and redraws
    uniformly (a fresh model is read per draw — a solved model cannot drive a
    second environment). ``--episodes`` sizes the pool, ``--epochs`` scales
    the iteration budget (episodes x epochs iterations, one gradient update
    each, like Algorithm 1). The canonical padding is the evaluation family's
    shape so the fitted PCA applies to the benchmark's instances.
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.fmsts.train import train

    instance_dir = checkpoint.parent / "fmsts_train_instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    for i in range(episodes):
        path = instance_dir / f"train_{i:03d}.mps"
        if not path.exists():
            next(generator).writeProblem(
                filename=str(path), genericnames=True, verbose=False
            )
    print(f"[train] {episodes} pool instances in {instance_dir}", flush=True)
    train(
        instance_dir,
        checkpoint,
        iterations=episodes * max(1, epochs),
        canonical=(eval_family["n_rows"], eval_family["n_cols"]),
        time_limit=time_limit,
        seed=42,
    )


def train_seqbnb(
    episodes: int,
    checkpoint: Path,
    train_family: dict,
    epochs: int,
    time_limit: float = 60.0,
) -> None:
    """The paper's DAgger over the joint regime: expert round 0, then the
    policy's own episodes with the expert's labels.

    ``--episodes`` sizes the on-disk pool (each instance's reference optimum
    x* is solved offline once, then cached) and the episodes per round;
    ``--epochs`` scales the DAgger round count (the paper: 10k expert
    samples, then 11 rounds of ~10k — 120k decisions in total).
    """
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.seqbnb.train import train

    instance_dir = checkpoint.parent / "seqbnb_train_instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    generator = SetCoverGenerator(
        rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family
    )
    for i in range(episodes):
        path = instance_dir / f"train_{i:03d}.mps"
        if not path.exists():
            next(generator).writeProblem(
                filename=str(path), genericnames=True, verbose=False
            )
    print(f"[train] {episodes} pool instances in {instance_dir}", flush=True)
    train(
        instance_dir,
        checkpoint,
        rounds=max(1, epochs),
        episodes_per_round=episodes,
        time_limit=time_limit,
    )


def train_mamba(
    trajectories: list, epochs: int, checkpoint: Path, train_family: dict
) -> None:
    """The official two-stage IL: contrastive pretraining, then the Mamba.

    ``epochs`` drives both stages (the paper runs 50 each); the trajectories
    are the collected relpscost-expert episodes, kept in decision order.
    """
    from l2b_bench.methods.mamba.train import train

    train(trajectories, checkpoint, epochs=epochs, clip_epochs=epochs)


def train_policy(
    samples: list,
    epochs: int,
    checkpoint: Path,
    train_family: dict,
    kind: str = "gasse",
) -> None:
    """The official imitation-learning loop: pretrain, then epochs of Adam.

    The loss is the selected method's: Gasse's candidate cross-entropy,
    Lookback's (one-hot target + the PAT distillation term, the paper's
    set-cover hyperparameters), or T-BranT's (candidate cross-entropy over
    the BranchFormer, AdamW + the Noam schedule).
    """
    import torch

    batch_size = METHOD_SPECS[kind].get("batch_size", BATCH_SIZE)

    if kind == "treegate":
        from l2b_bench.methods.treegate.model import TreeGatePolicy
        from l2b_bench.methods.treegate.train import (
            LR,
            LR_DECAY_EPOCHS,
            LR_DECAY_FACTOR,
            WEIGHT_DECAY,
            sample_to_device,
            sample_to_tensors,
        )
        from l2b_bench.methods.treegate.train import (
            process as process_treegate,
        )
        from l2b_bench.methods.treegate.train import (
            save_checkpoint as save_treegate_checkpoint,
        )

        model = TreeGatePolicy()
        shuffle = np.random.default_rng(0)

        def batches(device):
            # One node at a time, like the official loop — a batch is a list
            # of per-node tensor dicts.
            order = shuffle.permutation(len(samples))
            return [
                [
                    sample_to_device(sample_to_tensors(samples[i]), device)
                    for i in order[k : k + batch_size]
                ]
                for k in range(0, len(order), batch_size)
            ]

        device = get_device()
        print(f"[train] device: {device}")
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, LR_DECAY_EPOCHS, LR_DECAY_FACTOR
        )
        for epoch in range(1, epochs + 1):
            stats = process_treegate(model, batches(device), optimizer=optimizer)
            scheduler.step()
            kacc = " ".join(
                f"acc@{k}={a:.2f}"
                for k, a in zip((1, 3, 5, 10), stats["kacc"], strict=True)
            )
            print(f"[train] epoch {epoch:3d}  loss={stats['loss']:.4f}  {kacc}")

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_treegate_checkpoint(
            model.cpu(),
            checkpoint,
            extra={
                "n_samples": len(samples),
                "epochs": epochs,
                "train_family": train_family,
            },
        )
        print(f"[train] checkpoint saved to {checkpoint}")
        return

    if kind == "tbrant":
        from l2b_bench.methods.tbrant.model import BranchFormer
        from l2b_bench.methods.tbrant.train import (
            LR,
            WARMUP_EPOCHS,
            NoamLR,
            collate_stats,
        )
        from l2b_bench.methods.tbrant.train import (
            process as process_tbrant,
        )
        from l2b_bench.methods.tbrant.train import (
            save_checkpoint as save_tbrant_checkpoint,
        )
        from l2b_bench.methods.tbrant.train import (
            to_device as stats_to_device,
        )

        model = BranchFormer()
        shuffle = np.random.default_rng(0)

        def batches(device):
            order = shuffle.permutation(len(samples))
            return [
                stats_to_device(
                    collate_stats([samples[i] for i in order[k : k + batch_size]]),
                    device,
                )
                for k in range(0, len(order), batch_size)
            ]

        device = get_device()
        print(f"[train] device: {device}")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = NoamLR(optimizer, WARMUP_EPOCHS)
        for epoch in range(1, epochs + 1):
            stats = process_tbrant(model, batches(device), optimizer=optimizer)
            scheduler.step()
            kacc = " ".join(
                f"acc@{k}={a:.2f}"
                for k, a in zip((1, 3, 5, 10), stats["kacc"], strict=True)
            )
            print(f"[train] epoch {epoch:3d}  loss={stats['loss']:.4f}  {kacc}")

        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_tbrant_checkpoint(
            model.cpu(),
            checkpoint,
            extra={
                "n_samples": len(samples),
                "epochs": epochs,
                "train_family": train_family,
            },
        )
        print(f"[train] checkpoint saved to {checkpoint}")
        return

    if kind == "cambranch":
        from l2b_bench.methods.cambranch.train import PAIRS_PER_BATCH, subsample_budget
        from l2b_bench.methods.cambranch.train import train as train_cambranch

        # The paper's data budget: 10% of the strong-branching samples — the
        # 90k AMILPs are free by feature transformation (Thm 3.1).
        kept = subsample_budget(samples)
        print(
            f"[train] cambranch: {len(kept)} of {len(samples)} samples (the paper's 10% budget)"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        train_cambranch(
            kept,
            checkpoint,
            epochs=epochs,
            pairs_per_batch=PAIRS_PER_BATCH,
            extra={
                "n_samples": len(kept),
                "epochs": epochs,
                "train_family": train_family,
            },
        )
        return

    if kind == "dsccl":
        from l2b_bench.methods.dsccl.train import train as train_dsccl

        # The paper trains on the full Easy-level collection (100k samples)
        # with a fresh derivation draw per epoch — no budget subsampling.
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        train_dsccl(
            samples,
            checkpoint,
            epochs=epochs,
            extra={
                "n_samples": len(samples),
                "epochs": epochs,
                "train_family": train_family,
            },
        )
        return

    model = GNNPolicy()
    shuffle = np.random.default_rng(0)

    if kind == "lookback":
        from l2b_bench.methods.lookback.collect import pair_parent_children

        pairs = pair_parent_children(samples)
        print(f"[train] {len(pairs)} parent-child pairs out of {len(samples)} samples")

        def batches(device):
            # The batch holds children (global indices, shuffled); each paired
            # child's parent graph rides along in whatever batch its child
            # lands in — collate_pairs appends it, so shuffling never breaks
            # the parent-child pairing.
            order = shuffle.permutation(len(samples))
            return [
                lb_to_device(
                    collate_pairs(samples, list(order[k : k + batch_size]), pairs),
                    device,
                )
                for k in range(0, len(order), batch_size)
            ]

    else:

        def batches(device):
            order = shuffle.permutation(len(samples))
            return [
                to_device(
                    collate([samples[i] for i in order[k : k + batch_size]]), device
                )
                for k in range(0, len(order), batch_size)
            ]

    device = get_device()
    print(f"[train] device: {device}")
    model = model.to(device)
    n_layers = pretrain(
        model,
        [lb.batch if isinstance(lb, LookbackBatch) else lb for lb in batches(device)],
    )
    print(f"[train] froze {n_layers} PreNorm layers")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        if kind == "lookback":
            stats = process_lookback(model, batches(device), optimizer=optimizer)
        else:
            stats = process(model, batches(device), optimizer=optimizer)
        kacc = " ".join(
            f"acc@{k}={a:.2f}"
            for k, a in zip((1, 3, 5, 10), stats["kacc"], strict=True)
        )
        print(f"[train] epoch {epoch:3d}  loss={stats['loss']:.4f}  {kacc}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model.cpu(),
        checkpoint,
        extra={
            "n_samples": len(samples),
            "epochs": epochs,
            "train_family": train_family,
        },
    )
    print(f"[train] checkpoint saved to {checkpoint}")


def run_benchmark(checkpoint: Path, results_dir: Path, kind: str = "gasse") -> dict:
    """The trained policy vs the classical baselines on the pilot grid."""
    methods: dict[str, Method] = dict(BASELINE_METHODS)
    methods[kind] = METHOD_SPECS[kind]["method"](checkpoint)
    strate_root = results_dir / "instances" / "setcover_pilot"
    config = ExperimentConfig(
        cells=[
            Cell(
                method=name,
                strate="setcover_pilot",
                split="test",
                seeds=EVAL_SEEDS,
                profile="default",
                budget=dict(EVAL_BUDGET),
            )
            for name in methods
        ]
    )
    runner = Runner(
        config, methods, {"setcover_pilot": strate_root}, results_dir / "runs"
    )
    report = runner.run()
    for cell_key, counts in report.items():
        print(
            f"[runner] {cell_key}: {counts['executed']} executed, {counts['skipped']} skipped"
        )
    return summarize(runner, config, results_dir)


def summarize(runner: Runner, config: ExperimentConfig, results_dir: Path) -> dict:
    """Per-method mean nodes, paired ratios against the default solver (R6.2)."""
    nodes: dict[str, dict[str, float]] = {}
    statuses: dict[str, dict[str, str]] = {}
    for cell in config.cells:
        results = runner.collect_cell(cell)
        per_instance: dict[str, float] = {}
        per_status: dict[str, str] = {}
        for name in results[next(iter(results))]:
            per_instance[name] = float(
                np.mean([results[seed][name]["counters"]["nodes"] for seed in results])
            )
            per_status.update(
                {f"{seed}/{name}": results[seed][name]["status"] for seed in results}
            )
        nodes[cell.method] = per_instance
        statuses[cell.method] = per_status

    ratios = {
        method: paired_per_instance_ratios(per_instance, nodes["default"])
        for method, per_instance in nodes.items()
        if method != "default"
    }
    sgms = {method: sgm_of_ratios(nodes[method], nodes["default"]) for method in ratios}

    print("\nPilot summary — mean B&B nodes per instance (default solver = 1.000x):")
    header = f"{'instance':<10}" + "".join(f"{m:>12}" for m in nodes)
    print(header)
    for instance in next(iter(nodes.values())):
        row = f"{instance:<10}" + "".join(f"{nodes[m][instance]:>12.1f}" for m in nodes)
        print(row)
    print("\nSGM of paired node ratios vs default:")
    for method, sgm in sorted(sgms.items(), key=lambda kv: kv[1]):
        print(f"  {method:<10} {sgm:.3f}x")

    summary = {
        "nodes": nodes,
        "statuses": statuses,
        "ratios": ratios,
        "sgm": sgms,
        "n_instances": len(next(iter(nodes.values()))),
        "n_seeds": len(EVAL_SEEDS),
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    print(f"\n[summary] written to {results_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHOD_SPECS),
        default="gasse",
        help="which ported method to drive (default gasse)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="collection episodes (default 20)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="training epochs (default 20)"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "pilot_gasse",
        help="where instances, checkpoint, runs, and summary are written",
    )
    parser.add_argument(
        "--train-family", type=parse_family, default=TRAIN_FAMILY, metavar="RxCxDENSITY"
    )
    parser.add_argument(
        "--eval-family", type=parse_family, default=EVAL_FAMILY, metavar="RxCxDENSITY"
    )
    parser.add_argument(
        "--collect-time-limit",
        type=float,
        default=60.0,
        help="per-episode collection budget",
    )
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=1,
        help="parallel collection episode workers (each runs its own SCIP; 1 = sequential)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="re-collect and re-train even if a checkpoint exists (default: reuse it)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    write_eval_instances(results_dir / "instances" / "setcover_pilot", args.eval_family)
    checkpoint = results_dir / f"{args.method}.pt"
    if args.retrain or not checkpoint.exists():
        if args.method == "retrobranching":
            # RL: the agent explores on fresh training instances — no expert
            # data to collect (and --epochs has no role in the DQN loop).
            train_retrobranching(
                args.episodes,
                checkpoint,
                args.train_family,
                time_limit=args.collect_time_limit,
            )
        elif args.method == "tgppo":
            # RL from scratch under the paper's sandbox protocol — no expert
            # data to collect (--epochs again has no role in the PPO loop).
            train_tgppo(
                args.episodes,
                checkpoint,
                args.train_family,
                time_limit=args.collect_time_limit,
            )
        elif args.method == "bbmdp":
            # RL from scratch on the corrected vanilla MDP — no expert data
            # either; --epochs bounds the gradient-step budget.
            train_bbmdp(
                args.episodes,
                checkpoint,
                args.train_family,
                args.epochs,
                time_limit=args.collect_time_limit,
            )
        elif args.method == "fmsts":
            # RL from scratch with observed subtree-size Q-values over a fixed
            # redrawn pool — no expert data; --epochs scales the iterations.
            train_fmsts(
                args.episodes,
                checkpoint,
                args.train_family,
                args.epochs,
                args.eval_family,
                time_limit=args.collect_time_limit,
            )
        elif args.method == "seqbnb":
            # DAgger over the joint branching + node-selection regime: the
            # expert (strong branching + the x* oracle) drives round 0, the
            # policy its own episodes afterwards, every decision labeled.
            train_seqbnb(
                args.episodes,
                checkpoint,
                args.train_family,
                args.epochs,
                time_limit=args.collect_time_limit,
            )
        elif args.method == "mamba":
            # Two-stage IL over relpscost-expert trajectories: the collected
            # episodes stay grouped (the decision order is the path model's
            # input), and --epochs drives both the contrastive and the
            # sequence stage.
            trajectories = collect_expert_data(
                args.episodes,
                args.train_family,
                time_limit=args.collect_time_limit,
                cache=results_dir / METHOD_SPECS[args.method]["cache_name"],
                n_workers=args.collect_workers,
                kind=args.method,
            )
            if not trajectories or not any(trajectories):
                raise SystemExit(
                    "no expert trajectories collected — increase --episodes"
                )
            train_mamba(trajectories, args.epochs, checkpoint, args.train_family)
        elif args.method.startswith("scavuzzo"):
            # Two stages: IL pretraining (the gasse loss over the shared
            # prob-0.05 expert samples — all three variants pretrain alike),
            # then the variant's REINFORCE fine-tuning. A restarted campaign
            # reuses the IL checkpoint but always redoes the RL stage.
            il_checkpoint = results_dir / "scavuzzo_il.pt"
            if args.retrain or not il_checkpoint.exists():
                samples = collect_expert_data(
                    args.episodes,
                    args.train_family,
                    time_limit=args.collect_time_limit,
                    cache=results_dir / METHOD_SPECS[args.method]["cache_name"],
                    n_workers=args.collect_workers,
                    kind=args.method,
                )
                if not samples:
                    raise SystemExit(
                        "no expert samples collected — increase --episodes"
                    )
                train_policy(
                    samples, args.epochs, il_checkpoint, args.train_family, kind="gasse"
                )
            train_scavuzzo(
                il_checkpoint,
                checkpoint,
                args.episodes,
                args.train_family,
                METHOD_SPECS[args.method]["variant"],
                time_limit=args.collect_time_limit,
            )
        elif args.method == "sorrel":
            # Offline RL on the VHB demonstrations, then online PPO + SIL on
            # fresh training instances.
            samples = collect_expert_data(
                args.episodes,
                args.train_family,
                time_limit=args.collect_time_limit,
                cache=results_dir / METHOD_SPECS[args.method]["cache_name"],
                n_workers=args.collect_workers,
                kind=args.method,
            )
            if not samples:
                raise SystemExit(
                    "no VHB demonstrations collected — increase --episodes"
                )
            train_sorrel(
                samples,
                args.epochs,
                checkpoint,
                args.train_family,
                args.episodes,
                time_limit=args.collect_time_limit,
            )
        else:
            samples = collect_expert_data(
                args.episodes,
                args.train_family,
                time_limit=args.collect_time_limit,
                cache=results_dir / METHOD_SPECS[args.method]["cache_name"],
                n_workers=args.collect_workers,
                kind=args.method,
            )
            if not samples:
                raise SystemExit("no expert samples collected — increase --episodes")
            train_policy(
                samples, args.epochs, checkpoint, args.train_family, kind=args.method
            )
    else:
        print(f"[train] reusing checkpoint {checkpoint}")
    run_benchmark(checkpoint, results_dir, kind=args.method)


if __name__ == "__main__":
    main()
