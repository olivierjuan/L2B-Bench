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
from l2b_bench.methods.lookback.collect import collect_episode as lookback_collect_episode
from l2b_bench.methods.lookback.policy import lookback_method
from l2b_bench.methods.lookback.train import (
    LookbackBatch,
    collate_pairs,
    lb_to_device,
    process_lookback,
)

# The ported methods this pilot can drive, with their protocol knobs.
METHOD_SPECS = {
    "gasse": {"prob": 0.5, "method": gasse_method, "cache_name": "expert_samples.pt"},
    "lookback": {
        "prob": 0.05,  # the official node_record_prob
        "method": lookback_method,
        "cache_name": "expert_samples.pt",
    },
}

def parse_family(text: str) -> dict:
    """``"500x1000x0.05"`` → ``dict(n_rows=500, n_cols=1000, density=0.05)``."""
    rows, cols, density = text.split("x")
    return dict(n_rows=int(rows), n_cols=int(cols), density=float(density))


# The pilot's defaults — the paper's set covers, scaled down so a CPU run
# finishes in minutes. Training and evaluation draws come from different
# generator seeds (disjoint instances); the evaluation family is harder than
# the training one (default solver: ~600 B&B nodes per instance) so the node
# counts are not degenerate. Both families are CLI-overridable for the
# cluster campaigns.
TRAIN_FAMILY = dict(n_rows=200, n_cols=250, density=0.10)
TRAIN_RNG_SEED = 123
EVAL_FAMILY = dict(n_rows=500, n_cols=600, density=0.10)
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
    generator = SetCoverGenerator(rng=np.random.default_rng(EVAL_RNG_SEED), **eval_family)
    for i in range(N_EVAL_INSTANCES):
        model = next(generator)
        model.writeProblem(filename=str(split / f"sc_{i:02d}.mps"), genericnames=True, verbose=False)
    print(f"[instances] wrote {N_EVAL_INSTANCES} set covers to {split}")


def _collect_one_episode(job):
    """One collection episode in a worker process: its own env, its own SCIP.

    The instance file and base seed fully determine the episode, so parallel
    collection produces exactly the sequential campaign's data.
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    path, seed, kind, probability, time_limit = job
    from gyozas.dynamics.branching import BranchingDynamics, ExtraBranchingActions
    from gyozas.environment import Environment
    from gyozas.informations.solve_status import SolveStatus
    from gyozas.instances.files import FileGenerator
    from gyozas.observations import NodeBipartite
    from gyozas.seeds import SeedTriplet

    from l2b_bench.methods.gasse.collect import GASSE_SCIP_PARAMS, collect_episode

    scip_params = dict(GASSE_SCIP_PARAMS)
    scip_params["limits/time"] = time_limit
    scip_params["display/verblevel"] = 0
    env = Environment(
        instance_generator=FileGenerator(
            directory=path.parent, pattern=path.name, sampling_mode="replace", rng=np.random.default_rng(0)
        ),
        observation_function=NodeBipartite(),
        information_function=SolveStatus(),
        scip_params=scip_params,
        dynamics=BranchingDynamics(with_extra_actions=[ExtraBranchingActions.SKIP]),
    )
    try:
        triplet = SeedTriplet.from_base_seed(seed)
        env.seed(triplet.solver)
        triplet.seed_python()
        rng = np.random.default_rng(triplet.augmentation)
        if kind == "lookback":
            samples, n_nodes, status = lookback_collect_episode(env, rng, probability)
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
        print(f"[collect] reusing {len(samples)} cached expert samples from {cache}", flush=True)
        return samples

    from gyozas.instances import SetCoverGenerator

    # Deterministic instance pool: written once, reused across restarts.
    instance_dir = cache.parent / "train_instances" if cache is not None else None
    instance_dir = instance_dir or Path("train_instances")
    instance_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    generator = SetCoverGenerator(rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family)
    for i in range(n_episodes):
        path = instance_dir / f"train_{i:03d}.mps"
        if not path.exists():
            next(generator).writeProblem(filename=str(path), genericnames=True, verbose=False)
        paths.append(path)
    print(f"[collect] {n_episodes} training instances in {instance_dir}", flush=True)

    jobs = [(path, i, kind, METHOD_SPECS[kind]["prob"], time_limit) for i, path in enumerate(paths)]
    all_samples: list = []
    statuses: list[str] = []
    n_nodes_total = 0
    if n_workers <= 1:
        results = map(_collect_one_episode, jobs)
        for done, (samples, n_nodes, status) in enumerate(results, 1):
            all_samples.extend(samples)
            statuses.append(status)
            n_nodes_total += n_nodes
            print(f"[collect] episode {done}/{n_episodes}: {status}, {n_nodes} nodes", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_collect_one_episode, job): job for job in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                samples, n_nodes, status = future.result()
                all_samples.extend(samples)
                statuses.append(status)
                n_nodes_total += n_nodes
                print(f"[collect] episode {done}/{n_episodes}: {status}, {n_nodes} nodes", flush=True)

    solved = sum(1 for s in statuses if s == "optimal")
    print(
        f"[collect] {n_episodes} episodes ({solved} optimal), "
        f"{n_nodes_total} nodes, {len(all_samples)} expert samples",
        flush=True,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_samples, cache)
        print(f"[collect] samples cached to {cache}", flush=True)
    return all_samples


def train_policy(samples: list, epochs: int, checkpoint: Path, train_family: dict, kind: str = "gasse") -> None:
    """The official imitation-learning loop: pretrain, then epochs of Adam.

    The loss is the selected method's: Gasse's candidate cross-entropy, or
    Lookback's (one-hot target + the PAT distillation term, the paper's
    set-cover hyperparameters).
    """
    import torch

    model = GNNPolicy()
    shuffle = np.random.default_rng(0)

    if kind == "lookback":
        from l2b_bench.methods.lookback.collect import pair_parent_children

        pairs = pair_parent_children(samples)
        print(f"[train] {len(pairs)} parent-child pairs out of {len(samples)} samples")

        def batches(device):
            order = shuffle.permutation(len(samples))
            return [
                lb_to_device(
                    collate_pairs(
                        [samples[i] for i in order[k : k + BATCH_SIZE]],
                        {i - k: pairs[i] - k for i in order[k : k + BATCH_SIZE] if i in pairs and pairs[i] - k >= 0},
                    ),
                    device,
                )
                for k in range(0, len(order), BATCH_SIZE)
            ]

    else:

        def batches(device):
            order = shuffle.permutation(len(samples))
            return [
                to_device(collate([samples[i] for i in order[k : k + BATCH_SIZE]]), device)
                for k in range(0, len(order), BATCH_SIZE)
            ]

    device = get_device()
    print(f"[train] device: {device}")
    model = model.to(device)
    n_layers = pretrain(model, [lb.batch if isinstance(lb, LookbackBatch) else lb for lb in batches(device)])
    print(f"[train] froze {n_layers} PreNorm layers")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        if kind == "lookback":
            stats = process_lookback(model, batches(device), optimizer=optimizer)
        else:
            stats = process(model, batches(device), optimizer=optimizer)
        kacc = " ".join(f"acc@{k}={a:.2f}" for k, a in zip((1, 3, 5, 10), stats["kacc"], strict=True))
        print(f"[train] epoch {epoch:3d}  loss={stats['loss']:.4f}  {kacc}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model.cpu(),
        checkpoint,
        extra={"n_samples": len(samples), "epochs": epochs, "train_family": train_family},
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
    runner = Runner(config, methods, {"setcover_pilot": strate_root}, results_dir / "runs")
    report = runner.run()
    for cell_key, counts in report.items():
        print(f"[runner] {cell_key}: {counts['executed']} executed, {counts['skipped']} skipped")
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
            per_status.update({f"{seed}/{name}": results[seed][name]["status"] for seed in results})
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
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[summary] written to {results_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(METHOD_SPECS), default="gasse", help="which ported method to drive (default gasse)")
    parser.add_argument("--episodes", type=int, default=20, help="collection episodes (default 20)")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs (default 20)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "pilot_gasse",
        help="where instances, checkpoint, runs, and summary are written",
    )
    parser.add_argument("--train-family", type=parse_family, default=TRAIN_FAMILY, metavar="RxCxDENSITY")
    parser.add_argument("--eval-family", type=parse_family, default=EVAL_FAMILY, metavar="RxCxDENSITY")
    parser.add_argument("--collect-time-limit", type=float, default=60.0, help="per-episode collection budget")
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
        train_policy(samples, args.epochs, checkpoint, args.train_family, kind=args.method)
    else:
        print(f"[train] reusing checkpoint {checkpoint}")
    run_benchmark(checkpoint, results_dir, kind=args.method)


if __name__ == "__main__":
    main()
