"""The Gasse 2019 pilot campaign: collect → train → evaluate, end to end.

A small, fully reproducible demonstration that the benchmark loop works on
the harness — not a scientific result. Everything runs on CPU in minutes:

1. **Instances** — a handful of held-out set covers (the paper's synthetic
   family, scaled down) written as a pilot strate;
2. **Collect** — explore-then-strong-branch expert trajectories on training
   set covers (a different generator seed than the strate's);
3. **Train** — the paper's GCNN under the official imitation-learning loop;
4. **Evaluate** — the trained policy against the classical baselines
   through the runner: the same config-driven grid, records, and
   publication guard every method of the benchmark will use.

The pilot deliberately runs the *benchmark* path (the Runner), not a
hand-rolled evaluation loop: the point is to validate the harness and the
method-porting recipe, and to leave a template for the real campaigns.

Usage::

    uv run python scripts/pilot_gasse.py [--episodes 20] [--epochs 20]

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
from l2b_bench.methods.gasse.collect import collect
from l2b_bench.methods.gasse.policy import gasse_method
from l2b_bench.methods.gasse.train import (
    collate,
    get_device,
    pretrain,
    process,
    save_checkpoint,
    to_device,
)
from l2b_bench.methods.gasse.model import GNNPolicy

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


def collect_expert_data(n_episodes: int, train_family: dict, time_limit: float = 60.0) -> list:
    """Explore-then-strong-branch trajectories on the training family."""
    from gyozas.instances import SetCoverGenerator

    generator = SetCoverGenerator(rng=np.random.default_rng(TRAIN_RNG_SEED), **train_family)
    result = collect(
        generator,
        n_episodes=n_episodes,
        seeds=list(range(n_episodes)),
        expert_probability=0.5,  # the paper's exploration regime
        time_limit=time_limit,
    )
    solved = sum(1 for s in result.statuses if s == "optimal")
    print(
        f"[collect] {result.n_episodes} episodes ({solved} optimal), "
        f"{result.n_nodes} nodes, {len(result.samples)} expert samples"
    )
    print(f"[collect] multi-optimal nodes: {result.metadata.multi_optimal_distribution()}")
    return result.samples


def train_policy(samples: list, epochs: int, checkpoint: Path, train_family: dict) -> None:
    """The official imitation-learning loop: pretrain, then epochs of Adam."""
    import torch

    model = GNNPolicy()
    shuffle = np.random.default_rng(0)

    def batches(device):
        order = shuffle.permutation(len(samples))
        return [
            to_device(collate([samples[i] for i in order[k : k + BATCH_SIZE]]), device)
            for k in range(0, len(order), BATCH_SIZE)
        ]

    device = get_device()
    print(f"[train] device: {device}")
    model = model.to(device)
    n_layers = pretrain(model, batches(device))
    print(f"[train] froze {n_layers} PreNorm layers")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
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


def run_benchmark(checkpoint: Path, results_dir: Path) -> dict:
    """The trained policy vs the classical baselines on the pilot grid."""
    methods: dict[str, Method] = dict(BASELINE_METHODS)
    methods["gasse"] = gasse_method(checkpoint)
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
        "--retrain",
        action="store_true",
        help="re-collect and re-train even if a checkpoint exists (default: reuse it)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    write_eval_instances(results_dir / "instances" / "setcover_pilot", args.eval_family)
    checkpoint = results_dir / "gasse.pt"
    if args.retrain or not checkpoint.exists():
        samples = collect_expert_data(args.episodes, args.train_family, time_limit=args.collect_time_limit)
        if not samples:
            raise SystemExit("no expert samples collected — increase --episodes")
        train_policy(samples, args.epochs, checkpoint, args.train_family)
    else:
        print(f"[train] reusing checkpoint {checkpoint}")
    run_benchmark(checkpoint, results_dir)


if __name__ == "__main__":
    main()
