"""Diagnostic: top-1 agreement of the two checkpoints vs the SB expert.

Compares the GPU-campaign checkpoint (trained on 500x1000x0.05, 100 epochs)
and the CPU-pilot checkpoint (trained on 200x250x0.10, 20 epochs) on:
  - train-family nodes, reused from the cached expert samples (free);
  - fresh eval-family nodes (500x600x0.10), collected with p=1.0.

Both checkpoints run on CPU: if the GPU checkpoint is already bad here,
the device is exonerated and the problem is data/training/distribution.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from l2b_bench.methods.gasse.train import load_checkpoint  # noqa: E402

GPU_CKPT = REPO / "results" / "gasse_gpu_v1" / "gasse.pt"
CPU_CKPT = REPO / "results" / "pilot_gasse" / "gasse.pt"
TRAIN_SAMPLES = REPO / "results" / "gasse_gpu_v1" / "expert_samples.pt"


def evaluate(model, samples, limit=None, label=""):
    """Top-1 / top-3 agreement between the policy's ranking and the expert action."""
    if limit is not None:
        samples = samples[:limit]
    agree = top3 = total = 0
    with torch.no_grad():
        for sample in samples:
            candidates = np.asarray(sample.action_set)
            obs = sample.observation
            logits = model(
                torch.as_tensor(obs.row_features, dtype=torch.float32),
                torch.as_tensor(obs.edge_features.indices, dtype=torch.long),
                torch.as_tensor(obs.edge_features.values, dtype=torch.float32).reshape(-1, 1),
                torch.as_tensor(obs.variable_features, dtype=torch.float32),
            )
            cand_logits = logits[torch.as_tensor(candidates, dtype=torch.long)]
            ranked = candidates[cand_logits.argsort(descending=True).numpy()]
            expert_rank = int(np.where(ranked == sample.action)[0][0])
            agree += int(expert_rank == 0)
            top3 += int(expert_rank < 3)
            total += 1
    print(f"{label}: n={total} top1={agree / max(total, 1):.3f} top3={top3 / max(total, 1):.3f}",
          flush=True)


def main():
    gpu_model = load_checkpoint(GPU_CKPT, device=torch.device("cpu"))
    cpu_model = load_checkpoint(CPU_CKPT, device=torch.device("cpu"))

    # --- train-family nodes from the cached expert samples -------------------
    cached = torch.load(TRAIN_SAMPLES, weights_only=False)
    print(f"cached train-family samples: {len(cached)}", flush=True)
    rng = np.random.default_rng(0)
    subset = [cached[i] for i in rng.choice(len(cached), size=min(300, len(cached)), replace=False)]
    evaluate(cpu_model, subset, label="train-family | CPU-pilot ckpt")
    evaluate(gpu_model, subset, label="train-family | GPU ckpt     ")

    # --- fresh eval-family nodes (500x600x0.10), p=1.0 ------------------------
    from gyozas.instances import SetCoverGenerator

    from l2b_bench.methods.gasse.collect import collect

    result = collect(
        instance_generator=SetCoverGenerator(
            rng=np.random.default_rng(2024), n_rows=500, n_cols=600, density=0.10
        ),
        n_episodes=3,
        seeds=list(range(3)),
        expert_probability=1.0,
        time_limit=120.0,
        on_episode=lambda i, status, nodes, n: print(
            f"[episode {i}] {status}, {nodes} nodes, {n} samples", flush=True
        ),
    )
    print(f"fresh eval-family samples: {len(result.samples)} from {result.n_episodes} episodes",
          flush=True)
    evaluate(cpu_model, result.samples, label="eval-family  | CPU-pilot ckpt")
    evaluate(gpu_model, result.samples, label="eval-family  | GPU ckpt     ")


if __name__ == "__main__":
    main()
