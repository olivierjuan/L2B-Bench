"""TreeGate (Zarpellon 2021) — imitation learning training.

Port of ``train_test_IL.py`` from the official repository
(github.com/ds4dm/branch-search-trees):

- **Loss**: cross-entropy of the candidate scores against the expert's chosen
  candidate, computed **per node** and summed over the mini-batch then divided
  by the batch size (the official inner loop — the model sees one node at a
  time, no candidate-axis padding);
- **Optimizer**: Adam (weight decay 1e-5) with the MultiStepLR decay (x0.1 at
  epochs 20 and 30), the released checkpoint's regime;
- **Metrics**: top-k accuracy against the expert's choice (the harness's
  k = 1, 3, 5, 10 — the official checkpoint records k = 2, 3, 5, 10).

The hyperparameters below are the released
``TreeGatePolicy_hidden64_depth5_lr0.01`` checkpoint's ``args``.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l2b_bench.methods.gasse.train import TOP_K, get_device

from .collect import TreeGateSample
from .model import N_CANDIDATE_FEATURES, N_MIP_FEATURES, N_NODE_FEATURES, TreeGatePolicy

# The released TreeGatePolicy checkpoint's hyperparameters.
LR = 1e-2
WEIGHT_DECAY = 1e-5
LR_DECAY_EPOCHS = (20, 30)
LR_DECAY_FACTOR = 0.1


def sample_to_tensors(sample: TreeGateSample) -> dict[str, torch.Tensor]:
    """One collected node → the model's per-node inputs + the CE target."""
    candidates = np.asarray(sample.action_set)
    choice = int(np.where(candidates == sample.action)[0][0])
    return {
        "cands_state_mat": torch.as_tensor(sample.candidate_features, dtype=torch.float32),
        "node_state": torch.as_tensor(sample.node_features, dtype=torch.float32),
        "mip_state": torch.as_tensor(sample.mip_features, dtype=torch.float32),
        # (1,) — the batch dimension of the (1, S) per-node logits.
        "target": torch.tensor([choice], dtype=torch.long),
    }


def sample_to_device(tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in tensors.items()}


def process(
    model: TreeGatePolicy,
    batches: list[list[dict[str, torch.Tensor]]],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | np.ndarray]:
    """One pass over the batches: per-node CE summed over the batch + top-k.

    Mirrors the official loop exactly: one backward per mini-batch on the sum
    of the per-node losses divided by the batch size.
    """
    total_loss = 0.0
    total_kacc = np.zeros(len(TOP_K))
    n_batches = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            batch_loss = 0.0
            hits = np.zeros(len(TOP_K))
            for tensors in batch:
                logits = model(
                    tensors["cands_state_mat"], tensors["node_state"], tensors["mip_state"]
                ).transpose(1, 0)  # (1, S) — the official orientation
                batch_loss = batch_loss + F.cross_entropy(logits, tensors["target"])
                n_cands = logits.size(1)
                for j, k in enumerate(TOP_K):
                    if n_cands < k:
                        hits[j] += 1
                    else:
                        hits[j] += float(
                            tensors["target"].item() in logits.topk(k, dim=1).indices[0].tolist()
                        )
            batch_loss = batch_loss / len(batch)
            if optimizer is not None:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            total_loss += batch_loss.item()
            total_kacc += hits / len(batch)
            n_batches += 1

    return {"loss": total_loss / max(n_batches, 1), "kacc": total_kacc / max(n_batches, 1)}


def save_checkpoint(model: TreeGatePolicy, path: str | Path, extra: dict | None = None) -> None:
    """State dict + provenance metadata, same format as the other methods."""
    checkpoint = {"state_dict": model.state_dict()}
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> TreeGatePolicy:
    device = device or get_device()
    model = TreeGatePolicy()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()
    return model
