"""T-BranT (Lin 2022) — imitation learning training.

Port of ``train_test_IL.py`` from the official repository
(github.com/linjc16/TBranT):

- **Loss**: cross-entropy of the candidate logits against the expert's chosen
  candidate (a hard label — the index of the relpscost choice among the
  node's candidates);
- **Optimizer**: AdamW + the Noam scheduler (5 warm-up epochs, stepped once
  per epoch exactly like the official loop);
- **Metrics**: top-k accuracy against the expert's choice (the harness's
  k = 1, 3, 5, 10 — the official scripts report k = 2, 5).

Mini-batches pad the per-node candidate matrices to the batch's maximum and
mask the padding (the transformer and the BiMatching head both see the mask).
Checkpointing reuses the Gasse port's save/load (the state dict plus the
campaign's provenance metadata).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from l2b_bench.methods.gasse.train import TOP_K, get_device

from .collect import SolverStatsSample
from .model import N_CANDIDATE_FEATURES, N_MIP_FEATURES, N_NODE_FEATURES, BranchFormer

# The official train_LTBranT.sh hyperparameters.
LR = 1e-3
WARMUP_EPOCHS = 5


@dataclass
class StatsBatch:
    """A mini-batch of solver-stats samples, padded over the candidate axis."""

    candidate_features: torch.Tensor  # (B, S, 25)
    padding_mask: torch.Tensor  # (B, S) — 1 where the slot is padding
    node_features: torch.Tensor  # (B, 8)
    mip_features: torch.Tensor  # (B, 53)
    targets: torch.Tensor  # (B,) — the expert choice's rank among candidates


def collate_stats(samples: list[SolverStatsSample]) -> StatsBatch:
    n = len(samples)
    max_cands = max(len(sample.action_set) for sample in samples)
    candidate_features = torch.zeros(n, max_cands, N_CANDIDATE_FEATURES)
    padding_mask = torch.ones(n, max_cands, dtype=torch.long)  # 1 = padding
    node_features = torch.zeros(n, N_NODE_FEATURES)
    mip_features = torch.zeros(n, N_MIP_FEATURES)
    targets = torch.zeros(n, dtype=torch.long)
    for i, sample in enumerate(samples):
        k = len(sample.action_set)
        candidate_features[i, :k] = torch.as_tensor(sample.candidate_features, dtype=torch.float32)
        padding_mask[i, :k] = 0
        node_features[i] = torch.as_tensor(sample.node_features, dtype=torch.float32)
        mip_features[i] = torch.as_tensor(sample.mip_features, dtype=torch.float32)
        targets[i] = int(np.where(np.asarray(sample.action_set) == sample.action)[0][0])
    return StatsBatch(candidate_features, padding_mask, node_features, mip_features, targets)


def to_device(batch: StatsBatch, device: torch.device) -> StatsBatch:
    return StatsBatch(
        candidate_features=batch.candidate_features.to(device),
        padding_mask=batch.padding_mask.to(device),
        node_features=batch.node_features.to(device),
        mip_features=batch.mip_features.to(device),
        targets=batch.targets.to(device),
    )


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """The official Noam schedule, stepped once per epoch.

    Linear warm-up for ``warmup_epochs`` epochs, then decay proportional to
    the inverse square root of the epoch number.
    """

    def __init__(self, optimizer, warmup_epochs: int):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer)

    def get_lr(self):
        epoch = max(1, self.last_epoch)
        scale = self.warmup_epochs**0.5 * min(
            epoch**-0.5, epoch * self.warmup_epochs**-1.5
        )
        return [base_lr * scale for base_lr in self.base_lrs]


def process(
    model: BranchFormer,
    batches: list[StatsBatch],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | np.ndarray]:
    """One pass over the batches: CE against the expert choice + top-k accuracy."""
    total_loss = 0.0
    total_kacc = np.zeros(len(TOP_K))
    n_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            logits = model(
                batch.candidate_features, batch.padding_mask, batch.node_features, batch.mip_features
            )
            loss = F.cross_entropy(logits, batch.targets)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Per-sample top-k hits; a node with fewer than k candidates
            # trivially contains its target in any top-k.
            n_cands = (batch.padding_mask == 0).sum(dim=1)  # (B,)
            targets = batch.targets.unsqueeze(1)
            for j, k in enumerate(TOP_K):
                if k == 1:
                    hits = logits.argmax(dim=1) == batch.targets
                else:
                    hits = (logits.topk(k, dim=1).indices == targets).any(dim=1)
                hits = hits | (n_cands < k)
                total_kacc[j] += hits.float().sum().item()
            total_loss += loss.item() * len(batch.targets)
            n_samples += len(batch.targets)

    return {"loss": total_loss / n_samples, "kacc": total_kacc / n_samples}


def save_checkpoint(model: BranchFormer, path: str | Path, extra: dict | None = None) -> None:
    """State dict + provenance metadata, same format as the Gasse checkpoints."""
    checkpoint = {"state_dict": model.state_dict()}
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> BranchFormer:
    device = device or get_device()
    model = BranchFormer()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()
    return model
