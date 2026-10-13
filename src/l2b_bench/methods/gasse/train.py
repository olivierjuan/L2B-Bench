"""Gasse 2019 — imitation learning training.

Faithful port of ``03_train_gnn.py`` and ``utilities.py`` from the official
repository (github.com/ds4dm/learn2branch-ecole):

- **Pre-training**: the ``PreNormLayer`` statistics are collected over batches
  then frozen, one layer at a time, exactly like the official ``pretrain``.
- **Loss**: cross-entropy of the policy's per-variable scores restricted to
  the node's branching candidates, against the expert's chosen candidate
  (hard label), plus an entropy bonus (0.0 in the paper).
- **Metrics**: top-k accuracy against the expert's best score (k = 1, 3, 5,
  10), the official paper's metric.

The only difference is the data plumbing: samples come from
:func:`l2b_bench.methods.gasse.collect.collect` instead of the official
repository's gzipped pickles, and mini-batches are collated by hand (concatenate
features, offset edge indices) instead of through torch-geometric's ``Batch``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .collect import BranchingSample
from .model import GNNPolicy

TOP_K = (1, 3, 5, 10)


def get_device() -> "torch.device":
    """CUDA when available (the campaign nodes' A5000), CPU otherwise."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GraphBatch:
    """A mini-batch of bipartite graphs, collated by concatenation."""

    constraint_features: torch.Tensor  # (sum R, 5)
    edge_index: torch.Tensor  # (2, sum E) — constraint-major indices
    edge_attr: torch.Tensor  # (sum E, 1)
    variable_features: torch.Tensor  # (sum V, 19)
    candidates: list[torch.Tensor]  # per sample: candidate variable indices
    choices: torch.Tensor  # (B,) — chosen candidate's rank among candidates
    candidate_scores: list[torch.Tensor]  # per sample: expert scores of candidates


def sample_to_tensors(sample: BranchingSample) -> dict[str, torch.Tensor]:
    """Convert one collected sample to the model's tensor inputs."""
    obs = sample.observation
    candidates = torch.as_tensor(np.asarray(sample.action_set), dtype=torch.long)
    choice = int(np.where(candidates.numpy() == sample.action)[0][0])
    return {
        "constraint_features": torch.as_tensor(obs.row_features, dtype=torch.float32),
        # The observation's edge indices are constraint-major (row 0 =
        # constraint, row 1 = variable) — exactly the orientation the model
        # consumes, like ecole's.
        "edge_index": torch.as_tensor(obs.edge_features.indices, dtype=torch.long),
        "edge_attr": torch.as_tensor(obs.edge_features.values, dtype=torch.float32).reshape(-1, 1),
        "variable_features": torch.as_tensor(obs.variable_features, dtype=torch.float32),
        "candidates": candidates,
        "choice": torch.tensor(choice, dtype=torch.long),
        "candidate_scores": torch.as_tensor(
            [sample.scores[int(a)] for a in sample.action_set], dtype=torch.float32
        ),
    }


def collate(samples: list[BranchingSample]) -> GraphBatch:
    """Collate samples into one big disconnected bipartite graph (official batching)."""
    tensors = [sample_to_tensors(s) for s in samples]
    n_cons = [t["constraint_features"].shape[0] for t in tensors]
    n_vars = [t["variable_features"].shape[0] for t in tensors]
    edge_index = []
    edge_attr = []
    cons_offset = 0
    var_offset = 0
    for t, n_c, n_v in zip(tensors, n_cons, n_vars, strict=True):
        edge_index.append(t["edge_index"] + torch.tensor([[cons_offset], [var_offset]]))
        edge_attr.append(t["edge_attr"])
        cons_offset += n_c
        var_offset += n_v
    return GraphBatch(
        constraint_features=torch.cat([t["constraint_features"] for t in tensors]),
        edge_index=torch.cat(edge_index, dim=1),
        edge_attr=torch.cat(edge_attr),
        variable_features=torch.cat([t["variable_features"] for t in tensors]),
        candidates=[t["candidates"] + sum(n_vars[:i]) for i, t in enumerate(tensors)],
        choices=torch.stack([t["choice"] for t in tensors]),
        candidate_scores=[t["candidate_scores"] for t in tensors],
    )


def batched_tensors(batch: GraphBatch):
    """The model's forward arguments for a batch."""
    return (
        batch.constraint_features,
        batch.edge_index,
        batch.edge_attr,
        batch.variable_features,
    )


def to_device(batch: GraphBatch, device: "torch.device") -> GraphBatch:
    """A copy of the batch with every tensor on ``device``."""
    return GraphBatch(
        constraint_features=batch.constraint_features.to(device),
        edge_index=batch.edge_index.to(device),
        edge_attr=batch.edge_attr.to(device),
        variable_features=batch.variable_features.to(device),
        candidates=[c.to(device) for c in batch.candidates],
        choices=batch.choices.to(device),
        candidate_scores=[s.to(device) for s in batch.candidate_scores],
    )


def pretrain(policy: GNNPolicy, batches: list[GraphBatch]) -> int:
    """Freeze the PreNorm layers' statistics over the batches (official pre_train).

    Returns the number of layers that were pre-trained.
    """
    policy.pre_train_init()
    n_layers = 0
    while True:
        for batch in batches:
            if not policy.pre_train(*batched_tensors(batch)):
                break
        if policy.pre_train_next() is None:
            break
        n_layers += 1
    return n_layers


def process(
    policy: GNNPolicy,
    batches: list[GraphBatch],
    optimizer: torch.optim.Optimizer | None = None,
    entropy_bonus: float = 0.0,
) -> dict[str, float | np.ndarray]:
    """One pass over the batches: the official train/valid ``process``.

    The loss is the mean over samples of the cross-entropy of the candidate-
    restricted scores against the expert's choice; the entropy bonus is added
    exactly as in the official code (``loss = ce - bonus * entropy``).
    """
    mean_loss = 0.0
    mean_entropy = 0.0
    mean_kacc = np.zeros(len(TOP_K))
    n_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in batches:
            logits = policy(*batched_tensors(batch))
            ce_losses = []
            entropies = []
            kaccs = []
            for i, candidates in enumerate(batch.candidates):
                cand_logits = logits[candidates]
                choice = batch.choices[i]
                ce_losses.append(F.cross_entropy(cand_logits.unsqueeze(0), choice.unsqueeze(0)))
                entropies.append(
                    (-F.softmax(cand_logits, dim=-1) * F.log_softmax(cand_logits, dim=-1)).sum()
                )
                true_scores = batch.candidate_scores[i]
                true_best = true_scores.max()
                kacc = []
                for k in TOP_K:
                    if len(candidates) < k:
                        kacc.append(1.0)
                        continue
                    pred_top_k = cand_logits.topk(k).indices
                    kacc.append(bool((true_scores[pred_top_k] == true_best).any()))
                kaccs.append(kacc)
            cross_entropy_loss = torch.stack(ce_losses).mean()
            entropy = torch.stack(entropies).mean()
            loss = cross_entropy_loss - entropy_bonus * entropy

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += cross_entropy_loss.item() * len(batch.candidates)
            mean_entropy += entropy.item() * len(batch.candidates)
            mean_kacc += np.asarray(kaccs, dtype=float).sum(axis=0)
            n_samples += len(batch.candidates)

    return {
        "loss": mean_loss / n_samples,
        "entropy": mean_entropy / n_samples,
        "kacc": mean_kacc / n_samples,
    }


def save_checkpoint(policy: GNNPolicy, path: str | Path, extra: dict | None = None) -> None:
    """Save a policy checkpoint (state dict + any provenance)."""
    torch.save({"state_dict": policy.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: "torch.device | None" = None) -> GNNPolicy:
    """Load a policy from a :func:`save_checkpoint` file (weights only, eval mode)."""
    policy = GNNPolicy()
    policy.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["state_dict"])
    if device is not None:
        policy.to(device)
    policy.eval()
    return policy
