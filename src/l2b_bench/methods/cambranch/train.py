"""CAMBranch 2024 — the joint training loop: IL + InfoNCE + auxiliary loss.

Paper: "Contrastive Learning with Augmented MILPs for Learning to Branch"
(Lin et al., ICLR 2024 poster), Eq. 25-26:

    L = L_sup + lambda_1 * L_InfoNCE + lambda_2 * L_aux

with lambda_1 = 0.05 and lambda_2 = 0.01, over mini-batches of 64 samples =
32 (original, AMILP) pairs:

- ``L_sup`` — Gasse's candidate cross-entropy, on every sample of the batch
  (originals and AMILPs alike, whose labels coincide by Theorem 3.1);
- ``L_InfoNCE`` — Eq. 8 over the L2-normalized graph embeddings: each
  original's positive is its own AMILP, the negatives are the other pairs'
  embeddings; no temperature (the paper has none — the embeddings are
  unit-norm, so the similarities are cosine similarities);
- ``L_aux`` — the auxiliary term tying each pair's *branching* distributions:
  the candidate-masked softmax of the per-variable scores of the original and
  its AMILP, squared difference summed over the variables.

Optimization follows the paper: Adam over a grid {1e-3, 5e-4, 1e-4} (the
pilot takes 1e-3), batch 64, the learning rate cut to 20% after 10 epochs
without validation improvement. The pilot-scale loop trains a fixed number of
epochs on the training pairs (no validation split — like the other ports, the
official model-selection pass is documented as dropped); the LR schedule is
wired (:func:`PlateauScheduler`) so the campaign can arm it once a validation
loss is logged.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..gasse.train import (
    TOP_K,
    GraphBatch,
    batched_tensors,
    collate,
    get_device,
    pretrain,
    to_device,
)
from .augment import N_AUGMENTED, augment_pairs
from .model import CAMBranchGNN

#: The paper's loss weights (Eq. 25).
LAMBDA_1 = 0.05
LAMBDA_2 = 0.01

#: 64 samples per batch — 32 (original, AMILP) pairs.
PAIRS_PER_BATCH = 32

#: The paper's grid is {1e-3, 5e-4, 1e-4}; the pilot takes the first.
LR = 1e-3

#: 10 epochs without improvement, then the LR is cut to 20% (paper App. C).
PLATEAU_PATIENCE = 10
PLATEAU_FACTOR = 0.2

#: The paper's data budget: 10k strong-branching samples (10% of Gasse's
#: 100k) — the 90k AMILPs are free by feature transformation.
SUBSAMPLE_FRACTION = 0.1


def subsample_budget(
    samples: list,
    fraction: float = SUBSAMPLE_FRACTION,
    rng: np.random.Generator | None = None,
) -> list:
    """A uniform subsample of the collected originals — the paper's budget."""
    rng = rng or np.random.default_rng(0)
    n_keep = max(1, round(fraction * len(samples)))
    return [samples[i] for i in rng.permutation(len(samples))[:n_keep]]


@dataclass
class PairBatch:
    """A mini-batch of (original, AMILP) pairs as one big disconnected graph.

    The samples are interleaved — index ``2*i`` is a pair's original, ``2*i+1``
    its AMILP — so the graph embeddings coming back from the model split
    evenly into the two contrastive sides. ``n_vars``/``n_cons`` are the
    per-sample segment sizes the projection head pools over.
    """

    batch: GraphBatch
    n_vars: list[int]
    n_cons: list[int]


def collate_pairs(pairs: list[tuple]) -> PairBatch:
    """Collate the interleaved pair samples with gasse's concatenation."""
    samples = [sample for pair in pairs for sample in pair]
    return PairBatch(
        batch=collate(samples),
        n_vars=[s.observation.variable_features.shape[0] for s in samples],
        n_cons=[s.observation.row_features.shape[0] for s in samples],
    )


def pair_batches(
    pairs: list[tuple],
    device: torch.device,
    pairs_per_batch: int,
    shuffle: np.random.Generator,
) -> list[PairBatch]:
    """One epoch's batches: a fresh permutation of the pairs, on ``device``."""
    order = shuffle.permutation(len(pairs))
    return [
        _to_device(
            collate_pairs([pairs[i] for i in order[k : k + pairs_per_batch]]), device
        )
        for k in range(0, len(order), pairs_per_batch)
    ]


def _to_device(pb: PairBatch, device: torch.device) -> PairBatch:
    return PairBatch(
        batch=to_device(pb.batch, device), n_vars=pb.n_vars, n_cons=pb.n_cons
    )


def infonce_loss(g_original: torch.Tensor, g_augmented: torch.Tensor) -> torch.Tensor:
    """Eq. 8: the positives on the diagonal, no temperature (unit-norm
    embeddings — the logits are cosine similarities)."""
    targets = torch.arange(g_original.shape[0], device=g_original.device)
    return F.cross_entropy(g_original @ g_augmented.t(), targets)


def candidate_probs(
    sample_logits: torch.Tensor, local_candidates: torch.Tensor
) -> torch.Tensor:
    """Softmax of one sample's per-variable scores, masked to its branching
    candidates (the auxiliary term's distributions)."""
    masked = torch.full_like(sample_logits, -1e9)
    masked[local_candidates] = sample_logits[local_candidates]
    return F.softmax(masked, dim=-1)


def auxiliary_loss(p_original: torch.Tensor, p_augmented: torch.Tensor) -> torch.Tensor:
    """Eq. 26's auxiliary term: squared difference of the pair's candidate
    distributions, summed over the variables, averaged over the pairs."""
    return ((p_original - p_augmented) ** 2).sum(dim=-1).mean()


def process(
    model: CAMBranchGNN,
    batches: list[PairBatch],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | np.ndarray]:
    """One pass over the pair batches: the joint loss of Eq. 25.

    Returns the mean losses and the top-k accuracies (k = 1, 3, 5, 10)
    against the expert's best score, Gasse's metric, over every sample.
    """
    mean = {"loss": 0.0, "ce": 0.0, "infonce": 0.0, "aux": 0.0}
    mean_kacc = np.zeros(len(TOP_K))
    n_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for pb in batches:
            logits, g = model(
                *batched_tensors(pb.batch),
                n_vars_per_sample=pb.n_vars,
                n_cons_per_sample=pb.n_cons,
            )
            # The supervised term: gasse's candidate cross-entropy on every
            # sample — the AMILP label is the original's (Thm 3.1).
            ce_losses = []
            kaccs = []
            for i, candidates in enumerate(pb.batch.candidates):
                cand_logits = logits[candidates]
                choice = pb.batch.choices[i]
                ce_losses.append(
                    F.cross_entropy(cand_logits.unsqueeze(0), choice.unsqueeze(0))
                )
                true_scores = pb.batch.candidate_scores[i]
                true_best = true_scores.max()
                kacc = []
                for k in TOP_K:
                    if len(candidates) < k:
                        kacc.append(1.0)
                        continue
                    pred_top_k = cand_logits.topk(k).indices
                    kacc.append(bool((true_scores[pred_top_k] == true_best).any()))
                kaccs.append(kacc)
            ce = torch.stack(ce_losses).mean()

            # The contrastive terms, on the interleaved graph embeddings.
            info = infonce_loss(g[0::2], g[1::2])
            # The auxiliary term pairs each original with its own AMILP —
            # per-pair scalars (the samples of a batch may have different
            # variable counts, so the distributions never stack together).
            pair_aux = []
            offset = 0
            p_original = None
            for i, n_vars in enumerate(pb.n_vars):
                local_candidates = pb.batch.candidates[i] - offset
                probs = candidate_probs(
                    logits[offset : offset + n_vars], local_candidates
                )
                if i % 2 == 0:
                    p_original = probs
                else:
                    pair_aux.append(
                        auxiliary_loss(p_original.unsqueeze(0), probs.unsqueeze(0))
                    )
                offset += n_vars
            aux = torch.stack(pair_aux).mean()

            loss = ce + LAMBDA_1 * info + LAMBDA_2 * aux
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for key, value in (
                ("loss", loss),
                ("ce", ce),
                ("infonce", info),
                ("aux", aux),
            ):
                mean[key] += value.item() * len(pb.batch.candidates)
            mean_kacc += np.asarray(kaccs, dtype=float).sum(axis=0)
            n_samples += len(pb.batch.candidates)

    return {
        **{key: value / n_samples for key, value in mean.items()},
        "kacc": mean_kacc / n_samples,
    }


class PlateauScheduler:
    """The paper's schedule: after ``patience`` epochs without improvement the
    learning rate is cut to ``factor`` of its current value."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = PLATEAU_PATIENCE,
        factor: float = PLATEAU_FACTOR,
    ) -> None:
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, validation_loss: float) -> float:
        if validation_loss < self.best:
            self.best = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                for group in self.optimizer.param_groups:
                    group["lr"] *= self.factor
                self.bad_epochs = 0
        return self.optimizer.param_groups[0]["lr"]


def train(
    samples: list,
    checkpoint: Path,
    epochs: int = 100,
    n_augmented: int = N_AUGMENTED,
    pairs_per_batch: int = PAIRS_PER_BATCH,
    lr: float = LR,
    seed: int = 42,
    extra: dict | None = None,
) -> CAMBranchGNN:
    """The joint loop over the collected samples: draw the AMILP pairs,
    pretrain the PreNorm layers over them, then epochs of Adam on Eq. 25.

    The samples are the *subsampled* originals (the paper's 10% budget —
    :func:`subsample_budget` draws them); the free ×9 AMILPs happen here.
    ``extra`` is checkpoint provenance, saved alongside the state dict.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = augment_pairs(samples, rng, n_augmented=n_augmented)
    print(
        f"[train] {len(samples)} samples -> {len(pairs)} (original, AMILP) pairs",
        flush=True,
    )

    device = get_device()
    print(f"[train] device: {device}", flush=True)
    model = CAMBranchGNN().to(device)
    shuffle = np.random.default_rng(seed + 1)

    def batches() -> list[PairBatch]:
        return pair_batches(pairs, device, pairs_per_batch, shuffle)

    n_layers = pretrain(model, [pb.batch for pb in batches()])
    print(f"[train] froze {n_layers} PreNorm layers", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        stats = process(model, batches(), optimizer=optimizer)
        kacc = " ".join(
            f"acc@{k}={a:.2f}" for k, a in zip(TOP_K, stats["kacc"], strict=True)
        )
        print(
            f"[train] epoch {epoch:3d}  loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  "
            f"infonce={stats['infonce']:.4f}  aux={stats['aux']:.4f}  {kacc}",
            flush=True,
        )

    save_checkpoint(model.cpu(), checkpoint, extra=extra)
    print(f"[train] checkpoint saved to {checkpoint}", flush=True)
    return model


def save_checkpoint(
    model: CAMBranchGNN, path: str | Path, extra: dict | None = None
) -> None:
    """Save a policy checkpoint (state dict + any provenance)."""
    torch.save({"state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(
    path: str | Path, device: "torch.device | None" = None
) -> CAMBranchGNN:
    """Load a policy from a :func:`save_checkpoint` file (weights only, eval mode)."""
    model = CAMBranchGNN()
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location="cpu")["state_dict"]
    )
    if device is not None:
        model.to(device)
    model.eval()
    return model
