"""Lookback 2022 — training the GCNN with the ε-smooth target and the PAT term.

The model is the Gasse 2019 GCNN (identical weights layout); what Lookback
changes is the loss (Gupta et al., TMLR 2022, Eq. 3-6):

- **ε-smooth target** (Eq. 3): 1−ε on the strong-branching best, ε shared
  equally among the second-best strong-branching decisions (the argmax ties
  excluded);
- **Parent-as-Target (PAT) lookback term** (Eq. 5-6): when the child's
  expert choice is among the parent's second-best decisions (the lookback
  condition), a distillation cross-entropy pulls the child's candidate
  logits toward the parent model's logits on the same candidates, weighted
  by ``λ_PAT · N / Σ 1{L_i}``.

Hyperparameters: the paper reports a grid over {loss-target, λ_l2, λ_PAT}
and selects, for set cover, the one-hot target (``y``), λ_l2 = 0 and
λ_PAT = 0.1 — the defaults here. The ε value is not stated in the paper
text; 0.1 is a documented choice of this port.

Training infrastructure (pretrain, collate, checkpointing, top-k metrics)
is shared with the Gasse port; only the loss and the parent-child batch
plumbing are new.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from l2b_bench.methods.gasse.model import GNNPolicy
from l2b_bench.methods.gasse.train import (
    TOP_K,
    batched_tensors,
    collate,
    get_device,
    load_checkpoint,
    pretrain,
    save_checkpoint,
    to_device,
)

from .collect import LookbackSample, lookback_condition

# The paper's selected hyperparameters for set cover (Table 1 / App. I).
LAMBDA_PAT = 0.1
LOSS_TARGET = "y"  # "y" one-hot, "z" ε-smoothed
EPSILON = 0.1  # not stated in the paper — documented choice of this port


def smoothed_target(sample: LookbackSample, epsilon: float = EPSILON) -> np.ndarray:
    """The paper's ε-smooth target (Eq. 3) over the candidate set.

    1−ε on the expert's choice, ε divided equally among the second-best
    strong-branching decisions (ties on the second-best score, the choice
    excluded). With no finite second-best score, the mass stays on the
    choice.
    """
    candidates = np.asarray(sample.action_set)
    target = np.zeros(len(candidates))
    choice = int(np.where(candidates == sample.action)[0][0])
    target[choice] = 1.0 - epsilon

    rest = np.asarray([sample.scores[int(a)] for a in candidates], dtype=np.float64)
    rest[choice] = -np.inf
    finite = np.isfinite(rest)
    if not finite.any():
        target[choice] += epsilon
        return target
    second_best = rest[finite].max()
    tied = np.where(rest == second_best)[0]
    target[tied] += epsilon / len(tied)
    return target


@dataclass
class LookbackBatch:
    """A training batch: the samples plus their collected parents.

    ``batch`` collates children first (in ``samples`` order), then the
    parents appended at the back. ``parent_of`` maps a child's position to
    its parent's position; ``lookback`` is the paper's indicator (Eq. 4) per
    child; ``targets`` holds the per-child loss targets (one-hot or
    ε-smoothed distributions over the candidate set); ``var_offsets`` is
    each collated entry's offset into the batch's variable axis — the PAT
    term evaluates the parent's logits at the *child's* candidate variables,
    which share the local variable indexing within one solve.
    """

    batch: object  # a gasse GraphBatch
    parent_of: dict[int, int] = field(default_factory=dict)
    lookback: torch.Tensor | None = None  # (n_children,) bool
    targets: list[torch.Tensor] = field(default_factory=list)  # per child
    var_offsets: list[int] = field(default_factory=list)


def collate_pairs(
    samples: list[LookbackSample],
    pair_of: dict[int, int],
    loss_target: str = LOSS_TARGET,
    epsilon: float = EPSILON,
) -> LookbackBatch:
    """Collate children and their collected parents into one batch.

    ``pair_of`` maps a child sample's index to its parent sample's index
    (:func:`.collect.pair_parent_children`). Children are collated first (in
    ``samples`` order), then each parent is appended at the back.
    """
    entries: list[LookbackSample] = list(samples)
    parent_of: dict[int, int] = {}
    for child_i in sorted(pair_of):
        parent_of[child_i] = len(entries)
        entries.append(samples[pair_of[child_i]])
    var_offsets = []
    offset = 0
    for entry in entries:
        var_offsets.append(offset)
        offset += entry.observation.variable_features.shape[0]
    batch = collate(entries)
    lookback = torch.tensor(
        [
            i in pair_of and lookback_condition(samples[pair_of[i]], samples[i])
            for i in range(len(samples))
        ],
        dtype=torch.bool,
    )
    if loss_target == "z":
        targets = [
            torch.as_tensor(smoothed_target(sample, epsilon), dtype=torch.float32)
            for sample in samples
        ]
    else:
        targets = []
    return LookbackBatch(
        batch=batch, parent_of=parent_of, lookback=lookback, targets=targets, var_offsets=var_offsets
    )


def _to_device(batch: LookbackBatch, device: torch.device) -> LookbackBatch:
    return LookbackBatch(
        batch=to_device(batch.batch, device),
        parent_of=batch.parent_of,
        lookback=batch.lookback.to(device),
        targets=[t.to(device) for t in batch.targets],
        var_offsets=batch.var_offsets,
    )


def process_lookback(
    policy: GNNPolicy,
    batches: list[LookbackBatch],
    optimizer: torch.optim.Optimizer | None = None,
    loss_target: str = LOSS_TARGET,
    lambda_pat: float = LAMBDA_PAT,
) -> dict[str, float | np.ndarray]:
    """One pass over the batches with the Lookback loss (Eq. 5-6).

    The reported loss is the baseline cross-entropy (the PAT term is
    training-only, as in the official λ_PAT grid); the metrics are the
    official top-k accuracies against the expert's scores.
    """
    mean_loss = 0.0
    mean_pat = 0.0
    mean_kacc = np.zeros(len(TOP_K))
    n_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for lb in batches:
            logits = policy(*batched_tensors(lb.batch))
            ce_losses = []
            pat_losses = []
            kaccs = []
            n_children = len(lb.lookback)
            # The loss sums over children only — the parent entries at the
            # back of the batch exist solely for the PAT term.
            for i in range(n_children):
                candidates = lb.batch.candidates[i]
                cand_logits = logits[candidates]
                choice = lb.batch.choices[i]

                if loss_target == "z":
                    ce = -(lb.targets[i] * F.log_softmax(cand_logits, dim=-1)).sum()
                else:
                    ce = F.cross_entropy(cand_logits.unsqueeze(0), choice.unsqueeze(0))
                ce_losses.append(ce)

                if i in lb.parent_of and lb.lookback[i]:
                    # Cross-distillation (Parent-as-Target): the parent model's
                    # distribution over the child's candidates teaches the
                    # child. The child's candidates are mapped onto the
                    # parent's variable block (same local indexing within one
                    # solve — the shared LP columns).
                    child_offset = lb.var_offsets[i]
                    parent_offset = lb.var_offsets[lb.parent_of[i]]
                    parent_vars = candidates - child_offset + parent_offset
                    parent_logits = logits[parent_vars]
                    pat_losses.append(
                        -(
                            F.softmax(parent_logits, dim=-1) * F.log_softmax(cand_logits, dim=-1)
                        ).sum()
                    )

                true_scores = lb.batch.candidate_scores[i]
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
            if pat_losses:
                # Eq. 5-6: λ_PAT · N/Σ1{L} · Σ 1{L}·pat — the N/Σ1{L}
                # normalizer reduces the second term to a mean over the
                # lookback children.
                pat_loss = lambda_pat * torch.stack(pat_losses).mean()
            else:
                pat_loss = torch.zeros((), device=cross_entropy_loss.device)
            loss = cross_entropy_loss + pat_loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += cross_entropy_loss.item() * n_children
            mean_pat += pat_loss.item() * n_children
            mean_kacc += np.asarray(kaccs, dtype=float).sum(axis=0)
            n_samples += n_children

    return {
        "loss": mean_loss / n_children,
        "pat": mean_pat / n_children,
        "kacc": mean_kacc / n_children,
    }
