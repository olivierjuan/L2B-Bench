"""DSC-CL 2025 — the stratified contrastive training loop.

Paper: "SC-MILP" (Lu, Ma & Tao, arXiv 2511.21107), Eq 5 and Sec 4.4:

    L = L_sup + lambda * L_cons

- ``L_sup`` — Gasse's candidate cross-entropy, on the (possibly derived)
  samples; the derivations keep the original labels.
- ``L_cons`` — the dynamic stratified contrastive loss over a mini-batch of
  graph embeddings: for each node i, the positives P(i) are the batch's other
  same-cluster nodes, and the denominator runs over every j != i with each
  negative's similarity weighted by

      w(g_i, g_j) = sigmoid(alpha_|g_i - g_j|),
      alpha_k = 1 + sum_{j<=k} softplus(theta_j)   (alpha_0 = 1),

  the theta_j learnable — the "dynamic" part: how much the loss separates two
  clusters is learned, not fixed. Cosine similarity, temperature 0.08.
- ``lambda`` — a learnable balancing coefficient (its initial value is not
  specified; the pilot takes 0.1), set to 0 at inference (the projection is
  training-side only).

The nodes are grouped offline by K-means over their feature distributions
(m = 4 on setcover — the paper's elbow choice; the pilot clusters the
mean-pooled variable features of each sample, a declared choice: the paper
does not spell out the clustering representation). The paper says "the
training set follows the standard configuration in prior works" — the pilot
follows Gasse's (Adam 1e-3, batches of 32 samples). Each epoch re-draws every
sample's derivation (35% equivalent / 10% perturbed) so the scarce upstream
nodes keep yielding fresh views — a pilot reading of "upstream augmentation".
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
from .augment import derive_sample
from .model import DSCCLGNN

#: The contrastive temperature (Eq 5).
TAU = 0.08

#: The number of K-means clusters: 4 on setcover/cauctions/indset, 5 on
#: facilities (the paper's elbow); the pilot family is setcover.
M_CLUSTERS = 4

#: lambda's initial value is not specified — the pilot takes 0.1.
LAMBDA_INIT = 0.1

#: "The standard configuration in prior works" — the pilot follows Gasse's.
LR = 1e-3
BATCH_SIZE = 32


def assign_groups(samples: list, m: int = M_CLUSTERS, seed: int = 0) -> np.ndarray:
    """The offline K-means strata: one cluster id per sample, from the
    mean-pooled variable features (the paper's node feature distributions)."""
    from sklearn.cluster import KMeans

    features = np.stack(
        [
            np.asarray(s.observation.variable_features, dtype=float).mean(axis=0)
            for s in samples
        ]
    )
    n_clusters = min(m, len(samples))
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(
        features
    )


class StratifiedContrast(torch.nn.Module):
    """The learnable parts of Eq 5: the inter-cluster weights (theta) and the
    balancing coefficient lambda."""

    def __init__(self, m: int = M_CLUSTERS, lam: float = LAMBDA_INIT) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(m - 1))
        self.lam = torch.nn.Parameter(torch.tensor(float(lam)))

    def weight_matrix(self) -> torch.Tensor:
        """W[a, b] = sigmoid(alpha_|a - b|) with alpha_0 = 1 and alpha_k the
        cumulative softplus of the learnable theta."""
        softplus = F.softplus(self.theta)
        alpha = torch.cat(
            [torch.ones(1, device=softplus.device), 1.0 + softplus.cumsum(0)]
        )
        diff = (
            torch.arange(len(alpha), device=softplus.device).unsqueeze(0)
            - torch.arange(len(alpha), device=softplus.device).unsqueeze(1)
        ).abs()
        return torch.sigmoid(alpha[diff])

    def forward(self, embeddings: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """L_cons (Eq 5) over one mini-batch of L2-normalized graph
        embeddings with their cluster ids."""
        n = embeddings.shape[0]
        weights = self.weight_matrix()[groups][:, groups]
        logits = weights * (embeddings @ embeddings.t()) / TAU
        # The denominator excludes the diagonal (j != i).
        exp = torch.exp(logits).masked_fill(
            torch.eye(n, dtype=torch.bool, device=logits.device), 0.0
        )
        denominator = exp.sum(dim=1)
        losses = []
        for i in range(n):
            positives = (groups == groups[i]).clone()
            positives[i] = False
            if not bool(positives.any()):
                continue
            losses.append(-torch.log(exp[i, positives] / denominator[i]).mean())
        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()


@dataclass
class SampleBatch:
    """A mini-batch of samples (originals or derivations) as one big
    disconnected graph, with the per-sample segment sizes the graph embedding
    pools over and the samples' cluster ids."""

    batch: GraphBatch
    n_vars: list[int]
    n_cons: list[int]
    groups: list[int]


def collate_batch(samples: list, groups: list[int]) -> SampleBatch:
    """Collate the samples with gasse's concatenation."""
    return SampleBatch(
        batch=collate(samples),
        n_vars=[s.observation.variable_features.shape[0] for s in samples],
        n_cons=[s.observation.row_features.shape[0] for s in samples],
        groups=list(groups),
    )


def sample_batches(
    samples: list,
    groups: np.ndarray,
    device: torch.device,
    batch_size: int,
    shuffle: np.random.Generator,
) -> list[SampleBatch]:
    """One epoch's batches: a fresh permutation, on ``device``."""
    order = shuffle.permutation(len(samples))
    batches = [
        collate_batch(
            [samples[i] for i in order[k : k + batch_size]],
            [groups[i] for i in order[k : k + batch_size]],
        )
        for k in range(0, len(order), batch_size)
    ]
    return [
        SampleBatch(
            batch=to_device(b.batch, device),
            n_vars=b.n_vars,
            n_cons=b.n_cons,
            groups=b.groups,
        )
        for b in batches
    ]


def process(
    model: DSCCLGNN,
    contrast: StratifiedContrast,
    batches: list[SampleBatch],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | np.ndarray]:
    """One pass over the batches: L = L_sup + lambda * L_cons.

    Returns the mean losses, the top-k accuracies (k = 1, 3, 5, 10) against
    the expert's best score over every sample, and the current lambda.
    """
    mean = {"loss": 0.0, "ce": 0.0, "lcons": 0.0}
    mean_kacc = np.zeros(len(TOP_K))
    n_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for sb in batches:
            logits, embeddings = model(
                *batched_tensors(sb.batch),
                n_vars_per_sample=sb.n_vars,
                n_cons_per_sample=sb.n_cons,
            )
            # The supervised term: gasse's candidate cross-entropy.
            ce_losses = []
            kaccs = []
            for i, candidates in enumerate(sb.batch.candidates):
                cand_logits = logits[candidates]
                choice = sb.batch.choices[i]
                ce_losses.append(
                    F.cross_entropy(cand_logits.unsqueeze(0), choice.unsqueeze(0))
                )
                true_scores = sb.batch.candidate_scores[i]
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

            # The stratified contrastive term, on the graph embeddings.
            lcons = contrast(
                embeddings, torch.tensor(sb.groups, device=embeddings.device)
            )
            loss = ce + contrast.lam * lcons
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for key, value in (
                ("loss", loss),
                ("ce", ce),
                ("lcons", lcons),
            ):
                mean[key] += value.item() * len(sb.batch.candidates)
            mean_kacc += np.asarray(kaccs, dtype=float).sum(axis=0)
            n_samples += len(sb.batch.candidates)

    return {
        **{key: value / n_samples for key, value in mean.items()},
        "kacc": mean_kacc / n_samples,
        "lambda": float(contrast.lam.detach()),
    }


def train(
    samples: list,
    checkpoint: Path,
    epochs: int = 100,
    m: int = M_CLUSTERS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = 42,
    extra: dict | None = None,
) -> DSCCLGNN:
    """The loop over the collected samples: group them (offline K-means),
    pretrain the PreNorm layers over the originals, then epochs of Adam on
    L = L_sup + lambda L_cons with a fresh derivation draw per epoch.

    ``extra`` is checkpoint provenance, saved alongside the state dict.
    """
    torch.manual_seed(seed)
    groups = assign_groups(samples, m=m, seed=seed)
    sizes = np.bincount(groups, minlength=m)
    print(
        f"[train] {len(samples)} samples in {m} strata: {sizes.tolist()}",
        flush=True,
    )

    device = get_device()
    print(f"[train] device: {device}", flush=True)
    model = DSCCLGNN().to(device)
    contrast = StratifiedContrast(m=m).to(device)
    shuffle = np.random.default_rng(seed + 1)

    def batches(derive_rng: np.random.Generator | None) -> list[SampleBatch]:
        pool = (
            samples
            if derive_rng is None
            else [derive_sample(s, derive_rng) for s in samples]
        )
        return sample_batches(pool, groups, device, batch_size, shuffle)

    n_layers = pretrain(model, [sb.batch for sb in batches(None)])
    print(f"[train] froze {n_layers} PreNorm layers", flush=True)

    optimizer = torch.optim.Adam([*model.parameters(), *contrast.parameters()], lr=lr)
    for epoch in range(1, epochs + 1):
        stats = process(
            model,
            contrast,
            batches(np.random.default_rng(seed + 1 + epoch)),
            optimizer=optimizer,
        )
        kacc = " ".join(
            f"acc@{k}={a:.2f}" for k, a in zip(TOP_K, stats["kacc"], strict=True)
        )
        print(
            f"[train] epoch {epoch:3d}  loss={stats['loss']:.4f}  ce={stats['ce']:.4f}  "
            f"lcons={stats['lcons']:.4f}  lambda={stats['lambda']:.4f}  {kacc}",
            flush=True,
        )

    save_checkpoint(model.cpu(), checkpoint, extra=extra)
    print(f"[train] checkpoint saved to {checkpoint}", flush=True)
    return model


def save_checkpoint(
    model: DSCCLGNN, path: str | Path, extra: dict | None = None
) -> None:
    """Save a policy checkpoint (state dict + any provenance). The stratified
    head is training-side only (lambda = 0 at inference), so the state dict is
    the backbone's."""
    torch.save({"state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: "torch.device | None" = None) -> DSCCLGNN:
    """Load a policy from a :func:`save_checkpoint` file (weights only, eval mode)."""
    model = DSCCLGNN()
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location="cpu")["state_dict"]
    )
    if device is not None:
        model.to(device)
    model.eval()
    return model
