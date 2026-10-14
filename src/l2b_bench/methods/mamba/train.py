"""Mamba-Branching 2026 — the two-stage imitation training.

Port of ``main.py`` from the official repository
(github.com/doctor-watson626/Mamba-Branching):

- **Stage 1 — contrastive (``real_clip_process``)**: the TreeGateNet is
  pretrained by pulling the max-pooled candidate embedding towards the
  expert action's embedding and pushing it away from the other candidates'
  rows (pads included) — ``2 - cos(pre, strong) + mean cos(pre, weak)``,
  Adam lr 1e-4;
- **Stage 2 — sequence (``finetune_process_large``)**: each trajectory is
  sequentialized — every decision contributes its candidate rows then the
  chosen action's row, all 8-dim, one block per node — and the GroupedMamba
  reads the path; the logits are the mean of the Mamba output's 8 dims at
  each candidate's position, cross-entropy against the expert action. AdamW
  lr 1e-3 (weight decay 1e-2) over the Mamba **and** the TreeGateNet jointly
  (the embedder keeps training with the sequence loss).

The hyperparameters below are ``main.py``'s defaults (batch 32, 50 epochs
per stage, gradient-norm clip 1.0, max_seq_length 100). The official clamps
the logits to ±50 for its 'large'/'mid' data types; the default 'less' —
the regime ported here — does not clamp (``CLAMP_LOGITS = False``).

Trajectories are chunked to ``MAX_SEQ_LENGTH`` decisions
(``convert_pkl_to_h5``): sequences of length ≤ 1 are dropped, longer ones
are cut into max-length pieces (a trailing piece of length ≤ 1 is dropped
too).
"""

from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from l2b_bench.methods.gasse.train import get_device

from .collect import MambaSample
from .model import EMBED_DIM, GroupedMamba, TreeGateNet

# main.py's defaults.
CLIP_LR = 1e-4
CLIP_EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 0.01
EPOCHS = 50
BATCH_SIZE = 32
GRAD_CLIP = 1.0
MAX_SEQ_LENGTH = 100

# evaluate_all.py reads the sequence path over the last 49 blocks; the
# training chunking (main.py) uses the table's full length.
EVAL_MAX_SEQ_LEN = 49

CLAMP_LOGITS = False  # the official data_type='less' default


def chunk_trajectories(
    trajectories: list[list[MambaSample]], max_seq_length: int = MAX_SEQ_LENGTH
) -> list[list[MambaSample]]:
    """The official ``convert_pkl_to_h5`` chunking.

    A trajectory of length ≤ 1 carries no decision order to learn; a longer
    one is cut into ``max_seq_length``-decision pieces (the position-embedding
    table's size — position ids are node indices).
    """
    chunks: list[list[MambaSample]] = []
    for trajectory in trajectories:
        if len(trajectory) <= 1:
            continue
        for start in range(0, len(trajectory), max_seq_length):
            piece = trajectory[start : start + max_seq_length]
            if len(piece) > 1:
                chunks.append(piece)
    return chunks


def _action_index(sample: MambaSample) -> int:
    """The expert's choice as an index into the node's candidate rows."""
    return int(np.where(np.asarray(sample.action_set) == sample.action)[0][0])


def _tree_state(sample: MambaSample) -> np.ndarray:
    return np.concatenate([sample.node_features, sample.mip_features])


def clip_batch(samples: list[MambaSample]) -> dict[str, torch.Tensor]:
    """Pad one mini-batch of nodes for the contrastive stage."""
    n_cands = max(len(s.action_set) for s in samples)
    cands = np.zeros((len(samples), n_cands, samples[0].candidate_features.shape[-1]), dtype=np.float32)
    states = np.stack([_tree_state(s) for s in samples]).astype(np.float32)
    targets = np.empty(len(samples), dtype=np.int64)
    for i, sample in enumerate(samples):
        cands[i, : len(sample.action_set)] = sample.candidate_features
        targets[i] = _action_index(sample)
    return {
        "cands": torch.from_numpy(cands),
        "states": torch.from_numpy(states),
        "targets": torch.from_numpy(targets),
    }


def clip_loss(net: TreeGateNet, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """``2 - cos(pre, strong) + mean cos(pre, weak)`` over the mini-batch.

    ``pre`` is the max-pooled candidate embedding; the weak rows are every
    non-chosen candidate row, pads included (the official's weak set).
    """
    cands, states, targets = batch["cands"], batch["states"], batch["targets"]
    if torch.isnan(cands).any() or torch.isnan(states).any():
        return None
    embedded, _ = net(cands, states)  # (B, C, 8)
    pre = embedded.max(dim=1).values  # (B, 8)
    index = torch.arange(len(targets), device=targets.device)
    strong = embedded[index, targets]  # (B, 8)
    weak_mask = torch.ones_like(embedded[..., 0], dtype=torch.bool)
    weak_mask[index, targets] = False
    weak = embedded[weak_mask].view(len(targets), embedded.size(1) - 1, -1)  # (B, C-1, 8)
    loss = 2 - F.cosine_similarity(pre, strong).mean()
    loss = loss + F.cosine_similarity(pre.unsqueeze(1), weak, dim=-1).mean()
    if torch.isnan(loss):
        return None
    return loss


def sequence_batch(sequences: list[list[MambaSample]]) -> dict[str, torch.Tensor]:
    """Pad one mini-batch of trajectories for the sequence stage."""
    n_nodes = max(len(seq) for seq in sequences)
    n_cands = max(len(s.action_set) for seq in sequences for s in seq)
    n_features = sequences[0][0].candidate_features.shape[-1]
    cands = np.zeros((len(sequences), n_nodes, n_cands, n_features), dtype=np.float32)
    states = np.zeros((len(sequences), n_nodes, 8 + 53), dtype=np.float32)
    actions = np.zeros((len(sequences), n_nodes), dtype=np.int64)
    valid = np.zeros((len(sequences), n_nodes), dtype=bool)
    for b, sequence in enumerate(sequences):
        for i, sample in enumerate(sequence):
            cands[b, i, : len(sample.action_set)] = sample.candidate_features
            states[b, i] = _tree_state(sample)
            actions[b, i] = _action_index(sample)
            valid[b, i] = True
    return {
        "cands": torch.from_numpy(cands),
        "states": torch.from_numpy(states),
        "actions": torch.from_numpy(actions),
        "valid": torch.from_numpy(valid),
    }


def sequence_loss(
    net: TreeGateNet,
    mamba: GroupedMamba,
    batch: dict[str, torch.Tensor],
    clamp_logits: bool = CLAMP_LOGITS,
) -> tuple[torch.Tensor, float] | None:
    """The sequence stage's loss (and acc@1) for one mini-batch.

    Each node contributes ``n_cands + 1`` tokens — its candidate rows then
    the chosen action's row — all sharing the node's position id; the logits
    are the mean of the Mamba output's 8 dims at the candidate positions.
    """
    cands, states = batch["cands"], batch["states"]
    if torch.isnan(cands).any() or torch.isnan(states).any():
        return None
    batch_size, n_nodes, n_cands, _ = cands.shape

    embedded, _ = net(cands, states)  # (B, L, C, 8) — the 4D path
    action_embedding = embedded.gather(
        2, batch["actions"].view(batch_size, n_nodes, 1, 1).expand(-1, -1, 1, EMBED_DIM)
    ).squeeze(2)  # (B, L, 8)

    # Interleave: [candidates; action] per node — one block of C+1 tokens.
    inputs = cands.new_zeros(batch_size, n_nodes * (n_cands + 1), EMBED_DIM)
    position_ids = torch.zeros(batch_size, n_nodes * (n_cands + 1), dtype=torch.long, device=cands.device)
    candidate_positions = torch.zeros(
        batch_size, n_nodes * (n_cands + 1), dtype=torch.bool, device=cands.device
    )
    for i in range(n_nodes):
        start = i * (n_cands + 1)
        inputs[:, start : start + n_cands] = embedded[:, i]
        inputs[:, start + n_cands] = action_embedding[:, i]
        position_ids[:, start : start + n_cands + 1] = i
        candidate_positions[:, start : start + n_cands] = True

    output = mamba(inputs, position_ids)  # (B, L*(C+1), 8)
    candidate_output = output[candidate_positions].view(batch_size, n_nodes, n_cands, EMBED_DIM)
    logits = candidate_output.mean(dim=-1)  # (B, L, C)
    if clamp_logits:
        logits = logits.clamp(-50, 50)

    valid = batch["valid"]
    loss = F.cross_entropy(logits[valid], batch["actions"][valid])
    if torch.isnan(loss):
        return None
    with torch.no_grad():
        acc1 = (logits[valid].argmax(dim=-1) == batch["actions"][valid]).float().mean().item()
    return loss, acc1


def _batches(order: np.ndarray, batch_size: int, device: torch.device, collate: Any, items: list) -> list:
    return [
        {name: tensor.to(device) for name, tensor in collate([items[i] for i in order[k : k + batch_size]]).items()}
        for k in range(0, len(order), batch_size)
    ]


def train_clip(
    samples: list[MambaSample],
    epochs: int = CLIP_EPOCHS,
    device: torch.device | None = None,
    log: Any = print,
    seed: int = 0,
) -> TreeGateNet:
    """Stage 1 — the contrastive pretraining of the candidate embedder."""
    device = device or get_device()
    net = TreeGateNet().to(device)
    if not samples:
        return net
    optimizer = torch.optim.Adam(net.parameters(), lr=CLIP_LR)
    shuffle = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        total, n_batches = 0.0, 0
        for batch in _batches(shuffle.permutation(len(samples)), BATCH_SIZE, device, clip_batch, samples):
            loss = clip_loss(net, batch)
            if loss is None:  # the official skips NaN batches
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            optimizer.step()
            total, n_batches = total + loss.item(), n_batches + 1
        log(f"[train:mamba-clip] epoch {epoch:3d}  loss={total / max(n_batches, 1):.4f}")
    return net


def train_sequence(
    net: TreeGateNet,
    sequences: list[list[MambaSample]],
    epochs: int = EPOCHS,
    device: torch.device | None = None,
    log: Any = print,
    seed: int = 0,
) -> GroupedMamba:
    """Stage 2 — the Mamba over the sequentialized paths (joint with the embedder)."""
    device = device or get_device()
    mamba = GroupedMamba(max_seq_length=MAX_SEQ_LENGTH).to(device)
    if not sequences:
        return mamba
    optimizer = torch.optim.AdamW(
        chain(net.parameters(), mamba.parameters()), lr=LR, weight_decay=WEIGHT_DECAY
    )
    shuffle = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        total, acc, n_batches = 0.0, 0.0, 0
        for batch in _batches(shuffle.permutation(len(sequences)), BATCH_SIZE, device, sequence_batch, sequences):
            step = sequence_loss(net, mamba, batch)
            if step is None:  # the official skips NaN batches
                continue
            loss, acc1 = step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chain(net.parameters(), mamba.parameters()), GRAD_CLIP)
            optimizer.step()
            total, acc, n_batches = total + loss.item(), acc + acc1, n_batches + 1
        n = max(n_batches, 1)
        log(f"[train:mamba-seq] epoch {epoch:3d}  loss={total / n:.4f}  acc@1={acc / n:.3f}")
    return mamba


def save_checkpoint(
    net: TreeGateNet,
    mamba: GroupedMamba,
    path: str | Path,
    extra: dict | None = None,
) -> None:
    """Both stages' weights + the hparams needed to rebuild the nets."""
    checkpoint = {
        "clip_state_dict": net.state_dict(),
        "clip_hparams": dict(net.hparams),
        "mamba_state_dict": mamba.state_dict(),
        "max_seq_length": mamba.position_embed.num_embeddings,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> tuple[TreeGateNet, GroupedMamba]:
    """Rebuild (TreeGateNet, GroupedMamba) from a saved checkpoint."""
    device = device or get_device()
    data = torch.load(path, map_location=device, weights_only=True)
    net = TreeGateNet(**data["clip_hparams"])
    net.load_state_dict(data["clip_state_dict"])
    mamba = GroupedMamba(max_seq_length=int(data["max_seq_length"]))
    mamba.load_state_dict(data["mamba_state_dict"])
    net.to(device).eval()
    mamba.to(device).eval()
    return net, mamba


def train(
    trajectories: list[list[MambaSample]],
    checkpoint: Path,
    epochs: int = EPOCHS,
    clip_epochs: int = CLIP_EPOCHS,
    device: torch.device | None = None,
    log: Any = print,
) -> None:
    """The two-stage training: contrastive pretraining, then the sequence model.

    ``trajectories`` are whole solve trajectories (one per collected
    episode); the contrastive stage flattens them, the sequence stage chunks
    them to ``MAX_SEQ_LENGTH`` decisions.
    """
    device = device or get_device()
    log(f"[train:mamba] device: {device}")

    samples = [sample for trajectory in trajectories for sample in trajectory]
    log(f"[train:mamba] stage 1 (contrastive): {len(samples)} samples from {len(trajectories)} trajectories")
    net = train_clip(samples, epochs=clip_epochs, device=device, log=log)

    sequences = chunk_trajectories(trajectories)
    log(f"[train:mamba] stage 2 (sequence): {len(sequences)} chunks of <= {MAX_SEQ_LENGTH} decisions")
    mamba = train_sequence(net, sequences, epochs=epochs, device=device, log=log)

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        net.cpu(),
        mamba.cpu(),
        checkpoint,
        extra={"n_trajectories": len(trajectories), "n_samples": len(samples), "epochs": epochs},
    )
    log(f"[train:mamba] checkpoint saved to {checkpoint}")
