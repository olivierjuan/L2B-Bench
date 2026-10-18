"""Seq-BnB 2026 — the DAgger training loop (no official repository).

The paper's protocol, as reconstructed:

- **Round 0** — expert episodes: strong branching at the branching
  decisions, the x*-compatibility oracle at the node-selection decisions.
- **Rounds 1..R** — the current policy acts (its argmax); the expert's
  label is recorded on every decision regardless. Each round's fresh
  samples are aggregated with everything collected so far, and one epoch
  of minibatches updates the model.
- **Joint loss** — ``L = L_branch + lambda * L_ns`` with ``lambda = 0.1``,
  cross-entropy over each decision's action set (branching: the LP
  candidates; node selection: the open nodes).
- **Optimization** — Adam, lr 1e-5, batch size 64 (the paper's numbers).

The paper's sample budget (10k expert samples, then 11 rounds of ~10k —
120k decisions in total) is a campaign-time parameter here, as for every
ported method; the pilot runs a reduced budget. The per-round epoch count
is not specified in the paper — one pass over the aggregated data per
round (a pilot choice).

The loop checkpoints after every round (the policy weights plus a
sidecar state file holding the aggregated samples, the optimizer, and
the RNG), so a campaign killed at its wall time resumes where it stopped
— the same restartability contract as the other ports. A policy episode
can run to the per-episode time limit, so a DAgger round costs up to
``episodes_per_round * time_limit``; ten rounds do not fit every wall.
"""

import random
from pathlib import Path

import numpy as np
import torch

from .collect import PoolGenerator, SeqBnBSample, collect, make_env  # noqa: F401 (make_env re-exported for the pilot)
from .model import SeqBnBModel, collate, to_device
from .policy import SeqBnBPolicy

#: The paper's optimization (Adam lr 1e-5, batch 64) and joint-loss weight.
LR = 1e-5
BATCH_SIZE = 64
LAMBDA_NS = 0.1


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def joint_loss(
    model: SeqBnBModel,
    batch_samples: list[SeqBnBSample],
    device: torch.device,
) -> torch.Tensor:
    """``L_branch + lambda * L_ns`` over a minibatch: the mean
    cross-entropy per decision kind (a batch of one kind has no term for
    the other)."""
    batch = to_device(collate(batch_samples), device)
    logits = model(batch)
    branching, selection = [], []
    for i, sample_logits in enumerate(logits):
        target = batch.labels[i : i + 1]
        term = torch.nn.functional.cross_entropy(
            sample_logits.unsqueeze(0), target, reduction="sum"
        )
        (selection if int(batch.kinds[i]) == 0 else branching).append(term)
    loss = torch.zeros((), device=device)
    if branching:
        loss = loss + torch.stack(branching).mean()
    if selection:
        loss = loss + LAMBDA_NS * torch.stack(selection).mean()
    return loss


def _train_epoch(
    model: SeqBnBModel,
    optimizer: torch.optim.Optimizer,
    samples: list[SeqBnBSample],
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> float:
    """One pass of minibatch updates over the samples; the mean loss."""
    order = list(range(len(samples)))
    rng.shuffle(order)
    losses = []
    for start in range(0, len(order), batch_size):
        batch_samples = [samples[i] for i in order[start : start + batch_size]]
        optimizer.zero_grad()
        loss = joint_loss(model, batch_samples, device)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(model: SeqBnBModel, path: str | Path) -> None:
    """Save the policy checkpoint (weights only — the architecture is fixed)."""
    device = next(model.parameters()).device
    torch.save({"state_dict": model.cpu().state_dict()}, path)
    model.to(device)


def load_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> SeqBnBModel:
    """Load a policy checkpoint."""
    device = device or get_device()
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = SeqBnBModel()
    model.load_state_dict(data["state_dict"])
    return model.to(device).eval()


def _state_path(checkpoint: Path) -> Path:
    """The sidecar holding the loop's resume state (the checkpoint keeps
    the weights only, so the deployed policy loads a clean file)."""
    return Path(str(checkpoint) + ".state")


def train(
    instance_dir: Path,
    checkpoint: Path,
    rounds: int,
    episodes_per_round: int,
    base_seed: int = 42,
    time_limit: float | None = None,
    node_selection: bool = True,
) -> dict[str, float]:
    """The DAgger loop: expert round 0, then the policy's own episodes.

    Returns per-round summary counters (samples collected, mean loss).
    Restarts from the per-round checkpoint pair when both files exist."""
    device = get_device()
    model = SeqBnBModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    policy = SeqBnBPolicy(model)
    aggregated: list[SeqBnBSample] = []
    report: dict[str, float] = {}
    python_rng = random.Random(base_seed)
    start_round = 0
    seeds_used = 0
    state_path = _state_path(checkpoint)
    if checkpoint.exists() and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        aggregated = state["aggregated"]
        report = state["report"]
        python_rng.setstate(state["rng_state"])
        start_round = state["rounds_done"]
        seeds_used = state["seeds_used"]
        print(
            f"[train] resuming at round {start_round} "
            f"({len(aggregated)} samples, {seeds_used} episodes)",
            flush=True,
        )
    print(f"[train] device: {device}", flush=True)

    for round_index in range(start_round, rounds + 1):
        if round_index == 0:
            select_action = lambda s, d, e: e  # noqa: E731 — the expert round
        else:
            policy.model.eval()
            select_action = lambda s, d, e: policy.act(s)  # noqa: E731
        seeds = [base_seed + 1000 * round_index + i for i in range(episodes_per_round)]
        seeds_used += len(seeds)
        result = collect(
            instance_dir,
            n_episodes=episodes_per_round,
            seeds=seeds,
            select_action=select_action,
            time_limit=time_limit,
            node_selection=node_selection,
        )
        aggregated.extend(result.samples)
        loss = _train_epoch(
            model, optimizer, aggregated, BATCH_SIZE, python_rng, device
        )
        report[f"round_{round_index}"] = loss
        print(
            f"[round {round_index:2d}] episodes={result.n_episodes} "
            f"new_samples={len(result.samples)} total={len(aggregated)} loss={loss:.4f}",
            flush=True,
        )
        save_checkpoint(model, checkpoint)
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "rounds_done": round_index + 1,
                "aggregated": aggregated,
                "report": report,
                "rng_state": python_rng.getstate(),
                "optimizer": optimizer.state_dict(),
                "seeds_used": seeds_used,
            },
            state_path,
        )
        model.to(device)
    print(
        f"[train] saved {checkpoint} ({len(aggregated)} samples, {seeds_used} episodes)",
        flush=True,
    )
    return report
