"""Scavuzzo et al. 2022 (rl2branch) — the REINFORCE fine-tuning loop.

Faithful port of the official ``brain.py`` / ``agent.py`` / ``05_train_rl.py``
(github.com/lascavana/rl2branch), fine-tuning a Gasse imitation-learned
``GNNPolicy`` (the checkpoint of the IL stage, trained exactly like ``gasse``
with expert_probability 0.05) with policy-gradient RL:

- Training actions are *sampled* from ``Categorical(logits)`` over the node's
  branching candidates (greedy only at evaluation); 5% of each episode's
  transitions (``sample_rate``) are kept for the update.
- Credit assignment is one of the paper's three variants (``VARIANTS``):
  the flat ``mdp`` return — minus the number of nodes processed after the
  decision, read off the cumulative ``NNodes`` reward — or the tree-MDP
  return ``-(subtree_size) - 1`` (harness reward ``SubtreeSizes``), made
  well-defined by depth-first node selection (``tmdp+DFS``) or an objective
  limit at the training instances' optimum + 0.1 (``tmdp+ObjLim``).
- One epoch is 10 episodes; the update accumulates gradients over shuffled
  batches of 16 and takes a single Adam step (lr 1e-6). The loss is the mean
  over the epoch's samples of ``-returns * log_pi(action)`` minus the entropy
  bonus (1e-5) times the mean entropy.
- Episodes that exceed 50 000 iterations during training are discarded
  entirely (official stability guard).

Deviations from the official code, both pilot-scale: the official samples
its 10 epoch episodes with replacement from a fixed pool of 450 pre-solved
instances (20000 epochs); the pilot passes an iterator of fresh instances,
consumed sequentially. And the official computes the training optima in a
separate step (``02_solve_instances``); here ``with_objlimit`` solves each
instance (gasse parameters, then ``freeTransform``) right before handing it
to the environment — same numbers, no separate artifact.

The solver runs under the ``gasse_2019`` parameters plus
``timing/clocktype 2``; restarts stay disabled — a SCIP restart renumbers
the search tree and orphans the credit.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.environment import Environment
from gyozas.observations import NodeBipartite
from gyozas.rewards import NNodes, SubtreeSizes
from pyscipopt import Model

from ..gasse.model import GNNPolicy
from ..gasse.train import load_checkpoint, save_checkpoint

# config.default.json / brain.py, verbatim.
LR = 1e-6
ENTROPY_BONUS = 1e-5
SAMPLE_RATE = 0.05
EPISODES_PER_EPOCH = 10
BATCH_SIZE = 16
MAX_EPISODE_ITERATIONS = 50_000
OBJLIMIT_EPS = 0.1

#: The paper's three training variants (official ``--mode`` strings).
VARIANTS = ("mdp", "tmdp+DFS", "tmdp+ObjLim")

# agent.py: the gasse_2019 parameters plus the deterministic clock.
SCIP_PARAMS: dict[str, Any] = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
    "timing/clocktype": 2,
    "display/verblevel": 0,
}

# DFSBranchingDynamics: depth-first node selection during training.
DFS_PARAMS: dict[str, Any] = {
    "nodeselection/dfs/stdpriority": 666666,
    "nodeselection/dfs/memsavepriority": 666666,
}


@dataclass
class Transition:
    """One kept branching decision, credit assigned after the episode."""

    observation: Any  # the raw NodeBipartite observation at the decision
    action_set: np.ndarray  # the node's branching candidates
    action_idx: int  # the chosen candidate's rank within action_set
    node_id: int | None  # tmdp: the decision's focus node number
    cum_nnodes: float  # mdp: cumulative NNodes reward at the decision
    returns: float | None = None


def make_env(
    instances: Iterator[Model],
    variant: str,
    time_limit: float | None = None,
    seed: int = 0,
) -> Environment:
    """One training environment for the variant: NodeBipartite observation,
    NNodes (mdp) or SubtreeSizes (tmdp) reward, gasse params (+DFS for tmdp+DFS)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    params = dict(SCIP_PARAMS)
    if variant == "tmdp+DFS":
        params.update(DFS_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    env = Environment(
        instances,
        observation_function=NodeBipartite(),
        reward_function=NNodes() if variant == "mdp" else SubtreeSizes(),
        scip_params=params,
    )
    env.dynamics.seed(seed)
    return env


def with_objlimit(instances: Iterator[Model]) -> Iterator[Model]:
    """Yield models with SCIP's objective limit set just above their optimum.

    The official pre-computes the training optima (its step 02); here each
    instance is solved once under the gasse parameters, returned to its
    original state with ``freeTransform``, and handed on with
    ``setObjlimit(optimum + eps)`` (eps = +0.1 for a minimization problem,
    -0.1 otherwise). A solved model cannot drive a second environment — this
    is why the caller must pass fresh instances.
    """
    for model in instances:
        model.hideOutput()
        model.setParams(
            {
                "separating/maxrounds": 0,
                "presolving/maxrestarts": 0,
                "limits/time": 3600,
                "display/verblevel": 0,
            }
        )
        model.optimize()
        optimum = model.getObjVal()
        eps = -OBJLIMIT_EPS if model.getObjectiveSense() == "maximize" else OBJLIMIT_EPS
        model.freeTransform()
        model.setObjlimit(optimum + eps)
        yield model


def sample_action(
    policy: GNNPolicy, observation: Any, action_set: np.ndarray, device: torch.device
) -> int:
    """The official ``sample_action_idx``: one action sampled from the
    candidate-restricted Categorical (training exploration)."""
    obs = observation
    edge_index = torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
    logits = policy(
        torch.as_tensor(obs.row_features, dtype=torch.float32).to(device),
        edge_index.to(device),
        torch.as_tensor(obs.edge_features.values, dtype=torch.float32)
        .reshape(-1, 1)
        .to(device),
        torch.as_tensor(obs.variable_features, dtype=torch.float32).to(device),
    )
    candidates = torch.as_tensor(np.asarray(action_set), dtype=torch.long).to(device)
    return int(torch.distributions.Categorical(logits=logits[candidates]).sample())


def run_episode(
    env: Environment,
    policy: GNNPolicy,
    rng: np.random.Generator,
    device: torch.device,
    sample_rate: float = SAMPLE_RATE,
) -> tuple[list[Transition], float, int]:
    """One sampled episode; returns the kept transitions, the final cumulative
    NNodes reward, and the episode's iteration count."""
    reward_fn = env.reward_function
    tree = isinstance(reward_fn, SubtreeSizes)
    obs, action_set, reward, done, _info = env.reset()
    transitions: list[Transition] = []
    cum_nnodes = float(reward)  # the reset extraction's reward
    n_iter = 0
    while not done:
        # The reward function has already recorded the decision's focus node
        # (at reset for the first decision, at the previous step otherwise).
        node_id = reward_fn.node_ids[-1] if tree else None
        action_idx = sample_action(policy, obs, action_set, device)
        if rng.random() < sample_rate:
            transitions.append(
                Transition(obs, np.asarray(action_set), action_idx, node_id, cum_nnodes)
            )
        obs, action_set, reward, done, _info = env.step(
            int(np.asarray(action_set)[action_idx])
        )
        n_iter += 1
        cum_nnodes += float(reward)
    return transitions, cum_nnodes, n_iter


def assign_returns(
    transitions: list[Transition],
    env: Environment,
    final_cum_nnodes: float,
    variant: str,
) -> None:
    """Credit assignment: flat MDP from the cumulative NNodes, or the tree
    return -(subtree_size) - 1 keyed by each decision's focus node."""
    if variant == "mdp":
        for transition in transitions:
            transition.returns = transition.cum_nnodes - final_cum_nnodes
    else:
        sizes = env.reward_function.subtree_sizes
        assert sizes is not None, "assign_returns called before the episode finished"
        for transition in transitions:
            transition.returns = -sizes[transition.node_id] - 1


class TransitionBatch:
    """A mini-batch of transitions collated into one disconnected graph."""

    def __init__(self, transitions: list[Transition]) -> None:
        n_vars = [len(t.observation.variable_features) for t in transitions]
        edge_index = []
        cons_offset = 0
        var_offset = 0
        for transition, n_v in zip(transitions, n_vars, strict=True):
            obs = transition.observation
            edge_index.append(
                torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
                + torch.tensor([[cons_offset], [var_offset]], dtype=torch.long)
            )
            cons_offset += len(obs.row_features)
            var_offset += n_v
        self.constraint_features = torch.cat(
            [
                torch.as_tensor(t.observation.row_features, dtype=torch.float32)
                for t in transitions
            ]
        )
        self.edge_index = torch.cat(edge_index, dim=1)
        self.edge_attr = torch.cat(
            [
                torch.as_tensor(
                    t.observation.edge_features.values, dtype=torch.float32
                ).reshape(-1, 1)
                for t in transitions
            ]
        )
        self.variable_features = torch.cat(
            [
                torch.as_tensor(t.observation.variable_features, dtype=torch.float32)
                for t in transitions
            ]
        )
        # Each sample's candidates carry its variable offset in the collated graph.
        self.candidates = [
            torch.as_tensor(t.action_set, dtype=torch.long) + sum(n_vars[:i])
            for i, t in enumerate(transitions)
        ]
        self.actions = torch.tensor(
            [t.action_idx for t in transitions], dtype=torch.long
        )
        self.returns = torch.tensor(
            [t.returns for t in transitions], dtype=torch.float32
        )

    def to(self, device: torch.device) -> "TransitionBatch":
        self.constraint_features = self.constraint_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.variable_features = self.variable_features.to(device)
        self.candidates = [c.to(device) for c in self.candidates]
        self.actions = self.actions.to(device)
        self.returns = self.returns.to(device)
        return self


def reinforce_update(
    policy: GNNPolicy,
    transitions: list[Transition],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """One epoch's single Adam step: gradients accumulated over shuffled
    batches of 16, normalized by the epoch's full sample count."""
    n_samples = len(transitions)
    if n_samples < 1:
        return {"loss": 0.0, "reinforce_loss": 0.0, "entropy": 0.0}

    order = torch.randperm(n_samples)
    stats = {"loss": 0.0, "reinforce_loss": 0.0, "entropy": 0.0}
    optimizer.zero_grad()
    for k in range(0, n_samples, BATCH_SIZE):
        batch = TransitionBatch(
            [transitions[i] for i in order[k : k + BATCH_SIZE].tolist()]
        ).to(device)
        logits = policy(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )
        # One categorical per sample, over its own candidate slice.
        log_probs = []
        entropies = []
        for candidates, action in zip(batch.candidates, batch.actions, strict=True):
            dist = torch.distributions.Categorical(logits=logits[candidates])
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        reinforce_loss = -(batch.returns * torch.stack(log_probs)).sum() / n_samples
        entropy = torch.stack(entropies).sum() / n_samples
        loss = reinforce_loss - ENTROPY_BONUS * entropy
        loss.backward()
        stats["loss"] += float(loss.detach())
        stats["reinforce_loss"] += float(reinforce_loss.detach())
        stats["entropy"] += float(entropy.detach())
    optimizer.step()
    return stats


def train(
    instances: Iterator[Model],
    il_checkpoint: str | Path,
    checkpoint: str | Path,
    episodes: int,
    variant: str = "tmdp+ObjLim",
    time_limit: float | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    log=print,
) -> GNNPolicy:
    """The official RL loop: REINFORCE epochs of 10 episodes each.

    ``episodes`` is the total RL budget; ``instances`` must yield a fresh
    model per episode (they are consumed sequentially — the official samples
    with replacement from a fixed pool, see the module docstring). The policy
    starts from the imitation-learned ``il_checkpoint`` (the gasse-format
    GNNPolicy) and every epoch overwrites ``checkpoint``.
    """
    if variant == "tmdp+ObjLim":
        instances = with_objlimit(instances)
    device = device or get_device()
    env = make_env(instances, variant, time_limit=time_limit, seed=seed)
    policy = load_checkpoint(il_checkpoint, device=device)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    epochs = max(1, episodes // EPISODES_PER_EPOCH)
    episode_idx = 0
    epoch = 0
    while epoch < epochs:
        # One epoch: up to EPISODES_PER_EPOCH episodes (the instance iterator
        # may run dry first — a shorter final epoch is fine).
        epoch_samples: list[Transition] = []
        kept = 0
        for _ in range(EPISODES_PER_EPOCH):
            try:
                transitions, final_cum, n_iter = run_episode(env, policy, rng, device)
            except StopIteration:
                break
            episode_idx += 1
            if n_iter > MAX_EPISODE_ITERATIONS:
                # Official stability guard: too-large trees are discarded.
                continue
            assign_returns(transitions, env, final_cum, variant)
            epoch_samples.extend(transitions)
            kept += len(transitions)
        epoch += 1
        stats = reinforce_update(policy, epoch_samples, optimizer, device)
        log(
            f"[train] epoch {epoch:4d}  episodes={episode_idx}  "
            f"samples={len(epoch_samples)} (kept {kept})  "
            f"loss={stats['loss']:.4f}  entropy={stats['entropy']:.2f}"
        )
        # Checkpoint after every epoch — a cluster job killed at its wall time
        # keeps everything learned so far.
        save_checkpoint(
            policy.cpu(),
            checkpoint,
            extra={
                "variant": variant,
                "epochs": epoch,
                "episodes": episode_idx,
                "il_checkpoint": str(il_checkpoint),
                "seed": seed,
            },
        )
        policy.to(device)
    log(f"[train] checkpoint saved to {checkpoint}")
    return policy


def get_device() -> "torch.device":
    """CUDA when available (the campaign nodes' A5000), CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_checkpoint(
    path: str | Path, device: torch.device | None = None
) -> GNNPolicy:
    """The RL-fine-tuned policy (same format as the IL checkpoint)."""
    device = device or get_device()
    model = load_checkpoint(path, device=device)
    model.eval()
    return model
