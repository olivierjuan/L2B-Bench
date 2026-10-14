"""RetroBranching (Parsonson & Deisenroth 2022) — the DQN training loop.

Faithful port of the official ``experiments/dqn_trainer.py`` driving
``src/learners/dqn_learner.py`` under the ``retro.yaml`` /
``dqn_learner.yaml`` / ``dqn_agent.yaml`` configuration:

- The agent explores with probability epsilon = 0.025 (uniform random
  branching candidates), otherwise branches greedily on its Q values.
- Every transition of every episode enters a uniform replay buffer
  (capacity 100k), but only *after* the episode finishes: the
  ``retro_binary_fathomed`` reward is retrospective, so the harness's
  ``subtree_transitions`` decomposes the finished episode into sub-tree
  episodes first.
- The buffer fills to 20k transitions before any learning; then each batch
  of transitions added buys ``transitions // 10`` gradient steps (batch 128,
  MSE on 1-step TD targets, Adam lr 5e-5, gradient value clipping 10).
- The 1-step target is Double-DQN in math even though the official config
  leaves ``use_double_dqn`` off: the *online* network selects the greedy
  next action, the *target* network evaluates it,
  ``q_target = (1 - done) * gamma * Q_target(s')[argmax_a Q_online(s')]``.
- The target network follows every gradient step with a soft Polyak update
  (tau = 1e-4); the configured hard update every 10k steps is
  short-circuited by ``or soft_update_target_tau is not None`` in the
  official code and never fires.
- The ``n_step_return=3`` field the official stores in each transition is
  dead config: ``use_n_step_dqn`` is absent from the config and defaults to
  False, so the loss is 1-step TD. It is not ported.

The solver runs under the paper's ``gasse_2019`` parameters (no cuts, no
restarts — a SCIP restart renumbers the search tree and orphans the
sub-tree episodes), training on fresh set-cover instances as the official
``set_covering`` generator does.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.environment import Environment
from gyozas.observations import NodeBipartiteRetro
from gyozas.rewards import RetroBinaryFathomed, subtree_transitions
from pyscipopt import Model

from .model import RetroGCN
from .replay import GraphState, ReplayBuffer, Transition, graph_state

# dqn_learner.yaml / dqn_agent.yaml, verbatim.
LR = 0.00005
GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_CAPACITY = 100_000
BUFFER_MIN_LENGTH = 20_000
STEPS_PER_UPDATE = 10
EPSILON = 0.025
SOFT_UPDATE_TAU = 0.0001
GRADIENT_CLIP_VALUE = 10.0

# gasse_2019_scip_params.py, plus a quiet output and the episode budget.
SCIP_PARAMS: dict[str, Any] = {
    "separating/maxrounds": 0,
    "presolving/maxrestarts": 0,
    "limits/time": 3600,
    "display/verblevel": 0,
}


def get_device() -> "torch.device":
    """CUDA when available (the campaign nodes' A5000), CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(instances: Iterator[Model], time_limit: float | None = None, seed: int = 0) -> Environment:
    """One training environment: retro observation, retro reward, gasse params."""
    params = dict(SCIP_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    env = Environment(
        instances,
        observation_function=NodeBipartiteRetro(),
        reward_function=RetroBinaryFathomed(),
        scip_params=params,
    )
    env.dynamics.seed(seed)
    return env


def greedy_action(model: RetroGCN, state: GraphState, device: torch.device) -> int:
    """The candidate with the highest Q value (the official ``action_select``, epsilon 0)."""
    with torch.no_grad():
        q = model(
            state.constraint_features.to(device),
            state.edge_index.to(device),
            state.edge_attr.to(device),
            state.variable_features.to(device),
        )
    candidates = state.candidates.to(device)
    return int(candidates[q[candidates].argmax()])


def run_episode(
    env: Environment,
    model: RetroGCN,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[list[dict[str, Any]], Environment]:
    """One episode of the agent in the environment.

    Returns the per-decision experiences (``prev_state``, ``action``,
    ``state``, ``done`` — rewards are assigned retrospectively) and the
    environment, whose reward function holds the finished episode's
    sub-tree decomposition.
    """
    obs, action_set, _, done, _ = env.reset()
    experiences: list[dict[str, Any]] = []
    while not done:
        prev_state = graph_state(obs, action_set)
        if rng.random() < epsilon:
            # Uniform random exploration (the official RandomAgent).
            action = int(action_set[rng.integers(len(action_set))])
        else:
            action = greedy_action(model, prev_state, device)
        obs, action_set, _, done, _ = env.step(action)
        if done:
            # The official keeps the previous observation as the terminal
            # state so no None enters the buffer.
            next_state = prev_state
        else:
            next_state = graph_state(obs, action_set)
        experiences.append(
            {"prev_state": prev_state, "action": action, "state": next_state, "done": done}
        )
    return experiences, env


def add_episode_transitions(
    buffer: ReplayBuffer, experiences: list[dict[str, Any]], env: Environment
) -> int:
    """Retrospectively reward the episode and append its transitions.

    Every sub-tree episode assembled by the harness's ``subtree_transitions``
    enters the buffer (``prob_add_to_buffer=1``).

    A pre-solved instance yields zero decisions, and the reward's
    ``[{0: 0}]`` fallback would then index an empty experience list — the
    official code has the same latent crash and simply never sees the case
    with its generator. Nothing is learned from such an episode.
    """
    if not experiences:
        return 0
    reward_fn = env.reward_function
    assert isinstance(reward_fn, RetroBinaryFathomed)
    added = 0
    for episode in subtree_transitions(experiences, reward_fn.subtree_episodes):
        for experience in episode:
            buffer.append(
                Transition(
                    state=experience["prev_state"],
                    action=experience["action"],
                    reward=experience["reward"],
                    done=experience["done"],
                    next_state=experience["state"],
                )
            )
            added += 1
    return added


class TransitionBatch:
    """A mini-batch of transitions collated into one disconnected graph pair."""

    def __init__(self, transitions: list[Transition]) -> None:
        states = [t.state for t in transitions]
        next_states = [t.next_state for t in transitions]
        self._collate_states(states)
        self.next_constraint_features, self.next_edge_index, self.next_edge_attr, \
            self.next_variable_features, self.next_candidates, self.next_num_candidates = (
                self._collate_states_raw(next_states)
            )
        self.rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        self.dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        # Each action's index within the concatenated candidate list.
        action_idx = []
        offset = 0
        for transition, candidates in zip(transitions, states, strict=True):
            positions = (candidates.candidates == transition.action).nonzero(as_tuple=True)[0]
            assert len(positions) == 1, "action must be one of its state's candidates"
            action_idx.append(offset + int(positions[0]))
            offset += len(candidates.candidates)
        self.action_idx = torch.tensor(action_idx, dtype=torch.long)

    def _collate_states_raw(self, states: list[GraphState]):
        n_cons = [s.constraint_features.shape[0] for s in states]
        n_vars = [s.variable_features.shape[0] for s in states]
        edge_index = []
        cons_offset = 0
        var_offset = 0
        for state, n_c, n_v in zip(states, n_cons, n_vars, strict=True):
            edge_index.append(
                state.edge_index + torch.tensor([[cons_offset], [var_offset]], dtype=torch.long)
            )
            cons_offset += n_c
            var_offset += n_v
        candidates = torch.cat([s.candidates + sum(n_vars[:i]) for i, s in enumerate(states)])
        return (
            torch.cat([s.constraint_features for s in states]),
            torch.cat(edge_index, dim=1),
            torch.cat([s.edge_attr for s in states]),
            torch.cat([s.variable_features for s in states]),
            candidates,
            torch.tensor([len(s.candidates) for s in states]),
        )

    def _collate_states(self, states: list[GraphState]) -> None:
        (
            self.constraint_features,
            self.edge_index,
            self.edge_attr,
            self.variable_features,
            self.candidates,
            self.num_candidates,
        ) = self._collate_states_raw(states)

    def to(self, device: torch.device) -> "TransitionBatch":
        for name in (
            "constraint_features",
            "edge_index",
            "edge_attr",
            "variable_features",
            "candidates",
            "next_constraint_features",
            "next_edge_index",
            "next_edge_attr",
            "next_variable_features",
            "next_candidates",
            "rewards",
            "dones",
            "action_idx",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


def dqn_update(
    model: RetroGCN,
    target: RetroGCN,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    """One gradient step: 1-step Double-DQN target, MSE, soft target update.

    The batch shrinks below 128 only if the buffer is still smaller (the
    official floor of 20k transitions never lets that happen; tests and
    smoke runs with a tiny floor do).
    """
    batch = TransitionBatch(buffer.sample(min(BATCH_SIZE, len(buffer)), rng)).to(device)

    # Q(s, a) from the online network.
    q_values = model(
        batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features
    )
    q_action = q_values[batch.action_idx]

    with torch.no_grad():
        # The online network selects the greedy next action...
        next_q_online = model(
            batch.next_constraint_features,
            batch.next_edge_index,
            batch.next_edge_attr,
            batch.next_variable_features,
        )
        candidate_q = next_q_online[batch.next_candidates]
        # ...one argmax per sample among its own candidates...
        splits = batch.next_num_candidates.tolist()
        next_action = torch.stack([q.argmax() for q in torch.split(candidate_q, splits)])
        # ...the target network evaluates it (same candidate slices, gathered
        # at each sample's chosen action).
        next_q_target = target(
            batch.next_constraint_features,
            batch.next_edge_index,
            batch.next_edge_attr,
            batch.next_variable_features,
        )
        candidate_q_target = next_q_target[batch.next_candidates]
        q_next = torch.stack(
            [qt[a] for qt, a in zip(torch.split(candidate_q_target, splits), next_action)]
        )
        q_target = (1 - batch.dones) * GAMMA * q_next
        td_target = batch.rewards + q_target

    loss = torch.nn.functional.mse_loss(q_action, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), GRADIENT_CLIP_VALUE)
    optimizer.step()

    # Soft Polyak update after every gradient step.
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), model.parameters()):
            target_param.data.copy_(SOFT_UPDATE_TAU * param.data + (1.0 - SOFT_UPDATE_TAU) * target_param.data)

    return float(loss)


def train(
    instances: Iterator[Model],
    checkpoint: str | Path,
    episodes: int,
    buffer_min_length: int = BUFFER_MIN_LENGTH,
    time_limit: float | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    log=print,
) -> RetroGCN:
    """The official DQN loop: fill the buffer, then learn while acting.

    ``instances`` must yield a fresh model per episode (a solved model
    cannot drive a second environment). Episodes run until the instance is
    solved (or the ``time_limit`` budget is spent).
    """
    device = device or get_device()
    env = make_env(instances, time_limit=time_limit, seed=seed)
    model = RetroGCN().to(device)
    target = RetroGCN().to(device)
    target.load_state_dict(model.state_dict())
    for param in target.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    rng = np.random.default_rng(seed)

    def save() -> None:
        """Checkpoint after every episode — a cluster job killed at its wall
        time keeps everything learned so far."""
        save_checkpoint(
            model.cpu(),
            checkpoint,
            extra={
                "episodes": episode_idx,
                "buffer_min_length": buffer_min_length,
                "seed": seed,
                "total_updates": total_updates,
            },
        )
        model.to(device)

    # Phase 1: fill the buffer with the agent's own episodes.
    episode_idx = 0
    total_updates = 0
    while len(buffer) < buffer_min_length and episode_idx < episodes:
        experiences, env = run_episode(env, model, EPSILON, rng, device)
        added = add_episode_transitions(buffer, experiences, env)
        episode_idx += 1
        log(
            f"[train] fill episode {episode_idx:4d}  transitions={len(buffer):6d} "
            f"(+{added})  decisions={len(experiences)}"
        )
        save()

    # Phase 2: every 10 transitions added buys one gradient step.
    loss_sum = 0.0
    while episode_idx < episodes:
        experiences, env = run_episode(env, model, EPSILON, rng, device)
        added = add_episode_transitions(buffer, experiences, env)
        episode_idx += 1
        for _ in range(added // STEPS_PER_UPDATE):
            loss_sum += dqn_update(model, target, buffer, optimizer, rng, device)
            total_updates += 1
        log(
            f"[train] episode {episode_idx:4d}  transitions={len(buffer):6d} (+{added})  "
            f"decisions={len(experiences)}  updates={total_updates}  "
            f"loss={loss_sum / max(total_updates, 1):.4f}"
        )
        save()

    log(f"[train] checkpoint saved to {checkpoint}")
    return model


def save_checkpoint(model: RetroGCN, path: str | Path, extra: dict | None = None) -> None:
    """State dict + provenance metadata, same format as the other methods."""
    checkpoint = {"state_dict": model.state_dict()}
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> RetroGCN:
    device = device or get_device()
    model = RetroGCN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["state_dict"])
    model.to(device)
    model.eval()
    return model
