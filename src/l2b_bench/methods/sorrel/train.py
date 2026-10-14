"""SORREL 2024 — the two training stages (Feng et al., AAAI 2025).

Faithful port of ``brancher/models.py`` (``TD3BC`` and ``PPO``) and the two
official training scripts, on the harness's tree-MDP samples:

- **Offline (TD3BC)** — actor and critic are both Gasse's ``GNNPolicy``. The
  PreNorm warmup freezes only the *actor's* statistics (the official
  ``prenorm`` calls ``actor.prenorm_init``; the critic's PreNorm layers stay
  identity — kept verbatim); ``r_max`` is the max child reward over the warmup
  subset (every 10th sample, the official ``prenorm_loader``). Each batch
  first updates the critic — per-candidate Q with a Huber loss against the
  kappa-weighted tree backup of the two children's Q-targets (the target
  critic scores the child, the actor's softmax weighs it; an unvisited child
  contributes its reward with no bootstrap) — then every
  ``target_update_period`` steps the actor (``-lmbda*(probs*Q).sum + CE``
  with ``lmbda = alpha/|Q(a)|``, Q read from the *target* critic) followed by
  the soft update of the target critic (tau 1e-3).
- **Online (PPO + SIL)** — initialized from the TD3BC checkpoint. Rollouts
  sample actions (log-probs recorded); the critic's value is the max over the
  candidate logits; advantages are ``returns/r_max - value`` normalized over
  the whole rollout; one update per rollout accumulates gradients over
  minibatches (clipped surrogate, epsilon 0.1; critic ``0.5*err^2.sum/N``)
  and takes a single step. SIL draws at most 100 samples per instance from a
  per-instance buffer of best trajectories (by minus the node count): actor
  CE x 10 x ``clip(returns/r_max - value, 0)`` (one step per batch), critic
  ``0.5*weight^2/10`` — whose gradients *accumulate across batches*, the
  official ``sil`` never zeroes the critic's gradients; kept verbatim.

Official hyperparameters throughout (scripts/train_TD3BC.sh and
train_PPO.sh, set covering): lr 1e-4, batch 32, gamma 0.9, kappa 0.8,
alpha 0.01, epsilon 0.1, target update period 2, epoch size 624, 5 rollouts
of 20 instances per epoch.

Pilot-scale deviations, like every other RL port here: instances are consumed
sequentially instead of sampled with replacement from a fixed pool, the
validation rollouts and LR schedule are omitted (the checkpoint is
overwritten after every epoch instead — a job killed at its wall time keeps
everything learned), and the pre-epoch buffer-filling rollouts are
parameterized.
"""

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pyscipopt import Model

from ..gasse.model import GNNPolicy
from ..gasse.train import pretrain as _prenorm_warmup
from .collect import SorrelSample, TreeTracker, make_env, tree_returns
from .model import SorrelBatch, padded_logits

# The official hyperparameters (set covering).
LR = 1e-4
BATCH_SIZE = 32
EPOCH_SIZE = 624  # minibatches drawn with replacement per offline epoch
GAMMA = 0.9
KAPPA = 0.8
ALPHA = 0.01  # TD3BC's alpha (0.1 cauctions/facilities, 2.5 knapsack)
TAU = 1e-3
TARGET_UPDATE_PERIOD = 2
EPSILON = 0.1  # PPO clip range
GRAD_CLIP = 1.0

# The official online loop's shape (04_train_online.py).
EPISODES_PER_ROLLOUT = 20  # epoch_size
ROLLOUTS_PER_EPOCH = 5
PRE_ROLLOUTS = 10  # sampled buffer-filling rollouts before the first epoch
SIL_SAMPLES = 100  # drawn per best trajectory


def get_device() -> "torch.device":
    """CUDA when available (the campaign nodes' A5000), CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches_of(
    samples: list[SorrelSample], batch_size: int, rng: np.random.Generator, device: torch.device
) -> list[SorrelBatch]:
    """Shuffled minibatches of the samples, collated and on the device."""
    order = rng.permutation(len(samples))
    return [
        SorrelBatch([samples[i] for i in order[k : k + batch_size]]).to(device)
        for k in range(0, len(order), batch_size)
    ]


class TD3BC:
    """The offline stage: per-candidate-Q critic + BC-regularized actor."""

    def __init__(
        self,
        device: torch.device | None = None,
        lr: float = LR,
        gamma: float = GAMMA,
        kappa: float = KAPPA,
        alpha: float = ALPHA,
        target_update_period: int = TARGET_UPDATE_PERIOD,
    ) -> None:
        self.device = device or get_device()
        self.gamma = gamma
        self.kappa = kappa
        self.alpha = alpha
        self.target_update_period = target_update_period
        self.tau = TAU
        self.actor = GNNPolicy().to(self.device)
        self.critic = GNNPolicy().to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_criterion = torch.nn.HuberLoss()
        self.train_steps = 0
        self.r_max = 1.0

    def prenorm(self, samples: list[SorrelSample], rng: np.random.Generator) -> int:
        """Actor-only PreNorm warmup over the subset; ``r_max`` is its max
        child reward (the official collects both over the prenorm loader)."""
        batches = batches_of(samples, 2 * BATCH_SIZE, rng, self.device)
        n_layers = _prenorm_warmup(self.actor, [batch.states for batch in batches])
        rewards = torch.tensor(
            [r for s in samples for r in (s.reward1, s.reward2)], dtype=torch.float32
        )
        self.r_max = float(rewards.max())
        return n_layers

    def q_targets(
        self, group, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """The official ``Q_targets`` on one child group: the target critic's
        per-candidate Q weighed by the actor's softmax, reward/r_max plus the
        discounted bootstrap on visited children."""
        logits = padded_logits(self.actor(*group.forward_args()), group.candidates)
        action_probs = torch.softmax(logits, dim=-1)
        q_next = padded_logits(self.critic_target(*group.forward_args()), group.candidates).detach()
        q_next = q_next.masked_fill(q_next == -float("inf"), -1e6)
        q_next = (action_probs * q_next).sum(1, keepdim=True).detach()
        return rewards.unsqueeze(-1) / self.r_max + self.gamma * q_next * (1 - dones.unsqueeze(-1))

    def tree_q_targets(self, batch: SorrelBatch) -> torch.Tensor:
        """The kappa-weighted mix of the two children's Q-targets (the
        smaller-reward child carries weight kappa)."""
        q1 = self.q_targets(batch.children1, batch.rewards1, batch.dones1)
        q2 = self.q_targets(batch.children2, batch.rewards2, batch.dones2)
        coeff = torch.where(batch.rewards1 < batch.rewards2, self.kappa, 1 - self.kappa).unsqueeze(-1)
        return coeff * q1 + (1 - coeff) * q2

    def update_critic(self, batch: SorrelBatch) -> float:
        self.critic.train()
        with torch.no_grad():
            q_targets = self.tree_q_targets(batch)
        q = padded_logits(self.critic(*batch.states.forward_args()), batch.states.candidates)
        q_expected = q.gather(1, batch.actions.unsqueeze(1))
        loss = self.critic_criterion(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.critic_optimizer.step()
        return float(loss.detach())

    def update_actor(self, batch: SorrelBatch) -> float:
        self.actor.train()
        logits = padded_logits(self.actor(*batch.states.forward_args()), batch.states.candidates)
        with torch.no_grad():
            # The official reads Q from the *target* critic here.
            q = padded_logits(
                self.critic_target(*batch.states.forward_args()), batch.states.candidates
            )
            q = q.masked_fill(q == -float("inf"), -1e6)
            lmbda = self.alpha / q.gather(1, batch.actions.unsqueeze(-1)).abs().mean().detach()
        action_probs = torch.softmax(logits, dim=-1)
        loss = -lmbda * (action_probs * q).sum(1).mean() + F.cross_entropy(logits, batch.actions)
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.actor_optimizer.step()
        return float(loss.detach())

    def learn(self, batches: list[SorrelBatch]) -> dict[str, float]:
        critic_loss = 0.0
        actor_loss = 0.0
        n_actor = 0
        for batch in batches:
            critic_loss += self.update_critic(batch)
            self.train_steps += 1
            if self.train_steps % self.target_update_period == 0:
                actor_loss += self.update_actor(batch)
                n_actor += 1
                self.soft_update()
        return {
            "critic_loss": critic_loss / max(1, len(batches)),
            "actor_loss": actor_loss / max(1, n_actor),
        }

    def soft_update(self) -> None:
        for target_param, local_param in zip(
            self.critic_target.parameters(), self.critic.parameters(), strict=True
        ):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)


def train_td3bc(
    samples: list[SorrelSample],
    checkpoint: str | Path,
    epochs: int = 1,
    epoch_size: int = EPOCH_SIZE,
    batch_size: int = BATCH_SIZE,
    alpha: float = ALPHA,
    seed: int = 0,
    device: torch.device | None = None,
    log=print,
) -> TD3BC:
    """The official offline loop (03_train_offline.py): PreNorm warmup over
    every 10th sample, then epochs of ``epoch_size`` minibatches drawn with
    replacement from the demonstrations."""
    model = TD3BC(alpha=alpha, device=device)
    rng = np.random.default_rng(seed)
    n_layers = model.prenorm(samples[::10], rng)
    log(f"[td3bc] froze {n_layers} PreNorm layers (actor only), r_max={model.r_max:.4g}")
    for epoch in range(1, epochs + 1):
        # The official draws epoch_size*batch_size samples with replacement.
        order = rng.integers(0, len(samples), size=epoch_size * batch_size)
        batches = [
            SorrelBatch([samples[i] for i in order[k : k + batch_size]]).to(model.device)
            for k in range(0, len(order), batch_size)
        ]
        stats = model.learn(batches)
        log(
            f"[td3bc] epoch {epoch:4d}  critic={stats['critic_loss']:.4f}  "
            f"actor={stats['actor_loss']:.4f}"
        )
        save_td3bc(model, checkpoint, extra={"epochs": epoch, "n_samples": len(samples), "seed": seed})
    return model


def save_td3bc(model: TD3BC, path: str | Path, extra: dict | None = None) -> None:
    """The TD3BC checkpoint: actor, critic, and r_max (official state_dict)."""
    torch.save(
        {
            "actor": model.actor.state_dict(),
            "critic": model.critic.state_dict(),
            "r_max": model.r_max,
            **(extra or {}),
        },
        path,
    )


def _load_state(path: str | Path) -> dict[str, Any]:
    return torch.load(path, weights_only=True, map_location="cpu")


def load_actor(path: str | Path, device: torch.device | None = None) -> GNNPolicy:
    """The trained actor (either stage's checkpoint), eval mode."""
    device = device or get_device()
    state = _load_state(path)
    actor = GNNPolicy()
    actor.load_state_dict(state["actor"])
    actor.to(device)
    actor.eval()
    return actor


def policy_logits(
    policy: GNNPolicy, observation: Any, candidates: np.ndarray, device: torch.device
) -> torch.Tensor:
    """The policy's scores over one node's candidates (fresh graph forward)."""
    obs = observation
    edge_index = torch.as_tensor(obs.edge_features.indices, dtype=torch.long)
    logits = policy(
        torch.as_tensor(obs.row_features, dtype=torch.float32).to(device),
        edge_index.to(device),
        torch.as_tensor(obs.edge_features.values, dtype=torch.float32).reshape(-1, 1).to(device),
        torch.as_tensor(obs.variable_features, dtype=torch.float32).to(device),
    )
    cand = torch.as_tensor(np.asarray(candidates), dtype=torch.long).to(device)
    return logits[cand]


def run_episode(
    env,
    actor: GNNPolicy,
    device: torch.device,
    gamma: float = GAMMA,
    kappa: float = KAPPA,
    greedy: bool = False,
    length_limit: float | None = None,
) -> tuple[list[SorrelSample], int]:
    """One rollout (official Agent.run): every decision kept (sample_rate 1),
    actions sampled (or argmax, warmup) with log-probs recorded; the tree
    return of each decision is computed recursively over the tracker's tree
    once the episode ends. Returns the samples and the episode's node count."""
    obs, action_set, _reward, done, _info = env.reset()
    tracker = None if done else TreeTracker(env.model)
    records: list[tuple[Any, np.ndarray, int, int, float]] = []
    n_iter = 0
    while not done:
        candidates = np.asarray(action_set)
        cand_logits = policy_logits(actor, obs, candidates, device)
        if greedy:
            action_idx = int(cand_logits.argmax())
            log_prob = 0.0
        else:
            dist = torch.distributions.Categorical(torch.softmax(cand_logits, dim=-1))
            action = dist.sample()
            action_idx = int(action)
            log_prob = float(dist.log_prob(action))
        records.append((obs, candidates.copy(), int(candidates[action_idx]), action_idx, log_prob))
        obs, action_set, _reward, done, _info = env.step(int(candidates[action_idx]))
        n_iter += 1
        if length_limit is not None and n_iter >= length_limit:
            done = True
        elif not done:
            tracker.update(env.model)
    if tracker is None:
        return [], int(env.model.getNNodes()) if env.model is not None else 0
    children = tracker.children_of()
    returns = tree_returns(children, gamma, kappa)
    samples: list[SorrelSample] = []
    for i, (obs_i, candidates, action, action_idx, log_prob) in enumerate(records):
        positions, rewards = children[i]
        child_obs = []
        for child, reward in zip(positions, rewards, strict=True):
            if child is None:
                child_obs.append((None, None, reward))
            else:
                child_obs.append((records[child][0], records[child][1], reward))
        child_obs += [(None, None, 0.0)] * (2 - len(child_obs))
        samples.append(
            SorrelSample(
                observation=obs_i,
                action_set=candidates,
                action=action,
                candidate_scores=np.zeros(len(candidates)),  # the official zeros
                is_fsb=False,
                child1_obs=child_obs[0][0],
                child1_action_set=child_obs[0][1],
                reward1=float(child_obs[0][2]),
                child2_obs=child_obs[1][0],
                child2_action_set=child_obs[1][1],
                reward2=float(child_obs[1][2]),
                returns=returns.get(i, 0.0),
                log_prob=log_prob,
            )
        )
    return samples, int(env.model.getNNodes())


class PPOSIL:
    """The online stage: PPO over sampled rollouts plus self-imitation over
    the per-instance buffer of best trajectories."""

    def __init__(
        self,
        actor: GNNPolicy,
        critic: GNNPolicy,
        r_max: float,
        device: torch.device | None = None,
        lr: float = LR,
        epsilon: float = EPSILON,
    ) -> None:
        self.device = device or get_device()
        self.epsilon = epsilon
        self.r_max = r_max
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device | None = None) -> "PPOSIL":
        """Initialized from the TD3BC checkpoint (official pretrained_checkpoint)."""
        device = device or get_device()
        state = _load_state(path)
        actor = GNNPolicy()
        actor.load_state_dict(state["actor"])
        critic = GNNPolicy()
        critic.load_state_dict(state["critic"])
        return cls(actor, critic, float(state.get("r_max", 1.0)), device=device)

    def critic_value(self, group) -> torch.Tensor:
        """The PPO critic's value: the max over the candidate logits."""
        logits = padded_logits(self.critic(*group.forward_args()), group.candidates)
        return logits.max(1).values

    def compute_advantages(self, batches: list[SorrelBatch]) -> list[torch.Tensor]:
        """returns/r_max - value, normalized over the whole rollout."""
        all_advantages = []
        for batch in batches:
            value = self.critic_value(batch.states).detach()
            all_advantages.append(batch.returns / self.r_max - value)
        stacked = torch.cat(all_advantages)
        mean, std = stacked.mean(), stacked.std() + 1e-6
        return [(adv - mean) / std for adv in all_advantages]

    def ppo_update(self, batches: list[SorrelBatch]) -> dict[str, float]:
        self.actor.train()
        self.critic.train()
        advantages = self.compute_advantages(batches)
        n_samples = sum(len(batch.actions) for batch in batches)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss = 0.0
        actor_loss = 0.0
        for batch, adv in zip(batches, advantages, strict=True):
            value = self.critic_value(batch.states)
            error = batch.returns / self.r_max - value
            loss = 0.5 * error.pow(2).sum() / n_samples
            loss.backward()
            critic_loss += float(loss.detach())

            logits = padded_logits(self.actor(*batch.states.forward_args()), batch.states.candidates)
            dist = torch.distributions.Categorical(torch.softmax(logits, dim=-1))
            log_probs = dist.log_prob(batch.actions)
            ratios = torch.exp(log_probs - batch.log_probs)
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
            loss = -torch.min(surr1, surr2).sum() / n_samples
            loss.backward()
            actor_loss += float(loss.detach())
        # One accumulated step per rollout (official PPO.learn).
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return {"critic_loss": critic_loss / len(batches), "actor_loss": actor_loss / len(batches)}

    def sil_update(self, batches: list[SorrelBatch]) -> list[np.ndarray]:
        """Self-imitation: actor CE x 10 x clip(returns/r_max - value, 0) with
        one Adam step per batch; critic 0.5*weight^2/10, whose gradients
        accumulate across batches (the official never zeroes them here)."""
        priorities = []
        self.actor.train()
        for batch in batches:
            logits = padded_logits(self.actor(*batch.states.forward_args()), batch.states.candidates)
            # Not detached: the critic's gradient flows through the weight.
            value = self.critic_value(batch.states)
            weight = torch.clip(batch.returns / self.r_max - value, min=0)
            priorities.append((weight * self.r_max).detach().cpu().numpy())
            actor_loss = (
                F.cross_entropy(logits, batch.actions, reduction="none") * weight.detach() * 10
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            critic_loss = 0.5 * weight.pow(2).mean() / 10
            critic_loss.backward()
            self.critic_optimizer.step()
        return priorities

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "r_max": self.r_max,
                **(extra or {}),
            },
            path,
        )


class SILBuffer:
    """Per-instance buffer of best trajectories (official update_buffer /
    buffer2dataset): capacity 1, keyed by instance, a trajectory kept when its
    score (minus the node count) beats the incumbent."""

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = capacity
        self._buffer: dict[Any, list[tuple[float, list[SorrelSample]]]] = {}

    def add(self, key, trajectory: list[SorrelSample], score: float) -> None:
        sub_buffer = self._buffer.setdefault(key, [])
        if len(sub_buffer) < self.capacity:
            sub_buffer.append((score, trajectory))
            sub_buffer.sort(key=lambda x: x[0])
        elif sub_buffer[0][0] < score:
            sub_buffer.pop(0)
            sub_buffer.append((score, trajectory))
            sub_buffer.sort(key=lambda x: x[0])

    def dataset(self, rng: np.random.Generator) -> list[SorrelSample]:
        """At most SIL_SAMPLES samples drawn without replacement from each
        kept trajectory."""
        out: list[SorrelSample] = []
        for sub_buffer in self._buffer.values():
            for _score, trajectory in sub_buffer:
                if not trajectory:
                    continue
                k = min(SIL_SAMPLES, len(trajectory))
                out.extend(trajectory[i] for i in rng.choice(len(trajectory), size=k, replace=False))
        return out


def train_online(
    instances: Iterator[Model],
    td3bc_checkpoint: str | Path,
    checkpoint: str | Path,
    episodes: int,
    time_limit: float | None = None,
    seed: int = 0,
    pre_rollouts: int = PRE_ROLLOUTS,
    rollouts_per_epoch: int = ROLLOUTS_PER_EPOCH,
    warmup: bool = True,
    device: torch.device | None = None,
    log=print,
) -> PPOSIL:
    """The official online loop (04_train_online.py): one greedy warmup
    rollout filling the SIL buffer, ``pre_rollouts`` sampled buffer-filling
    rollouts, then epochs of ``rollouts_per_epoch`` (rollout + one PPO
    update) followed by a SIL pass — the checkpoint overwritten after every
    epoch."""
    device = device or get_device()
    env = make_env(instances, time_limit=time_limit, seed=seed)
    agent = PPOSIL.from_checkpoint(td3bc_checkpoint, device=device)
    buffer = SILBuffer()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    state = {"n_episodes": 0, "instance": 0}

    def rollout(greedy: bool) -> list[SorrelSample]:
        samples_all: list[SorrelSample] = []
        for _ in range(EPISODES_PER_ROLLOUT):
            if state["n_episodes"] >= episodes:
                break
            try:
                samples, nb_nodes = run_episode(env, agent.actor, device, greedy=greedy)
            except StopIteration:
                break
            buffer.add(state["instance"], samples, -nb_nodes)
            state["n_episodes"] += 1
            state["instance"] += 1
            samples_all.extend(samples)
        return samples_all

    log(f"[ppo] online stage from {td3bc_checkpoint} (r_max={agent.r_max:.4g})")
    if warmup:
        rollout(greedy=True)
    for _ in range(pre_rollouts):
        if state["n_episodes"] >= episodes:
            break
        rollout(greedy=False)

    epoch = 0
    # ``or epoch == 0``: a budget fully consumed by the warmup rollout must
    # still produce one (SIL-only) epoch and a checkpoint.
    while state["n_episodes"] < episodes or epoch == 0:
        epoch += 1
        ppo_stats = {"critic_loss": 0.0, "actor_loss": 0.0}
        n_rollouts = 0
        for _ in range(rollouts_per_epoch):
            if state["n_episodes"] >= episodes:
                break
            samples = rollout(greedy=False)
            if not samples:
                break
            batches = batches_of(samples, BATCH_SIZE, rng, device)
            stats = agent.ppo_update(batches)
            ppo_stats["critic_loss"] += stats["critic_loss"]
            ppo_stats["actor_loss"] += stats["actor_loss"]
            n_rollouts += 1
        sil_batches = batches_of(buffer.dataset(rng), BATCH_SIZE, rng, device)
        if sil_batches:
            agent.sil_update(sil_batches)
        log(
            f"[ppo] epoch {epoch:4d}  episodes={state['n_episodes']}  "
            f"critic={ppo_stats['critic_loss'] / max(1, n_rollouts):.4f}  "
            f"actor={ppo_stats['actor_loss'] / max(1, n_rollouts):.4f}"
        )
        # Checkpoint after every epoch — a job killed at its wall time keeps
        # everything learned so far.
        agent.save(
            checkpoint,
            extra={
                "stage": "PPO+SIL",
                "epoch": epoch,
                "episodes": state["n_episodes"],
                "td3bc_checkpoint": str(td3bc_checkpoint),
                "seed": seed,
            },
        )
        if n_rollouts == 0:
            break
    log(f"[ppo] checkpoint saved to {checkpoint}")
    return agent
