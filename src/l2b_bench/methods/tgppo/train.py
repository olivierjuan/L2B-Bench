"""TGPPO 2025 (Tree-Gate PPO) — the PPO training loop.

Faithful port of the official repository (github.com/benmhamed-a/tgppo-aaai-26,
``code/train.py`` + ``project/{agent,memory,environment,brancher}``), adapted
to the harness's ``Environment`` (which drives SCIP through the same
branchrule-plugin mechanism the official ``Brancher`` uses):

- Rollout: at every branching decision the actor *samples* an action from its
  softmax over the node's LP branching candidates (the official also samples
  at evaluation, with per-job fixed seeds); the transition — state, action,
  value, log-prob — is completed by the reward of the *next* extraction (the
  official stores the previous decision at every ``branchexeclp`` call and
  the last one in ``branchfree``), so the final transition alone carries
  ``done=True``.
- Update: PPO with GAE over the whole collected batch — clipped surrogate
  (one AdamW step per minibatch, gradient-norm clip 0.5), separate actor and
  critic optimizers (weight decay 1e-2), entropy bonus; advantages normalized
  across the batch; a bootstrap value from the critic on the last state when
  the batch does not end ``done``.
- Reward: the difficulty-adaptive H3 (harness ``DifficultyAdaptive``), fed by
  a per-instance baseline solve; step rewards in [-1, 1], terminal bonus by
  final status.
- Solver: the official 'sandbox' setting — all 54 primal heuristics off, the
  SB-related probing/checking switches off, feastol/dualfeastol 1e-6, and a
  cutoff (objective limit) at the instance's optimum from the baseline solve.
  Variable permutation per episode comes from the harness's own
  ``set_seed_on_model`` (permutevars + permutationseed + randomseedshift +
  lpseed), matching the official ``randomization/permutevars`` protocol.

Deviations from the official code, both pilot-scale: the official collects
its 125 rollout episodes per iteration (25 instances x 5 SCIP seeds) with a
process pool over a fixed instance directory and its baselines from a
precomputed info dict (Mendeley); the pilot solves each baseline inline
(the same sandbox setting, default branching) right before the episode and
consumes fresh instances sequentially, one PPO update every
``EPISODES_PER_UPDATE`` episodes. The official hyperparameters come from a
nested Optuna search; the defaults here are the official test-time values
(clip 0.18, entropy 3e-4, gamma 0.99, lambda 0.95) at pilot scale.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gyozas.environment import Environment
from gyozas.observations import BranchingStateObservation
from gyozas.rewards import DifficultyAdaptive
from pyscipopt import Model

from .model import N_CANDIDATE_FEATURES, N_MIP_FEATURES, N_NODE_FEATURES, Actor, Critic

# One PPO update per 125 episodes — the official 25 instances x 5 seeds.
EPISODES_PER_UPDATE = 125

# The official test-time PPO constants (build_eval_env in code/test.py).
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
POLICY_CLIP = 0.18
ENTROPY_WEIGHT = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 64
N_EPOCHS = 1
GRAD_CLIP = 0.5
WEIGHT_DECAY = 1e-2

# The 54 primal heuristics the 'sandbox' setting disables (settings.py).
_HEURISTICS_OFF = (
    "actconsdiving",
    "bound",
    "clique",
    "coefdiving",
    "completesol",
    "conflictdiving",
    "crossover",
    "dins",
    "distributiondiving",
    "dualval",
    "farkasdiving",
    "feaspump",
    "fixandinfer",
    "fracdiving",
    "gins",
    "guideddiving",
    "zeroobj",
    "indicator",
    "intdiving",
    "intshifting",
    "linesearchdiving",
    "localbranching",
    "locks",
    "lpface",
    "alns",
    "nlpdiving",
    "mutation",
    "multistart",
    "mpec",
    "objpscostdiving",
    "octane",
    "ofins",
    "oneopt",
    "proximity",
    "pscostdiving",
    "randrounding",
    "rens",
    "reoptsols",
    "repair",
    "rins",
    "rootsoldiving",
    "rounding",
    "shiftandpropagate",
    "shifting",
    "simplerounding",
    "subnlp",
    "trivial",
    "trivialnegation",
    "trysol",
    "twoopt",
    "undercover",
    "vbounds",
    "veclendiving",
    "zirounding",
)

#: The official 'sandbox' SCIP setting (init_params in utils/settings.py).
SANDBOX_PARAMS: dict[str, Any] = {
    "display/verblevel": 0,
    "limits/nodes": -1,
    "presolving/maxrounds": -1,
    "separating/maxrounds": -1,
    "separating/maxroundsroot": -1,
    "reoptimization/enable": False,
    "conflict/usesb": False,
    "branching/fullstrong/probingbounds": False,
    "branching/relpscost/probingbounds": False,
    "branching/checksol": False,
    "branching/fullstrong/reevalage": 0,
    "numerics/feastol": 1e-6,
    "numerics/dualfeastol": 1e-6,
    **{f"heuristics/{name}/freq": -1 for name in _HEURISTICS_OFF},
}


@dataclass
class Transition:
    """One branching decision, completed by the next extraction's reward."""

    cands_state: np.ndarray  # (n_candidates, 25)
    node_state: np.ndarray  # (8,)
    mip_state: np.ndarray  # (53,)
    action: int  # the chosen candidate's rank within the candidate set
    reward: float = 0.0
    done: bool = False
    value: float = 0.0
    log_prob: float = 0.0


@dataclass
class BaselineStats:
    """The per-instance baseline solve feeding H3 and the sandbox cutoff."""

    nodes: float
    gap: float
    pdi: float
    status: str
    cutoff: float | None  # the optimum, when the baseline solved it


def baseline_solve(model: Model, time_limit: float | None = None) -> BaselineStats:
    """One sandbox solve without the learned brancher (the official info dict).

    Solves ``model`` under the sandbox parameters with SCIP's default
    branching, then returns it to its original state with ``freeTransform``.
    The optimum (when found) doubles as the episode's cutoff.
    """
    model.hideOutput()
    params = dict(SANDBOX_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    model.setParams(params)
    model.optimize()
    stats = BaselineStats(
        nodes=float(model.getNNodes()),
        gap=float(model.getGap()),
        pdi=float(model.getPrimalDualIntegral()),
        status=model.getStatus(),
        cutoff=float(model.getObjVal()) if model.getStatus() == "optimal" else None,
    )
    model.freeTransform()
    return stats


def make_env(
    instances: Iterator[Model],
    reward_fn: DifficultyAdaptive,
    time_limit: float | None = None,
    seed: int = 0,
) -> Environment:
    """One training environment: BranchingState observation, H3 reward,
    sandbox parameters, per-episode variable permutation."""
    params = dict(SANDBOX_PARAMS)
    if time_limit is not None:
        params["limits/time"] = time_limit
    env = Environment(
        instances,
        observation_function=BranchingStateObservation(),
        reward_function=reward_fn,
        scip_params=params,
    )
    env.dynamics.seed(seed)
    return env


def with_baselines(
    instances: Iterator[Model],
    reward_fn: DifficultyAdaptive,
    time_limit: float | None = None,
) -> Iterator[Model]:
    """Yield models with their baseline injected into the reward and the
    sandbox cutoff (objective limit at the optimum) set on the model."""
    for model in instances:
        stats = baseline_solve(model, time_limit=time_limit)
        reward_fn.configure(
            baseline_nodes=stats.nodes,
            baseline_gap=stats.gap,
            baseline_pdi=stats.pdi,
            solver_status=stats.status,
            time_limit=time_limit or 3600.0,
        )
        if stats.cutoff is not None:
            model.setObjlimit(stats.cutoff)
        yield model


def choose_action(
    actor: Actor,
    critic: Critic,
    cands_state: np.ndarray,
    node_state: np.ndarray,
    mip_state: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[int, float, float]:
    """The official Agent.choose_action on a single node: sample (or argmax)
    from the actor's softmax, plus the critic's value and the log-prob."""
    cands = torch.as_tensor(np.asarray(cands_state), dtype=torch.float32, device=device)
    node = torch.as_tensor(np.asarray(node_state), dtype=torch.float32, device=device)
    mip = torch.as_tensor(np.asarray(mip_state), dtype=torch.float32, device=device)
    actor.eval()
    critic.eval()
    with torch.no_grad():
        probs = actor(cands.unsqueeze(0), node.unsqueeze(0), mip.unsqueeze(0))
        value = critic(cands.unsqueeze(0), node.unsqueeze(0), mip.unsqueeze(0)).squeeze(
            -1
        )[0]
        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = torch.log(
                probs.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
            )
        else:
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
    # reshape(-1)[0] rather than [0]: with a batch of one, torch may return
    # 0-dim tensors for the sampled action's log-prob.
    return (
        int(action.reshape(-1)[0]),
        float(value.reshape(-1)[0]),
        float(log_prob.reshape(-1)[0]),
    )


def run_episode(
    env: Environment, actor: Actor, critic: Critic, device: torch.device
) -> tuple[list[Transition], dict[str, Any]]:
    """One sampled rollout; returns the completed transitions and the final
    solve statistics (the official episode info)."""
    obs, action_set, _reward, done, _info = env.reset()
    transitions: list[Transition] = []
    while not done:
        assert obs is not None, "a live decision must carry a BranchingState"
        action, value, log_prob = choose_action(
            actor,
            critic,
            obs.candidate_features,
            obs.node_features,
            obs.mip_features,
            device,
        )
        pending = Transition(
            cands_state=np.asarray(obs.candidate_features, dtype=np.float32),
            node_state=np.asarray(obs.node_features, dtype=np.float32),
            mip_state=np.asarray(obs.mip_features, dtype=np.float32),
            action=action,
            value=value,
            log_prob=log_prob,
        )
        obs, action_set, reward, done, _info = env.step(
            int(np.asarray(action_set)[action])
        )
        pending.reward = float(reward)
        pending.done = bool(done)
        transitions.append(pending)
    model = env.model
    info = (
        {
            "status": model.getStatus(),
            "nnodes": model.getNNodes(),
            "gap": model.getGap(),
            "pdi": model.getPrimalDualIntegral(),
        }
        if model is not None
        else {}
    )
    return transitions, info


def collate(transitions: list[Transition]) -> dict[str, torch.Tensor]:
    """Pad the batch's candidate sets to one rectangular block (official
    Memory.get_batch_generator): True marks padding in ``mask``."""
    max_cands = max(t.cands_state.shape[0] for t in transitions)
    n = len(transitions)
    cands = torch.zeros(n, max_cands, N_CANDIDATE_FEATURES)
    mask = torch.zeros(n, max_cands, dtype=torch.bool)
    node = torch.zeros(n, N_NODE_FEATURES)
    mip = torch.zeros(n, N_MIP_FEATURES)
    for i, t in enumerate(transitions):
        k = t.cands_state.shape[0]
        cands[i, :k] = torch.as_tensor(t.cands_state)
        mask[i, k:] = True
        node[i] = torch.as_tensor(t.node_state)
        mip[i] = torch.as_tensor(t.mip_state)
    return {
        "cands": cands,
        "mask": mask,
        "node": node,
        "mip": mip,
        "actions": torch.tensor([t.action for t in transitions], dtype=torch.long),
        "rewards": torch.tensor([t.reward for t in transitions], dtype=torch.float32),
        "dones": torch.tensor([t.done for t in transitions], dtype=torch.float32),
        "values": torch.tensor([t.value for t in transitions], dtype=torch.float32),
        "log_probs": torch.tensor(
            [t.log_prob for t in transitions], dtype=torch.float32
        ),
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation with a scalar bootstrap value (the
    official Agent._compute_gae, verbatim)."""
    t_ = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    next_value = bootstrap_value
    gae = torch.zeros((), device=rewards.device)
    for t in reversed(range(t_)):
        nonterminal = 1.0 - dones[t]
        nv = next_value if t == t_ - 1 else values[t + 1]
        if t < t_ - 1:
            nv = nv * (1.0 - dones[t + 1])
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
    return advantages, returns


def ppo_update(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
    policy_clip: float = POLICY_CLIP,
    entropy_weight: float = ENTROPY_WEIGHT,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = N_EPOCHS,
) -> dict[str, float]:
    """One PPO update over the collected batch (the official Agent.learn):
    GAE, advantage normalization, then n_epochs of shuffled padded minibatches
    with the clipped surrogate and a separate critic MSE."""
    if len(transitions) < batch_size:
        return {}

    data = collate(transitions)
    rewards = data["rewards"].to(device)
    dones = data["dones"].to(device)
    old_values = data["values"].to(device)
    node_states = data["node"].to(device)
    mip_states = data["mip"].to(device)

    # Bootstrap value for the state after the last step (padded to the batch's
    # candidate width, as the official does).
    last = transitions[-1]
    last_cands = torch.as_tensor(last.cands_state, dtype=torch.float32, device=device)
    width = data["cands"].shape[1]
    pad = width - last_cands.shape[0]
    if pad > 0:
        last_cands = torch.cat(
            [last_cands, torch.zeros(pad, N_CANDIDATE_FEATURES, device=device)]
        )
        last_mask = torch.cat(
            [
                torch.zeros(last.cands_state.shape[0], dtype=torch.bool, device=device),
                torch.ones(pad, dtype=torch.bool, device=device),
            ]
        )
    else:
        last_mask = torch.zeros(last_cands.shape[0], dtype=torch.bool, device=device)
    if dones[-1] > 0.5:
        bootstrap_value = torch.tensor(0.0, device=device)
    else:
        with torch.no_grad():
            bootstrap_value = critic(
                last_cands.unsqueeze(0),
                node_states[-1:],
                mip_states[-1:],
                last_mask.unsqueeze(0),
            ).squeeze(-1)[0]

    advantages, returns = compute_gae(rewards, old_values, dones, bootstrap_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor.train()
    critic.train()
    stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    n_updates = 0
    indices = np.arange(len(transitions))
    for _epoch in range(n_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = collate([transitions[i] for i in batch_idx])
            cands = batch["cands"].to(device)
            mask = batch["mask"].to(device)
            actions = batch["actions"].to(device)
            mb_old_log_probs = batch["log_probs"].to(device)
            mb_advantages = advantages[batch_idx]
            mb_returns = returns[batch_idx]

            probs = actor(
                cands, batch["node"].to(device), batch["mip"].to(device), mask
            )
            dist = torch.distributions.Categorical(probs=probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - policy_clip, 1 + policy_clip) * mb_advantages
            actor_loss = (
                -torch.min(surr1, surr2).mean() - entropy_weight * entropy.mean()
            )

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
            actor_optimizer.step()

            values = critic(
                cands, batch["node"].to(device), batch["mip"].to(device), mask
            ).view(-1)
            critic_loss = torch.nn.functional.mse_loss(values, mb_returns.view(-1))

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
            critic_optimizer.step()

            stats["actor_loss"] += float(actor_loss.detach())
            stats["critic_loss"] += float(critic_loss.detach())
            stats["entropy"] += float(entropy.mean().detach())
            n_updates += 1

    if n_updates == 0:
        return {}
    stats = {k: v / n_updates for k, v in stats.items()}
    stats["episode_return"] = float(rewards.sum())
    stats["n_updates"] = n_updates
    return stats


def save_checkpoint(
    actor: Actor,
    critic: Critic,
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> None:
    """The official checkpoint layout: actor and critic state dicts, plus the
    architecture hyperparameters needed to rebuild them."""
    actor.cpu()
    critic.cpu()
    payload: dict[str, Any] = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "hparams": actor.hparams,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[Actor, Critic]:
    """Rebuild the actor-critic pair from a checkpoint (the official
    build_models_from_ckpt: same architecture, weights loaded)."""
    device = device or get_device()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = Actor(**payload["hparams"])
    critic = Critic(**payload["hparams"])
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    actor.to(device).eval()
    critic.to(device).eval()
    return actor, critic


def get_device() -> "torch.device":
    """CUDA when available (the campaign nodes' A5000), CPU otherwise."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    instances: Iterator[Model],
    checkpoint: str | Path,
    episodes: int,
    time_limit: float | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    log=print,
) -> tuple[Actor, Critic]:
    """The official RL loop: collect ``EPISODES_PER_UPDATE`` rollout episodes,
    then one PPO update; repeat until ``episodes`` are consumed.

    ``instances`` must yield a fresh, unsolved model per episode (each is
    baseline-solved inline under the sandbox setting, then handed to the
    environment with the cutoff set). The checkpoint is overwritten after
    every update — a cluster job killed at its wall time keeps what it
    learned — and a numbered snapshot ``<checkpoint>.uNNN`` is kept per
    update so a convergence campaign can evaluate every budget.
    """
    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    actor = Actor().to(device)
    critic = Critic().to(device)
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=ACTOR_LR, weight_decay=WEIGHT_DECAY
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=CRITIC_LR, weight_decay=WEIGHT_DECAY
    )

    reward_fn = DifficultyAdaptive()
    env = make_env(
        with_baselines(instances, reward_fn, time_limit=time_limit),
        reward_fn,
        time_limit=time_limit,
        seed=seed,
    )

    n_updates = 0
    episode_idx = 0
    batch: list[Transition] = []
    while episode_idx < episodes:
        try:
            transitions, info = run_episode(env, actor, critic, device)
        except StopIteration:
            break
        episode_idx += 1
        batch.extend(transitions)
        if episode_idx % EPISODES_PER_UPDATE == 0 or episode_idx == episodes:
            n_updates += 1
            stats = ppo_update(
                actor, critic, actor_optimizer, critic_optimizer, batch, device
            )
            if stats:
                log(
                    f"[train] update {n_updates:3d}  episodes={episode_idx}  "
                    f"transitions={len(batch)}  actor={stats['actor_loss']:.4f}  "
                    f"critic={stats['critic_loss']:.4f}  entropy={stats['entropy']:.4f}  "
                    f"last_status={info.get('status')}"
                )
            batch = []
            actor.cpu()
            critic.cpu()
            extra = {"episodes": episode_idx, "updates": n_updates, "seed": seed}
            save_checkpoint(actor, critic, checkpoint, extra=extra)
            # A numbered copy per update: convergence campaigns evaluate each
            # snapshot on the pilot grid to trace SGM vs training budget.
            save_checkpoint(
                actor, critic, Path(f"{checkpoint}.u{n_updates:03d}"), extra=extra
            )
            actor.to(device)
            critic.to(device)
    log(f"[train] checkpoint saved to {checkpoint}")
    return actor, critic
