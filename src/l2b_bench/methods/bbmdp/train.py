"""BBMDP 2025 — the Tree DQN agent and its training loop.

Faithful port of ``rl/agents/tree_dqn.py`` (``TreeDQNAgent``) and
``rl/learners/tree_dqn.py`` (``DQNTrainer``) from the official repository,
adapted to the harness's episode collection (:mod:`.collect`):

- **DDQN over multiple children** — for each transition, every visited child
  state is scored by the online network (argmax over its own candidates,
  padded blocks so the argmax stays inside the sample), evaluated by the
  target network, and the per-child targets are *summed* over the real
  children (the official masked sum, ``children_mask`` zeroing the fill
  states); a transition with no visited child targets its reward alone.
- **HL-Gauss element-wise loss** — the target ``reward + gamma * sum`` is
  negative by construction; it is mapped through ``log2(-z + 1e-8)`` and
  asserted inside the bins' support, then compared to the current logits by
  cross-entropy against the erf-smoothed bin distribution. The 1-step and
  n-step losses are added element-wise before the PER weighting.
- **Training loop** — one episode per iteration (pre-solved draws skipped,
  like the official redraw); once the replay buffer holds more than
  ``MEMORY_MIN_SIZE`` transitions, ``min(len(episode) // 10, 500)`` gradient
  steps run, each decaying epsilon (1.0 -> 0.025) and the softmax
  temperature (1.0 -> 1e-3), annealing beta (0.4 -> 1.0 over the first half
  of the gradient budget), and softly updating the target network
  (tau 1e-4). The budget counts gradient steps (200k officially).

Pilot-scale deviations (documented, like the other RL pilots): the official
validation pass — 20 instances greedy every 100 episodes, best checkpoint
kept — is dropped; this loop keeps the final policy and writes a resumable
checkpoint every ``SAVE_EVERY_EPISODES`` episodes, plus a numbered snapshot
``<checkpoint>.gNNNNNN`` every 100 episodes (keyed on the official budget
unit, gradient steps) so a convergence campaign can evaluate every budget.
"""

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn

from .collect import GAMMA, N_STEP, make_env, run_episode
from .model import NUM_BINS, DQNPolicy, StateBatch, TreeHLGaussLoss, pad_per_sample
from .replay import PrioritizedReplayBuffer

#: The official hyperparameters (experiments/params/learner.py,
#: real_test_params — the dqn_bbmdp configuration).
MEMORY_SIZE = 100_000
MEMORY_MIN_SIZE = 20_000
BATCH_SIZE = 128
LR = 5e-5
ALPHA = 0.6
BETA_0 = 0.4
PRIOR_EPS = 1e-3
STEPS_PER_UPDATE = 10  # gradient steps per episode transition, capped:
MAX_GRADIENT_STEPS_PER_EPISODE = 500
MAX_EPSILON = 1.0
MIN_EPSILON = 0.025
EPSILON_DECAY = 1e-4
TEMPERATURE_DECAY = 1e-5
TAU = 1e-4
GRADIENT_CLIP_NORM = 10.0

#: Pilot knob: checkpoint cadence (episodes).
SNAPSHOT_EVERY_EPISODES = 100
SAVE_EVERY_EPISODES = 10


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TreeDQNAgent:
    """The official ``TreeDQNAgent``: DDQN selection, HL-Gauss value head,
    soft target updates."""

    def __init__(self, device: torch.device | None = None, lr: float = LR, tau: float = TAU) -> None:
        self.device = device or get_device()
        self.dqn = DQNPolicy(classification=True).to(self.device)
        self.dqn_target = copy.deepcopy(self.dqn)
        for p in self.dqn_target.parameters():
            p.requires_grad_(False)
        self.loss_fn = TreeHLGaussLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)
        self.tau = tau

    # -- scoring ---------------------------------------------------------

    def to_values_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """The official ``_to_value_from_logits``: ``Q = -2^E[centers]``."""
        return -torch.pow(2, self.loss_fn.transform_from_probs(logits.softmax(dim=-1)))

    def q_values(self, observation: Any) -> torch.Tensor:
        """The official ``_compute_Q_value``: per-variable decoded Q values."""
        device = self.device
        edge_index = torch.as_tensor(np.asarray(observation.edge_features.indices), dtype=torch.long)
        with torch.no_grad():
            logits = self.dqn(
                torch.as_tensor(observation.row_features, dtype=torch.float32).to(device),
                edge_index.to(device),
                torch.as_tensor(observation.edge_features.values, dtype=torch.float32)
                .reshape(-1, 1)
                .to(device),
                torch.as_tensor(observation.variable_features, dtype=torch.float32).to(device),
            )
            return self.to_values_from_logits(logits).squeeze(-1)

    def select_action(
        self,
        observation: Any,
        action_set: np.ndarray,
        epsilon: float,
        temperature: float,
        rng: random.Random,
    ) -> tuple[int, int]:
        """The official ``select_action``: epsilon-greedy random choice, else
        a sample from softmax(Q / T) over the candidates (training) — the
        greedy argmax is the policy bridge's job."""
        if rng.random() < epsilon:
            action_idx = rng.randrange(len(action_set))
            return int(action_set[action_idx]), action_idx
        with torch.no_grad():
            q = self.q_values(observation).cpu()
            candidates = torch.as_tensor(np.asarray(action_set), dtype=torch.long)
            scores = q[candidates]
            probs = torch.softmax(scores / temperature, dim=0)
            action_idx = int(torch.multinomial(probs, 1).item())
        return int(action_set[action_idx]), action_idx

    # -- learning --------------------------------------------------------

    def compute_tree_dqn_loss(self, batch: dict, gamma: float, n_step: bool) -> torch.Tensor:
        """The official ``_compute_tree_dqn_loss``: the element-wise
        HL-Gauss loss over the DDQN multi-children target."""
        transitions = batch["transitions"]
        next_states = batch["n_step_next_states"] if n_step else batch["next_states"]
        children_mask = batch["n_step_children_mask"] if n_step else batch["children_mask"]
        batch_size = len(transitions)

        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            [[t.n_step_reward if n_step else t.reward] for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [[float(t.n_step_done if n_step else t.done)] for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        # The current Q logits at the taken action: [batch, 1, bins].
        actions = actions.view(-1, 1, 1).expand(batch_size, 1, NUM_BINS)
        states = StateBatch([t.state for t in transitions]).to(self.device)
        current_q_value = self.dqn(*states.forward_args())
        current_q_value = pad_per_sample(current_q_value, states.nb_variables).gather(1, actions)

        if children_mask is not None:
            # DDQN: the online net selects each child's action, the target
            # net evaluates it; the per-child targets are summed over the
            # real children (the mask zeroes the fill states).
            mask = torch.as_tensor(children_mask, dtype=torch.float32, device=self.device).unsqueeze(-1)
            slot_target_values = []
            slot_online_values = []
            slot_batches = []
            for slot_states in next_states:
                slot = StateBatch(slot_states).to(self.device)
                slot_batches.append(slot)
                with torch.no_grad():
                    target_logits = self.dqn_target(*slot.forward_args())
                    online_logits = self.dqn(*slot.forward_args())
                candidates = torch.cat(slot.candidates)
                slot_target_values.append(
                    pad_per_sample(self.to_values_from_logits(target_logits)[candidates], slot.nb_candidates)
                )
                slot_online_values.append(
                    pad_per_sample(self.to_values_from_logits(online_logits)[candidates], slot.nb_candidates)
                )
            # The official _get_next_actions: re-mask the padding then argmax.
            next_actions = [
                (v.masked_fill(self._padding_mask(v, slot), -1e9)).argmax(dim=1, keepdim=True)
                for v, slot in zip(slot_online_values, slot_batches, strict=True)
            ]
            per_child_target = torch.stack(
                [
                    v.gather(1, a)
                    for v, a in zip(slot_target_values, next_actions, strict=True)
                ]
            )
            next_target_q_value = (per_child_target * mask).sum(dim=0)
            not_done = 1 - dones
            target = torch.addcmul(rewards, not_done, next_target_q_value, value=gamma).detach()
        else:
            target = rewards.detach()

        # The categorical head: log2(-target) inside the bins' support.
        assert target.max() <= 0, "the BBMDP target must be negative"
        current_q_value = current_q_value.squeeze(1)
        target = torch.log2(-target + 1e-8).squeeze(-1)
        assert (
            target.min() >= self.loss_fn.min_value and target.max() <= self.loss_fn.max_value
        ), f"target {target.min():.2f}..{target.max():.2f} outside the bins' support"
        return self.loss_fn(current_q_value, target)

    def _padding_mask(self, padded: torch.Tensor, slot: StateBatch) -> torch.Tensor:
        positions = torch.arange(padded.size(1), device=self.device).unsqueeze(0)
        return positions >= slot.nb_candidates.unsqueeze(1)

    def target_update(self) -> None:
        """The official soft update: target <- tau * online + (1 - tau) * target."""
        with torch.no_grad():
            for target_param, param in zip(self.dqn_target.parameters(), self.dqn.parameters(), strict=True):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        state = {"dqn": self.dqn.state_dict()}
        if extra:
            state.update(extra)
        torch.save(state, path)


def load_dqn(path: str | Path, device: torch.device | None = None) -> DQNPolicy:
    device = device or get_device()
    dqn = DQNPolicy(classification=True)
    dqn.load_state_dict(torch.load(path, map_location=device, weights_only=True)["dqn"])
    return dqn.to(device).eval()


def _update_model(
    agent: TreeDQNAgent,
    memory: PrioritizedReplayBuffer,
    beta: float,
    sampling_rng: random.Random,
) -> float:
    """The official ``update_model``: one PER batch, the combined 1-step +
    n-step element-wise loss under the importance weights, gradient-norm
    clip, and the priorities written back."""
    batch = memory.sample_batch(beta=beta, rng=sampling_rng)
    # The official shapes: element-wise losses [B] and importance weights
    # [B, 1] — their product broadcasts to the outer product, so the mean
    # is mean(loss) * mean(weights) and the per-sample IS weights act as a
    # global scale. Kept verbatim for fidelity.
    weights = torch.as_tensor(batch["weights"], dtype=torch.float32).reshape(-1, 1).to(agent.device)

    elementwise_loss = agent.compute_tree_dqn_loss(batch, GAMMA, n_step=False)
    elementwise_loss = elementwise_loss + agent.compute_tree_dqn_loss(batch, GAMMA**N_STEP, n_step=True)
    loss = torch.mean(elementwise_loss * weights)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(agent.dqn.parameters(), GRADIENT_CLIP_NORM)
    agent.optimizer.step()
    agent.optimizer.zero_grad()

    # PER: the priorities are the element-wise losses plus prior_eps.
    new_priorities = elementwise_loss.detach().cpu().numpy() + PRIOR_EPS
    memory.update_priorities(batch["indices"], new_priorities)
    return float(loss.item())


def train(
    instances,
    checkpoint: Path,
    episodes: int = 100,
    n_epochs: int = 62_500,
    time_limit: float = 300.0,
    seed: int = 42,
) -> int:
    """The official ``DQNTrainer.train`` loop: episodes until the gradient-step
    budget (``n_epochs``, 200k officially) or the episode budget is spent.

    ``instances`` is an iterator of models; pre-solved draws yield no
    transitions and simply move to the next instance, like the official
    redraw loop. Returns the number of gradient steps run.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    python_rng = random.Random(seed)
    sampling_rng = random.Random(seed + 1)

    env = make_env(instances, time_limit=time_limit, seed=seed)
    agent = TreeDQNAgent()
    memory = PrioritizedReplayBuffer(MEMORY_SIZE, batch_size=BATCH_SIZE, alpha=ALPHA)
    print(f"[train] device: {agent.device}", flush=True)

    epsilon, temperature = MAX_EPSILON, 1.0
    beta = BETA_0
    gradient_steps = 0
    for episode in range(1, episodes + 1):
        if gradient_steps >= n_epochs:
            break
        # Default args pin the schedule values at episode start (they only
        # change between episodes, at gradient-step time).
        def policy(obs, cands, epsilon=epsilon, temperature=temperature):
            return agent.select_action(obs, cands, epsilon, temperature, python_rng)

        transitions, n_nodes, status = run_episode(env, policy)
        if transitions is None:
            print(f"[episode {episode:4d}] rejected ({status})", flush=True)
            continue
        for transition in transitions:
            memory.store(transition)

        if len(memory) > MEMORY_MIN_SIZE:
            n_gradient_steps = min(len(transitions) // STEPS_PER_UPDATE, MAX_GRADIENT_STEPS_PER_EPISODE)
            for _ in range(n_gradient_steps):
                # The official run_epochs: beta anneals over the first half
                # of the gradient budget, then epsilon and the temperature
                # decay once per gradient step.
                fraction = min(2 * gradient_steps / n_epochs, 1.0)
                beta = BETA_0 + fraction * (1.0 - BETA_0)
                _update_model(agent, memory, beta, sampling_rng)
                gradient_steps += 1
                epsilon = max(MIN_EPSILON, epsilon - (MAX_EPSILON - MIN_EPSILON) * EPSILON_DECAY)
                temperature = max(1e-3, temperature - (1.0 - 1e-3) * TEMPERATURE_DECAY)
                agent.target_update()
        print(
            f"[episode {episode:4d}] {status}  decisions={n_nodes}  "
            f"memory={len(memory)}  grad_steps={gradient_steps}  "
            f"eps={epsilon:.3f}  T={temperature:.4f}  beta={beta:.3f}",
            flush=True,
        )
        if episode % SAVE_EVERY_EPISODES == 0:
            agent.save(checkpoint)
        if episode % SNAPSHOT_EVERY_EPISODES == 0:
            # A numbered copy keyed on the official budget unit (gradient
            # steps): convergence campaigns evaluate each snapshot on the
            # pilot grid to trace SGM vs budget.
            agent.save(
                Path(f"{checkpoint}.g{gradient_steps:06d}"),
                extra={"episodes": episode, "grad_steps": gradient_steps},
            )
    agent.save(checkpoint)
    print(f"[train] saved {checkpoint} after {gradient_steps} gradient steps", flush=True)
    return gradient_steps
