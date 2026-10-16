"""PlanB&B — sequential training: self-play, uniform replay, the four
MuZero losses.

Faithful port of the official repository (github.com/abfariah/planbb,
``core/train.py`` ``update_weights``/``adjust_lr`` + ``core/reanalyze_worker.py``
``make_batch``/``_prepare_policy_value_no_re``/``_prepare_node_value_target``),
without the ray actor topology:

- **Replay** — every stored trajectory contributes its visited positions;
  batches sample ``(trajectory, position)`` pairs uniformly (the official's
  released ``--use_priority --use_max_priority`` degenerates to uniform
  sampling) and are thrown away after the step (no priority updates — the
  official's ``new_priority = value_priority * 0``).
- **Targets** — the non-reanalyze path: the policy target is the stored
  MCTS improved-policy distribution, the value target is the td reward sum
  plus the target model's scalar value of the n-step bootstrap frontier
  (``initial_inference`` at the frontier nodes, the official's
  ``use_mcts_root_values=False``). The official's ``_prepare_node_value_target``
  quirk is ported as applied: the discount powers are computed but the
  reduceat runs on the raw values (the discount multiplication on the
  previous line is overwritten) — identical at the released gamma = 1.
- **update_weights** — the tree-MDP branch verbatim: oracle hidden states
  from the representation on all 3u+1 window slots, the dynamics unroll
  over the recorded actions with per-slot next-observation routing (cached
  vs emergency), the policy CE against the stored distributions, the value
  CE against the bootstrap targets weighted by the open-node appearance
  counts, the breward CE on the ghost-masked slots, and the SimSiam
  consistency between the dynamics and oracle projections — all inside the
  released torch-amp region (``autocast("cuda")`` + ``GradScaler``).
- **The loop** — the released schedule: warmup then 0.95^(step/1000) decay,
  the temperature at 1.0 for the first 20% then 0.99^(step/500), a target
  model refreshed every 200 steps, the checkpoint every 1000.

Pilot-scale deviations (documented here, not in the ported math):

- sequential self-play — the official's ``p_mcts_num`` only parallelizes
  independent episodes, never the search itself;
- reanalyze is off (the released configuration never reanalyzes — the
  reanalyzed targets only enter through ``config.reanalyze``, False by
  default and absent from ``train_agents.sh``);
- the validation pass is dropped, like every RL pilot in this benchmark;
  ``start_transitions`` is a parameter (the official 40k transitions are
  sized for 11 parallel actors, not a pilot).
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch

from .collect import NODE_LIMIT, PlanBBTrajectory, make_env, run_selfplay_episode
from .model import (
    PlanBBNet,
    amp_autocast,
    format_batch_actions,
    geometric_batch,
    get_idx,
    get_target_index,
    pad_array,
)
from .search import MCTS

logger = logging.getLogger(__name__)

#: The released hyperparameters (config/milp/__init__.py + config_base.py).
BATCH_SIZE = 96
LR_INIT = 1e-3
LR_DECAY_RATE = 0.95
LR_DECAY_STEPS = 1_000
LR_WARM_UP = 0.01
WEIGHT_DECAY = 2e-5
MAX_GRAD_NORM = 5
TEMPERATURE_DECAY = 0.99
TEMPERATURE_DECAY_STEP = 500
TARGET_MODEL_INTERVAL = 200
SAVE_CKPT_INTERVAL = 1_000


def consist_loss_func(f1, f2, alpha: float = 0.5) -> torch.Tensor:
    """The official SimSiam consistency: 1 - (cosine + normalized-distance)/2
    over the L2-normalized projections."""
    f1 = F.normalize(f1, p=2.0, dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2.0, dim=-1, eps=1e-5)
    cosine_sim = (f1 * f2).sum(dim=1)
    euclidean_dist = torch.norm(f1 - f2, p=2.0, dim=1)
    max_dist = torch.norm(f1 + f2, p=2.0, dim=1)
    normalized_dist = 1 - euclidean_dist / max_dist
    return 1 - (alpha * cosine_sim + (1 - alpha) * normalized_dist)


def adjust_lr(optimizer, step_count: int, training_steps: int, warm_up: float = LR_WARM_UP) -> float:
    """The official schedule: linear warmup over ``warm_up`` of the training
    steps, then ``lr_decay_rate^((step - warm) // decay_steps)`` floored at
    1e-5."""
    lr_warm_step = int(training_steps * warm_up)
    if step_count < lr_warm_step:
        lr = LR_INIT * (step_count + 1) / lr_warm_step
        lr = max(lr, 1e-5)
    else:
        lr = LR_INIT * LR_DECAY_RATE ** ((step_count - lr_warm_step) // LR_DECAY_STEPS)
        lr = max(lr, 1e-5)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def visit_softmax_temperature(trained_steps: int, training_steps: int) -> float:
    """The official ``visit_softmax_temperature_fn``: 1.0 over the first 20%
    of training, then ``temperature_decay^((steps - 20%) // 500)``."""
    if trained_steps < 0.2 * training_steps:
        return 1.0
    index = trained_steps - 0.2 * training_steps
    return 1.0 * (TEMPERATURE_DECAY ** (index // TEMPERATURE_DECAY_STEP))


@dataclass
class TrainBatch:
    """One sampled batch, the official ``make_batch`` + non-reanalyze target
    outputs."""

    obs_windows: list  # [B][window] the ghost-completed observations
    action_nums: list  # [B]
    obs_idx_lst: list  # [B][window] window slot -> episode step idx
    obs_mask_lst: list  # [B][window] is_real per window slot
    action_batch: list  # [B][unroll]
    mask_batch: list  # [B][unroll]
    value_weights: list  # [B][traj_obs]
    target_policy: np.ndarray  # [traj_horizon][B][max_vars]
    target_value: np.ndarray  # [traj_obs][B][1]


def sample_batch(
    replay: List[PlanBBTrajectory],
    batch_size: int,
    rng: np.random.Generator,
) -> List[Tuple[PlanBBTrajectory, int]]:
    """Uniform sampling over (trajectory, position) pairs — the released
    ``use_max_priority`` degenerates the priority tree to uniform."""
    positions = [(traj, pos) for traj in replay for pos in range(len(traj))]
    idxs = rng.choice(len(positions), size=batch_size, replace=len(positions) < batch_size)
    return [positions[i] for i in idxs]


def make_batch(
    samples: List[Tuple[PlanBBTrajectory, int]],
    unroll_steps: int,
) -> Tuple[List[object], List[object], List[object], List[object], List[object], List[object], List[object]]:
    """The official ``make_batch`` inputs: per-sample windows, actions, masks,
    weights, plus the context the value target needs."""
    obs_lst, obs_idx_lst, obs_mask_lst, actions_lst, mask_lst = [], [], [], [], []
    action_nums = []
    value_loss_obs_weight_lst = []
    td_reward_sums_lst, discount_power_lst, n_steps_td_observations_lst = [], [], []

    action_horizon = unroll_steps
    for milp, milp_pos in samples:
        bo = milp.batch_obs(milp_pos)
        node_obs_to_node_idx = bo.batch_nodes_idx

        # add mask for invalid steps (out of trajectory)
        state_indices = np.array(
            [node_obs_to_node_idx[3 * i] for i in range(action_horizon - bo.pad_len)]
        )
        _actions = np.asarray(milp.actions)[state_indices].tolist()
        _mask = [1 for _ in range(len(_actions[1:]))]
        _actions += [-1 for _ in range(bo.pad_len)]
        _mask += [0 for _ in range(bo.pad_len)]
        m_bar = node_obs_to_node_idx[-1] == node_obs_to_node_idx[-4]
        _mask += [1 - m_bar]

        obs_lst.append(bo.batch_observations)
        obs_idx_lst.append(node_obs_to_node_idx)
        obs_mask_lst.append(bo.is_real_nodes)
        actions_lst.append(_actions)
        mask_lst.append(_mask)
        assert len(_actions) == action_horizon
        action_nums.append(milp.action_num)
        value_loss_obs_weight_lst.append(bo.value_loss_obs_weight)
        td_reward_sums_lst.append(bo.td_reward_sums)
        discount_power_lst.append(bo.n_steps_discount_powers)
        n_steps_td_observations_lst.append(bo.n_steps_td_observations)

    return (
        obs_lst,
        obs_idx_lst,
        obs_mask_lst,
        actions_lst,
        mask_lst,
        action_nums,
        value_loss_obs_weight_lst,
        td_reward_sums_lst,
        discount_power_lst,
        n_steps_td_observations_lst,
    )


def make_target_policy(samples, unroll_steps: int) -> np.ndarray:
    """The official policy target: the stored MCTS improved-policy
    distributions at the window's root slots, zero-padded to the batch's
    variable count."""
    traj_horizon = unroll_steps + 1
    child_visits = []
    action_nums = []
    for milp, milp_pos in samples:
        bo_nodes_idx = milp.batch_nodes_idx[milp_pos]
        state_indices = np.array([bo_nodes_idx[3 * i] for i in range(traj_horizon)])
        child_visits.append(np.asarray(milp.child_visits)[state_indices])
        action_nums.append(milp.action_num)
    return pad_array(child_visits, action_nums).transpose(1, 0, 2)


def prepare_node_value_target(
    model: PlanBBNet,
    td_reward_sums_lst,
    discount_power_lst,
    n_steps_td_observations_lst,
    device: torch.device,
    unroll_steps: int,
) -> np.ndarray:
    """The official ``_prepare_node_value_target`` (non-reanalyze): per
    window slot, the td reward sum plus the model's scalar value of the
    n-step bootstrap frontier. Ported as applied — the reduceat runs on the
    raw frontier values (the official's discount multiplication on the
    previous line is immediately overwritten; identical at gamma = 1)."""
    batch_size = len(td_reward_sums_lst)
    traj_obs = 3 * unroll_steps + 1
    node_values = []
    with torch.no_grad():
        for obs_idx in range(traj_obs):
            sums_r = []
            open_nodes_lst = []
            build_back_batch_values = []
            for batch_idx in range(batch_size):
                sum_r = td_reward_sums_lst[batch_idx][obs_idx]
                obss = n_steps_td_observations_lst[batch_idx][obs_idx]
                sums_r.append(sum_r)
                build_back_batch_values.append(len(obss))
                if len(obss) > 0:
                    open_nodes_lst.append(obss)

            sums_r = np.array(sums_r).reshape(batch_size, 1)

            if sum(build_back_batch_values) == 0:
                node_values.append(sums_r)
                continue

            batch_obs_lst = [x for ll in open_nodes_lst for x in ll]
            obs = geometric_batch(batch_obs_lst).to(device)
            with amp_autocast():
                network_output = model.initial_inference(obs, root=True)

            batch_indices = np.array(build_back_batch_values)
            batch_indices = np.cumsum(batch_indices, axis=0) - np.array(batch_indices)
            # Prevent np.add.reduceat index error (the official guard).
            overreach_mask = batch_indices >= network_output.value.shape[0]
            ghost_node_mask = np.array(build_back_batch_values) == 0
            indices_mask = overreach_mask * ghost_node_mask
            batch_indices[indices_mask] -= 1

            obs_values = np.add.reduceat(
                network_output.value.detach().cpu().numpy(), batch_indices
            ).reshape(batch_size, 1)
            # Correct np.add.reduceat (the official mask).
            mask = np.array(build_back_batch_values) == 0
            obs_values[mask] = 0.0
            target_value = sums_r + obs_values
            node_values.append(target_value)

    return np.array(node_values)


def update_weights(
    model: PlanBBNet,
    batch: TrainBatch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    unroll_steps: int,
    scaler: "torch.amp.GradScaler | None" = None,
) -> dict:
    """The official ``update_weights`` tree-MDP branch (torch amp: the
    inference passes run under ``autocast("cuda")`` and the backward under a
    ``GradScaler`` — the released ``amp_type``; no priority updates). Returns
    the loss components for logging."""
    traj_obs = 3 * unroll_steps + 1
    batch_size = len(batch.obs_windows)
    obs_batch_ori = [
        [window[slot] for window in batch.obs_windows] for slot in range(traj_obs)
    ]  # [traj_obs][B]

    # The oracle pass: hidden states + emergency heads on every slot (the
    # official keeps the gradients — the first step's losses train the
    # representation through slot 0).
    with amp_autocast():
        output_batch_target = [
            model.initial_inference(geometric_batch(slot_obs).to(device), root=True, tree_mdp=True)
            for slot_obs in obs_batch_ori
        ]
    obs_batch_target = [output.hidden_state for output in output_batch_target]
    emergency_policy = [output.policy_logits for output in output_batch_target]
    emergency_value = [output.value for output in output_batch_target]
    init_output = output_batch_target[0]
    obs_batch_init = obs_batch_target[0].clone()

    action_batch = torch.from_numpy(np.array(batch.action_batch).transpose(1, 0)).to(device).long()
    mask_batch = torch.from_numpy(np.array(batch.mask_batch).transpose(1, 0)).to(device).float()
    target_value = torch.from_numpy(np.array(batch.target_value)).to(device).float()
    target_policy = torch.from_numpy(np.array(batch.target_policy)).to(device).float()
    target_index = get_target_index(unroll_steps)
    obs_idx_batch = torch.from_numpy(np.array(get_idx(batch.obs_idx_lst))).to(device).long().transpose(1, 0)
    obs_mask_batch = torch.from_numpy(np.array(batch.obs_mask_lst)).to(device).float()
    value_weights = torch.from_numpy(np.array(batch.value_weights)).to(device).float()
    value_weights = F.normalize(value_weights, p=1, dim=1) * (unroll_steps + 1)
    value_weights = value_weights.transpose(1, 0)
    target_breward = (obs_mask_batch == 1).long().transpose(1, 0)

    obs_idx_batch = obs_idx_batch[:traj_obs]
    target_breward = target_breward[:traj_obs]

    assert batch_size == target_value.size(1)

    value_logits = init_output.value
    breward = init_output.breward
    policy_logits = init_output.policy_logits

    # loss of the first step
    value_loss = model.value_loss(value_logits.squeeze(-1), target_value[0, :], tree_mdp=True)
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[0, :]).sum(1)
    consistency_loss = torch.zeros(batch_size, device=device)

    gradient_scale = 1 / (unroll_steps + 1)

    batch_node_values = [value_logits]
    batch_policy_logits = [policy_logits]
    batch_obs = [obs_batch_init]
    batch_breward = [breward]

    # loss of the unrolled steps — the official's single autocast region:
    # the unroll, the reward loss and the consistency loss all run inside
    # it (core/train.py lines 201-413).
    with amp_autocast():
        for step_i in range(unroll_steps):
            current_idx = 3 * step_i
            hidden_state_batch = batch_obs[current_idx]

            actions = format_batch_actions(action_batch[step_i, :], hidden_state_batch.nb_variables)
            left_output, right_output = model.recurrent_inference(hidden_state_batch, actions, tree_mdp=True)

            left_node_v, left_node_b, left_node_p, left_node_o = (
                left_output.value,
                left_output.breward,
                left_output.policy_logits,
                left_output.hidden_state,
            )
            right_node_v, right_node_b, right_node_p, right_node_o = (
                right_output.value,
                right_output.breward,
                right_output.policy_logits,
                right_output.hidden_state,
            )

            batch_obs += [left_node_o, right_node_o]
            batch_breward += [left_node_b, right_node_b]
            batch_node_values += [left_node_v, right_node_v]
            batch_policy_logits += [left_node_p, right_node_p]
            next_current_idx = current_idx + 3

            # Build next batch obs: cached window state or the slot oracle.
            next_obs_idx = obs_idx_batch[next_current_idx]
            next_obs, next_policy_logits, next_node_value = [], [], []
            for batch_idx in range(batch_size):
                obs_idx = next_obs_idx[batch_idx]
                if obs_idx < next_current_idx:
                    next_obs.append(batch_obs[obs_idx][batch_idx])
                    next_node_value.append(batch_node_values[obs_idx][batch_idx])
                    next_policy_logits.append(batch_policy_logits[obs_idx][batch_idx])
                else:
                    assert obs_idx == next_current_idx
                    next_obs.append(obs_batch_target[next_current_idx][batch_idx])
                    next_policy_logits.append(emergency_policy[next_current_idx][batch_idx])
                    next_node_value.append(emergency_value[next_current_idx][batch_idx].clone())

            next_obs = Batch.from_data_list(next_obs)
            next_node_value = torch.cat(next_node_value).view(batch_size, -1)
            next_policy_logits = torch.vstack(next_policy_logits)

            batch_obs.append(next_obs)
            batch_node_values.append(next_node_value)
            batch_policy_logits.append(next_policy_logits)

            policy_loss += (
                -(torch.log_softmax(next_policy_logits, dim=1) * target_policy[step_i + 1, :]).sum(1)
                * mask_batch[step_i, :]
            )
            value_loss += (
                model.value_loss(left_node_v, target_value[current_idx + 1, :], tree_mdp=True)
                * value_weights[current_idx + 1]
            )
            value_loss += (
                model.value_loss(right_node_v, target_value[current_idx + 2, :], tree_mdp=True)
                * value_weights[current_idx + 2]
            )
            value_loss += (
                model.value_loss(next_node_value, target_value[current_idx + 3, :], tree_mdp=True)
                * value_weights[current_idx + 3]
            )

        # Reward loss
        batch_breward = torch.stack(batch_breward).transpose(2, 1)
        breward_loss = F.cross_entropy(batch_breward, target_breward[target_index], reduction="none").sum(0)

        # Consistency: dynamics projections vs the no-grad oracle projections.
        for obs, obs_target in zip(batch_obs, obs_batch_target):
            dynamic_proj = model.project(obs, with_grad=True)
            observation_proj = model.project(obs_target, with_grad=False)
            consistency_loss += consist_loss_func(dynamic_proj, observation_proj)

    loss = consistency_loss + policy_loss + value_loss + breward_loss
    weighted_loss = loss.mean()

    total_loss = weighted_loss
    total_loss.register_hook(lambda grad: grad * gradient_scale)

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    else:
        total_loss.backward()
    clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return {
        "loss": total_loss.item(),
        "policy": policy_loss.mean().item(),
        "value": value_loss.mean().item(),
        "breward": breward_loss.mean().item(),
        "consistency": consistency_loss.mean().item(),
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    instances,
    checkpoint,
    episodes: int = 100,
    training_steps: int = 1_000,
    time_limit: float = 300.0,
    seed: int = 42,
    start_transitions: int = 500,
    batch_size: int = BATCH_SIZE,
    node_limit: int = NODE_LIMIT,
    device: torch.device | None = None,
    unroll_steps: int = 3,
) -> int:
    """The sequential pilot training loop: fill the replay with random-action
    self-play (the official's pre-training phase, MCTS statistics still
    stored), then alternate self-play episodes with gradient steps until the
    training-step or episode budget is spent.

    ``instances`` is an iterator of models. Returns the number of gradient
    steps run; the final weights (and a resumable checkpoint every
    ``SAVE_CKPT_INTERVAL`` steps) are written to ``checkpoint``.
    """
    import copy

    device = device or get_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = make_env(instances, node_limit=node_limit, time_limit=time_limit, seed=seed)
    model = PlanBBNet().to(device)
    target_model = copy.deepcopy(model)
    for p in target_model.parameters():
        p.requires_grad_(False)
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    # The released torch-amp training (GradScaler; disabled without CUDA so
    # the CPU tests take the plain backward path).
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mcts = MCTS()
    print(f"[planbb] device: {device}", flush=True)

    replay: List[PlanBBTrajectory] = []
    n_transitions = 0
    trained_steps = 0
    n_episodes = 0

    def run_episode(start_training: bool) -> None:
        nonlocal n_transitions
        temperature = visit_softmax_temperature(trained_steps, training_steps)
        model.eval()
        trajectory, _n, _status = run_selfplay_episode(
            env,
            model,
            device,
            temperature=temperature,
            start_training=start_training,
            rng=rng,
            mcts=mcts,
        )
        model.train_mode()
        model.train()
        if trajectory is not None:
            replay.append(trajectory)
            n_transitions += len(trajectory)

    # Phase 1: random-action self-play until the replay is warm.
    while n_transitions < start_transitions and n_episodes < episodes:
        run_episode(start_training=False)
        n_episodes += 1

    # Phase 2: alternate self-play and gradient steps.
    while trained_steps < training_steps and n_episodes < episodes:
        run_episode(start_training=True)
        n_episodes += 1
        episodes_left = max(1, episodes - n_episodes)
        for _ in range(max(1, training_steps // episodes_left)):
            if trained_steps >= training_steps or n_transitions < 2:
                break
            samples = sample_batch(replay, batch_size, rng)
            (
                obs_lst,
                obs_idx_lst,
                obs_mask_lst,
                actions_lst,
                mask_lst,
                _action_nums,
                value_weights,
                td_reward_sums,
                discount_powers,
                n_steps_td_observations,
            ) = make_batch(samples, unroll_steps)
            target_model.eval()
            target_value = prepare_node_value_target(
                target_model,
                td_reward_sums,
                discount_powers,
                n_steps_td_observations,
                device,
                unroll_steps,
            )
            batch = TrainBatch(
                obs_windows=obs_lst,
                action_nums=_action_nums,
                obs_idx_lst=obs_idx_lst,
                obs_mask_lst=obs_mask_lst,
                action_batch=actions_lst,
                mask_batch=mask_lst,
                value_weights=value_weights,
                target_policy=make_target_policy(samples, unroll_steps),
                target_value=target_value,
            )
            model.train()
            losses = update_weights(model, batch, optimizer, device, unroll_steps, scaler=scaler)
            trained_steps += 1

            # The target model lags the trainer (the official refreshes it
            # every target_model_interval steps).
            if trained_steps % TARGET_MODEL_INTERVAL == 0:
                target_model.load_state_dict(model.state_dict())
            if trained_steps % SAVE_CKPT_INTERVAL == 0:
                torch.save(model.state_dict(), checkpoint)
            if trained_steps % 100 == 0:
                print(
                    f"[planbb] step {trained_steps}/{training_steps} "
                    f"episodes={n_episodes} transitions={n_transitions} "
                    f"loss={losses['loss']:.4f} policy={losses['policy']:.4f} "
                    f"value={losses['value']:.4f}",
                    flush=True,
                )

    torch.save(model.state_dict(), checkpoint)
    print(
        f"[planbb] done: {trained_steps} gradient steps, {n_episodes} episodes, "
        f"{n_transitions} transitions -> {checkpoint}",
        flush=True,
    )
    return trained_steps
