"""End-to-end tests of the TGPPO 2025 (Tree-Gate PPO) port.

The instances are small set covers at the size where the sandbox cutoff
still leaves branching decisions: with the objective limit at the optimum,
root reduced-cost fixing collapses easier instances to a single node (the
official trained on hard MILPs where this does not happen). The point of
the tests is that the whole machinery runs on the harness (BranchingState
observation → sampled actor in the sandbox environment → H3 rewards from
an inline baseline solve → one PPO update over padded minibatches), not
that the agent learns anything.
"""

import numpy as np
import pytest
import torch
from gyozas.instances import SetCoverGenerator
from gyozas.observations import BranchingStateObservation
from gyozas.rewards import DifficultyAdaptive

from l2b_bench.methods.tgppo import (
    SANDBOX_PARAMS,
    Actor,
    Critic,
    Transition,
    baseline_solve,
    collate,
    compute_gae,
    load_checkpoint,
    make_env,
    ppo_update,
    run_episode,
    save_checkpoint,
    tgppo_method,
    train,
    with_baselines,
)


def _instances(n, rows=250, cols=500, seed=7):
    """Small set covers whose sandbox-cutoff episodes still branch."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.hideOutput()
        yield model


def _nets(seed=0):
    torch.manual_seed(seed)
    return Actor(hidden_dim=32, num_heads=2, num_layers=1), Critic(
        hidden_dim=32, num_heads=2, num_layers=1
    )


class TestModel:
    def test_actor_outputs_a_distribution_over_candidates(self):
        """The actor returns probabilities: non-negative, summing to 1 over
        the valid candidates, exactly 0 on padded slots."""
        actor = _nets()[0]
        cands = torch.randn(3, 5, 25)
        node = torch.randn(3, 8)
        mip = torch.randn(3, 53)
        mask = torch.zeros(3, 5, dtype=torch.bool)
        mask[0, 3:] = True
        mask[2, 4:] = True
        probs = actor(cands, node, mip, mask)
        assert probs.shape == (3, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
        assert (probs >= 0).all()
        assert probs[0, 3:].sum() == 0.0 and probs[2, 4:].sum() == 0.0

    def test_critic_outputs_one_value_per_state(self):
        critic = _nets()[1]
        value = critic(torch.randn(4, 6, 25), torch.randn(4, 8), torch.randn(4, 53))
        assert value.shape == (4, 1)
        assert torch.isfinite(value).all()

    def test_no_mask_means_no_padding(self):
        """A None mask is equivalent to an all-False mask."""
        actor = _nets()[0]
        cands, node, mip = torch.randn(2, 4, 25), torch.randn(2, 8), torch.randn(2, 53)
        with torch.no_grad():
            a = actor(cands, node, mip, None)
            b = actor(cands, node, mip, torch.zeros(2, 4, dtype=torch.bool))
        assert torch.allclose(a, b)


class TestBaseline:
    def test_baseline_solve_returns_stats_and_frees_the_model(self):
        """The baseline is a full sandbox solve; the optimum doubles as the
        cutoff and the model still drives an environment afterwards."""
        model = next(_instances(1))
        stats = baseline_solve(model, time_limit=60.0)
        assert stats.status == "optimal"
        assert stats.nodes >= 1
        assert stats.gap == 0.0
        assert stats.cutoff is not None and stats.cutoff > 0

    def test_with_baselines_configures_the_reward_and_sets_the_cutoff(self):
        reward_fn = DifficultyAdaptive()
        models = list(with_baselines(_instances(2), reward_fn, time_limit=60.0))
        assert len(models) == 2
        for model in models:
            assert 0 < model.getObjlimit() < 1000
        assert reward_fn.baseline_nodes >= 1


class TestEnv:
    def test_make_env_wires_the_official_regime(self):
        env = make_env(_instances(1), DifficultyAdaptive(), time_limit=60.0, seed=0)
        assert isinstance(env.observation_function, BranchingStateObservation)
        assert isinstance(env.reward_function, DifficultyAdaptive)
        for key in ("conflict/usesb", "numerics/feastol", "heuristics/fracdiving/freq"):
            assert env.scip_params[key] == SANDBOX_PARAMS[key]
        assert env.scip_params["limits/time"] == 60.0


class TestEpisode:
    def test_transitions_are_well_formed(self):
        """One transition per decision: the action ranks within the recorded
        candidate set, only the last transition is done, step rewards are
        clipped to [-1, 1]."""
        actor, critic = _nets()
        reward_fn = DifficultyAdaptive()
        env = make_env(
            with_baselines(_instances(2), reward_fn, time_limit=60.0),
            reward_fn,
            time_limit=60.0,
            seed=0,
        )
        transitions, info = run_episode(env, actor, critic, torch.device("cpu"))
        assert len(transitions) >= 1
        for t in transitions:
            assert 0 <= t.action < t.cands_state.shape[0]
            assert t.cands_state.shape[1] == 25
            assert t.node_state.shape == (8,)
            assert t.mip_state.shape == (53,)
        assert [t.done for t in transitions] == [False] * (len(transitions) - 1) + [
            True
        ]
        assert all(-1.0 <= t.reward <= 1.0 for t in transitions[:-1])
        assert info["status"] in ("optimal", "timelimit", "nodelimit")
        assert info["nnodes"] >= 1

    def test_terminal_reward_matches_the_official_formula(self):
        actor, critic = _nets()
        reward_fn = DifficultyAdaptive()
        env = make_env(
            with_baselines(_instances(1), reward_fn, time_limit=60.0),
            reward_fn,
            time_limit=60.0,
            seed=0,
        )
        transitions, info = run_episode(env, actor, critic, torch.device("cpu"))
        terminal = transitions[-1].reward
        status = info["status"]
        speedup = min(reward_fn.baseline_nodes / max(info["nnodes"], 1), 5.0)
        if status == "optimal":
            assert terminal == pytest.approx(1.0 + 3.0 * speedup)
        elif status == "timelimit":
            # 0.5*speedup + 0.3*gap_gain + 0.2*pdi_gain, each gain in (-1, 1).
            assert 0.5 * speedup - 0.5 <= terminal <= 0.5 * speedup + 0.5
        else:  # infeasible/unbounded or a limit stop
            assert terminal == pytest.approx(
                0.8 + 2.0 * speedup
            ) or terminal == pytest.approx(0.3 * speedup)


class TestGAE:
    def test_gae_matches_a_hand_computed_case(self):
        """Two steps, no done flags, bootstrap 0: the official recursion."""
        rewards = torch.tensor([1.0, 1.0])
        values = torch.tensor([0.5, 0.5])
        dones = torch.tensor([0.0, 0.0])
        bootstrap = torch.tensor(0.0)
        gamma, lam = 0.9, 0.8
        advantages, returns = compute_gae(rewards, values, dones, bootstrap, gamma, lam)
        # delta_2 = 1 + 0 - 0.5 = 0.5; A_2 = 0.5
        # delta_1 = 1 + 0.9*0.5 - 0.5 = 0.95; A_1 = 0.95 + 0.9*0.8*0.5 = 1.31
        assert advantages[1] == pytest.approx(0.5)
        assert advantages[0] == pytest.approx(1.31)
        assert returns[0] == pytest.approx(1.31 + 0.5)
        assert returns[1] == pytest.approx(1.0)

    def test_done_cuts_the_bootstrap(self):
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.0])
        dones = torch.tensor([1.0])
        advantages, returns = compute_gae(rewards, values, dones, torch.tensor(3.0))
        assert advantages[0] == pytest.approx(1.0)
        assert returns[0] == pytest.approx(1.0)


class TestUpdate:
    @pytest.fixture(scope="class")
    def synthetic_transitions(self):
        rng = np.random.default_rng(1)

        def one():
            n = int(rng.integers(2, 6))
            return Transition(
                cands_state=rng.normal(size=(n, 25)).astype(np.float32),
                node_state=rng.normal(size=8).astype(np.float32),
                mip_state=rng.normal(size=53).astype(np.float32),
                action=int(rng.integers(n)),
                reward=float(rng.normal()),
                done=False,
                value=float(rng.normal()),
                log_prob=float(-rng.random()),
            )

        return [one() for _ in range(40)]

    def test_collate_pads_candidate_sets(self, synthetic_transitions):
        batch = collate(synthetic_transitions)
        width = batch["cands"].shape[1]
        assert batch["cands"].shape == (40, width, 25)
        for i, t in enumerate(synthetic_transitions):
            k = t.cands_state.shape[0]
            assert not batch["mask"][i, :k].any()
            assert batch["mask"][i, k:].all()
            assert np.allclose(batch["cands"][i, :k].numpy(), t.cands_state)

    def test_ppo_update_moves_the_nets(self, synthetic_transitions):
        actor, critic = _nets()
        before = {n: p.clone() for n, p in actor.named_parameters()}
        actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
        stats = ppo_update(
            actor,
            critic,
            actor_opt,
            critic_opt,
            synthetic_transitions,
            torch.device("cpu"),
            batch_size=16,
        )
        assert np.isfinite(stats["actor_loss"])
        assert np.isfinite(stats["critic_loss"])
        assert any(not torch.equal(p, before[n]) for n, p in actor.named_parameters())

    def test_too_few_transitions_is_a_no_op(self, synthetic_transitions):
        actor, critic = _nets()
        before = {n: p.clone() for n, p in actor.named_parameters()}
        actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
        stats = ppo_update(
            actor,
            critic,
            actor_opt,
            critic_opt,
            synthetic_transitions[:8],
            torch.device("cpu"),
            batch_size=64,
        )
        assert stats == {}
        assert all(torch.equal(p, before[n]) for n, p in actor.named_parameters())


class TestTrain:
    def test_train_saves_a_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "tgppo.pt"
        train(
            _instances(2, seed=11),
            checkpoint,
            episodes=2,
            time_limit=30.0,
            seed=0,
            device=torch.device("cpu"),
            log=lambda *_: None,
        )
        assert checkpoint.exists()
        actor, critic = load_checkpoint(checkpoint, device=torch.device("cpu"))
        assert isinstance(actor, Actor)
        assert isinstance(critic, Critic)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["episodes"] == 2
        assert payload["updates"] == 1

    def test_train_writes_a_snapshot_per_update(self, tmp_path):
        snapshot = tmp_path / "tgppo.pt.u001"
        train(
            _instances(2, seed=11),
            tmp_path / "tgppo.pt",
            episodes=2,
            time_limit=30.0,
            seed=0,
            device=torch.device("cpu"),
            log=lambda *_: None,
        )
        assert snapshot.exists()
        actor, critic = load_checkpoint(snapshot, device=torch.device("cpu"))
        assert isinstance(actor, Actor)
        assert isinstance(critic, Critic)


class TestPolicy:
    def test_tgppo_method_is_runner_ready(self, tmp_path):
        checkpoint = tmp_path / "tgppo.pt"
        actor, critic = _nets()
        save_checkpoint(actor, critic, checkpoint)
        method = tgppo_method(checkpoint)
        assert method.name == "tgppo"
        assert method.fidelity == "tgppo"
        policy = method.policy()
        reward_fn = DifficultyAdaptive()
        env = make_env(
            with_baselines(_instances(3, seed=11), reward_fn, time_limit=30.0),
            reward_fn,
            time_limit=30.0,
            seed=0,
        )
        obs, action_set, _reward, done, _info = env.reset()
        while obs is None and not done:  # skip pre-solved instances
            obs, action_set, _reward, done, _info = env.reset()
        if obs is not None:
            action = policy.select_action(obs, action_set)
            assert action in np.asarray(action_set).tolist()


def _fit_argmax_feature(fixed: bool, steps: int = 150) -> float:
    """Supervised probe: train the actor to pick the candidate with the
    largest feature 0; returns the final cross-entropy."""
    torch.manual_seed(0)
    actor = Actor(fixed=fixed)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=1e-2)
    n_cands, batch = 20, 32
    for _ in range(steps):
        cands = torch.randn(batch, n_cands, 25)
        target = cands[:, :, 0].argmax(-1)
        probs = actor(cands, torch.randn(batch, 8), torch.randn(batch, 53))
        loss = torch.nn.functional.nll_loss(torch.log(probs + 1e-12), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss)


class TestFixedVariant:
    """The released architecture's collapse, and the tgppo-fixed variant."""

    def test_released_actor_outputs_a_uniform_policy(self):
        # Pinned finding (same result on the official modules): every
        # candidate gets the same probability, at 1e-6.
        torch.manual_seed(0)
        probs = Actor().eval()(torch.randn(2, 40, 25), torch.randn(2, 8), torch.randn(2, 53))
        assert torch.allclose(probs, torch.full_like(probs, 1 / 40), atol=1e-6)

    def test_released_actor_cannot_learn_a_trivial_ranking(self):
        uniform = float(np.log(20))
        assert _fit_argmax_feature(fixed=False) == pytest.approx(uniform, abs=1e-3)

    def test_fixed_actor_learns_a_trivial_ranking(self):
        assert _fit_argmax_feature(fixed=True) < 0.85 * float(np.log(20))

    def test_fixed_flag_round_trips_through_the_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "tgppo-fixed.pt"
        save_checkpoint(Actor(fixed=True), Critic(fixed=True), checkpoint)
        actor, critic = load_checkpoint(checkpoint, device=torch.device("cpu"))
        assert actor.fixed and critic.fixed
        method = tgppo_method(checkpoint, fixed=True)
        assert method.name == "tgppo-fixed"
        assert method.fidelity == "tgppo"
        x = (torch.randn(3, 7, 25), torch.randn(3, 8), torch.randn(3, 53))
        assert critic(*x).shape == (3, 1)
        assert actor(*x).shape == (3, 7)
