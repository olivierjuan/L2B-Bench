"""End-to-end tests of the SORREL 2024 (Feng et al., AAAI 2025) port.

The instances are tiny set covers — the point is that the whole regime runs
on the harness (VHB demonstrations over the pseudo branching candidates →
tree-MDP samples with two children's graphs → TD3BC offline → PPO + SIL
online), not that the agent learns anything. The math tests pin the official
formulas (kappa-weighted tree backup, lmbda, advantage normalization, SIL
weights) against hand computation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.gasse.model import GNNPolicy
from l2b_bench.methods.sorrel import (
    PPOSIL,
    SORREL_SCIP_PARAMS,
    TD3BC,
    SILBuffer,
    collect_episode,
    load_actor,
    make_env,
    run_episode,
    sorrel_method,
    train_online,
    train_td3bc,
    tree_returns,
)
from l2b_bench.methods.sorrel.model import SorrelBatch, padded_logits


def _instances(n, rows=120, cols=150, seed=7):
    """Tiny set covers, one fresh model per episode."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.hideOutput()
        yield model


def _dummy_observation():
    """A one-constraint/one-variable bipartite graph, linked by one edge so
    the PreNorm warmup's edge layer sees a non-empty input."""
    return SimpleNamespace(
        row_features=np.zeros((1, 5), dtype=np.float32),
        edge_features=SimpleNamespace(
            indices=np.array([[0], [0]], dtype=np.int64), values=np.array([1.0], dtype=np.float32)
        ),
        variable_features=np.zeros((1, 19), dtype=np.float32),
    )


def _synthetic_sample(reward1=0.0, reward2=0.0, visited1=False, visited2=False, returns=0.0):
    """A tree-MDP transition over dummy graphs, hand-controlled rewards."""
    from l2b_bench.methods.sorrel.collect import SorrelSample

    obs = _dummy_observation()
    return SorrelSample(
        observation=obs,
        action_set=np.array([0]),
        action=0,
        candidate_scores=np.array([0.0]),
        is_fsb=False,
        child1_obs=obs if visited1 else None,
        child1_action_set=np.array([0]) if visited1 else None,
        reward1=reward1,
        child2_obs=obs if visited2 else None,
        child2_action_set=np.array([0]) if visited2 else None,
        reward2=reward2,
        returns=returns,
        log_prob=0.0,
    )


class TestTreeReturns:
    def test_hand_computed_tree(self):
        # Node 0 branches into 1 (r=1) and 2 (r=2); 1 has one visited child 3
        # (r=0.5, a leaf); 2 has one unvisited child (r=3, no bootstrap).
        children = [
            ([1, 2], [1.0, 2.0]),
            ([3], [0.5]),
            ([None], [3.0]),
            ([], []),
        ]
        returns = tree_returns(children, gamma=0.9, kappa=0.8)
        assert returns[3] == 0.0
        assert returns[1] == pytest.approx(0.5)
        assert returns[2] == pytest.approx(3.0)
        # The smaller-reward child carries kappa.
        assert returns[0] == pytest.approx(0.8 * (1.0 + 0.9 * 0.5) + 0.2 * (2.0 + 0.9 * 3.0))

    def test_unvisited_children_contribute_reward_only(self):
        returns = tree_returns([([None, None], [1.0, 2.0])], gamma=0.9, kappa=0.8)
        assert returns[0] == pytest.approx(0.8 * 1.0 + 0.2 * 2.0)

    def test_kappa_goes_to_the_smaller_reward(self):
        # Swapping the rewards swaps the weights.
        returns = tree_returns([([None, None], [2.0, 1.0])], gamma=0.9, kappa=0.8)
        assert returns[0] == pytest.approx(0.2 * 2.0 + 0.8 * 1.0)

    def test_empty_tree(self):
        assert tree_returns([], gamma=0.9, kappa=0.8) == {}


class TestScipProtocol:
    def test_heuristics_off_and_restarts_off(self):
        assert SORREL_SCIP_PARAMS["separating/maxrounds"] == 0
        assert SORREL_SCIP_PARAMS["presolving/maxrestarts"] == 0
        assert SORREL_SCIP_PARAMS["estimation/restarts/restartpolicy"] == "n"
        assert SORREL_SCIP_PARAMS["heuristics/rounding/freq"] == -1
        assert len([k for k in SORREL_SCIP_PARAMS if k.startswith("heuristics/")]) == 54

    def test_make_env_uses_the_pseudo_candidates(self):
        env = make_env(_instances(1), time_limit=10.0, seed=0)
        try:
            assert env.dynamics.pseudo_candidates
        finally:
            env.close()


class TestCollection:
    def test_vhb_episode_records_tree_transitions(self):
        # fsb_probability 0: the weak expert (pseudocosts) decides everywhere.
        env = make_env(_instances(1), time_limit=60.0, seed=0)
        try:
            rng = np.random.default_rng(0)
            samples, n_iter, _status = collect_episode(
                env, rng, fsb_probability=0.0, length_limit=30, sampling_threshold=5
            )
        finally:
            env.close()
        assert n_iter >= 5
        assert samples, "the episode must clear the sampling threshold"
        for sample in samples:
            assert sample.action in sample.action_set
            assert not sample.is_fsb
            assert len(sample.candidate_scores) == len(sample.action_set)
            # Set covers are minimization: a child's dual bound never
            # decreases, so the tree rewards are non-negative.
            assert sample.reward1 >= -1e-9
            assert sample.reward2 >= -1e-9
            # An unvisited child has neither observation nor action set.
            assert (sample.child1_obs is None) == (sample.child1_action_set is None)
            assert (sample.child2_obs is None) == (sample.child2_action_set is None)

    def test_short_episodes_are_dropped(self):
        env = make_env(_instances(1), time_limit=60.0, seed=0)
        try:
            rng = np.random.default_rng(0)
            samples, _n_iter, _status = collect_episode(
                env, rng, fsb_probability=0.0, length_limit=30, sampling_threshold=10_000
            )
        finally:
            env.close()
        assert samples == []


class TestCollation:
    def test_dummy_graphs_for_unvisited_children(self):
        batch = SorrelBatch(
            [
                _synthetic_sample(visited1=False, visited2=False),
                _synthetic_sample(visited1=True, visited2=True),
            ]
        )
        assert batch.dones1.tolist() == [1, 0]
        assert batch.dones2.tolist() == [1, 0]
        # Two one-variable state graphs -> candidate offsets 0 and 1.
        assert batch.states.candidates[0].tolist() == [0]
        assert batch.states.candidates[1].tolist() == [1]
        # The unvisited child's graph is the official dummy: one constraint,
        # one variable, one candidate, no edges (the second sample's visited
        # child carries the synthetic single edge).
        assert batch.children1.constraint_features.shape == (2, 5)
        assert batch.children1.edge_attr.shape == (1, 1)
        assert batch.rewards1.tolist() == [0.0, 0.0]

    def test_padded_logits_fill_with_minus_inf(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        padded = padded_logits(logits, [torch.tensor([0, 1]), torch.tensor([2])])
        assert padded.shape == (2, 2)
        assert torch.allclose(padded[0], torch.tensor([1.0, 2.0]))
        assert padded[1].tolist() == [3.0, -float("inf")]
        # A softmax over the block is a distribution over the candidates.
        probs = torch.softmax(padded, dim=-1)
        assert probs[1, 1] == 0.0
        assert torch.allclose(probs.sum(1), torch.ones(2))


class TestTD3BC:
    def _model(self, alpha=0.01):
        torch.manual_seed(0)
        return TD3BC(alpha=alpha, device=torch.device("cpu"))

    def test_tree_q_targets_kappa_mixing(self):
        # Both children unvisited (done): Q_i = reward_i / r_max exactly —
        # no network output enters, so the mix is hand-checkable.
        model = self._model()
        model.r_max = 2.0
        batch = SorrelBatch([_synthetic_sample(reward1=1.0, reward2=2.0)])
        q = model.tree_q_targets(batch)
        # reward1 < reward2 -> kappa on child 1.
        assert q.item() == pytest.approx(0.8 * (1.0 / 2.0) + 0.2 * (2.0 / 2.0))

    def test_actor_loss_formula(self):
        model = self._model()
        model.r_max = 1.0
        batch = SorrelBatch([_synthetic_sample(), _synthetic_sample(visited1=True)])
        model.update_critic(batch)  # bring the critic target up to date
        # The official loss: -lmbda*(probs*Q).sum(1).mean() + CE, with
        # lmbda = alpha/|Q(a)|.mean() and Q from the *target* critic.
        model.actor.train()
        logits = padded_logits(model.actor(*batch.states.forward_args()), batch.states.candidates)
        with torch.no_grad():
            q = padded_logits(
                model.critic_target(*batch.states.forward_args()), batch.states.candidates
            )
            q = q.masked_fill(q == -float("inf"), -1e6)
            lmbda = model.alpha / q.gather(1, batch.actions.unsqueeze(-1)).abs().mean()
        import torch.nn.functional as F

        expected = (
            -(lmbda * (torch.softmax(logits, dim=-1) * q).sum(1).mean())
            + F.cross_entropy(logits, batch.actions)
        )
        assert model.update_actor(batch) == pytest.approx(float(expected.detach()))

    def test_prenorm_freezes_only_the_actor_and_sets_r_max(self):
        model = self._model()
        samples = [_synthetic_sample(reward1=1.0, reward2=3.0) for _ in range(11)]
        n_layers = model.prenorm(samples[::10], np.random.default_rng(0))
        assert n_layers > 0
        assert model.r_max == pytest.approx(3.0)
        # The actor's PreNorm statistics are frozen; the critic's layers were
        # never warmed, so they stay identity (shift 0, scale 1).
        from l2b_bench.methods.gasse.model import PreNormLayer

        for module in model.actor.modules():
            if isinstance(module, PreNormLayer):
                assert not module.waiting_updates
        critic_norms = [m for m in model.critic.modules() if isinstance(m, PreNormLayer)]
        assert critic_norms and all(not m.waiting_updates for m in critic_norms)
        assert all(torch.equal(m.scale, torch.ones_like(m.scale)) for m in critic_norms)

    def test_checkpoint_roundtrip(self, tmp_path):
        samples = [_synthetic_sample(reward1=0.5, reward2=1.0) for _ in range(4)]
        path = tmp_path / "td3bc.pt"
        model = train_td3bc(samples, path, epochs=1, epoch_size=1, seed=0, log=lambda *_: None)
        assert path.exists()
        actor = load_actor(path)
        assert isinstance(actor, GNNPolicy)
        agent = PPOSIL.from_checkpoint(path)
        assert agent.r_max == pytest.approx(model.r_max)
        state = torch.load(path, weights_only=True)
        assert state["epochs"] == 1


class TestPPOSIL:
    def _agent(self, r_max=2.0):
        torch.manual_seed(0)
        agent = PPOSIL(GNNPolicy(), GNNPolicy(), r_max, device=torch.device("cpu"))
        return agent

    def test_advantages_are_globally_normalized(self):
        agent = self._agent()
        batch = SorrelBatch(
            [_synthetic_sample(returns=r) for r in (0.0, 1.0, 2.0, 3.0, 4.0)]
        )
        advantages = agent.compute_advantages([batch])
        stacked = torch.cat(advantages)
        assert stacked.mean() == pytest.approx(0.0, abs=1e-6)
        assert stacked.std() == pytest.approx(1.0, abs=1e-3)

    def test_ppo_update_runs(self):
        agent = self._agent()
        batch = SorrelBatch([_synthetic_sample(returns=1.0), _synthetic_sample(returns=2.0)])
        stats = agent.ppo_update([batch])
        assert np.isfinite(stats["actor_loss"]) and np.isfinite(stats["critic_loss"])

    def test_sil_weights_are_clipped_advantages(self):
        agent = self._agent()
        # returns/r_max - value, clipped at 0; value comes from the critic's
        # max over the (single) candidate's logits.
        batch = SorrelBatch([_synthetic_sample(returns=100.0), _synthetic_sample(returns=0.0)])
        priorities = agent.sil_update([batch])
        assert len(priorities) == 1
        assert (priorities[0] >= 0).all()
        assert priorities[0][0] > 0  # the high-return sample imitates itself

    def test_batches_stay_on_the_cpu_until_consumed(self):
        """Regression (campaign OOM): ``batches_of`` used to move every
        batch to the device eagerly — one epoch's SIL buffer exhausted
        the campaign card's 22 GiB. Batches are collated on the CPU, and
        ``SorrelBatch.to`` returns a moved copy without pinning the
        original (``compute_advantages`` iterating the batch list must
        not leave the states behind on the device)."""
        from l2b_bench.methods.sorrel.train import batches_of, get_device

        samples = [_synthetic_sample(returns=1.0) for _ in range(5)]
        batches = batches_of(samples, 2, np.random.default_rng(0))
        for batch in batches:
            assert batch.states.constraint_features.device.type == "cpu"
        moved = batches[0].to(get_device())
        assert moved is not batches[0]
        assert batches[0].states.constraint_features.device.type == "cpu"
        assert moved.states.constraint_features.device == get_device()

    def test_sil_buffer_keeps_the_best_trajectory(self):
        buffer = SILBuffer()
        rng = np.random.default_rng(0)
        good = [_synthetic_sample(returns=1.0) for _ in range(5)]
        bad = [_synthetic_sample(returns=0.0) for _ in range(5)]
        buffer.add("a", bad, score=-10.0)
        buffer.add("a", good, score=-3.0)
        buffer.add("b", bad, score=-7.0)
        dataset = buffer.dataset(rng)
        # Only the best trajectory per instance, at most 100 samples each.
        assert len(dataset) == 10
        assert all(sample is not None for sample in dataset)


class TestOnlineStage:
    def test_rollout_records_returns_and_log_probs(self):
        env = make_env(_instances(1), time_limit=60.0, seed=0)
        torch.manual_seed(0)
        actor = GNNPolicy()
        try:
            samples, nb_nodes = run_episode(env, actor, torch.device("cpu"), length_limit=25)
        finally:
            env.close()
        assert nb_nodes >= 1
        assert samples, "the rollout must produce transitions"
        for sample in samples:
            assert sample.log_prob is not None
            assert sample.returns is not None

    def test_train_online_end_to_end(self, tmp_path):
        # Two VHB samples suffice for the offline stage; the online budget of
        # two episodes is consumed by the greedy warmup rollout, which must
        # still produce one (SIL-only) epoch and a checkpoint.
        samples = [_synthetic_sample(reward1=0.5, reward2=1.0) for _ in range(4)]
        td3bc = tmp_path / "td3bc.pt"
        train_td3bc(samples, td3bc, epochs=1, epoch_size=1, seed=0, log=lambda *_: None)
        checkpoint = tmp_path / "ppo.pt"
        train_online(
            _instances(2),
            td3bc,
            checkpoint,
            episodes=2,
            time_limit=60.0,
            seed=0,
            pre_rollouts=0,
            log=lambda *_: None,
        )
        assert checkpoint.exists()
        actor = load_actor(checkpoint)
        assert isinstance(actor, GNNPolicy)
        # The per-epoch numbered snapshot (epoch 1) is directly loadable.
        snapshot = tmp_path / "ppo.pt.p0001"
        assert snapshot.exists()
        assert isinstance(load_actor(snapshot), GNNPolicy)


class TestPolicyBridge:
    def test_method_declares_the_sorrel_fidelity(self, tmp_path):
        samples = [_synthetic_sample() for _ in range(2)]
        path = tmp_path / "actor.pt"
        train_td3bc(samples, path, epochs=1, epoch_size=1, seed=0, log=lambda *_: None)
        method = sorrel_method(path)
        assert method.name == "sorrel"
        assert method.fidelity == "sorrel"
        policy = method.policy()
        action = policy.select_action(_dummy_observation(), np.array([0]))
        assert action == 0
