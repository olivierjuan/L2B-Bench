"""End-to-end tests of the RetroBranching (Parsonson, Laterza & Barrett 2022) port.

The instances are tiny set covers — the point is that the protocol runs on
the harness (retro observation → DQN agent in the environment → sub-tree
episode decomposition → replay → 1-step Double-DQN updates), not that the
agent learns anything.
"""

from dataclasses import replace

import numpy as np
import pytest
import torch
from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.retrobranching import (
    ReplayBuffer,
    RetroGCN,
    Transition,
    add_episode_transitions,
    dqn_update,
    load_checkpoint,
    make_env,
    retrobranching_method,
    run_episode,
    save_checkpoint,
    train,
)
from l2b_bench.methods.retrobranching.replay import GraphState, graph_state
from l2b_bench.methods.retrobranching.train import (
    BATCH_SIZE,
    SOFT_UPDATE_TAU,
    TransitionBatch,
)


def _instances(n, rows=150, cols=200, seed=7):
    """Tiny set covers, one fresh model per episode."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.hideOutput()
        yield model


@pytest.fixture(scope="module")
def episodes():
    """Three agent episodes (untrained network, epsilon 0.025) + their buffer."""
    device = torch.device("cpu")
    env = make_env(_instances(3), time_limit=10.0, seed=0)
    model = RetroGCN().to(device)
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(10_000)
    all_experiences = []
    for _ in range(3):
        experiences, env = run_episode(env, model, 0.025, rng, device)
        add_episode_transitions(buffer, experiences, env)
        all_experiences.append(experiences)
    return all_experiences, buffer


class TestModel:
    def test_forward_shape_and_range(self, episodes):
        """One Q value per variable; at initialisation the near-zero head
        outputs pass through -LeakyReLU almost unchanged (the positive side
        is clipped to 0, the negative side scaled by 0.01)."""
        model = RetroGCN()
        model.eval()
        episode = next(e for e in episodes[0] if e)  # the first non pre-solved episode
        state = episode[0]["prev_state"]
        with torch.no_grad():
            q = model(
                state.constraint_features,
                state.edge_index,
                state.edge_attr,
                state.variable_features,
            )
        assert q.shape == (state.variable_features.shape[0],)
        assert torch.isfinite(q).all()
        assert q.abs().max() < 1e-4

    def test_initialisation(self):
        """The official init: Linear weights ~ N(0, 0.01), biases zero."""
        model = RetroGCN()
        weights = [m.weight for m in model.modules() if isinstance(m, torch.nn.Linear)]
        biases = [m.bias for m in model.modules() if isinstance(m, torch.nn.Linear) and m.bias is not None]
        assert len(weights) >= 10
        for w in weights:
            assert w.std() < 0.05
            assert w.mean().abs() < 0.01
        for b in biases:
            assert torch.all(b == 0)


class TestEpisode:
    def test_experiences_are_well_formed(self, episodes):
        all_experiences, _ = episodes
        for episode in all_experiences:
            for experience in episode:
                assert set(experience) == {"prev_state", "action", "state", "done"}
                assert experience["action"] in experience["prev_state"].candidates.tolist()

    def test_transitions_partition_the_decisions(self, episodes):
        """The buffer holds exactly the decisions the sub-tree decomposition
        kept (CHANGE #2 can drop an episode's final transition), with the
        binary retrospective rewards in {0, -1}."""
        all_experiences, buffer = episodes
        total = sum(len(episode) for episode in all_experiences)
        assert len(buffer) > 0
        assert len(buffer) <= total
        for transition in buffer.buffer[: len(buffer)]:
            assert transition.reward in (0.0, -1.0)
            assert transition.action in transition.state.candidates.tolist()

    def test_pre_solved_episode_adds_nothing(self):
        """A pre-solved instance yields zero decisions: nothing to learn
        (the official code would crash indexing its [{0: 0}] fallback)."""
        env = make_env(_instances(1), time_limit=10.0, seed=0)
        device = torch.device("cpu")
        model = RetroGCN()
        experiences, env = run_episode(env, model, 0.025, np.random.default_rng(0), device)
        if not experiences:
            buffer = ReplayBuffer(10)
            assert add_episode_transitions(buffer, experiences, env) == 0
            assert len(buffer) == 0


class TestDQN:
    @pytest.fixture(scope="class")
    def synthetic_buffer(self):
        """A buffer of synthetic transitions — enough to sample batch 128."""
        rng = np.random.default_rng(1)
        buffer = ReplayBuffer(500)
        for _ in range(200):
            buffer.append(_synthetic_transition(rng))
        return buffer

    def test_batch_collation_offsets(self, synthetic_buffer):
        """The collated graph is one disconnected graph: edge indices and
        candidates shifted per sample, action_idx pointing at each action's
        position inside its own sample's candidate slice."""
        transitions = list(synthetic_buffer.buffer[:4])
        batch = TransitionBatch(transitions)
        expected_action_idx = []
        expected_candidates = []
        var_offset = 0
        cand_offset = 0
        for transition in transitions:
            state = transition.state
            # candidates carry the per-sample variable offset...
            expected_candidates.extend((state.candidates + var_offset).tolist())
            position = int((state.candidates == transition.action).nonzero()[0])
            expected_action_idx.append(cand_offset + position)
            var_offset += state.variable_features.shape[0]
            cand_offset += len(state.candidates)
        assert batch.candidates.tolist() == expected_candidates
        assert batch.action_idx.tolist() == expected_action_idx
        assert batch.num_candidates.tolist() == [len(t.state.candidates) for t in transitions]
        assert batch.next_num_candidates.tolist() == [len(t.next_state.candidates) for t in transitions]
        # the collated edge index stays inside the concatenated graph
        assert batch.edge_index[0].max() < batch.constraint_features.shape[0]
        assert batch.edge_index[1].max() < batch.variable_features.shape[0]

    def test_update_is_finite_and_moves_the_networks(self, synthetic_buffer):
        model = RetroGCN()
        target = RetroGCN()
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = {n: p.clone() for n, p in model.named_parameters()}
        target_before = {n: p.clone() for n, p in target.named_parameters()}
        loss = dqn_update(model, target, synthetic_buffer, optimizer, np.random.default_rng(2), torch.device("cpu"))
        assert np.isfinite(loss)
        moved = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
        assert len(moved) > 0
        # The soft Polyak update nudges the target toward the online network —
        # every parameter the gradient moved, by at most tau-sized steps. (The
        # LayerNorm(1) of the edge embedding sees an exactly zero gradient:
        # normalising a single element always yields 0.)
        target_moved = 0
        for name, param in target.named_parameters():
            delta = (param - target_before[name]).abs().max()
            assert delta <= SOFT_UPDATE_TAU + 1e-8
            if delta > 0:
                target_moved += 1
        assert target_moved > 0

    def test_terminal_transitions_aim_at_the_reward(self, synthetic_buffer):
        """With done=True everywhere the bootstrap vanishes: the TD target is
        the reward itself, so the loss is MSE(q_action, rewards)."""
        transitions = [replace(t, done=True) for t in synthetic_buffer.buffer[:BATCH_SIZE]]
        buffer = ReplayBuffer(BATCH_SIZE)
        for t in transitions:
            buffer.append(t)
        model = RetroGCN()
        target = RetroGCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # frozen
        # The same rng draws the same sample as inside dqn_update.
        loss = dqn_update(model, target, buffer, optimizer, np.random.default_rng(3), torch.device("cpu"))
        sampled = buffer.sample(BATCH_SIZE, np.random.default_rng(3))
        batch = TransitionBatch(sampled)
        with torch.no_grad():
            q = model(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
        expected = torch.nn.functional.mse_loss(q[batch.action_idx], batch.rewards)
        assert loss == pytest.approx(float(expected), rel=1e-5)


class TestTrain:
    def test_train_saves_a_checkpoint(self, tmp_path):
        """A handful of episodes with a tiny buffer floor: the full loop
        (fill → learn while acting → save) runs and the checkpoint loads."""
        checkpoint = tmp_path / "retrobranching.pt"
        model = train(
            _instances(6, seed=11),
            checkpoint,
            episodes=6,
            buffer_min_length=10,
            time_limit=10.0,
            seed=0,
            device=torch.device("cpu"),
            log=lambda *_: None,
        )
        assert checkpoint.exists()
        loaded = load_checkpoint(checkpoint, device=torch.device("cpu"))
        assert isinstance(loaded, RetroGCN)
        for a, b in zip(loaded.parameters(), model.parameters()):
            assert torch.equal(a, b)


class TestPolicy:
    def test_retrobranching_method_is_runner_ready(self, tmp_path):
        path = tmp_path / "retrobranching.pt"
        save_checkpoint(RetroGCN(), path)
        method = retrobranching_method(path)
        assert method.name == "retrobranching"
        assert method.fidelity == "retrobranching"
        policy = method.policy()
        env = make_env(_instances(5, seed=11), time_limit=10.0, seed=0)
        obs, action_set, _, _, _ = env.reset()
        while obs is None:  # skip pre-solved instances
            obs, action_set, _, _, _ = env.reset()
        action = policy.select_action(obs, action_set)
        assert action in np.asarray(action_set).tolist()

    def test_greedy_action_picks_the_highest_q(self):
        """select_action must be the greedy action over the candidates."""
        from l2b_bench.methods.retrobranching.policy import RetroPolicy

        model = RetroGCN()
        retro = RetroPolicy(model)
        state = graph_state(*_synthetic_obs(np.random.default_rng(4)))
        with torch.no_grad():
            q = model(
                state.constraint_features,
                state.edge_index,
                state.edge_attr,
                state.variable_features,
            )
        best = int(state.candidates[q[state.candidates].argmax()])
        assert retro.select_action(_synthetic_obs_of(state), state.candidates.tolist()) == best


# --- synthetic helpers -----------------------------------------------------


def _synthetic_graph(rng, n_cons=4, n_vars=6, n_cands=3):
    n_edges = rng.integers(6, 12)
    edge_index = torch.stack(
        [
            torch.tensor(rng.integers(0, n_cons, n_edges), dtype=torch.long),
            torch.tensor(rng.integers(0, n_vars, n_edges), dtype=torch.long),
        ]
    )
    return GraphState(
        constraint_features=torch.tensor(rng.normal(size=(n_cons, 5)), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(rng.normal(size=(n_edges, 1)), dtype=torch.float32),
        variable_features=torch.tensor(rng.normal(size=(n_vars, 43)), dtype=torch.float32),
        candidates=torch.tensor(np.sort(rng.choice(n_vars, size=n_cands, replace=False)), dtype=torch.long),
    )


def _synthetic_transition(rng):
    state = _synthetic_graph(rng)
    next_state = _synthetic_graph(rng)
    action = int(state.candidates[rng.integers(len(state.candidates))])
    return Transition(
        state=state,
        action=action,
        reward=float(rng.choice([0.0, -1.0])),
        done=bool(rng.random() < 0.3),
        next_state=next_state,
    )


def _synthetic_obs(rng):
    """A raw observation compatible with ``graph_state``."""
    n_cons, n_vars, n_edges = 4, 6, 10
    row = rng.normal(size=(n_cons, 5))
    values = rng.normal(size=n_edges)
    indices = np.stack(
        [rng.integers(0, n_cons, n_edges), rng.integers(0, n_vars, n_edges)]
    )
    variable_features = rng.normal(size=(n_vars, 43))
    candidates = np.sort(rng.choice(n_vars, size=3, replace=False))

    class _Edges:
        pass

    class _Obs:
        pass

    edges = _Edges()
    edges.indices = indices
    edges.values = values
    obs = _Obs()
    obs.row_features = row
    obs.edge_features = edges
    obs.variable_features = variable_features
    return obs, candidates


def _synthetic_obs_of(state: GraphState):
    """Rebuild a raw observation from a GraphState (for select_action)."""

    class _Edges:
        pass

    class _Obs:
        pass

    edges = _Edges()
    edges.indices = state.edge_index.numpy()
    edges.values = state.edge_attr.numpy().reshape(-1)
    obs = _Obs()
    obs.row_features = state.constraint_features.numpy()
    obs.edge_features = edges
    obs.variable_features = state.variable_features.numpy()
    return obs
