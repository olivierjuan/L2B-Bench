"""End-to-end tests of the BBMDP 2025 (Strang et al., NeurIPS 2025) port.

The instances are tiny set covers — the point is that the whole regime runs
on the harness (vanilla-MDP episodes under the DFS protocol → retrospective
k-step returns over the tracked tree → prioritized replay → DDQN with the
HL-Gauss value head), not that the agent learns anything. The math tests pin
the official formulas (the RewardAgent BFS, the erf-smoothed bin
distribution, the DDQN multi-children target, the episode gap processing)
against hand computation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.bbmdp import (
    BBMDPTransition,
    TreeDQNAgent,
    bbmdp_returns,
    make_env,
    process_episode,
    run_episode,
)
from l2b_bench.methods.bbmdp.model import (
    NUM_BINS,
    StateBatch,
    TreeHLGaussLoss,
    pad_per_sample,
)
from l2b_bench.methods.bbmdp.policy import BBMDPPolicy
from l2b_bench.methods.bbmdp.replay import PrioritizedReplayBuffer
from l2b_bench.methods.bbmdp.train import load_dqn


def _instances(n, rows=120, cols=150, seed=7):
    """Tiny set covers, one fresh model per episode."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.hideOutput()
        yield model


def _dummy_observation(n_vars=2):
    """A one-constraint bipartite graph over ``n_vars`` variables, linked by
    one edge per variable — the 43-feature retro shape."""
    return SimpleNamespace(
        row_features=np.zeros((1, 5), dtype=np.float32),
        edge_features=SimpleNamespace(
            indices=np.array([np.zeros(n_vars, dtype=np.int64), np.arange(n_vars)], dtype=np.int64),
            values=np.ones(n_vars, dtype=np.float32),
        ),
        variable_features=np.zeros((n_vars, 43), dtype=np.float32),
    )


def _transition(action=0, reward=-1.0, n_step_reward=-3.0, children=0, n_step_children=0):
    """A hand-controlled vanilla-MDP transition over dummy graphs;
    ``children``/``n_step_children`` count the dummy child states."""
    return BBMDPTransition(
        observation=_dummy_observation(),
        action_set=np.arange(2),
        action=int(action),
        action_idx=int(action) % 2,
        depth=0,
        reward=reward,
        next_states=[(_dummy_observation(), np.arange(2)) for _ in range(children)],
        done=children == 0,
        n_step_reward=n_step_reward,
        n_step_next_states=[(_dummy_observation(), np.arange(2)) for _ in range(n_step_children)],
        n_step_done=n_step_children == 0,
    )


class TestRewardAgentBFS:
    """The official ``RewardAgent.extract`` BFS, hand-computed."""

    def test_chain_gamma_one(self):
        # 0 -> 1 -> 2 -> 3: the 1-step child is the successor, the 3-step
        # frontier the node three pops away (reward -1 per pop).
        returns = bbmdp_returns([[1], [2], [3], []], n_step=3, gamma=1.0)
        assert returns[0] == {
            "next_states": [1],
            "reward": -1.0,
            "n_step_next_states": [3],
            "n_step_reward": -3.0,
        }
        # A leaf pops itself then the frontier empties: the n-step captures
        # keep the last values (reward -1, empty frontier).
        assert returns[3] == {
            "next_states": [],
            "reward": -1.0,
            "n_step_next_states": [],
            "n_step_reward": -1.0,
        }

    def test_early_empty_frontier_keeps_the_last_capture(self):
        # Chain of two with n_step 3: only two pops ever happen, so the
        # n-step reward freezes at -2 after the frontier empties.
        returns = bbmdp_returns([[1], []], n_step=3, gamma=1.0)
        assert returns[0]["n_step_reward"] == -2.0
        assert returns[0]["n_step_next_states"] == []

    def test_branching_gamma_half(self):
        # 0 branches into 1 and 3, 1 branches into 2; gamma 0.5, n_step 2:
        # pops are 0 then 1, so the n-step reward is -(0.5^0 + 0.5^1) and
        # the frontier [2, 3] (sorted by visit order).
        returns = bbmdp_returns([[1, 3], [2], [], []], n_step=2, gamma=0.5)
        assert returns[0]["reward"] == -1.0
        assert returns[0]["next_states"] == [1, 3]
        assert returns[0]["n_step_reward"] == pytest.approx(-1.5)
        assert returns[0]["n_step_next_states"] == [2, 3]

    def test_frontier_is_sorted_by_visit_order(self):
        # Node 0's children are visited at steps 5 and 2: the frontier after
        # one pop orders them 2, 5 whatever their discovery order.
        children = [[] for _ in range(6)]
        children[0] = [5, 2]
        returns = bbmdp_returns(children, n_step=3, gamma=1.0)
        assert returns[0]["next_states"] == [2, 5]


class TestTreeHLGaussLoss:
    """The categorical head over log2(-z): 18 bins on [-1.5, 16.5]."""

    @pytest.fixture()
    def loss(self):
        return TreeHLGaussLoss()

    def test_support_and_sigma(self, loss):
        assert torch.allclose(loss.support, torch.linspace(-1.5, 16.5, NUM_BINS + 1))
        # sigma is expressed in bin-width units (bin width 1 here).
        assert loss.sigma == pytest.approx(0.75)
        assert loss.centers.shape == (NUM_BINS,)

    def test_probs_are_a_distribution_around_the_target(self, loss):
        probs = loss.transform_to_probs(torch.tensor([0.0, 8.0, 16.0]))
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5)
        assert (probs >= 0).all()
        # The mass concentrates on the bins around each target.
        assert probs[0, :3].sum() > 0.5
        assert probs[1, 7:10].sum() > 0.5
        assert probs[2, 15:].sum() > 0.5

    def test_decode_round_trips_inside_the_support(self, loss):
        target = torch.tensor([1.0, 8.3])
        decoded = loss.transform_from_probs(loss.transform_to_probs(target))
        assert torch.allclose(decoded, target, atol=0.1)


class TestPadPerSample:
    def test_logits_blocks(self):
        values = torch.arange(5.0).reshape(5, 1).expand(5, 3)
        padded = pad_per_sample(values, torch.tensor([2, 3]))
        assert padded.shape == (2, 3, 3)
        assert torch.allclose(padded[0, 2], torch.full((3,), -1e9))
        assert torch.allclose(padded[1, :3, 0], torch.tensor([2.0, 3.0, 4.0]))

    def test_value_rows(self):
        # The decoded 1-D values take the official's other padding branch.
        values = torch.tensor([10.0, 11.0, 12.0, 13.0])
        padded = pad_per_sample(values, torch.tensor([3, 1]))
        assert padded.shape == (2, 3)
        assert torch.allclose(padded, torch.tensor([[10.0, 11.0, 12.0], [13.0, -1e9, -1e9]]))


class TestProcessEpisode:
    """The official ``process_episode_into_bbmdp`` checks."""

    def _lists(self, n):
        return (
            [_dummy_observation() for _ in range(n)],
            [np.arange(2) for _ in range(n)],
            list(range(n)),
            list(range(n)),
            [0] + [1] * (n - 1),
        )

    def _tree_returns(self, n):
        return bbmdp_returns([[i + 1] if i + 1 < n else [] for i in range(n)])

    def test_aligned_episode(self):
        observations, action_sets, actions, action_idxs, depths = self._lists(3)
        transitions = process_episode(
            observations, action_sets, actions, action_idxs, depths, self._tree_returns(3), 3
        )
        assert len(transitions) == 3
        # The chain's leaves are done, the others are not.
        assert [t.done for t in transitions] == [False, False, True]
        assert transitions[0].reward == -1.0

    def test_gap_one_drops_the_extra_tree_node(self):
        observations, action_sets, actions, action_idxs, depths = self._lists(3)
        tree_returns = self._tree_returns(4)
        # The extra node (position 3) is a child of node 2: it must not
        # survive as a next state.
        tree_returns[2] = {
            "next_states": [3],
            "reward": -1.0,
            "n_step_next_states": [],
            "n_step_reward": -2.0,
        }
        transitions = process_episode(
            observations, action_sets, actions, action_idxs, depths, tree_returns, 3
        )
        assert len(transitions) == 3
        assert transitions[2].next_states == []
        assert transitions[2].done

    def test_gap_two_rejects(self):
        observations, action_sets, actions, action_idxs, depths = self._lists(3)
        assert (
            process_episode(
                observations, action_sets, actions, action_idxs, depths, self._tree_returns(5), 3
            )
            is None
        )

    def test_short_episode_rejects(self):
        observations, action_sets, actions, action_idxs, depths = self._lists(1)
        assert (
            process_episode(
                observations, action_sets, actions, action_idxs, depths, self._tree_returns(1), 1
            )
            is None
        )

    def test_double_root_drops_the_first_transition(self):
        observations, action_sets, actions, action_idxs, depths = self._lists(3)
        depths = [0, 0, 1]  # two consecutive decisions at the root
        transitions = process_episode(
            observations, action_sets, actions, action_idxs, depths, self._tree_returns(3), 3
        )
        assert len(transitions) == 2
        assert [t.action for t in transitions] == [1, 2]


class TestStateBatch:
    def test_collation_offsets_candidates(self):
        obs1, obs2 = _dummy_observation(2), _dummy_observation(3)
        batch = StateBatch([(obs1, np.array([0, 1])), (obs2, np.array([0, 2]))])
        assert batch.nb_variables.tolist() == [2, 3]
        assert batch.nb_candidates.tolist() == [2, 2]
        # The second sample's candidates carry its variable offset.
        assert batch.candidates[1].tolist() == [2, 4]
        assert batch.edge_index.shape[1] == 2 + 3


class TestPrioritizedReplay:
    def test_fill_state_is_a_zeroed_copy(self):
        memory = PrioritizedReplayBuffer(4, batch_size=1, alpha=0.6)
        transition = _transition(children=1)
        memory.store(transition)
        next_states, mask = memory._preprocess_next_states([[], [transition.next_states[0]]])
        # The sample without children gets the fill state (zero features) in
        # its padded slot, masked out.
        fill_obs = next_states[0][0][0]
        assert (np.asarray(fill_obs.variable_features) == 0).all()
        assert (np.asarray(fill_obs.row_features) == 0).all()
        assert mask.tolist() == [[0.0, 1.0]]
        # The stored transition keeps its real observation object (the
        # official mutates state_buf[0] in place — the documented deviation).
        assert transition.observation is not None

    def test_weights_are_normalized_and_priorities_update(self):
        import random

        memory = PrioritizedReplayBuffer(8, batch_size=2, alpha=0.6)
        for i in range(4):
            memory.store(_transition(action=i))
        batch = memory.sample_batch(beta=0.4, rng=random.Random(0))
        assert len(batch["transitions"]) == 2
        assert (batch["weights"] <= 1.0 + 1e-9).all()
        # A spiked priority raises the max priority.
        memory.update_priorities([0], [100.0])
        assert memory.max_priority == 100.0


class _ConstantNet:
    """A stub scoring every variable with the same logits row."""

    def __init__(self, row):
        self.row = row

    def __call__(self, constraint_features, edge_index, edge_attr, variable_features):
        return self.row.expand(variable_features.shape[0], -1)


class TestTreeDQNLoss:
    """The DDQN multi-children target and the HL-Gauss element-wise loss."""

    @pytest.fixture()
    def agent(self):
        torch.manual_seed(0)
        return TreeDQNAgent(device=torch.device("cpu"))

    def test_no_children_targets_the_reward(self, agent):
        # Every transition childless: the target is the reward alone, so the
        # loss is the HL-Gauss CE at log2(-reward).
        batch = {
            "transitions": [_transition(action=0, reward=-2.0), _transition(action=1, reward=-4.0)],
            "next_states": None,
            "children_mask": None,
            "n_step_next_states": None,
            "n_step_children_mask": None,
        }
        loss = agent.compute_tree_dqn_loss(batch, gamma=1.0, n_step=False)
        assert loss.shape == (2,)  # one element-wise loss per sample
        assert (loss > 0).all()
        # The hand-computed target: log2(2 + 1e-8) and log2(4 + 1e-8).
        states = StateBatch([t.state for t in batch["transitions"]])
        logits = pad_per_sample(agent.dqn(*states.forward_args()), states.nb_variables)
        at_action = logits[torch.arange(2), torch.tensor([0, 1]), :]
        expected = agent.loss_fn(
            at_action,
            torch.log2(-torch.tensor([[-2.0], [-4.0]]) + 1e-8).squeeze(-1),
        )
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_children_are_summed_and_masked(self, agent):
        # Two children on the first sample, none (done) on the second: with
        # the networks stubbed to constant logits, the target is
        # reward + gamma * (sum of the children's decoded Q values) — and
        # the fill state's slot contributes nothing.
        agent.dqn = _ConstantNet(torch.full((1, NUM_BINS), 0.1))
        agent.dqn_target = _ConstantNet(torch.full((1, NUM_BINS), 0.1))
        first = _transition(action=0, reward=-1.0, children=2)
        second = _transition(action=1, reward=-5.0, children=0)
        memory = PrioritizedReplayBuffer(4)
        memory.store(first)
        per_slot, mask = memory._preprocess_next_states([first.next_states, second.next_states])
        batch = {
            "transitions": [first, second],
            "next_states": per_slot,
            "children_mask": mask,
            "n_step_next_states": None,
            "n_step_children_mask": None,
        }
        loss = agent.compute_tree_dqn_loss(batch, gamma=1.0, n_step=False)
        assert loss.shape == (2,)
        # The constant-logit Q value, decoded once.
        q = float(agent.to_values_from_logits(torch.full((1, NUM_BINS), 0.1)))
        targets = torch.log2(-torch.tensor([[-1.0 + 2 * q], [-5.0]]) + 1e-8).squeeze(-1)
        expected = agent.loss_fn(torch.full((2, NUM_BINS), 0.1), targets)
        assert torch.allclose(loss, expected, atol=1e-5)

    def test_gradients_flow_to_the_online_net(self, agent):
        first = _transition(action=0, reward=-1.0, children=1)
        batch = {
            "transitions": [first],
            "next_states": None,
            "children_mask": None,
            "n_step_next_states": None,
            "n_step_children_mask": None,
        }
        loss = agent.compute_tree_dqn_loss(batch, gamma=1.0, n_step=False)
        loss.sum().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.dqn.parameters())


class TestEpisode:
    """The full regime on tiny set covers (harness + SCIP)."""

    def test_env_declares_the_official_protocol(self):
        env = make_env(iter([]))
        assert env.scip_params["separating/maxrounds"] == 0
        assert env.scip_params["nodeselection/dfs/stdpriority"] == 300000
        assert env.scip_params["nodeselection/dfs/memsavepriority"] == 300000
        assert env.scip_params["estimation/restarts/restartpolicy"] == "n"

    def test_run_episode_builds_aligned_transitions(self):
        env = make_env(_instances(8), time_limit=20.0, seed=0)
        rng = np.random.default_rng(0)

        def policy(obs, candidates):
            idx = int(rng.integers(len(candidates)))
            return int(candidates[idx]), idx

        transitions, n_nodes, _status = run_episode(env, policy)
        attempts = 1
        while transitions is None and attempts < 8:
            transitions, n_nodes, _status = run_episode(env, policy)
            attempts += 1
        assert transitions is not None, "every tiny episode was rejected"
        assert n_nodes == len(transitions)
        assert transitions[0].depth == 0
        # The vanilla-MDP rewards: -1 per pop (gamma 1), leaves done.
        assert {t.reward for t in transitions} <= {-1.0}
        assert {t.n_step_reward for t in transitions} <= {-1.0, -2.0, -3.0}
        assert any(t.done for t in transitions)
        assert all(t.action in t.action_set for t in transitions)

    def test_train_smoke_end_to_end(self, tmp_path, monkeypatch):
        # The whole loop on toy scale: episodes, PER sampling, both losses,
        # the schedules, the checkpoint — then the greedy policy bridge.
        # (The package exports the train *function*, which shadows the
        # module attribute — importlib gets the module for monkeypatching.)
        import importlib

        bbmdp_train = importlib.import_module("l2b_bench.methods.bbmdp.train")

        monkeypatch.setattr(bbmdp_train, "MEMORY_MIN_SIZE", 5)
        monkeypatch.setattr(bbmdp_train, "BATCH_SIZE", 4)
        monkeypatch.setattr(bbmdp_train, "SAVE_EVERY_EPISODES", 1)
        checkpoint = tmp_path / "bbmdp.pt"
        gradient_steps = bbmdp_train.train(
            _instances(12, rows=200, cols=250),
            checkpoint,
            episodes=3,
            n_epochs=3,
            time_limit=15.0,
            seed=0,
        )
        assert checkpoint.exists()
        assert gradient_steps > 0  # at least one episode escaped the presolve
        policy = BBMDPPolicy.from_checkpoint(checkpoint)
        assert isinstance(load_dqn(checkpoint), type(policy.model))
        env = make_env(_instances(4), time_limit=15.0, seed=1)
        obs, action_set, _r, done, _info = env.reset()
        attempts = 0
        while done and attempts < 3:
            obs, action_set, _r, done, _info = env.reset()
            attempts += 1
        assert not done
        action = policy.select_action(obs, action_set)
        assert action in np.asarray(action_set)
