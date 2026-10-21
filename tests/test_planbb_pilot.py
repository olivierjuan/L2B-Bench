"""End-to-end tests of the PlanB&B (Strang et al., AAAI 2026) port.

The instances are tiny set covers — the point is that the whole regime runs
on the harness (Gumbel MCTS self-play over the tree MDP → ghost-completed
state windows → the four MuZero losses → the greedy policy bridge), not that
the agent learns anything. The math tests pin the official formulas (the
HL-Gauss decode with its trailing dim, the cminimax normalize, the Gumbel
Top-k root selection, the td-capped BFS reward sums, the ghost-child
completion, the warmup/decay schedules) against hand computation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gyozas.instances import SetCoverGenerator
from torch_geometric.data import Batch

from l2b_bench.methods.planbb import (
    MCTS,
    PlanBBNet,
    PlanBBPolicy,
    PlanBBTrajectory,
    make_env,
    run_selfplay_episode,
    select_action,
)
from l2b_bench.methods.planbb.collect import SearchTreeAdapter, zero_obs
from l2b_bench.methods.planbb.model import (
    NUM_BINS,
    PlanBBObs,
    TreeHLGaussLoss,
    format_batch_actions,
    geometric_batch,
    get_idx,
    get_target_index,
    pad_array,
    pad_tensor,
)
from l2b_bench.methods.planbb.search import (
    MinMaxStats,
    Node,
    select_action as mcts_select_action,
)
from l2b_bench.methods.planbb.train import (
    adjust_lr,
    consist_loss_func,
    visit_softmax_temperature,
)


def _instances(n, rows=100, cols=120, seed=7):
    """Tiny set covers, one fresh model per episode."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.hideOutput()
        yield model


def _dummy_observation(n_vars=2):
    """A one-constraint bipartite graph over ``n_vars`` variables, linked by
    one edge per variable — the 43-feature planbb shape."""
    return SimpleNamespace(
        row_features=np.zeros((1, 5), dtype=np.float32),
        edge_features=SimpleNamespace(
            indices=np.array([np.zeros(n_vars, dtype=np.int64), np.arange(n_vars)], dtype=np.int64),
            values=np.ones(n_vars, dtype=np.float32),
        ),
        variable_features=np.zeros((n_vars, 43), dtype=np.float32),
    )


def _planbb_obs(n_vars=2, candidates=None):
    if candidates is None:
        candidates = np.arange(n_vars)
    return PlanBBObs(_dummy_observation(n_vars), np.asarray(candidates))


def _fake_tracker(nodes, visited):
    """A minimal stand-in for sorrel's ``TreeTracker``: ``_nodes`` maps id to
    (parent, lb) in insertion order, ``visited`` lists the visited ids."""
    return SimpleNamespace(_nodes=nodes, visited=visited)


class TestTreeHLGaussLoss:
    """The categorical head over log2(-z): 18 bins on [-1.5, 16.5], with the
    planbb trailing-dim decode."""

    @pytest.fixture()
    def loss(self):
        return TreeHLGaussLoss()

    def test_decode_keeps_the_trailing_dim(self, loss):
        # The official's [B, 1] decode — the bbmdp head returns [B]; the MCTS
        # ``value * breward`` broadcast needs the singleton.
        probs = torch.full((3, NUM_BINS), 1.0 / NUM_BINS)
        decoded = loss.transform_from_probs(probs)
        assert decoded.shape == (3, 1)

    def test_value_transform_is_minus_two_pow_centers(self, loss):
        # centers[k] = k - 1 on the unit-wide bins; a near-one-hot softmax at
        # bin 7 decodes to -2^6.
        logits = torch.full((1, NUM_BINS), -10.0)
        logits[0, 7] = 10.0
        value = loss.value_transform(logits)
        assert value.shape == (1, 1)
        assert float(value[0, 0]) == pytest.approx(-64.0, abs=1e-3)

    def test_tree_mdp_ce_matches_the_target_distribution(self, loss):
        # tree_mdp: CE against the bin distribution of log2(-target) = 1.0,
        # which centers on bin 2 (0.495 mass there, 0.23 on each neighbor).
        target = torch.full((1, 1), -2.0)
        expected = F.cross_entropy(
            torch.eye(NUM_BINS)[[2]],
            loss.transform_to_probs(torch.log2(-target).squeeze(-1)),
        )
        got = loss(torch.eye(NUM_BINS)[[2]], target, tree_mdp=True)
        assert torch.allclose(got, expected, atol=1e-6)
        # Concentrating on the right bin beats concentrating on a wrong one.
        wrong = loss(torch.eye(NUM_BINS)[[5]], target, tree_mdp=True)
        assert float(got) < float(wrong)

    def test_decode_round_trips_inside_the_support(self, loss):
        target = torch.tensor([1.0, 8.3])
        decoded = loss.transform_from_probs(loss.transform_to_probs(target)).squeeze(-1)
        assert torch.allclose(decoded, target, atol=0.1)


class TestModelHelpers:
    def test_get_target_index_skips_the_next_slots(self):
        # The breward loss covers the 2u+1 non-"next" slots of the window.
        assert get_target_index(3).tolist() == [0, 1, 2, 4, 5, 7, 8]

    def test_format_batch_actions_offsets_by_the_variable_prefix(self):
        got = format_batch_actions(torch.tensor([1, 2]), torch.tensor([2, 3]))
        assert got.tolist() == [1, 4]

    def test_pad_tensor_mask_value_depends_on_width(self):
        one_wide = torch.arange(5.0).reshape(5, 1)
        padded = pad_tensor(one_wide, torch.tensor([2, 3]))
        assert padded.shape == (2, 3, 1)
        assert padded[0, 2, 0] == -6e4  # the official argmax-safe mask
        two_wide = pad_tensor(torch.ones(5, 2), torch.tensor([2, 3]))
        assert (two_wide[0, 2] == 0).all()  # probabilities pad with 0

    def test_pad_array_zero_pads_policy_rows(self):
        padded = pad_array([np.ones((1, 2)), np.ones((1, 5))], [2, 5])
        assert padded.shape == (2, 1, 5)
        assert (padded[0, 0, 2:] == 0).all()

    def test_get_idx_maps_repeats_to_their_first_slot(self):
        assert get_idx([[1, 0, 1]]) == [[0, 1, 0]]


class TestInitialInference:
    """The dual-nature value head over a padded batch of two graphs."""

    @pytest.fixture()
    def net(self):
        torch.manual_seed(0)
        net = PlanBBNet()
        net.eval()
        return net

    @pytest.fixture()
    def batched(self):
        # Two graphs, 2 and 5 variables: the second forces padding.
        return geometric_batch([_planbb_obs(2, [0, 1]), _planbb_obs(5, [1, 3])])

    def _pooled(self, net, batched):
        """The official pooling by hand: padded per-variable mean times the
        padding correction."""
        with torch.no_grad():
            hidden = net.representation(batched)
            _logits, pseudo_q, breward = net.prediction(hidden)
            pseudo_q = pad_tensor(pseudo_q, batched.nb_variables, pad_value=0)
            breward = pad_tensor(breward, batched.nb_variables)
            max_variables = pseudo_q.size(-2)
            correction = (max_variables / batched.nb_variables).unsqueeze(-1)
            value = pseudo_q.mean(dim=-2) * correction
            breward = breward.mean(dim=-2)
        return value, breward

    def test_mcts_mode_scales_the_decoded_value_by_breward(self, net, batched):
        net.mcts_mode()
        # The self-play loop runs the search under no_grad, like the official.
        with torch.no_grad():
            out = net.initial_inference(batched, root=True)
        assert isinstance(out.value, np.ndarray)
        assert out.value.shape == (2, 1)
        assert out.policy_logits.shape == (2, 5)
        # Non-candidates carry the -6e4 mask, candidates do not.
        assert out.policy_logits[0, 2] == -6e4
        assert out.policy_logits[0, 0] > -6e4
        assert out.policy_logits[1, 0] == -6e4
        assert out.policy_logits[1, 3] > -6e4
        value, breward = self._pooled(net, batched)
        decoded = net.value_loss.value_transform(value)
        b = F.softmax(breward, dim=-1)[..., 1].unsqueeze(-1)
        assert np.allclose(out.value, (decoded * b).numpy(), atol=1e-5)

    def test_tree_mdp_mode_returns_bin_logits(self, net, batched):
        net.train_mode()
        out = net.initial_inference(batched, root=True, tree_mdp=True)
        assert torch.is_tensor(out.value)
        assert out.value.shape == (2, NUM_BINS)
        value, _ = self._pooled(net, batched)
        assert torch.allclose(out.value, value, atol=1e-6)

    def test_decoded_mode_returns_the_scalar_value(self, net, batched):
        net.train_mode()
        out = net.initial_inference(batched, root=True)
        assert out.value.shape == (2, 1)
        value, _ = self._pooled(net, batched)
        assert torch.allclose(out.value, net.value_loss.value_transform(value), atol=1e-6)


class TestConsistencyAndSchedules:
    def test_consist_loss_is_zero_on_identical_projections(self):
        f = torch.randn(4, 16)
        assert float(consist_loss_func(f, f.clone()).sum()) == pytest.approx(0.0, abs=1e-5)

    def test_consist_loss_on_orthogonal_projections(self):
        f1 = torch.tensor([[1.0, 0.0]])
        f2 = torch.tensor([[0.0, 1.0]])
        # cosine 0 and euclidean = max = sqrt(2) -> normalized distance 0:
        # both similarity terms vanish, the loss tops out at 1.
        assert float(consist_loss_func(f1, f2)) == pytest.approx(1.0)

    def test_adjust_lr_warms_up_then_decays(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        # 1000 steps, 1% warmup -> 10 warmup steps.
        assert adjust_lr(optimizer, 0, 1000) == pytest.approx(1e-4)
        assert adjust_lr(optimizer, 9, 1000) == pytest.approx(1e-3)
        assert adjust_lr(optimizer, 10, 1000) == pytest.approx(1e-3)  # decay^0
        assert adjust_lr(optimizer, 1010, 1000) == pytest.approx(1e-3 * 0.95)

    def test_visit_softmax_temperature(self):
        assert visit_softmax_temperature(0, 1000) == 1.0
        assert visit_softmax_temperature(199, 1000) == 1.0
        # The exponent (steps - 20%) // 500 is still 0 at the boundary.
        assert visit_softmax_temperature(200, 1000) == 1.0
        assert visit_softmax_temperature(699, 1000) == 1.0
        assert visit_softmax_temperature(700, 1000) == pytest.approx(0.99)
        assert visit_softmax_temperature(1200, 1000) == pytest.approx(0.99**2)


class _StubNet:
    """Constant heads for the search math test (never touches the graphs)."""

    def __init__(self, value, breward, n_actions, hidden):
        self.value = value
        self.breward = breward
        self.n_actions = n_actions
        self.hidden = hidden

    def recurrent_inference(self, hidden_states, actions):
        batch_size = hidden_states.batch_size

        def make():
            return SimpleNamespace(
                policy_logits=torch.zeros(batch_size, self.n_actions),
                value=torch.full((batch_size, 1), self.value),
                breward=torch.full((batch_size, 1), self.breward),
                hidden_state=Batch.from_data_list([self.hidden for _ in range(batch_size)]),
            )

        return make(), make()


class TestGumbelMCTS:
    """The cminimax normalize and the Gumbel Top-k root selection."""

    def test_normalize_clamps_and_respects_the_delta_floor(self):
        stats = MinMaxStats()
        stats.update(-5.0)
        stats.update(1.0)
        assert stats.normalize(-5.0) == 0.0
        assert stats.normalize(1.0) == 1.0
        assert stats.normalize(-2.0) == pytest.approx(0.5)
        # A range narrower than value_delta_max divides by the floor instead.
        tight = MinMaxStats()
        tight.set_static_val(0.01, 50, 0.1)
        tight.update(0.102)
        tight.update(0.097)
        assert tight.normalize(0.1) == pytest.approx(0.3)

    def test_root_selection_is_gumbel_top_k_over_the_candidates(self):
        root = Node(1.0, -1, None, 1.0, 4)
        root.pool = []  # the owning root's node pool
        root.set_root_action_set([1, 0, 1, 0])
        root.expand(0, 0, [0.1, 0.2, 3.0, 0.4], 4)
        stats = MinMaxStats()
        stats.set_static_val(0.01, 50, 0.1)
        # Simulation 0: scores are gumbel + prior over the candidates only —
        # a huge noise on a non-candidate must not survive the -10000 mask.
        noise = [0.0, 100.0, 0.0, 100.0]
        action = mcts_select_action(
            root, stats, num_simulations=4, simulation_idx=0,
            gumble_noise=noise, current_num_top_actions=2,
        )
        assert root.selected_children_idx == [2, 0]  # prior 3.0 then 0.1
        # Equal visits: the first selected candidate goes first.
        assert action == 2

    def test_search_with_constant_heads(self):
        """Two full searches against a stub net: constant value and
        branchability, uniform policy — the backup math by hand. The MCTS
        object is reused across the two searches, like the training loop
        does, which exercises the per-search phase reset."""
        mcts = MCTS(num_simulations=4, num_top_actions=2)
        graph = geometric_batch([_planbb_obs(4)]).to_data_list()[0]

        def run(breward):
            model = _StubNet(value=-5.0, breward=breward, n_actions=4, hidden=graph)
            values, policies, best, _info = mcts.search(
                model,
                batch_size=1,
                action_nums=[4],
                root_action_sets=[[1, 0, 1, 0]],
                root_states=[[0]],
                root_state_obs=[[graph]],
                root_policy_logits=[[[0.1, 0.2, 3.0, 0.4]]],
                root_state_node_values=[[-2.0]],
                root_brewards=[[breward]],
                device=torch.device("cpu"),
                use_gumble_noise=False,
            )
            return values, policies, best

        # Branchable current node: r = -1 and both children inserted, so the
        # per-simulation state updates are -2 - 1 - 5 - 5 + 2 = -11 (sims 0-1
        # stop at the root's children) then -11 - 1 - 5 - 5 + 5 = -17 (sims
        # 2-3 descend one level deeper through the expanded children). The
        # backups shift by -1 per level on the way up, so the root sees
        # [-2 (prepare), -12, -12, -19, -19].
        values, policies, best = run(0.9)
        assert float(values[0]) == pytest.approx((-2.0 - 12.0 - 12.0 - 19.0 - 19.0) / 5)
        assert best[0] == 2  # the candidate with the 3.0 prior
        assert policies.shape == (1, 4)
        assert policies[0].sum() == pytest.approx(1.0)

        # Unbranchable: r = 0 and no children inserted (absorbing state), so
        # the updates are -10 then -15; the root sees
        # [-2, -11, -11, -17, -17].
        values, _policies, _best = run(0.4)
        assert float(values[0]) == pytest.approx((-2.0 - 11.0 - 11.0 - 17.0 - 17.0) / 5)


class TestTrajectory:
    """The official window math on a hand-built chain 0 -> 1 -> 2 -> 3."""

    @pytest.fixture()
    def chain(self):
        """A solved 4-node chain: 4 observations (depths 0..3), 4 actions and
        rewards, one visited child per node."""
        n = 4
        traj = PlanBBTrajectory()
        traj.init(_planbb_obs(), depth=0)
        for i in range(n):
            next_obs = _planbb_obs() if i + 1 < n else None
            traj.store_search_stats([0.5, 0.5], -1.0)
            traj.append(i % 2, next_obs, depth=i + 1)
        nodes = {i: ((None, 0.0) if i == 0 else (i - 1, 0.0)) for i in range(n)}
        adapter = SearchTreeAdapter(_fake_tracker(nodes, list(range(n))))
        return traj, adapter

    def test_adapter_reads_insertion_order_children(self):
        nodes = {0: (None, 0.0), 2: (0, 0.0), 1: (0, 0.0)}
        adapter = SearchTreeAdapter(_fake_tracker(nodes, [0, 2, 1]))
        assert adapter.visited_node_ids == [0, 2, 1]
        assert adapter.successors(0) == [2, 1]

    def test_reward_sums_are_the_td_capped_bfs(self, chain):
        traj, adapter = chain
        assert traj.milp_solved(adapter)
        # From node 0 the BFS pops 0, 1, 2 before hitting td_steps = 3; the
        # frontier empties earlier from the deeper nodes.
        assert traj.reward_sums == [-3.0, -3.0, -2.0, -1.0]

    def test_state_obs_window_and_ghosts(self, chain):
        traj, adapter = chain
        traj.milp_solved(adapter)
        (
            batch_nodes,
            batch_nodes_idx,
            is_real,
            state_open_nodes_idx,
            pad_len,
            traj_index,
            td_value,
        ) = traj.state_obs(0, padding=True)
        # horizon 6 over 4 observations -> 2 padded steps; the window is
        # 3 * (unroll + td) + 1 = 19 nodes.
        assert pad_len == 2
        assert len(batch_nodes) == 19
        assert len(batch_nodes_idx) == 19
        # Step 0: node 0 with one visited child -> left ghost, right real.
        assert batch_nodes[:3] == [0, 0, 1]
        assert is_real[:3] == [1, 0.3, 1]
        # The terminal node has no visited child -> both ghosts.
        assert batch_nodes[9:12] == [3, 3, 3]
        assert is_real[9:12] == [1, 0.3, 0.7]
        # The open-node stacks are padded to horizon + 1.
        assert len(state_open_nodes_idx) == 7
        assert len(traj_index) == 4  # one entry per real open-node stack
        # The td bootstrap of slot 0 points at the last stack's front node
        # (node 3, first seen as node 2's right child at position 8).
        assert td_value[0] == [8]
        assert td_value[3] == []  # the trajectory is over by slot 3

    def test_build_milp_states_and_batch_obs(self, chain):
        traj, adapter = chain
        traj.milp_solved(adapter)
        traj_obs = 3 * traj.unroll_steps + 1
        for pos in range(len(traj.observations)):
            assert len(traj.td_reward_sums[pos]) == traj_obs
            assert len(traj.value_loss_obs_weight[pos]) == traj_obs
        bo = traj.batch_obs(0)
        assert len(bo.batch_observations) == len(bo.batch_nodes_idx)
        assert len(bo.is_real_nodes) == len(bo.batch_nodes_idx)
        # Ghosts are zeroed: branchability (-1) = 0, is_best_sibling (-2) set
        # on the left ghost only.
        for obs, is_real in zip(bo.batch_observations, bo.is_real_nodes):
            if is_real != 1:
                assert (obs.variable_features[:, -1] == 0).all()
                assert set(np.unique(obs.variable_features[:, -2])) == {1 if is_real == 0.3 else 0}
        assert bo.num_obs >= 0

    def test_double_root_drops_the_first_transition(self):
        # depths [0, 0, 1, 2]: the first two decisions both sit at the root.
        traj = PlanBBTrajectory()
        traj.init(_planbb_obs(), depth=0)
        for i in range(4):
            next_obs = _planbb_obs() if i + 1 < 4 else None
            traj.store_search_stats([0.5, 0.5], -1.0)
            traj.append(i % 2, next_obs, depth=[0, 1, 2][i] if i < 3 else 2)
        nodes = {i: ((None, 0.0) if i == 0 else (i - 1, 0.0)) for i in range(4)}
        adapter = SearchTreeAdapter(_fake_tracker(nodes, list(range(4))))
        assert traj.milp_solved(adapter)
        # 4 visited nodes vs 3 remaining observations: gap 1, accepted.
        assert traj.transition_gap == 1
        assert len(traj.observations) == 3
        assert traj.depths == [0, 1, 2]

    def test_gap_two_rejects(self):
        traj = PlanBBTrajectory()
        traj.init(_planbb_obs(), depth=0)
        for i in range(5):
            next_obs = _planbb_obs() if i + 1 < 5 else None
            traj.store_search_stats([0.5, 0.5], -1.0)
            traj.append(i % 2, next_obs, depth=i + 1)
        nodes = {i: ((None, 0.0) if i == 0 else (i - 1, 0.0)) for i in range(7)}
        adapter = SearchTreeAdapter(_fake_tracker(nodes, list(range(7))))
        assert not traj.milp_solved(adapter)

    def test_over_long_episode_rejects(self):
        traj = PlanBBTrajectory(node_limit=20)  # 20 // 2 - 10 = 0
        traj.init(_planbb_obs(), depth=0)
        traj.append(0, None, depth=1)  # done: no next observation
        nodes = {0: (None, 0.0), 1: (0, 0.0)}
        adapter = SearchTreeAdapter(_fake_tracker(nodes, [0, 1]))
        assert not traj.milp_solved(adapter)

    def test_zero_obs_only_accepts_ghosts(self):
        obs = _planbb_obs()
        with pytest.raises(AssertionError):
            zero_obs(obs, 1.0)

    def test_select_action_entropy(self):
        # Counts [4, 1] at temperature 1: probs [0.8, 0.2]; the entropy is
        # hand-computed. Deterministic mode argmaxes the raw counts.
        action_pos, ent = select_action([4, 1], temperature=1.0, deterministic=True)
        assert action_pos == 0
        assert ent == pytest.approx(-(0.8 * np.log2(0.8) + 0.2 * np.log2(0.2)))
        np.random.seed(0)
        action_pos, _ent = select_action([4, 1], temperature=1.0, deterministic=False)
        assert action_pos in (0, 1)


class TestEpisode:
    """The full regime on tiny set covers (harness + SCIP)."""

    def test_env_declares_the_official_protocol(self):
        env = make_env(iter([]))
        assert env.scip_params["separating/maxrounds"] == 0
        assert env.scip_params["nodeselection/dfs/stdpriority"] == 300000
        assert env.scip_params["nodeselection/dfs/memsavepriority"] == 300000
        assert env.scip_params["estimation/restarts/restartpolicy"] == "n"

    def test_run_selfplay_episode_builds_a_valid_trajectory(self):
        torch.manual_seed(0)
        np.random.seed(0)
        env = make_env(_instances(8), time_limit=30.0, seed=0)
        model = PlanBBNet()
        rng = np.random.default_rng(0)
        mcts = MCTS(num_simulations=8, num_top_actions=8)

        trajectory = None
        attempts = 0
        while trajectory is None and attempts < 8:
            trajectory, n_decisions, _status = run_selfplay_episode(
                env,
                model,
                torch.device("cpu"),
                start_training=False,
                rng=rng,
                mcts=mcts,
            )
            attempts += 1
        assert trajectory is not None, "every tiny episode was rejected"
        assert n_decisions == len(trajectory)
        assert len(trajectory.observations) == len(trajectory.actions)
        assert len(trajectory.observations) == len(trajectory.child_visits)
        # The official constant reward and normalized stored distributions.
        assert set(trajectory.rewards) == {-1.0}
        assert all(abs(sum(cv) - 1.0) < 1e-9 for cv in trajectory.child_visits)
        assert all(
            action in obs.action_set
            for action, obs in zip(trajectory.actions, trajectory.observations)
        )
        # The windows were built: the position-0 batch has ghost-zeroed obs.
        bo = trajectory.batch_obs(0)
        assert len(bo.batch_observations) == len(bo.batch_nodes_idx)

    def test_train_smoke_end_to_end(self, tmp_path, monkeypatch):
        # The whole loop on toy scale: random-action warmup, MCTS self-play,
        # all four losses, the schedules, the checkpoint — then the greedy
        # policy bridge. (The package exports the train *function*, which
        # shadows the module attribute — importlib gets the module.)
        import importlib

        planbb_train = importlib.import_module("l2b_bench.methods.planbb.train")
        monkeypatch.setattr(
            planbb_train, "MCTS", lambda: MCTS(num_simulations=4, num_top_actions=4)
        )

        checkpoint = tmp_path / "planbb.pt"
        monkeypatch.setattr(planbb_train, "SNAPSHOT_EVERY_STEPS", 2)
        trained_steps = planbb_train.train(
            _instances(6),
            checkpoint,
            episodes=3,
            training_steps=4,
            time_limit=30.0,
            seed=0,
            start_transitions=5,
            batch_size=4,
            device=torch.device("cpu"),
        )
        assert checkpoint.exists()
        assert trained_steps > 0  # at least one episode escaped the presolve
        # The numbered weight snapshots are deployable checkpoints.
        snapshots = sorted(tmp_path.glob("planbb.pt.s*"))
        assert len(snapshots) >= 1
        assert PlanBBPolicy.from_checkpoint(snapshots[0]) is not None

        policy = PlanBBPolicy.from_checkpoint(checkpoint)
        env = make_env(_instances(4), time_limit=30.0, seed=1)
        obs, action_set, _r, done, _info = env.reset()
        attempts = 0
        while done and attempts < 3:
            obs, action_set, _r, done, _info = env.reset()
            attempts += 1
        assert not done
        action = policy.select_action(obs, action_set)
        assert action in np.asarray(action_set)

    def test_train_resume_after_preemption(self, tmp_path, monkeypatch):
        # The campaign QoS is preemptible: a mid-training snapshot
        # (<checkpoint>.train.pt) must let a requeued run pick up where the
        # interrupted one stopped, and the model checkpoint must appear only
        # on completion (the pilot gates on its existence).
        import importlib

        planbb_train = importlib.import_module("l2b_bench.methods.planbb.train")
        monkeypatch.setattr(
            planbb_train, "MCTS", lambda: MCTS(num_simulations=4, num_top_actions=4)
        )
        monkeypatch.setattr(planbb_train, "SAVE_CKPT_INTERVAL", 2)

        real_update = planbb_train.update_weights
        calls = {"n": 0}

        def crashing_update(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 3:  # the preemption, mid-training
                raise KeyboardInterrupt
            return real_update(*args, **kwargs)

        monkeypatch.setattr(planbb_train, "update_weights", crashing_update)

        checkpoint = tmp_path / "planbb.pt"
        resume_path = checkpoint.with_suffix(".train.pt")

        # 100x120 covers are almost always integral at the root (one branchable
        # in twenty), which would strand the resumed run with an empty replay;
        # 200x300 covers branch often enough for self-play to refill it.
        instances = lambda: _instances(20, rows=200, cols=300)

        with pytest.raises(KeyboardInterrupt):
            planbb_train.train(
                instances(),
                checkpoint,
                episodes=15,
                training_steps=5,
                time_limit=30.0,
                seed=0,
                start_transitions=5,
                batch_size=4,
                device=torch.device("cpu"),
            )
        assert resume_path.exists()  # the snapshot from step 2 (or 4)
        assert not checkpoint.exists()  # not complete — the pilot must retrain

        # The resumed run skips the episodes the first one consumed and needs
        # one trainable episode to finish the remaining steps (the per-episode
        # ration recomputes as episodes deplete).
        trained_steps = planbb_train.train(
            instances(),
            checkpoint,
            episodes=15,
            training_steps=5,
            time_limit=30.0,
            seed=0,
            start_transitions=5,
            batch_size=4,
            device=torch.device("cpu"),
        )
        assert trained_steps == 5  # the budget was reached, not restarted
        assert checkpoint.exists()  # completion writes the model checkpoint
        assert not resume_path.exists()  # and retires the snapshot


class _Trajectory:
    """Stub with the one thing the replay reads: the trajectory length."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


class TestPrioritizedReplay:
    def test_new_positions_enter_at_the_max_priority(self):
        from l2b_bench.methods.planbb.train import PrioritizedReplay

        replay = PrioritizedReplay()
        replay.add(_Trajectory(3))
        assert np.array_equal(replay.priorities, np.ones(3))
        replay.priorities[1] = 4.0
        replay.add(_Trajectory(2))
        assert np.array_equal(replay.priorities[3:], [4.0, 4.0])

    def test_drawn_positions_are_consumed_once(self):
        from l2b_bench.methods.planbb.train import PrioritizedReplay

        replay = PrioritizedReplay()
        replay.add(_Trajectory(50))
        rng = np.random.default_rng(0)
        batches = []
        while replay.can_sample():
            _samples, drawn = replay.sample(8, rng)
            batches.append(drawn.tolist())
            replay.consume(drawn)
        # Full batches draw distinct positions while enough remain (6 x 8 = 48);
        # the tail — 2 positions for a batch of 8 — falls back to replacement,
        # as the official ``repl`` does. Every position is used, then none.
        assert all(len(set(b)) == 8 for b in batches[:-1])
        assert set(sum(batches, [])) == set(range(50))
        assert not replay.can_sample()

    def test_sampling_is_proportional_to_priority_to_the_alpha(self):
        from l2b_bench.methods.planbb.train import PrioritizedReplay

        replay = PrioritizedReplay()
        replay.add(_Trajectory(2))
        replay.priorities[:] = [1.0, 4.0]  # alpha 0.6: weights 1 and 4**0.6 = 2.297
        rng = np.random.default_rng(1)
        counts = np.zeros(2)
        for _ in range(4000):
            _samples, drawn = replay.sample(1, rng)
            counts[drawn[0]] += 1
        expected = 2.297 / (1 + 2.297)
        assert counts[1] / counts.sum() == pytest.approx(expected, abs=0.03)
