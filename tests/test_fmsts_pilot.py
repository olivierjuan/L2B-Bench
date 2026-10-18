"""End-to-end tests of the FMSTS 2020 (Etheve, Ales, Bissuel, Juan &
Kedad-Sidhoum, arXiv 2005.10026) port.

The instances are small set covers — the point is that the whole regime runs
on the harness (bespoke PCA observation -> epsilon-greedy episodes under the
paper's DFS protocol -> observed subtree-size Q-values over the full tree ->
prioritized replay -> the MDA's one weighted-MSE update per iteration), not
that the agent learns anything. The math tests pin the paper's formulas (the
subtree recursion, the Eq 5 weighting, the Eq 6 priorities, the multiplicative
dueling product) against hand computation.
"""

import random
from types import SimpleNamespace

import numpy as np
import torch
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.environment import Environment
from gyozas.informations.solve_status import SolveStatus
from gyozas.instances import FileGenerator, SetCoverGenerator
from gyozas.runner import Method
from pyscipopt import Model

from l2b_bench.methods.fmsts import (
    FMSTS_SCIP_PARAMS,
    EpisodeTree,
    FMSTSExperience,
    FMSTSObservation,
    FMSTSReplay,
    RawInstanceObservation,
    fit_pca,
    fmsts_method,
    instance_vector,
    make_env,
    run_episode,
    subtree_sizes,
)
from l2b_bench.methods.fmsts.model import N_DYNAMIC_FEATURES, MDANetwork
from l2b_bench.methods.fmsts.policy import FMSTSPolicy
from l2b_bench.methods.fmsts.train import load_checkpoint, save_checkpoint


def _instances(n, rows=120, cols=150, density=0.10, seed=7):
    """Small set covers, one fresh model per draw. The default family branches
    about one draw in three (the others solve at the root, with no decision)."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=density)
        model.hideOutput()
        yield model


def _write_pool(directory, n, rows, cols, seed=3):
    """An on-disk training pool, as the pilot writes it."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for i in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.writeProblem(
            filename=str(directory / f"train_{i:03d}.mps"), genericnames=True
        )


def _identity_observation(canonical):
    """An FMSTS observation whose PCA is the identity's first three rows:
    the static vector is the instance vector's first three coordinates."""
    n_rows, n_cols = canonical
    dim = n_rows * n_cols + n_rows + n_cols
    return FMSTSObservation(np.zeros(dim), np.eye(3, dim), canonical)


def _reset_until_decision(env):
    """Reset until an instance actually branches: a small set cover can be
    solved at the root (integral LP), and those episodes carry no decision."""
    while True:
        try:
            obs, action_set, reward, done, info = env.reset()
        except StopIteration:
            raise AssertionError(
                "no drawn instance required a branching decision"
            ) from None
        if not done:
            return obs, action_set, reward, done, info


#: The test instances are 120x150 set covers inside a 130x160 canvas.
CANONICAL = (130, 160)
CANONICAL_DIM = 130 * 160 + 130 + 160


class TestInstanceVector:
    """The padded (A, b, c) vector, read at the root of a paused solve."""

    def test_layout_and_padding(self):
        env = make_env(
            _instances(8), RawInstanceObservation(canonical=CANONICAL), seed=0
        )
        vector, _action_set, _reward, _done, _info = _reset_until_decision(env)
        vector = np.asarray(vector)
        n_rows, n_cols, R, C = 120, 150, *CANONICAL
        assert vector.shape == (R * C + R + C,)
        # The A block: the 120x150 matrix in the top-left of the 130x160 canvas.
        matrix = vector[: R * C].reshape(R, C)
        assert matrix[n_rows:, :].sum() == 0  # padded rows
        assert matrix[:, n_cols:].sum() == 0  # padded columns
        assert matrix[:n_rows, :n_cols].sum() > 0
        # The b block: one side per row (>= 1 set covers), then zero padding.
        sides = vector[R * C : R * C + R]
        assert set(sides[:n_rows].tolist()) == {1.0}
        assert sides[n_rows:].sum() == 0
        # The c block: the objective coefficients, then zero padding.
        objective = vector[R * C + R :]
        assert (objective[:n_cols] > 0).all()
        assert objective[n_cols:].sum() == 0

    def test_protocol_independent(self, tmp_path):
        # The vector is the original problem's (A, b, c): identical on a
        # freshly read model, under the paper's no-presolve/no-cut protocol,
        # and under the default solver (presolving and cutting on) — the
        # benchmark evaluates under the default protocol, so the static
        # vector must not depend on it.
        model = SetCoverGenerator(rng=np.random.default_rng(11)).generate_instance(
            n_rows=300, n_cols=360, density=0.10
        )
        model.writeProblem(filename=str(tmp_path / "inst.mps"), genericnames=True)
        fresh = Model()
        fresh.readProblem(filename=str(tmp_path / "inst.mps"))
        expected = instance_vector(fresh, (310, 370))

        for name, scip_params in (("paper", dict(FMSTS_SCIP_PARAMS)), ("default", {})):
            params = dict(scip_params)
            params["display/verblevel"] = 0
            env = Environment(
                FileGenerator(
                    tmp_path, pattern="inst.mps", sampling_mode="replace", rng=0
                ),
                observation_function=RawInstanceObservation((310, 370)),
                information_function=SolveStatus(),
                scip_params=params,
                dynamics=BranchingDynamics(),
            )
            env.dynamics.seed(0)
            obs, _action_set, _reward, done, _info = _reset_until_decision(env)
            env.dynamics.close()
            np.testing.assert_allclose(np.asarray(obs), expected)
            assert done is False  # _reset_until_decision guarantees a decision


class TestDynamicFeatures:
    def test_root_one_hot_states_at_depth_zero(self):
        env = make_env(_instances(8), _identity_observation(CANONICAL), seed=0)
        obs, _action_set, _reward, _done, _info = _reset_until_decision(env)
        # The variable axis is padded to the canonical width.
        assert obs.dynamic.shape == (CANONICAL[1], N_DYNAMIC_FEATURES)
        assert (obs.dynamic[:, 0] == 0.0).all()  # depth
        assert (obs.dynamic[:, 1] >= 0.0).all()  # distance to the bounds
        # One-hot branching state: every real variable sits in exactly one
        # state. The root heuristics and propagation have typically already
        # fixed a few binaries by the first decision — the fixings are part
        # of the paper's state (read from the local domains).
        assert (obs.dynamic[:150, 2:5].sum(axis=1) == 1.0).all()
        assert (obs.dynamic[150:, 1:] == 0.0).all()  # the padded rows stay zero

    def test_branching_fixes_the_branched_variable(self):
        env = make_env(_instances(8), _identity_observation(CANONICAL), seed=0)
        obs, action_set, _reward, done, _info = _reset_until_decision(env)
        branched = int(np.asarray(action_set)[0])
        obs, _action_set, _reward, done, _info = env.step(branched)
        if not done:
            # The child's local domain has the branched variable fixed: its
            # one-hot leaves the free state.
            assert obs.dynamic[branched, 4] == 0.0
            assert obs.dynamic[branched, 2] + obs.dynamic[branched, 3] == 1.0
            assert obs.dynamic[branched, 0] == 1.0  # one level deeper

    def test_static_is_the_pca_of_the_vector(self):
        # mean 0, the first three canonical basis rows: the static vector is
        # the instance vector's first three coordinates.
        env = make_env(_instances(8), _identity_observation(CANONICAL), seed=0)
        obs, _action_set, _reward, _done, _info = _reset_until_decision(env)
        raw = instance_vector(env.model, CANONICAL)
        np.testing.assert_allclose(obs.static, raw[:3])


class TestSubtreeSizes:
    """The subtree recursion over hand-built trees."""

    def _tree(self, visited, parent):
        tree = EpisodeTree()
        tree.visited = list(visited)
        tree.parent = dict(parent)
        return tree

    def test_chain_with_unrecorded_last_children(self):
        # 0 -> 1 -> 2, the last decision's children never recorded: both are
        # unvisited leaves, so its subtree is 3 nodes.
        tree = self._tree([0, 1, 2], {0: None, 1: 0, 2: 1})
        assert subtree_sizes(tree) == [5, 4, 3]

    def test_leaf_children_count_one(self):
        # Root 10 with two children: 11 (visited, itself one unvisited child
        # 12) and 2 (unvisited leaf): V(10) = 1 + 2 + 1.
        tree = self._tree([10, 11], {10: None, 11: 10, 2: 10, 12: 11})
        assert subtree_sizes(tree) == [4, 2]

    def test_absorbed_events_add_dive_pruned_children(self):
        # The child processed between two decisions (never in an open-nodes
        # snapshot) enters through the dynamics' node events.
        tree = self._tree([10, 11], {10: None, 11: 10})
        dynamics = SimpleNamespace(
            infeasible_nodes=[(None, 12, None, 11, 0.0), (None, 13, None, 11, 0.0)],
            feasible_nodes=[],
        )
        tree.absorb(dynamics)
        assert subtree_sizes(tree) == [4, 3]

    def test_total_size_counts_every_recorded_node(self):
        # sizes[0] is the whole tree: every recorded node, plus the last
        # decision's two unrecorded children when it has none.
        tree = self._tree([10, 11], {10: None, 11: 10, 3: 10, 12: 11})
        assert subtree_sizes(tree)[0] == len(tree.parent)  # 11's child recorded
        tree = self._tree([10, 11, 12], {10: None, 11: 10, 12: 11})
        assert subtree_sizes(tree)[0] == len(tree.parent) + 2  # the unrecorded pair


class TestMDA:
    def test_shapes_and_product_structure(self):
        torch.manual_seed(0)
        model = MDANetwork(n_pca=4)
        static = torch.randn(3, 4)
        dynamic = torch.randn(3, 7, N_DYNAMIC_FEATURES)
        values = model(static, dynamic)
        assert values.shape == (3, 7)
        # The multiplicative form: the static branch's linear unit times the
        # dynamic branch's per-variable units.
        scale = model.static_branch(static).squeeze(-1)
        inputs = torch.cat([static.unsqueeze(1).expand(-1, 7, -1), dynamic], dim=-1)
        per_variable = model.dynamic_branch(inputs).squeeze(-1)
        torch.testing.assert_close(values, scale.unsqueeze(-1) * per_variable)

    def test_gradients_flow(self):
        model = MDANetwork(n_pca=4)
        loss = model(torch.randn(2, 4), torch.randn(2, 5, N_DYNAMIC_FEATURES)).sum()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())


    def test_q_at_actions_chunks_long_episodes_exactly(self):
        # The Q-hat priorities are computed one episode at a time; a long
        # random episode reaches tens of thousands of decisions and one
        # monolithic [episode, n_vars, hidden] activation block overflows
        # the GPU. The chunked pass must be exact.
        from types import SimpleNamespace as _NS

        from l2b_bench.methods.fmsts.train import _q_at_actions

        torch.manual_seed(0)
        model = MDANetwork(n_pca=3)
        rng = np.random.default_rng(0)
        experiences = [
            _NS(
                observation=_NS(
                    static=rng.normal(size=3),
                    dynamic=rng.normal(size=(4, N_DYNAMIC_FEATURES)),
                ),
                action=int(rng.integers(4)),
            )
            for _ in range(10)
        ]
        chunked = _q_at_actions(model, experiences, torch.device("cpu"), chunk=3)
        # The unchunked forward on the same episode.
        static = torch.as_tensor(
            np.stack([e.observation.static for e in experiences]), dtype=torch.float32
        )
        dynamic = torch.as_tensor(
            np.stack([e.observation.dynamic for e in experiences]), dtype=torch.float32
        )
        actions = torch.as_tensor([e.action for e in experiences], dtype=torch.long)
        with torch.no_grad():
            full = model(static, dynamic).gather(1, actions.unsqueeze(1)).squeeze(1)
        torch.testing.assert_close(chunked, full)
        assert chunked.shape == (10,)


class TestReplay:
    def test_priorities_drive_sampling(self):
        buffer = FMSTSReplay(size=4)
        observation = SimpleNamespace(
            static=np.zeros(3, dtype=np.float32),
            dynamic=np.zeros((4, 5), dtype=np.float32),
        )

        def experience(q_target):
            return FMSTSExperience(
                observation=observation,
                action_set=np.arange(4),
                action=1,
                action_idx=1,
                depth=0,
                q_target=q_target,
                instance_weight=1.0 / q_target,
            )

        buffer.store(experience(3.0), priority=1.0)
        buffer.store(experience(4.0), priority=1000.0)
        buffer.store(experience(5.0), priority=1.0)
        buffer.store(experience(6.0), priority=1.0)
        rng = random.Random(0)
        counts = np.zeros(4)
        for _ in range(200):
            indices, _experiences = buffer.sample(2, rng)
            for index in indices:
                counts[index] += 1
        # The oversized priority dominates the draw.
        assert counts[1] > counts[0] + counts[2] + counts[3]


class TestEpisode:
    """The full regime on small set covers (harness + SCIP)."""

    def test_env_declares_the_paper_protocol(self):
        env = make_env(iter([]), RawInstanceObservation(canonical=(10, 10)))
        assert env.scip_params["presolving/maxrounds"] == 0
        assert env.scip_params["separating/maxrounds"] == 0
        assert env.scip_params["estimation/restarts/restartpolicy"] == "n"
        assert env.scip_params["nodeselection/dfs/stdpriority"] == 300000
        assert env.scip_params["nodeselection/dfs/memsavepriority"] == 300000

    def test_run_episode_observes_the_subtree_sizes(self):
        observation = _identity_observation(CANONICAL)
        env = make_env(_instances(24), observation, time_limit=60.0, seed=0)
        rng = np.random.default_rng(0)

        def policy(obs, candidates):
            idx = int(rng.integers(len(candidates)))
            return int(candidates[idx]), idx

        experiences, n_decisions, status = run_episode(env, policy)
        attempts = 1
        while experiences is None and attempts < 16:
            experiences, n_decisions, status = run_episode(env, policy)
            attempts += 1
        assert experiences is not None, "every episode was rejected"
        assert status == "optimal"
        assert n_decisions == len(experiences)
        assert experiences[0].depth == 0
        # Every decision owns its two children: Q >= 3 (Prop 2's subtree).
        assert all(e.q_target >= 3.0 for e in experiences)
        # The root's Q is the whole tree, and Eq 5's weight is its inverse.
        tree_size = 1.0 / experiences[0].instance_weight
        assert tree_size == experiences[0].q_target
        assert {round(1.0 / e.instance_weight) for e in experiences} == {
            round(tree_size)
        }
        assert all(e.action in e.action_set for e in experiences)
        # The observations are the bespoke ones (static PCA + dynamics), on
        # the canonical variable axis.
        assert all(e.observation.static.shape == (3,) for e in experiences)
        assert all(
            e.observation.dynamic.shape == (CANONICAL[1], N_DYNAMIC_FEATURES)
            for e in experiences
        )

    def test_rejected_when_not_solved(self):
        # A 1-second limit on an evaluation-family instance (500x600) rejects
        # the episode: no fully-expanded tree, no observable Q-values.
        observation = _identity_observation((500, 600))
        env = make_env(
            _instances(4, rows=500, cols=600, density=0.10),
            observation,
            time_limit=1.0,
            seed=0,
        )
        rng = np.random.default_rng(0)

        def policy(obs, candidates):
            idx = int(rng.integers(len(candidates)))
            return int(candidates[idx]), idx

        experiences, _n, status = run_episode(env, policy)
        attempts = 1
        while experiences is not None and attempts < 4:
            experiences, _n, status = run_episode(env, policy)
            attempts += 1
        assert experiences is None, "a time-limited episode must be rejected"
        assert status == "timelimit"


class TestTrain:
    def test_fit_pca(self, tmp_path):
        # A pool of 200x250 instances (most of them branch): the PCA
        # (scikit-learn) is fitted over their padded (A, b, c) vectors —
        # mean in the vector space, 3 components.
        _write_pool(tmp_path, n=12, rows=200, cols=250)
        mean, components = fit_pca(tmp_path, canonical=(200, 250), n_components=3)
        dim = 200 * 250 + 200 + 250
        assert mean.shape == (dim,)
        assert components.shape == (3, dim)

    def test_train_smoke_end_to_end(self, tmp_path, monkeypatch):
        import importlib

        fmsts_train = importlib.import_module("l2b_bench.methods.fmsts.train")

        monkeypatch.setattr(fmsts_train, "BATCH_SIZE", 4)
        monkeypatch.setattr(fmsts_train, "SNAPSHOT_EVERY_ITERATIONS", 3)
        pool = tmp_path / "pool"
        pool.mkdir()
        _write_pool(pool, n=6, rows=200, cols=250)
        checkpoint = tmp_path / "fmsts.pt"
        n_updates = fmsts_train.train(
            pool,
            checkpoint,
            iterations=6,
            canonical=(200, 250),
            time_limit=60.0,
            seed=0,
        )
        assert checkpoint.exists()
        assert n_updates > 0
        # The per-budget snapshots (cadence 3; a rejected episode skips its
        # snapshot, so the count depends on the redraws).
        snapshots = sorted(tmp_path.glob("fmsts.pt.i*"))
        assert 1 <= len(snapshots) <= 2
        _m, mean, _c, _can = load_checkpoint(snapshots[0])
        assert mean.shape == (200 * 250 + 200 + 250,)
        # The checkpoint carries the fitted PCA, weights-only loadable.
        _model, mean, components, canonical = load_checkpoint(checkpoint)
        assert canonical == (200, 250)
        dim = 200 * 250 + 200 + 250
        assert mean.shape == (dim,)
        assert 1 <= components.shape[0] <= 16
        assert components.shape[1] == dim

    def test_policy_and_method_bridge(self, tmp_path):
        # A saved checkpoint drives the greedy policy and the Method table
        # entry — the observation is the method's own (fitted PCA inside).
        torch.manual_seed(0)
        model = MDANetwork(n_pca=3)
        save_checkpoint(
            model,
            np.zeros(CANONICAL_DIM),
            np.eye(3, CANONICAL_DIM),
            CANONICAL,
            tmp_path / "fmsts.pt",
        )
        method = fmsts_method(tmp_path / "fmsts.pt")
        assert isinstance(method, Method)
        assert method.name == "fmsts"
        assert method.fidelity == "fmsts"
        assert isinstance(method.observation(), FMSTSObservation)
        policy = FMSTSPolicy.from_checkpoint(tmp_path / "fmsts.pt")
        env = make_env(_instances(8), method.observation(), seed=0)
        obs, action_set, _reward, _done, _info = _reset_until_decision(env)
        action = policy.select_action(obs, action_set)
        assert action in np.asarray(action_set)
