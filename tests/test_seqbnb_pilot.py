"""Tests for the Seq-BnB 2026 port (arXiv 2610.14835): the collector first —
typed token sequences, the x*-compatibility node-selection oracle, and joint
episodes under BranchingNodeSelectionDynamics."""

import numpy as np
import pytest
from gyozas.dynamics import DecisionKind
from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.seqbnb.collect import (
    ACTION_TOKEN_DIM,
    GAP_CEILING,
    L_MAX,
    NODE_TOKEN_DIM,
    PoolGenerator,
    ReferenceCache,
    _window,
    ancestor_bounds,
    collect,
    compatible_with,
    make_env,
    node_token,
    oracle_node_selection,
    run_episode,
    solve_reference,
)


def _write_pool(directory, n, rows=120, cols=150, seed=3):
    """An on-disk training pool, as the pilot writes it."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for i in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.writeProblem(
            filename=str(directory / f"train_{i:03d}.mps"), genericnames=True
        )


class _FakeVar:
    """A branching variable: the oracle reads its name."""

    def __init__(self, name):
        self.name = name


class _FakeNode:
    """The node API the tokens and the oracle read (probe-verified on real
    SCIP nodes): number, depth, bounds, parent chain with branchings."""

    def __init__(
        self, number, depth=0, lower=0.0, estimate=0.0, parent=None, branchings=None
    ):
        self._number = number
        self._depth = depth
        self._lower = lower
        self._estimate = estimate
        self._parent = parent
        self._branchings = branchings

    def getNumber(self):
        return self._number

    def getDepth(self):
        return self._depth

    def getLowerbound(self):
        return self._lower

    def getEstimate(self):
        return self._estimate

    def getParent(self):
        return self._parent

    def getParentBranchings(self):
        return self._branchings


def _chain(node, *branchings):
    """Parent chain above ``node``, one (vars, vals, btypes) triple per level."""
    current = node
    for branching in branchings:
        parent = _FakeNode(-1, current.getDepth() - 1)
        current._parent = parent
        current._branchings = branching
        current = parent
    return current


class TestNodeToken:
    def test_eight_features_in_paper_order(self):
        node = _FakeNode(7, depth=3, lower=1.5, estimate=2.25)
        token = node_token(node, visited={7}, fathomed=set(), incumbent={7}, gap=0.1)
        np.testing.assert_allclose(token, [1.5, 2.25, 3.0, 1.0, 0.0, 1.0, 7.0, 0.1])
        assert token.shape == (NODE_TOKEN_DIM,)

    def test_flags_default_to_zero(self):
        token = node_token(
            _FakeNode(1), visited=set(), fathomed=set(), incumbent=set(), gap=0.0
        )
        assert token[3] == 0.0 and token[4] == 0.0 and token[5] == 0.0


class TestWindow:
    def test_open_tokens_always_retained(self):
        table = {i: np.full(NODE_TOKEN_DIM, i) for i in range(L_MAX + 10)}
        order = list(range(L_MAX + 10))
        open_ids = np.array([L_MAX + 9], dtype=np.int64)
        _tokens, ids = _window(table, open_ids, order)
        assert L_MAX + 9 in ids  # open: retained
        assert 0 not in ids  # oldest evicted by the window cap
        assert len(ids) == L_MAX

    def test_small_tables_keep_everything(self):
        table = {5: np.zeros(NODE_TOKEN_DIM), 2: np.zeros(NODE_TOKEN_DIM)}
        tokens, ids = _window(table, np.array([5], dtype=np.int64), [5, 2])
        assert sorted(ids.tolist()) == [2, 5]
        assert tokens.shape == (2, NODE_TOKEN_DIM)

    def test_history_snapshot_is_windowed(self):
        """The action history is windowed like the token table — an
        unbounded history made the campaign OOM (quadratic attention in the
        episode length)."""
        from l2b_bench.methods.seqbnb.collect import EpisodeTracker

        tracker = EpisodeTracker()
        tracker.history = [np.full(ACTION_TOKEN_DIM, i) for i in range(L_MAX + 10)]
        tracker.history_kinds = list(range(L_MAX + 10))
        past, kinds = tracker._snapshot()
        assert past.shape == (L_MAX, ACTION_TOKEN_DIM)
        assert kinds.shape == (L_MAX,)
        assert past[0][0] == 10  # the oldest entries are the evicted ones
        assert past[-1][0] == L_MAX + 9  # the most recent action is kept
        assert kinds[0] == 10


class TestCompatibilityOracle:
    def test_ancestor_bounds_accumulate_the_chain(self):
        node = _FakeNode(3, depth=2)
        _chain(
            node,
            ([_FakeVar("x1")], [0.5], [0]),  # x1 >= 0.5
            (
                [_FakeVar("x0"), _FakeVar("x2")],
                [1.0, 0.0],
                [1, 0],
            ),  # x0 <= 1.0, x2 >= 0.0
        )
        bounds = ancestor_bounds(node)
        assert ("x1", 0.5, 0) in bounds
        assert ("x0", 1.0, 1) in bounds
        assert ("x2", 0.0, 0) in bounds

    def test_compatible_when_x_star_survives_every_bound(self):
        x_star = {"x0": 0.0, "x1": 1.0}
        node = _FakeNode(1, depth=1)
        _chain(node, ([_FakeVar("x1")], [0.5], [0]))  # x1 >= 0.5, x*1 = 1
        assert compatible_with(node, x_star)
        _chain(node, ([_FakeVar("x1")], [0.5], [1]))  # x1 <= 0.5, x*1 = 1
        assert not compatible_with(node, x_star)

    def test_unknown_variable_counts_as_incompatible(self):
        node = _FakeNode(1, depth=1)
        _chain(node, ([_FakeVar("t_x9")], [0.5], [0]))  # presolve-renamed: not in x*
        assert not compatible_with(node, {"x0": 1.0})

    def test_oracle_picks_deepest_compatible(self):
        shallow = _FakeNode(1, depth=1, lower=0.0, estimate=0.0)
        deep = _FakeNode(2, depth=5, lower=3.0, estimate=4.0)
        nodes = [shallow, deep]
        # Unbounded nodes are always compatible: deepest first.
        assert oracle_node_selection(nodes, {}) == 2
        # Nothing compatible (every node excludes x*): fallback ordering,
        # lower bound first.
        _chain(shallow, ([_FakeVar("x")], [0.5], [1]))  # x <= 0.5
        _chain(deep, ([_FakeVar("x")], [0.5], [1]))  # x <= 0.5
        assert oracle_node_selection(nodes, {"x": 1.0}) == 1


class TestReferenceOptimum:
    def test_solve_reference_matches_a_direct_solve(self, tmp_path):
        generator = SetCoverGenerator(rng=np.random.default_rng(11))
        model = generator.generate_instance(n_rows=60, n_cols=80, density=0.10)
        model.writeProblem(filename=str(tmp_path / "inst.mps"), genericnames=True)
        x_star = solve_reference(tmp_path / "inst.mps")
        # A set cover: integral optimum, every original variable present.
        assert set(x_star.values()) <= {0.0, 1.0}
        assert len(x_star) == 80

    def test_cache_solves_once_per_file(self, tmp_path):
        _write_pool(tmp_path, 1, rows=60, cols=80)
        cache = ReferenceCache()
        first = cache.get(tmp_path / "train_000.mps")
        second = cache.get(tmp_path / "train_000.mps")
        assert first is second  # memoized, not re-solved


class TestEpisode:
    def _episode(self, tmp_path, n_files=4):
        """Env + reference resolver over a pool of the 200x250 family —
        those instances branch reliably (the 120x150 family mostly closes
        at the root, so a branching draw is not guaranteed)."""
        _write_pool(tmp_path, n_files, rows=200, cols=250)
        generator = PoolGenerator(tmp_path, rng=np.random.default_rng(0))
        env = make_env(generator)
        references = ReferenceCache()

        def reference_for(_env):
            assert generator.current_path is not None
            return references.get(generator.current_path)

        return env, reference_for

    def test_expert_episode_records_both_decision_kinds(self, tmp_path):
        env, reference_for = self._episode(tmp_path)
        kinds_seen = set()
        try:
            for _ in range(8):
                samples, _n_nodes, _status = run_episode(
                    env, lambda s, d, e: e, reference_for
                )
                kinds_seen.update(s.kind for s in samples)
                if DecisionKind.BRANCHING in kinds_seen:
                    break
        finally:
            env.close()
        assert DecisionKind.NODE_SELECTION in kinds_seen
        assert DecisionKind.BRANCHING in kinds_seen, "no drawn instance branched"

    def test_sample_layout_matches_the_token_protocol(self, tmp_path):
        env, reference_for = self._episode(tmp_path)
        samples = []
        try:
            for _ in range(8):
                episode_samples, _n_nodes, _status = run_episode(
                    env, lambda s, d, e: e, reference_for
                )
                samples.extend(episode_samples)
                if any(s.kind == DecisionKind.BRANCHING for s in episode_samples):
                    break
        finally:
            env.close()
        branching = [s for s in samples if s.kind == DecisionKind.BRANCHING]
        assert branching, "no drawn instance branched"
        for sample in samples:
            assert sample.tokens.shape[1] == NODE_TOKEN_DIM
            assert len(sample.token_ids) == len(sample.tokens)
            assert sample.history.shape[1] == ACTION_TOKEN_DIM
            assert sample.history.shape[0] == sample.t
            assert sample.gap <= GAP_CEILING
            if sample.kind == DecisionKind.NODE_SELECTION:
                assert sample.open_ids is not None and sample.candidates is None
                assert sample.expert_action in set(sample.open_ids.tolist())
            else:
                assert sample.candidates is not None and sample.open_ids is None
                assert sample.expert_action in set(sample.candidates.tolist())
                # A branching decision has a graph to condition on.
                assert sample.graph is not None

    def test_history_carries_the_past_decisions(self, tmp_path):
        env, reference_for = self._episode(tmp_path)
        samples = []
        try:
            for _ in range(8):
                episode_samples, _n_nodes, _status = run_episode(
                    env, lambda s, d, e: e, reference_for
                )
                samples = episode_samples
                if len(samples) >= 3:
                    break
        finally:
            env.close()
        assert len(samples) >= 3, "no episode made three decisions"
        last = samples[-1]
        assert last.t == len(samples) - 1
        # The history's first row is the first decision's [g, |O|, t=0, ...].
        assert last.history[0][2] == 0.0
        assert last.history[-1][2] == last.t - 1

    def test_collect_runs_the_paper_protocol(self, tmp_path):
        _write_pool(tmp_path, 2, rows=120, cols=150)
        result = collect(
            tmp_path, n_episodes=2, seeds=[41, 42], select_action=lambda s, d, e: e
        )
        assert result.n_episodes == 2
        assert len(result.statuses) == 2
        assert result.samples


class TestPoolGenerator:
    def test_remembers_the_current_file(self, tmp_path):
        _write_pool(tmp_path, 3, rows=60, cols=80)
        generator = PoolGenerator(tmp_path, rng=np.random.default_rng(0))
        seen = set()
        for _ in range(9):
            model = next(generator)
            seen.add(generator.current_path.name)
            model.hideOutput()
        assert len(seen) == 3  # every pool file was visited

    def test_empty_pool_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PoolGenerator(tmp_path)


class TestSeqBnBModel:
    def _samples(self, tmp_path, n_files=6):
        """Real collected samples (both kinds, at least one branching)."""
        _write_pool(tmp_path, n_files, rows=200, cols=250)
        generator = PoolGenerator(tmp_path, rng=np.random.default_rng(0))
        env = make_env(generator)
        references = ReferenceCache()
        samples = []
        try:
            for _ in range(8):
                episode, _n_nodes, _status = run_episode(
                    env,
                    lambda s, d, e: e,
                    lambda _e: references.get(generator.current_path),
                )
                samples.extend(episode)
                if any(s.kind == DecisionKind.BRANCHING for s in episode):
                    return samples
        finally:
            env.close()
        raise AssertionError("no drawn instance branched")

    def test_collate_and_forward_shapes(self, tmp_path):
        from l2b_bench.methods.seqbnb.model import collate

        samples = self._samples(tmp_path)
        batch = collate(samples)
        # Node-selection samples score the open nodes; branching samples the
        # LP candidates — one logit per action, per sample.
        from l2b_bench.methods.seqbnb.model import SeqBnBModel

        model = SeqBnBModel()
        logits = model(batch)
        assert len(logits) == len(samples)
        for sample, sample_logits in zip(samples, logits, strict=True):
            n_actions = len(
                sample.open_ids
                if sample.kind == DecisionKind.NODE_SELECTION
                else sample.candidates
            )
            assert sample_logits.shape == (n_actions,)

    def test_joint_loss_backpropagates(self, tmp_path):
        import torch

        from l2b_bench.methods.seqbnb.model import SeqBnBModel, collate

        samples = self._samples(tmp_path)
        model = SeqBnBModel()
        batch = collate(samples)
        logits = model(batch)
        loss = sum(
            torch.nn.functional.cross_entropy(
                logit.unsqueeze(0), batch.labels[i : i + 1], reduction="sum"
            )
            for i, logit in enumerate(logits)
        )
        loss.backward()
        gradients = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert gradients and any(float(g.abs().sum()) > 0 for g in gradients)

    def test_repeated_epochs_stay_finite(self, tmp_path):
        """Regression (GPU campaign): torch 2.13's fused SDPA backward
        kernels produce NaN gradients on CUDA when the attention carries a
        padding mask — the model pins the exact math backend (see
        ``SeqBnBModel.forward``). Several epochs of real updates must keep
        losses and weights finite. (CPU runs the exact backend anyway; the
        assertion guards the wrapper and the training loop.)"""
        import torch

        from l2b_bench.methods.seqbnb.model import SeqBnBModel
        from l2b_bench.methods.seqbnb.train import BATCH_SIZE, LR, joint_loss

        samples = self._samples(tmp_path, n_files=3)
        torch.manual_seed(0)
        model = SeqBnBModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(8):
            for start in range(0, len(samples), BATCH_SIZE):
                batch = samples[start : start + BATCH_SIZE]
                optimizer.zero_grad()
                loss = joint_loss(model, batch, torch.device("cpu"))
                loss.backward()
                assert torch.isfinite(loss)
                assert all(
                    torch.isfinite(p).all()
                    for p in model.parameters()
                    if p.requires_grad
                )
                optimizer.step()


class TestPolicyAndTraining:
    def _policy_episode(self, tmp_path, model=None):
        """Env + a policy acting greedily over the joint regime (the
        benchmark's own path: attach + select_action on Decision objects)."""
        from l2b_bench.methods.seqbnb.policy import SeqBnBPolicy

        _write_pool(tmp_path, 2, rows=200, cols=250)
        generator = PoolGenerator(tmp_path, rng=np.random.default_rng(0))
        env = make_env(generator)
        policy = (
            SeqBnBPolicy(model) if model is not None else SeqBnBPolicy(_tiny_model())
        )
        policy.attach(env)
        return env, policy

    def test_policy_runs_a_joint_episode_end_to_end(self, tmp_path):
        env, policy = self._policy_episode(tmp_path)
        try:
            _obs, decision, _reward, done, _info = env.reset()
            kinds = []
            steps = 0
            while not done and steps < 50:
                assert decision is not None
                kinds.append(decision.kind)
                action = policy.select_action(None, decision)
                _obs, decision, _reward, done, _info = env.step(action)
                steps += 1
            assert steps == 50 or done
            assert DecisionKind.NODE_SELECTION in kinds
        finally:
            env.close()

    def test_joint_loss_matches_the_papers_form(self):
        """L = L_branch + 0.1 * L_ns, each a mean CE over its kind."""
        import torch

        from l2b_bench.methods.seqbnb.model import SeqBnBModel
        from l2b_bench.methods.seqbnb.train import LAMBDA_NS, joint_loss

        # Zero logits give a uniform distribution: CE = log(n_actions) per
        # sample, so the joint loss is exactly L_br + lambda * L_ns.
        device = torch.device("cpu")

        class _Stub(SeqBnBModel):
            def forward(self, batch):
                return [
                    torch.zeros(len(batch.open_positions[i]))
                    if int(batch.kinds[i]) == 0
                    else torch.zeros(len(batch.candidate_rows[i]))
                    for i in range(batch.kinds.shape[0])
                ]

        stub = _Stub()
        samples = [
            _synthetic_sample(DecisionKind.NODE_SELECTION),
            _synthetic_sample(DecisionKind.BRANCHING),
        ]
        # Zero logits give CE = log(n_actions) for both — check the weights.
        loss = joint_loss(stub, samples, device)
        ns_loss = torch.log(torch.tensor(float(3)))  # 3 open nodes
        br_loss = torch.log(torch.tensor(float(4)))  # 4 candidates
        torch.testing.assert_close(loss, br_loss + LAMBDA_NS * ns_loss)

    def test_dagger_train_runs_and_saves(self, tmp_path):
        from l2b_bench.methods.seqbnb.model import SeqBnBModel
        from l2b_bench.methods.seqbnb.train import load_checkpoint, train

        _write_pool(tmp_path, 2, rows=120, cols=150)
        checkpoint = tmp_path / "seqbnb.pt"
        report = train(
            tmp_path,
            checkpoint,
            rounds=1,
            episodes_per_round=1,
            base_seed=7,
            time_limit=60.0,
        )
        assert "round_0" in report and "round_1" in report
        assert checkpoint.exists()
        assert isinstance(load_checkpoint(checkpoint), SeqBnBModel)


def _tiny_model():
    from l2b_bench.methods.seqbnb.model import SeqBnBModel

    model = SeqBnBModel()
    model.eval()
    return model


def _synthetic_sample(kind):
    """A minimal handcrafted sample (3 open nodes or 4 candidates)."""
    from l2b_bench.methods.seqbnb.collect import ACTION_TOKEN_DIM, SeqBnBSample

    n = 3 if kind == DecisionKind.NODE_SELECTION else 4
    return SeqBnBSample(
        kind=kind,
        tokens=np.zeros((n, 8)),
        token_ids=np.arange(1, n + 1),
        open_ids=np.arange(1, n + 1) if kind == DecisionKind.NODE_SELECTION else None,
        candidates=None if kind == DecisionKind.NODE_SELECTION else np.arange(n),
        history=np.zeros((0, ACTION_TOKEN_DIM)),
        history_kinds=np.zeros((0,), dtype=np.int64),
        action=-1,
        expert_action=1 if kind == DecisionKind.NODE_SELECTION else 2,
        t=0,
        n_open=n,
        gap=0.0,
        graph=None,
    )
