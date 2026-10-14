"""End-to-end tests of the Scavuzzo et al. 2022 (rl2branch) port.

The instances are tiny set covers — the point is that all three of the
paper's variants run on the harness (NodeBipartite observation → sampled
policy in the environment → credit assignment → one REINFORCE step per
epoch), not that the agent learns anything.
"""

import numpy as np
import pytest
import torch
from gyozas.instances import SetCoverGenerator
from gyozas.rewards import NNodes, SubtreeSizes

from l2b_bench.methods.gasse.model import GNNPolicy
from l2b_bench.methods.gasse.train import save_checkpoint
from l2b_bench.methods.scavuzzo import (
    VARIANTS,
    Transition,
    assign_returns,
    load_trained_checkpoint,
    make_env,
    reinforce_update,
    run_episode,
    scavuzzo_method,
    train,
    with_objlimit,
)
from l2b_bench.methods.scavuzzo.train import DFS_PARAMS


def _instances(n, rows=120, cols=150, seed=7):
    """Tiny set covers, one fresh model per episode."""
    generator = SetCoverGenerator(rng=np.random.default_rng(seed))
    for _ in range(n):
        model = generator.generate_instance(n_rows=rows, n_cols=cols, density=0.10)
        model.hideOutput()
        yield model


def _policy(seed=0):
    """A fresh GNNPolicy with fixed seed (stand-in for the IL checkpoint)."""
    torch.manual_seed(seed)
    return GNNPolicy()


@pytest.fixture(scope="module")
def episodes():
    """One episode per variant with the same untrained policy."""
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    policy = _policy()
    out = {}
    for variant in VARIANTS:
        env = make_env(_instances(3), variant, time_limit=30.0, seed=0)
        transitions, final_cum, n_iter = run_episode(env, policy, rng, device)
        assign_returns(transitions, env, final_cum, variant)
        out[variant] = (transitions, final_cum, n_iter, env)
    return out


class TestEnv:
    def test_mdp_env_uses_nnodes_tmdp_uses_subtree_sizes(self):
        assert isinstance(make_env(_instances(1), "mdp").reward_function, NNodes)
        for variant in ("tmdp+DFS", "tmdp+ObjLim"):
            assert isinstance(
                make_env(_instances(1), variant).reward_function, SubtreeSizes
            )

    def test_dfs_variant_forces_depth_first_selection(self):
        """Only tmdp+DFS carries the official node-selector overrides."""
        for variant, params in (
            ("mdp", {}),
            ("tmdp+ObjLim", {}),
            ("tmdp+DFS", DFS_PARAMS),
        ):
            env = make_env(_instances(1), variant)
            for key, value in params.items():
                assert env.scip_params[key] == value
            for key in DFS_PARAMS:
                if variant != "tmdp+DFS":
                    assert key not in env.scip_params

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="unknown variant"):
            make_env(_instances(1), "tmdp")


class TestEpisode:
    def test_transitions_are_well_formed(self, episodes):
        """Kept transitions sample ~5% of decisions, each action in its
        candidate set, node ids recorded for the tree variants."""
        for variant in VARIANTS:
            transitions, _, n_iter, _ = episodes[variant]
            assert 0 <= len(transitions) <= n_iter
            for t in transitions:
                assert 0 <= t.action_idx < len(t.action_set)
                if variant == "mdp":
                    assert t.node_id is None
                else:
                    assert t.node_id is not None

    def test_mdp_returns_are_node_costs(self, episodes):
        """The flat-MDP return is minus the number of nodes processed after
        the decision: non-positive, and the later the decision the closer to
        zero (the cumulative node count only grows)."""
        transitions, final_cum, _n_iter, _ = episodes["mdp"]
        returns = [t.returns for t in transitions]
        assert len(returns) >= 1
        assert all(r <= 0 for r in returns)
        assert all(
            t.returns == pytest.approx(t.cum_nnodes - final_cum) for t in transitions
        )
        assert returns == sorted(returns)

    def test_tmdp_returns_are_subtree_costs(self, episodes):
        """The tree-MDP return is -(subtree_size) - 1, and a subtree holds at
        least the decision's two children: return <= -3."""
        for variant in ("tmdp+DFS", "tmdp+ObjLim"):
            transitions, _, _, env = episodes[variant]
            sizes = env.reward_function.subtree_sizes
            assert sizes is not None
            for t in transitions:
                assert t.returns == pytest.approx(-sizes[t.node_id] - 1)
                assert t.returns <= -3

    def test_objlimit_yields_solved_limit_models(self):
        """with_objlimit solves each instance, then hands on the freed model
        with the objective limit just above its optimum."""
        from pyscipopt import Model

        models = list(with_objlimit(_instances(2)))
        assert len(models) == 2
        for model in models:
            assert isinstance(model, Model)
            # The optimum of a 120x150x0.10 set cover is a small positive
            # integer; the limit sits 0.1 above it.
            assert 0 < model.getObjlimit() < 1000

    def test_env_runs_on_objlimited_models(self):
        """A freed-and-objlimited model still drives a full episode."""
        device = torch.device("cpu")
        env = make_env(
            with_objlimit(_instances(3)), "tmdp+ObjLim", time_limit=30.0, seed=0
        )
        transitions, final_cum, n_iter = run_episode(
            env, _policy(), np.random.default_rng(0), device
        )
        assign_returns(transitions, env, final_cum, "tmdp+ObjLim")
        assert n_iter >= 0  # small instances are often pre-solved under the limit


class TestUpdate:
    @pytest.fixture(scope="class")
    def synthetic_transitions(self):
        """Enough synthetic transitions for a few shuffled batches of 16."""
        rng = np.random.default_rng(1)
        return [_synthetic_transition(rng) for _ in range(40)]

    def test_one_step_per_epoch(self, synthetic_transitions):
        """The update takes a single Adam step over accumulated batch
        gradients, and moves the policy."""
        policy = _policy()
        before = {n: p.clone() for n, p in policy.named_parameters()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        stats = reinforce_update(
            policy, synthetic_transitions, optimizer, torch.device("cpu")
        )
        moved = [
            n for n, p in policy.named_parameters() if not torch.equal(p, before[n])
        ]
        assert np.isfinite(stats["loss"])
        assert stats["entropy"] > 0
        assert len(moved) > 0

    def test_loss_matches_the_reinforce_formula(
        self, synthetic_transitions, monkeypatch
    ):
        """With the entropy bonus at zero the loss is exactly the mean of
        -returns * log_pi(action) recomputed from a fresh forward pass."""
        # importlib, not `import ... as`: the package's `train` attribute is
        # the train() *function* (re-exported), which shadows the submodule.
        import importlib

        scavuzzo_train = importlib.import_module("l2b_bench.methods.scavuzzo.train")
        monkeypatch.setattr(scavuzzo_train, "ENTROPY_BONUS", 0.0)
        policy = _policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0)
        transitions = synthetic_transitions[:16]  # a single batch
        stats = reinforce_update(policy, transitions, optimizer, torch.device("cpu"))
        from l2b_bench.methods.scavuzzo.train import TransitionBatch

        batch = TransitionBatch(transitions)
        logits = policy(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
        )
        log_probs = []
        for candidates, action in zip(batch.candidates, batch.actions, strict=True):
            dist = torch.distributions.Categorical(logits=logits[candidates])
            log_probs.append(dist.log_prob(action))
        expected = -(batch.returns * torch.stack(log_probs)).sum() / len(transitions)
        assert stats["reinforce_loss"] == pytest.approx(float(expected), rel=1e-5)

    def test_empty_epoch_is_a_no_op(self):
        policy = _policy()
        before = {n: p.clone() for n, p in policy.named_parameters()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        stats = reinforce_update(policy, [], optimizer, torch.device("cpu"))
        assert stats == {"loss": 0.0, "reinforce_loss": 0.0, "entropy": 0.0}
        for name, param in policy.named_parameters():
            assert torch.equal(param, before[name])


class TestTrain:
    def test_train_saves_a_checkpoint(self, tmp_path):
        """The full loop (episodes → credit → REINFORCE → save) runs for one
        variant per family: mdp (no pre-solve) and tmdp+ObjLim (with the
        optimum pre-solves)."""
        il_checkpoint = tmp_path / "scavuzzo_il.pt"
        save_checkpoint(_policy(), il_checkpoint)
        for variant in ("mdp", "tmdp+ObjLim"):
            checkpoint = tmp_path / f"scavuzzo-{variant}.pt"
            train(
                _instances(8, seed=11),
                il_checkpoint,
                checkpoint,
                episodes=8,
                variant=variant,
                time_limit=20.0,
                seed=0,
                device=torch.device("cpu"),
                log=lambda *_: None,
            )
            assert checkpoint.exists()
            loaded = load_trained_checkpoint(checkpoint, device=torch.device("cpu"))
            assert isinstance(loaded, GNNPolicy)
            extra = torch.load(checkpoint, weights_only=True)
            assert extra["variant"] == variant
            assert extra["epochs"] >= 1
            assert extra["episodes"] >= 1


class TestPolicy:
    def test_scavuzzo_method_is_runner_ready(self, tmp_path):
        path = tmp_path / "scavuzzo.pt"
        save_checkpoint(_policy(), path)
        for variant in VARIANTS:
            method = scavuzzo_method(path, variant)
            assert method.name == f"scavuzzo-{variant.lower().replace('+', '-')}"
            assert method.fidelity == "scavuzzo"
            policy = method.policy()
            env = make_env(_instances(5, seed=11), variant, time_limit=20.0, seed=0)
            obs, action_set, _, _done, _ = env.reset()
            while obs is None:  # skip pre-solved instances
                obs, action_set, _, _done, _ = env.reset()
            action = policy.select_action(obs, action_set)
            assert action in np.asarray(action_set).tolist()

    def test_unknown_variant_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown variant"):
            scavuzzo_method(tmp_path / "x.pt", "tmdp")


# --- synthetic helpers -----------------------------------------------------


def _synthetic_graph(rng, n_cons=4, n_vars=6, n_cands=3):
    n_edges = rng.integers(6, 12)
    return {
        "row_features": rng.normal(size=(n_cons, 5)),
        "edge_index": np.stack(
            [rng.integers(0, n_cons, n_edges), rng.integers(0, n_vars, n_edges)]
        ),
        "edge_values": rng.normal(size=n_edges),
        "variable_features": rng.normal(size=(n_vars, 19)),
        "candidates": np.sort(rng.choice(n_vars, size=n_cands, replace=False)),
    }


class _Edges:
    pass


class _Obs:
    pass


def _obs_of(graph):
    edges = _Edges()
    edges.indices = graph["edge_index"]
    edges.values = graph["edge_values"]
    obs = _Obs()
    obs.row_features = graph["row_features"]
    obs.edge_features = edges
    obs.variable_features = graph["variable_features"]
    return obs


def _synthetic_transition(rng):
    graph = _synthetic_graph(rng)
    action_idx = int(rng.integers(len(graph["candidates"])))
    return Transition(
        observation=_obs_of(graph),
        action_set=graph["candidates"],
        action_idx=action_idx,
        node_id=int(rng.integers(1, 100)),
        cum_nnodes=float(rng.integers(0, 50)),
        returns=float(-rng.integers(1, 20)),
    )
