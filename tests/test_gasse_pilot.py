"""End-to-end tests of the Gasse 2019 pilot: collect → train → policy.

The instances are tiny set covers (a few branching nodes each) — the point is
that the whole loop runs on the harness, not that the model learns anything.
"""

import numpy as np
import pytest
import torch

from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.gasse.collect import collect
from l2b_bench.methods.gasse.model import GNNPolicy
from l2b_bench.methods.gasse.policy import GassePolicy, gasse_method
from l2b_bench.methods.gasse.train import (
    batched_tensors,
    collate,
    load_checkpoint,
    pretrain,
    process,
    sample_to_tensors,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def collected():
    """A small expert dataset from tiny set covers (a handful of samples)."""
    generator = SetCoverGenerator(n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7))
    result = collect(generator, n_episodes=4, seeds=[0, 1, 2, 3], expert_probability=1.0, time_limit=10.0)
    return result


class TestCollect:
    def test_samples_are_consistent(self, collected):
        assert collected.n_episodes == 4
        assert len(collected.samples) > 0
        assert collected.metadata.n_decisions == len(collected.samples)
        for sample in collected.samples:
            assert sample.action in sample.action_set
            assert sample.observation.variable_features.shape[1] == 19  # the paper's features
            assert sample.observation.row_features.shape[1] == 5
            scores = sample.scores[sample.action_set]
            assert np.isfinite(scores).any()
            assert sample.scores[sample.action] == pytest.approx(scores.max())

    def test_collect_is_reproducible(self, collected):
        generator = SetCoverGenerator(n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7))
        again = collect(generator, n_episodes=4, seeds=[0, 1, 2, 3], expert_probability=1.0, time_limit=10.0)
        assert again.n_nodes == collected.n_nodes
        assert [s.action for s in again.samples] == [s.action for s in collected.samples]

    def test_multi_optimal_record_is_populated(self, collected):
        # R4.3: the acc@1 context — some nodes may have several optimal actions.
        distribution = collected.metadata.multi_optimal_distribution()
        assert sum(distribution.values()) == len(collected.samples)


class TestModel:
    def test_forward_scores_every_variable(self, collected):
        policy = GNNPolicy()
        tensors = sample_to_tensors(collected.samples[0])
        logits = policy(
            tensors["constraint_features"],
            tensors["edge_index"],
            tensors["edge_attr"],
            tensors["variable_features"],
        )
        assert logits.shape == (collected.samples[0].observation.variable_features.shape[0],)
        assert torch_isfinite(logits)

    def test_collate_matches_single(self, collected):
        """A batch must give the same per-sample logits as single-graph passes."""
        policy = GNNPolicy().eval()
        batch = collate(collected.samples[:2])
        with torch.no_grad():
            batched = policy(*batched_tensors(batch))
        var_offset = 0
        for sample in collected.samples[:2]:
            tensors = sample_to_tensors(sample)
            with torch.no_grad():
                single = policy(
                    tensors["constraint_features"],
                    tensors["edge_index"],
                    tensors["edge_attr"],
                    tensors["variable_features"],
                )
            candidates = batch.candidates[0] if var_offset == 0 else batch.candidates[1]
            candidates = candidates - var_offset
            assert torch_allclose(batched[candidates + var_offset], single[candidates])
            var_offset += tensors["variable_features"].shape[0]


def torch_isfinite(t):
    return bool(np.isfinite(t.detach().numpy()).all())


def torch_allclose(a, b):
    return bool(np.allclose(a.detach().numpy(), b.detach().numpy(), atol=1e-5))


class TestTrain:
    def test_pretrain_then_train_runs_the_official_loop(self, collected):
        model = GNNPolicy()
        batches = [collate(collected.samples[i : i + 2]) for i in range(0, len(collected.samples), 2)]
        n_layers = pretrain(model, batches)
        assert n_layers > 0  # the PreNorm layers were frozen one by one

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stats = process(model, batches)  # a valid pass first
        for _ in range(3):
            stats = process(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])
        assert stats["kacc"].shape == (4,)  # top-k accuracies, k = 1, 3, 5, 10

    def test_checkpoint_roundtrip(self, collected, tmp_path):
        model = GNNPolicy()
        batches = [collate(collected.samples)]
        pretrain(model, batches)
        path = tmp_path / "gasse.pt"
        save_checkpoint(model, path, extra={"n_samples": len(collected.samples)})
        loaded = load_checkpoint(path)
        tensors = sample_to_tensors(collected.samples[0])
        logits = loaded(
            tensors["constraint_features"],
            tensors["edge_index"],
            tensors["edge_attr"],
            tensors["variable_features"],
        )
        assert torch_isfinite(logits)


class TestPolicy:
    def test_policy_selects_within_the_action_set(self, collected, tmp_path):
        model = GNNPolicy()
        batches = [collate(collected.samples)]
        pretrain(model, batches)
        path = tmp_path / "gasse.pt"
        save_checkpoint(model, path)
        policy = GassePolicy.from_checkpoint(path)
        for sample in collected.samples:
            action = policy.select_action(sample.observation, sample.action_set)
            assert action in sample.action_set

    def test_gasse_method_is_runner_ready(self, collected, tmp_path):
        model = GNNPolicy()
        pretrain(model, [collate(collected.samples)])
        path = tmp_path / "gasse.pt"
        save_checkpoint(model, path)
        method = gasse_method(path)
        assert method.name == "gasse"
        assert method.fidelity == "gasse"
        policy = method.policy()
        sample = collected.samples[0]
        assert policy.select_action(sample.observation, sample.action_set) in sample.action_set


class TestRelpscostControlArm:
    """R6.4: the same protocol and observation with the scalable expert's
    labels — the gasse-relpscost fidelity profile's collection."""

    @pytest.fixture(scope="class")
    def relpscost_collected(self):
        generator = SetCoverGenerator(
            n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7)
        )
        return collect(
            generator,
            n_episodes=4,
            seeds=[0, 1, 2, 3],
            expert_probability=1.0,
            time_limit=10.0,
            expert="relpscost",
        )

    def test_samples_carry_relpscost_labels(self, relpscost_collected):
        """Every recorded action is the relpscost argmax over its candidates
        (finite on the candidates, −inf elsewhere in the vector)."""
        from gyozas.experts import RelpscostExpert

        expert = RelpscostExpert()
        for sample in relpscost_collected.samples:
            scores = sample.scores
            # The vector spans the LP columns; candidates are finite, and
            # the recorded action is their argmax.
            assert np.isfinite(scores[sample.action_set]).all()
            assert sample.action in set(sample.action_set.tolist())
            assert scores[sample.action] == scores[sample.action_set].max()

    def test_sample_shape_matches_the_gasse_protocol(self, relpscost_collected, collected):
        """Same layout as the SB-labeled samples: the training path (collate
        + process) consumes either without knowing the expert."""
        sb = collected.samples[0]
        rpc = relpscost_collected.samples[0]
        assert type(rpc) is type(sb)
        assert rpc.observation.row_features.shape[-1] == sb.observation.row_features.shape[-1]
        assert rpc.observation.variable_features.shape[-1] == sb.observation.variable_features.shape[-1]

    def test_unknown_expert_is_rejected(self):
        generator = SetCoverGenerator(n_rows=60, n_cols=80, density=0.10, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="unknown expert"):
            collect(
                generator,
                n_episodes=1,
                seeds=[0],
                expert_probability=1.0,
                time_limit=10.0,
                expert="oracle",
            )

    def test_method_wrapper_declares_the_control_profile(self, relpscost_collected, tmp_path):
        from l2b_bench.methods.gasse.policy import gasse_relpscost_method

        checkpoint = tmp_path / "gasse_rpc.pt"
        save_checkpoint(GNNPolicy(), checkpoint, extra={})
        method = gasse_relpscost_method(checkpoint)
        assert method.name == "gasse-relpscost"
        assert method.fidelity == "gasse-relpscost"
        # The same policy contract as gasse — the trained GCNN.
        policy = method.policy()
        assert isinstance(policy, GassePolicy)
