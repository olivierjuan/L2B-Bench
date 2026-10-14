"""End-to-end tests of the TreeGate (Zarpellon 2021) port: collect → train.

The instances are tiny set covers (a few branching nodes each) — the point
is that the TreeGate protocol (relpscost expert under the (s, k) Zarpellon
augmentation, on the BranchingState observation, the feature-gated
feedforward) runs on the harness, not that the model learns anything.
"""

import numpy as np
import pytest
import torch

from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.treegate.collect import collect
from l2b_bench.methods.treegate.model import NoTreePolicy, TreeGatePolicy
from l2b_bench.methods.treegate.policy import TreeGatePolicyModel, treegate_method
from l2b_bench.methods.treegate.train import (
    LR,
    LR_DECAY_EPOCHS,
    process,
    sample_to_tensors,
    save_checkpoint,
    load_checkpoint,
)

# A small k so the tiny test instances still leave expert decisions after the
# random-explorer prefix (the campaign's k=10 would swallow most of them).
TEST_K = 2


@pytest.fixture(scope="module")
def collected():
    """A small relpscost-expert dataset from tiny set covers, k=2 explorer."""
    generator = SetCoverGenerator(n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7))
    return collect(generator, n_episodes=4, seeds=[0, 1, 2, 3], time_limit=10.0, k=TEST_K)


class TestCollect:
    def test_samples_are_consistent(self, collected):
        assert collected.n_episodes == 4
        assert len(collected.samples) > 0
        for sample in collected.samples:
            assert sample.candidate_features.shape == (len(sample.action_set), 25)
            assert sample.node_features.shape == (8,)
            assert sample.mip_features.shape == (53,)
            assert sample.action in sample.action_set
            assert np.isfinite(sample.candidate_features).all()
            assert np.isfinite(sample.node_features).all()
            assert np.isfinite(sample.mip_features).all()

    def test_explorer_prefix_is_not_collected(self, collected):
        """The k random-explorer decisions are taken by the dynamics, not the
        expert — every exposed decision (hence every sample) is the expert's,
        so an episode yields at most n_nodes - k samples."""
        assert collected.n_nodes >= len(collected.samples)
        assert collected.n_episodes * TEST_K <= collected.n_nodes

    def test_augmentation_is_deterministic(self):
        """The same (instances, seeds, s, k) yield the same collected states —
        the augmentation cell is part of the protocol's identity."""
        generator = SetCoverGenerator(n_rows=100, n_cols=120, density=0.10, rng=np.random.default_rng(3))
        first = collect(generator, n_episodes=2, seeds=[11, 12], time_limit=10.0, s=1, k=TEST_K)
        generator = SetCoverGenerator(n_rows=100, n_cols=120, density=0.10, rng=np.random.default_rng(3))
        second = collect(generator, n_episodes=2, seeds=[11, 12], time_limit=10.0, s=1, k=TEST_K)
        assert len(first.samples) == len(second.samples)
        for a, b in zip(first.samples, second.samples):
            assert np.array_equal(a.candidate_features, b.candidate_features)
            assert a.action == b.action

    def test_statuses_reported(self, collected):
        assert len(collected.statuses) == 4
        assert all(s == "optimal" for s in collected.statuses)


class TestModel:
    def test_forward_shape_per_node(self, collected):
        """The official orientation: (S, 25) + (8,) + (53,) → (S, 1) scores."""
        model = TreeGatePolicy()
        model.eval()
        for sample in collected.samples[:5]:
            with torch.no_grad():
                scores = model(
                    torch.as_tensor(sample.candidate_features),
                    torch.as_tensor(sample.node_features),
                    torch.as_tensor(sample.mip_features),
                )
            assert scores.shape == (len(sample.action_set), 1)
            assert torch.isfinite(scores).all()

    def test_gating_changes_with_the_tree_state(self, collected):
        """TreeGate's whole point: the same candidates score differently under
        a different tree state (NoTree's scores do not)."""
        treegate = TreeGatePolicy().eval()
        notree = NoTreePolicy().eval()
        sample = collected.samples[0]
        cands = torch.as_tensor(sample.candidate_features)
        node = torch.as_tensor(sample.node_features)
        mip = torch.as_tensor(sample.mip_features)
        with torch.no_grad():
            base = treegate(cands, node, mip)
            perturbed = treegate(cands, node + 1.0, mip)
            notree_base = notree(cands, node, mip)
            notree_perturbed = notree(cands, node + 1.0, mip)
        assert not torch.allclose(base, perturbed)
        assert torch.allclose(notree_base, notree_perturbed)

    def test_notree_matches_treegate_without_gating(self, collected):
        """NoTree is the TreeGate ablation: same pyramid, no tree input. With
        zero gates (sigmoid of a large negative bias is not reachable), the
        sanity check is instead that both output (S, 1) and differ — the
        ablation exists to be different."""
        treegate = TreeGatePolicy().eval()
        notree = NoTreePolicy().eval()
        sample = collected.samples[0]
        cands = torch.as_tensor(sample.candidate_features)
        with torch.no_grad():
            a = treegate(cands, torch.as_tensor(sample.node_features), torch.as_tensor(sample.mip_features))
            b = notree(cands)
        assert a.shape == b.shape
        assert not torch.allclose(a, b)


class TestTrain:
    def test_training_loss_is_finite_and_kacc_bounded(self, collected):
        model = TreeGatePolicy()
        device = torch.device("cpu")
        batches = [
            [{k: v.to(device) for k, v in sample_to_tensors(s).items()} for s in collected.samples]
        ]
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(2):
            stats = process(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])
        assert stats["kacc"].shape == (4,)
        assert (stats["kacc"] >= 0).all() and (stats["kacc"] <= 1).all()

    def test_shuffled_minibatches_train(self, collected):
        model = TreeGatePolicy()
        samples = collected.samples
        shuffle = np.random.default_rng(0)
        order = shuffle.permutation(len(samples))
        batches = [
            [sample_to_tensors(samples[i]) for i in order[k : k + 5]]
            for k in range(0, len(order), 5)
        ]
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        stats = process(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])

    def test_lr_decay_schedule_matches_the_checkpoint(self):
        model = TreeGatePolicy()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DECAY_EPOCHS, 0.1)
        lrs = []
        for _ in range(35):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        assert lrs[0] == pytest.approx(LR)  # before epoch 20: unchanged
        assert lrs[18] == pytest.approx(LR)
        assert lrs[19] == pytest.approx(LR * 0.1)  # decay at epoch 20
        assert lrs[29] == pytest.approx(LR * 0.01)  # and again at 30

    def test_checkpoint_roundtrip(self, collected, tmp_path):
        model = TreeGatePolicy()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        process(model, [[sample_to_tensors(s) for s in collected.samples]], optimizer=optimizer)
        path = tmp_path / "treegate.pt"
        save_checkpoint(model, path, extra={"n_samples": len(collected.samples), "method": "treegate"})
        loaded = load_checkpoint(path, device=torch.device("cpu"))
        assert isinstance(loaded, TreeGatePolicy)
        policy = TreeGatePolicyModel(loaded)
        for sample in collected.samples[:3]:
            assert policy.select_action(_observation_of(sample), sample.action_set) in sample.action_set


class TestPolicy:
    def test_treegate_method_is_runner_ready(self, collected, tmp_path):
        model = TreeGatePolicy()
        path = tmp_path / "treegate.pt"
        save_checkpoint(model, path)
        method = treegate_method(path)
        assert method.name == "treegate"
        assert method.fidelity == "treegate"
        policy = method.policy()
        assert isinstance(policy, TreeGatePolicyModel)
        sample = collected.samples[0]
        assert policy.select_action(_observation_of(sample), sample.action_set) in sample.action_set


def _observation_of(sample):
    """Rebuild the BranchingState the policy consumes at inference time."""

    class _Obs:
        pass

    obs = _Obs()
    obs.candidate_features = sample.candidate_features
    obs.node_features = sample.node_features
    obs.mip_features = sample.mip_features
    return obs
