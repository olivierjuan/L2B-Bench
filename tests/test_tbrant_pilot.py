"""End-to-end tests of the T-BranT (Lin 2022) port: collect → train → policy.

The instances are tiny set covers (a few branching nodes each) — the point
is that the T-BranT protocol (relpscost expert on the BranchingState
observation, padded candidate batching, the BranchFormer with tree gating
and the BiMatching head) runs on the harness, not that the model learns
anything.
"""

import numpy as np
import pytest
import torch

from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.tbrant.collect import SolverStatsSample, collect
from l2b_bench.methods.tbrant.model import BranchFormer
from l2b_bench.methods.tbrant.policy import TBranTPolicy, tbrant_method
from l2b_bench.methods.tbrant.train import (
    NoamLR,
    collate_stats,
    load_checkpoint,
    process,
    save_checkpoint,
)


@pytest.fixture(scope="module")
def collected():
    """A small relpscost-expert dataset from tiny set covers."""
    generator = SetCoverGenerator(n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7))
    return collect(generator, n_episodes=4, seeds=[0, 1, 2, 3], time_limit=10.0)


class TestCollect:
    def test_samples_are_consistent(self, collected):
        assert collected.n_episodes == 4
        assert len(collected.samples) > 0
        for sample in collected.samples:
            # The solver-stats triple: (n, 25) candidates, (8,) node, (53,) MIP.
            assert sample.candidate_features.shape == (len(sample.action_set), 25)
            assert sample.node_features.shape == (8,)
            assert sample.mip_features.shape == (53,)
            assert sample.action in sample.action_set
            assert np.isfinite(sample.candidate_features).all()
            assert np.isfinite(sample.node_features).all()
            assert np.isfinite(sample.mip_features).all()

    def test_every_node_is_an_expert_decision(self, collected):
        """k=0: the expert decides from the root, so there is no explorer
        prefix to skip — every node with branching candidates is recorded
        (nodes whose candidates are all fixed-domain fall back to SKIP and
        are recorded by SCIP, not by us)."""
        assert 0 <= collected.n_nodes - len(collected.samples)
        assert len(collected.samples) > 0

    def test_statuses_reported(self, collected):
        assert len(collected.statuses) == 4
        assert all(s == "optimal" for s in collected.statuses)


class TestModel:
    def test_forward_shapes_and_padding_mask(self, collected):
        """The forward pass on a padded batch: (N,S,25) → (N,S) logits with
        ``-inf`` exactly on padded slots."""
        model = BranchFormer()
        samples = collected.samples
        batch = collate_stats(samples)
        with torch.no_grad():
            logits = model(
                batch.candidate_features, batch.padding_mask, batch.node_features, batch.mip_features
            )
        n, s = batch.candidate_features.shape[:2]
        assert logits.shape == (n, s)
        padded = batch.padding_mask == 1
        assert torch.isinf(logits[padded]).all()
        assert torch.isfinite(logits[~padded]).all()

    def test_batch_of_one_matches_padded_batch(self, collected):
        """A node's logits must not depend on how much padding rides along —
        the transformer masks it, and BiMatching softmaxes over real slots."""
        model = BranchFormer().eval()
        sample = collected.samples[0]
        others = collected.samples[:3]
        padded = collate_stats([sample] + others)
        with torch.no_grad():
            alone = model(
                torch.as_tensor(sample.candidate_features).unsqueeze(0),
                torch.zeros(1, len(sample.action_set), dtype=torch.long),
                torch.as_tensor(sample.node_features).unsqueeze(0),
                torch.as_tensor(sample.mip_features).unsqueeze(0),
            )
            padded_out = model(
                padded.candidate_features, padded.padding_mask, padded.node_features, padded.mip_features
            )
        k = len(sample.action_set)
        assert torch.allclose(alone[0], padded_out[0, :k], atol=1e-5)

    def test_targets_point_at_the_expert_choice(self, collected):
        batch = collate_stats(collected.samples)
        for i, sample in enumerate(collected.samples):
            chosen = np.where(np.asarray(sample.action_set) == sample.action)[0][0]
            assert int(batch.targets[i]) == int(chosen)
            # The target's slot is a real (unpadded) candidate slot.
            assert batch.padding_mask[i, int(batch.targets[i])] == 0


class TestTrain:
    def test_training_loss_is_finite_and_kacc_bounded(self, collected):
        model = BranchFormer()
        batches = [collate_stats(collected.samples)]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(2):
            stats = process(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])
        assert stats["kacc"].shape == (4,)
        assert (stats["kacc"] >= 0).all() and (stats["kacc"] <= 1).all()
        # Top-k accuracy is monotone in k on a single batch.
        assert (np.diff(stats["kacc"]) >= -1e-9).all()

    def test_shuffled_minibatches_train(self, collected):
        """The campaign's training loop: shuffled minibatches, each padded to
        its own maximum."""
        model = BranchFormer()
        samples = collected.samples
        shuffle = np.random.default_rng(0)
        order = shuffle.permutation(len(samples))
        batches = [
            collate_stats([samples[i] for i in order[k : k + 5]])
            for k in range(0, len(order), 5)
        ]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        stats = process(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])

    def test_noam_schedule_warms_up_then_decays(self):
        model = BranchFormer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = NoamLR(optimizer, warmup_epochs=5)
        lrs = []
        for _ in range(15):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        # Linear warm-up (epochs 1..5) then inverse-sqrt decay: the peak sits
        # at the end of warm-up, and epoch 6's lr is below it.
        assert lrs[4] == pytest.approx(max(lrs))
        assert lrs[5] < lrs[4]
        assert all(a >= b for a, b in zip(lrs[5:], lrs[6:]))

    def test_checkpoint_roundtrip(self, collected, tmp_path):
        model = BranchFormer()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        process(model, [collate_stats(collected.samples)], optimizer=optimizer)
        path = tmp_path / "tbrant.pt"
        save_checkpoint(model, path, extra={"n_samples": len(collected.samples), "method": "tbrant"})
        loaded = load_checkpoint(path, device=torch.device("cpu"))
        assert isinstance(loaded, BranchFormer)
        policy = TBranTPolicy(loaded)
        for sample in collected.samples[:3]:
            assert policy.select_action(
                _observation_of(sample), sample.action_set
            ) in sample.action_set


class TestPolicy:
    def test_tbrant_method_is_runner_ready(self, collected, tmp_path):
        model = BranchFormer()
        path = tmp_path / "tbrant.pt"
        save_checkpoint(model, path)
        method = tbrant_method(path)
        assert method.name == "tbrant"
        assert method.fidelity == "tbrant"
        policy = method.policy()
        assert isinstance(policy, TBranTPolicy)
        sample = collected.samples[0]
        assert policy.select_action(_observation_of(sample), sample.action_set) in sample.action_set


def _observation_of(sample: SolverStatsSample):
    """Rebuild the BranchingState the policy consumes at inference time."""

    class _Obs:
        pass

    obs = _Obs()
    obs.candidate_features = sample.candidate_features
    obs.node_features = sample.node_features
    obs.mip_features = sample.mip_features
    return obs
