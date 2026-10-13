"""End-to-end tests of the Lookback 2022 port: collect → pair → train.

The instances are tiny set covers (a few branching nodes each) — the point
is that the Lookback protocol (root + lookback collection, parent-child
pairs, PAT loss) runs on the harness, not that the model learns anything.
"""

import numpy as np
import pytest
import torch

from gyozas.instances import SetCoverGenerator

from l2b_bench.methods.gasse.model import GNNPolicy
from l2b_bench.methods.gasse.policy import GassePolicy
from l2b_bench.methods.gasse.train import batched_tensors, pretrain, save_checkpoint
from l2b_bench.methods.lookback.collect import (
    collect,
    lookback_condition,
    pair_parent_children,
)
from l2b_bench.methods.lookback.policy import lookback_method
from l2b_bench.methods.lookback.train import (
    EPSILON,
    collate_pairs,
    process_lookback,
    smoothed_target,
)

def all_children(samples: list) -> list[int]:
    """The full-dataset batch: every sample a child, in collection order."""
    return list(range(len(samples)))


@pytest.fixture(scope="module")
def collected():
    """A small expert dataset from tiny set covers, queried at every node.

    ``node_record_prob=1.0`` makes every node an expert query, so the root's
    children (and everything else) are collected and parent-child pairs exist.
    """
    generator = SetCoverGenerator(n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7))
    return collect(generator, n_episodes=4, seeds=[0, 1, 2, 3], node_record_prob=1.0, time_limit=10.0)


class TestCollect:
    def test_samples_are_consistent(self, collected):
        assert collected.n_episodes == 4
        assert len(collected.samples) > 0
        for sample in collected.samples:
            assert sample.action in sample.action_set
            scores = sample.scores[sample.action_set]
            assert np.isfinite(scores).any()
            assert sample.scores[sample.action] == pytest.approx(scores.max())

    def test_root_is_always_collected_first(self, collected):
        for sample in collected.samples[:1]:
            assert sample.depth == 0
            assert sample.parent_number == 0
            assert sample.node_number == 1  # SCIP's root
            assert not sample.child_collection

    def test_lookback_children_are_collected(self, collected):
        """With p=1.0 every node is queried; children of queried nodes carry
        the child_collection flag and form parent-child pairs."""
        children = [s for s in collected.samples if s.child_collection]
        assert children, "no lookback child collected"
        pairs = pair_parent_children(collected.samples)
        assert pairs, "no parent-child pair"
        for child_i, parent_i in pairs.items():
            child, parent = collected.samples[child_i], collected.samples[parent_i]
            assert child.parent_number == parent.node_number
            assert parent.node_number in {s.node_number for s in collected.samples}

    def test_lookback_condition_matches_the_paper(self, collected):
        pairs = pair_parent_children(collected.samples)
        checked = 0
        for child_i, parent_i in pairs.items():
            parent, child = collected.samples[parent_i], collected.samples[child_i]
            # Recompute the indicator by hand: the child's choice among the
            # parent's second-best (argmax ties, parent's best excluded).
            scores = np.asarray(parent.scores, dtype=np.float64).copy()
            scores[parent.action] = -np.inf
            finite = np.isfinite(scores)
            if not finite.any():
                assert not lookback_condition(parent, child)
                continue
            expected = child.action in set(np.where(scores == scores[finite].max())[0].tolist())
            assert lookback_condition(parent, child) == expected
            checked += 1
        assert checked


class TestTrain:
    def test_smoothed_target_is_a_distribution(self, collected):
        for sample in collected.samples[:10]:
            target = smoothed_target(sample, epsilon=0.1)
            assert target.sum() == pytest.approx(1.0)
            choice = int(np.where(np.asarray(sample.action_set) == sample.action)[0][0])
            assert target[choice] == pytest.approx(1.0 - EPSILON)
            assert (target >= 0).all()

    def test_pat_term_runs_and_train_loss_is_finite(self, collected):
        model = GNNPolicy()
        pairs = pair_parent_children(collected.samples)
        batches = [collate_pairs(collected.samples, all_children(collected.samples), pairs)]
        pretrain(model, [lb.batch for lb in batches])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        process_lookback(model, batches)  # a valid pass first
        for _ in range(2):
            stats = process_lookback(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])
        assert np.isfinite(stats["pat"])
        assert stats["kacc"].shape == (4,)

    def test_shuffled_batches_keep_every_pair(self, collected):
        """The campaign's training loop: shuffled minibatches smaller than
        the dataset, each pair's parent riding along in its child's batch.

        This is the regression that killed campaign job 3806280 — chunk-local
        index remapping broke on global pair indices once the dataset spanned
        more than one chunk.
        """
        model = GNNPolicy()
        samples = collected.samples
        pairs = pair_parent_children(samples)
        shuffle = np.random.default_rng(0)
        order = shuffle.permutation(len(samples))
        batch_size = 5  # several chunks, each smaller than the dataset
        batches = [
            collate_pairs(samples, list(order[k : k + batch_size]), pairs)
            for k in range(0, len(order), batch_size)
        ]
        pretrain(model, [lb.batch for lb in batches])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stats = process_lookback(model, batches, optimizer=optimizer)
        assert np.isfinite(stats["loss"])
        assert np.isfinite(stats["pat"])
        # Every pair contributes its PAT term no matter where the shuffle put
        # the child: each batch's parent_of covers all its children's pairs.
        total_children = sum(len(lb.lookback) for lb in batches)
        total_pairs = sum(len(lb.parent_of) for lb in batches)
        assert total_children == len(samples)
        assert total_pairs == len(pairs)

    def test_loss_target_z_trains_too(self, collected):
        model = GNNPolicy()
        pairs = pair_parent_children(collected.samples)
        batches = [collate_pairs(collected.samples, all_children(collected.samples), pairs, loss_target="z")]
        pretrain(model, [lb.batch for lb in batches])
        stats = process_lookback(model, batches, loss_target="z")
        assert np.isfinite(stats["loss"])

    def test_checkpoint_roundtrip(self, collected, tmp_path):
        model = GNNPolicy()
        pairs = pair_parent_children(collected.samples)
        lb = collate_pairs(collected.samples, all_children(collected.samples), pairs)
        pretrain(model, [lb.batch])
        path = tmp_path / "lookback.pt"
        save_checkpoint(model, path, extra={"n_samples": len(collected.samples), "method": "lookback"})
        policy = GassePolicy.from_checkpoint(path)
        for sample in collected.samples[:3]:
            assert policy.select_action(sample.observation, sample.action_set) in sample.action_set


class TestPolicy:
    def test_lookback_method_is_runner_ready(self, collected, tmp_path):
        model = GNNPolicy()
        pairs = pair_parent_children(collected.samples)
        pretrain(model, [collate_pairs(collected.samples, all_children(collected.samples), pairs).batch])
        path = tmp_path / "lookback.pt"
        save_checkpoint(model, path)
        method = lookback_method(path)
        assert method.name == "lookback"
        assert method.fidelity == "lookback"
        policy = method.policy()
        assert isinstance(policy, GassePolicy)
        sample = collected.samples[0]
        assert policy.select_action(sample.observation, sample.action_set) in sample.action_set
