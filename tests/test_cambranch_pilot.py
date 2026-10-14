"""End-to-end tests of the CAMBranch 2024 pilot: augment → train → policy.

The instances are tiny set covers — the point is that the variable-shift
transformation matches a re-solved AMILP feature for feature and that the
whole joint loop runs, not that the model learns anything.
"""

import numpy as np
import pytest
import torch
from gyozas.dynamics.branching import BranchingDynamics
from gyozas.instances import SetCoverGenerator
from gyozas.observations import (
    ECOLE_COL_FEATURE_NAMES,
    ECOLE_ROW_FEATURE_NAMES,
    NodeBipartite,
)
from gyozas.observations.structs import BipartiteGraph, EdgeFeatures
from pyscipopt import Model, quicksum

from l2b_bench.methods.cambranch.augment import (
    N_AUGMENTED,
    SHIFT_SUPPORT,
    augment_pairs,
    draw_shift,
    shift_sample,
)
from l2b_bench.methods.cambranch.model import CAMBranchGNN
from l2b_bench.methods.cambranch.policy import CAMBranchPolicy, cambranch_method
from l2b_bench.methods.cambranch.train import (
    LAMBDA_1,
    LAMBDA_2,
    PlateauScheduler,
    auxiliary_loss,
    candidate_probs,
    collate_pairs,
    infonce_loss,
    load_checkpoint,
    pair_batches,
    process,
    subsample_budget,
    train,
)
from l2b_bench.methods.gasse.collect import BranchingSample, collect
from l2b_bench.methods.gasse.train import batched_tensors, collate

VAR_IDX = {name: i for i, name in enumerate(ECOLE_COL_FEATURE_NAMES)}
ROW_IDX = {name: i for i, name in enumerate(ECOLE_ROW_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sample(n_vars: int = 6, n_rows: int = 4, seed: int = 0) -> BranchingSample:
    """A synthetic sample: mixed binary/integer/continuous variables and a
    hand-built edge structure, for exercising the transformation rules."""
    rng = np.random.default_rng(seed)
    variable_features = rng.random((n_vars, len(ECOLE_COL_FEATURE_NAMES)))
    # Variable types: binary, binary, integer, integer, continuous, continuous.
    for name in ("binary", "integer", "implicit_integer", "continuous"):
        variable_features[:, VAR_IDX[name]] = 0.0
    variable_features[[0, 1], VAR_IDX["binary"]] = 1.0
    variable_features[[2, 3], VAR_IDX["integer"]] = 1.0
    variable_features[[4, 5], VAR_IDX["continuous"]] = 1.0
    row_features = rng.random((n_rows, len(ECOLE_ROW_FEATURE_NAMES)))
    # A hand-built sparse structure: row i touches variables i and i+1.
    indices = np.array(
        [
            [r for r in range(n_rows) for _ in (0, 1)],
            [c for r in range(n_rows) for c in (r, r + 1)],
        ],
        dtype=np.int64,
    )
    values = rng.random(indices.shape[1])
    observation = BipartiteGraph(
        variable_features=variable_features,
        row_features=row_features,
        edge_features=EdgeFeatures(indices=indices, values=values),
    )
    action_set = np.array([1, 3, 4])
    return BranchingSample(
        observation=observation,
        action_set=action_set,
        action=3,
        scores=rng.random(n_vars),
    )


@pytest.fixture(scope="module")
def collected():
    """A small expert dataset from tiny set covers (a handful of samples)."""
    generator = SetCoverGenerator(
        n_rows=150, n_cols=200, density=0.10, rng=np.random.default_rng(7)
    )
    result = collect(
        generator,
        n_episodes=4,
        seeds=[0, 1, 2, 3],
        expert_probability=1.0,
        time_limit=10.0,
    )
    return result


# ---------------------------------------------------------------------------
# The transformation (unit rules)
# ---------------------------------------------------------------------------


class TestShiftSample:
    def test_primal_features_shift_by_s(self):
        sample = make_sample()
        shift = np.array([2, -1, 0, 1, -2, 2])
        augmented = shift_sample(sample, shift)
        var = augmented.observation.variable_features
        for name in ("sol_val", "best_incumbent_val", "avg_incumbent_val"):
            assert np.allclose(
                var[:, VAR_IDX[name]],
                sample.observation.variable_features[:, VAR_IDX[name]] + shift,
            )

    def test_binary_flips_to_integer_only_where_shifted(self):
        sample = make_sample()
        shift = np.array([2, 0, -1, 1, 0, 0])  # var 0 shifted, var 1 not
        var = shift_sample(sample, shift).observation.variable_features
        # Binary + s != 0 -> integer.
        assert var[0, VAR_IDX["binary"]] == 0.0 and var[0, VAR_IDX["integer"]] == 1.0
        # Binary + s == 0 -> stays binary.
        assert var[1, VAR_IDX["binary"]] == 1.0 and var[1, VAR_IDX["integer"]] == 0.0
        # Integer variables are untouched by the type columns whatever the shift.
        assert var[2, VAR_IDX["binary"]] == 0.0 and var[2, VAR_IDX["integer"]] == 1.0
        assert var[3, VAR_IDX["binary"]] == 0.0 and var[3, VAR_IDX["integer"]] == 1.0
        # Continuous too.
        assert var[4, VAR_IDX["continuous"]] == 1.0

    def test_every_other_variable_feature_is_invariant(self):
        sample = make_sample()
        shift = np.array([1, -2, 2, 0, -1, 1])
        var = shift_sample(sample, shift).observation.variable_features
        original = sample.observation.variable_features
        shifted_cols = {
            "sol_val",
            "best_incumbent_val",
            "avg_incumbent_val",
            "binary",
            "integer",
        }
        for name in ECOLE_COL_FEATURE_NAMES:
            if name not in shifted_cols:
                assert np.allclose(var[:, VAR_IDX[name]], original[:, VAR_IDX[name]]), (
                    name
                )

    def test_row_bias_shifts_by_the_edge_weighted_sum(self):
        sample = make_sample()
        shift = np.array([2, -1, 1, 0, 3, -2])
        augmented = shift_sample(sample, shift)
        edge = sample.observation.edge_features
        rows, cols = np.asarray(edge.indices)
        expected = np.zeros(augmented.observation.row_features.shape[0])
        for e in range(len(cols)):
            expected[rows[e]] += edge.values[e] * shift[cols[e]]
        assert np.allclose(
            augmented.observation.row_features[:, ROW_IDX["bias"]],
            sample.observation.row_features[:, ROW_IDX["bias"]] + expected,
        )
        # The other row features are invariant.
        for name in ("obj_cosine", "is_tight", "dual_sol", "age"):
            assert np.allclose(
                augmented.observation.row_features[:, ROW_IDX[name]],
                sample.observation.row_features[:, ROW_IDX[name]],
            ), name

    def test_labels_edges_and_originals_are_untouched(self):
        sample = make_sample()
        augmented = shift_sample(sample, np.array([1, 1, 1, 1, 1, 1]))
        # Thm 3.1: the action set, action and expert scores carry over.
        assert augmented.action_set is sample.action_set
        assert augmented.action == sample.action
        assert augmented.scores is sample.scores
        # The edge structure is invariant (and shared, never mutated).
        assert augmented.observation.edge_features is sample.observation.edge_features
        # The original observation is left unmutated.
        assert np.allclose(
            sample.observation.variable_features,
            make_sample().observation.variable_features,
        )


class TestDrawShift:
    def test_shifts_are_integers_on_discrete_variables_only(self):
        sample = make_sample()
        rng = np.random.default_rng(0)
        seen_nonzero = False
        for _ in range(10):
            shift = draw_shift(sample.observation.variable_features, rng)
            assert set(np.unique(shift)) <= set(SHIFT_SUPPORT.tolist())
            assert np.all(shift[[4, 5]] == 0)  # continuous: untouched
            seen_nonzero |= bool((shift[[0, 1, 2, 3]] != 0).any())
        assert seen_nonzero  # the discrete variables do get shifted

    def test_zero_shift_gives_the_identity_augmentation(self):
        sample = make_sample()
        augmented = shift_sample(sample, np.zeros(6))
        assert np.allclose(
            augmented.observation.variable_features,
            sample.observation.variable_features,
        )
        assert np.allclose(
            augmented.observation.row_features, sample.observation.row_features
        )


class TestAugmentPairs:
    def test_every_sample_yields_n_augmented_pairs(self):
        samples = [make_sample(seed=s) for s in range(3)]
        pairs = augment_pairs(samples, np.random.default_rng(0), n_augmented=4)
        assert len(pairs) == 12
        # The originals are the samples themselves (shared, not copied).
        assert [id(p[0]) for p in pairs] == [id(s) for s in samples for _ in range(4)]

    def test_augmented_copies_differ_from_their_original(self):
        pairs = augment_pairs([make_sample()], np.random.default_rng(1))
        changed = any(
            not np.allclose(
                p[1].observation.variable_features, p[0].observation.variable_features
            )
            for p in pairs
        )
        assert changed  # at least one drawn shift is nonzero

    def test_default_count_is_the_papers_nine(self):
        assert N_AUGMENTED == 9
        assert len(augment_pairs([make_sample()], np.random.default_rng(2))) == 9


# ---------------------------------------------------------------------------
# The empirical check: the transformation vs a re-solved AMILP
# ---------------------------------------------------------------------------


class TestEmpiricalShift:
    def test_transformed_features_match_a_resolved_amilp(self):
        """The paper's Tables 4-5, validated against SCIP itself: build a set
        cover A and its AMILP B = A + s by hand, extract both root
        observations, and check that ``shift_sample`` on A's observation
        reproduces B's features.

        Integer [0, 1] variables on both sides (SCIP upgrades rows over
        binary variables to specialized constraints, which would diverge the
        row structures); the objective's constant offset -c^T s changes no
        feature. The ``age``, ``best_incumbent_val`` and ``avg_incumbent_val``
        columns are exempt: they depend on the LP counter and on which
        heuristic found the incumbent — re-solve artifacts the method never
        sees (it transforms collected features, it does not re-solve).
        """
        rng = np.random.default_rng(7)
        n_rows, n_cols = 150, 150
        matrix = (rng.random((n_rows, n_cols)) < 0.05).astype(float)
        for i in range(n_rows):
            if not matrix[i].any():
                matrix[i, rng.integers(n_cols)] = 1.0
        cost = rng.integers(1, 101, size=n_cols).astype(float)
        shift = rng.integers(-2, 3, size=n_cols)

        def build(shifted):
            model = Model()
            model.hideOutput()
            model.setMinimize()
            for param in (
                "presolving/maxrounds",
                "separating/maxrounds",
                "separating/maxroundsroot",
                "presolving/maxrestarts",
            ):
                model.setParam(param, 0)
            variables = []
            for j in range(n_cols):
                lb, ub = (
                    (float(shift[j]), float(1 + shift[j])) if shifted else (0.0, 1.0)
                )
                variables.append(
                    model.addVar(
                        vtype="I", lb=lb, ub=ub, name=f"x_{j}", obj=float(cost[j])
                    )
                )
            for i in range(n_rows):
                lhs = quicksum(matrix[i, j] * variables[j] for j in range(n_cols))
                if shifted:
                    model.addCons(lhs >= 1.0 + float(matrix[i] @ shift))
                else:
                    model.addCons(lhs >= 1.0)
            return model

        def root_observation(model):
            dynamics = BranchingDynamics()
            done, action_set = dynamics.reset(model)
            assert not done and len(action_set) > 0
            observation = NodeBipartite().extract(model, done=False)
            dynamics.close()
            return observation

        original, shifted_model = build(False), build(True)
        obs_a, obs_b = root_observation(original), root_observation(shifted_model)

        # SCIP permutes the transformed variables its own way (differently
        # per model) — align B onto A by variable name.
        names_a = [v.name for v in original.getVars(transformed=True)]
        names_b = [v.name for v in shifted_model.getVars(transformed=True)]
        assert sorted(names_a) == sorted(names_b)
        perm = np.array([names_b.index(n) for n in names_a])
        var_b = obs_b.variable_features[perm]
        edge_indices_b = np.asarray(obs_b.edge_features.indices).copy()
        inverse = np.empty(n_cols, dtype=int)
        inverse[perm] = np.arange(n_cols)
        edge_indices_b[1, :] = inverse[edge_indices_b[1, :]]

        sample = BranchingSample(
            observation=obs_a,
            action_set=np.arange(n_cols),
            action=0,
            scores=np.zeros(n_cols),
        )
        transformed = shift_sample(sample, shift)

        # Every variable column the method relies on matches B's, except the
        # re-solve artifacts (age, incumbents).
        exempt = {"age", "best_incumbent_val", "avg_incumbent_val"}
        for name in ECOLE_COL_FEATURE_NAMES:
            if name in exempt:
                continue
            assert np.allclose(
                transformed.observation.variable_features[:, VAR_IDX[name]],
                var_b[:, VAR_IDX[name]],
            ), f"variable feature {name}"
        # Every row column except age.
        for name in ECOLE_ROW_FEATURE_NAMES:
            if name == "age":
                continue
            assert np.allclose(
                transformed.observation.row_features[:, ROW_IDX[name]],
                obs_b.row_features[:, ROW_IDX[name]],
            ), f"row feature {name}"
        assert np.allclose(
            transformed.observation.edge_features.values, obs_b.edge_features.values
        )

        # SCIP orders each model's row nonzeros its own way — the edge
        # structure compares as a (row, aligned column, value) multiset.
        def edge_triples(indices, values):
            return sorted(
                zip(
                    indices[0].tolist(),
                    indices[1].tolist(),
                    np.round(values, 10).tolist(),
                )
            )

        assert edge_triples(
            np.asarray(transformed.observation.edge_features.indices),
            np.asarray(transformed.observation.edge_features.values),
        ) == edge_triples(edge_indices_b, np.asarray(obs_b.edge_features.values))


# ---------------------------------------------------------------------------
# The model and the losses
# ---------------------------------------------------------------------------


class TestModel:
    def test_forward_without_segments_is_the_gasse_contract(self, collected):
        model = CAMBranchGNN()
        sample = collected.samples[0]
        obs = sample.observation
        logits = model(
            torch.as_tensor(obs.row_features, dtype=torch.float32),
            torch.as_tensor(obs.edge_features.indices, dtype=torch.long),
            torch.as_tensor(obs.edge_features.values, dtype=torch.float32).reshape(
                -1, 1
            ),
            torch.as_tensor(obs.variable_features, dtype=torch.float32),
        )
        assert logits.shape == (obs.variable_features.shape[0],)
        assert bool(np.isfinite(logits.detach().numpy()).all())

    def test_graph_embeddings_are_unit_norm_and_per_sample(self, collected):
        model = CAMBranchGNN().eval()
        pairs = augment_pairs(
            collected.samples[:3], np.random.default_rng(0), n_augmented=1
        )
        pb = collate_pairs(pairs)
        with torch.no_grad():
            logits, g = model(
                *batched_tensors(pb.batch),
                n_vars_per_sample=pb.n_vars,
                n_cons_per_sample=pb.n_cons,
            )
        assert logits.shape == (sum(pb.n_vars),)
        assert g.shape == (len(pairs) * 2, 64)
        assert torch.allclose(g.norm(dim=-1), torch.ones(len(pairs) * 2), atol=1e-5)
        # Distinct graphs give distinct embeddings (pair 0 vs pair 1 originals).
        assert not torch.allclose(g[0], g[2])

    def test_segments_do_not_change_the_logits(self, collected):
        model = CAMBranchGNN().eval()
        pairs = augment_pairs(
            collected.samples[:2], np.random.default_rng(0), n_augmented=1
        )
        pb = collate_pairs(pairs)
        with torch.no_grad():
            plain = model(*batched_tensors(pb.batch))
            segmented, _ = model(
                *batched_tensors(pb.batch),
                n_vars_per_sample=pb.n_vars,
                n_cons_per_sample=pb.n_cons,
            )
        assert torch.allclose(plain, segmented, atol=1e-6)

    def test_gradients_flow_through_the_projection_head(self, collected):
        model = CAMBranchGNN()
        pairs = augment_pairs(
            collected.samples[:2], np.random.default_rng(0), n_augmented=1
        )
        pb = collate_pairs(pairs)
        logits, g = model(
            *batched_tensors(pb.batch),
            n_vars_per_sample=pb.n_vars,
            n_cons_per_sample=pb.n_cons,
        )
        (logits.sum() + g.sum()).backward()
        assert model.graph_projection[0].weight.grad is not None


class TestLosses:
    def test_infonce_prefers_matching_pairs_by_the_cosine_margin(self):
        # Unit-norm embeddings, no temperature: the logits are cosine
        # similarities in [-1, 1]. Perfect pairs score log(1 + e^-1), fully
        # mismatched ones log(1 + e) — one apart either way.
        identity = torch.eye(2)
        assert infonce_loss(identity, identity).item() == pytest.approx(
            np.log(1 + np.exp(-1.0)), rel=1e-5
        )
        swapped = torch.flip(identity, dims=(1,))
        assert infonce_loss(identity, swapped).item() == pytest.approx(
            np.log(1 + np.exp(1.0)), rel=1e-5
        )

    def test_candidate_probs_masks_to_the_candidates(self):
        logits = torch.tensor([1.0, 5.0, 3.0])
        probs = candidate_probs(logits, torch.tensor([0, 2]))
        assert probs[1].item() == pytest.approx(0.0, abs=1e-4)
        total = np.exp(1.0) + np.exp(3.0)
        assert probs[0].item() == pytest.approx(np.exp(1.0) / total, rel=1e-5)
        assert probs[2].item() == pytest.approx(np.exp(3.0) / total, rel=1e-5)

    def test_auxiliary_loss_sums_squared_differences(self):
        p_original = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        p_augmented = torch.tensor([[0.0, 1.0], [0.5, 0.5]])
        # Pair 0 differs on both variables (2.0), pair 1 not at all.
        assert auxiliary_loss(p_original, p_augmented).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


class TestTrain:
    def test_collate_pairs_interleaves_and_offsets(self, collected):
        pairs = augment_pairs(
            collected.samples[:3], np.random.default_rng(0), n_augmented=1
        )
        pb = collate_pairs(pairs)
        flat = [sample for pair in pairs for sample in pair]
        assert pb.n_vars == [s.observation.variable_features.shape[0] for s in flat]
        reference = collate(flat)
        assert torch.equal(pb.batch.candidates[0], reference.candidates[0])
        assert torch.equal(pb.batch.choices, reference.choices)

    def test_process_runs_the_joint_loss(self, collected):
        model = CAMBranchGNN()
        pairs = augment_pairs(
            collected.samples, np.random.default_rng(0), n_augmented=1
        )
        # One single batch, so the component means combine exactly.
        batches = pair_batches(
            pairs,
            torch.device("cpu"),
            pairs_per_batch=10_000,
            shuffle=np.random.default_rng(0),
        )
        stats = process(model, batches)
        for key in ("loss", "ce", "infonce", "aux"):
            assert np.isfinite(stats[key]), key
        assert stats["kacc"].shape == (4,)
        # The joint loss ties the components with the paper's weights.
        assert stats["loss"] == pytest.approx(
            stats["ce"] + LAMBDA_1 * stats["infonce"] + LAMBDA_2 * stats["aux"],
            rel=1e-4,
        )

    def test_process_handles_samples_of_different_sizes(self):
        # The auxiliary term pairs an original with its own AMILP — a batch
        # of mixed-size graphs must not try to stack the distributions.
        samples = [
            make_sample(n_vars=6, n_rows=4, seed=0),
            make_sample(n_vars=8, n_rows=5, seed=1),
        ]
        pairs = augment_pairs(samples, np.random.default_rng(0), n_augmented=1)
        model = CAMBranchGNN()
        stats = process(model, [collate_pairs(pairs)])
        for key in ("loss", "ce", "infonce", "aux"):
            assert np.isfinite(stats[key]), key

    def test_process_steps_the_optimizer(self, collected):
        model = CAMBranchGNN()
        pairs = augment_pairs(
            collected.samples[:4], np.random.default_rng(0), n_augmented=1
        )
        batches = pair_batches(
            pairs,
            torch.device("cpu"),
            pairs_per_batch=4,
            shuffle=np.random.default_rng(0),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.output_module[0].weight.detach().clone()
        process(model, batches, optimizer=optimizer)
        assert not torch.allclose(model.output_module[0].weight.detach(), before)

    def test_train_end_to_end_and_checkpoint_roundtrip(self, collected, tmp_path):
        checkpoint = tmp_path / "cambranch.pt"
        train(
            collected.samples,
            checkpoint,
            epochs=2,
            n_augmented=2,
            pairs_per_batch=8,
            extra={"n_samples": len(collected.samples)},
        )
        assert checkpoint.exists()
        loaded = load_checkpoint(checkpoint)
        sample = collected.samples[0]
        logits = loaded(
            torch.as_tensor(sample.observation.row_features, dtype=torch.float32),
            torch.as_tensor(sample.observation.edge_features.indices, dtype=torch.long),
            torch.as_tensor(
                sample.observation.edge_features.values, dtype=torch.float32
            ).reshape(-1, 1),
            torch.as_tensor(sample.observation.variable_features, dtype=torch.float32),
        )
        assert bool(np.isfinite(logits.detach().numpy()).all())

    def test_subsample_budget_matches_the_papers_ten_percent(self, collected):
        kept = subsample_budget(collected.samples, rng=np.random.default_rng(0))
        assert len(kept) == max(1, round(0.1 * len(collected.samples)))
        # Kept samples are originals (shared identities), never copies.
        assert all(
            any(kept_one is original for original in collected.samples)
            for kept_one in kept
        )

    def test_plateau_scheduler_cuts_after_patience(self, collected):
        model = CAMBranchGNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = PlateauScheduler(optimizer)
        scheduler.step(1.0)  # the baseline
        for _ in range(10):
            lr = scheduler.step(1.0)  # no improvement
        assert lr == pytest.approx(0.2e-3)
        scheduler.step(0.5)  # improvement resets the counter
        scheduler.step(1.0)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.2e-3)


class TestPolicy:
    def test_policy_selects_within_the_action_set(self, collected, tmp_path):
        checkpoint = tmp_path / "cambranch.pt"
        train(collected.samples, checkpoint, epochs=1, n_augmented=1, pairs_per_batch=8)
        policy = CAMBranchPolicy.from_checkpoint(checkpoint)
        for sample in collected.samples:
            action = policy.select_action(sample.observation, sample.action_set)
            assert action in sample.action_set

    def test_cambranch_method_is_runner_ready(self, collected, tmp_path):
        checkpoint = tmp_path / "cambranch.pt"
        train(collected.samples, checkpoint, epochs=1, n_augmented=1, pairs_per_batch=8)
        method = cambranch_method(checkpoint)
        assert method.name == "cambranch"
        assert method.fidelity == "cambranch"
        policy = method.policy()
        sample = collected.samples[0]
        assert (
            policy.select_action(sample.observation, sample.action_set)
            in sample.action_set
        )
