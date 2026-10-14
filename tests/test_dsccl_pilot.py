"""End-to-end tests of the DSC-CL 2025 pilot: derive → group → train → policy.

The instances are tiny set covers — the point is that the LT-MILP feature
transformation matches a re-solved transformed MILP feature for feature
(validated against SCIP itself), that the RC-MILP row matches an appended
redundant constraint, and that the whole stratified loop runs, not that the
model learns anything.
"""

import math

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
from pyscipopt import SCIP_PARAMSETTING, Model, quicksum

from l2b_bench.methods.dsccl.augment import (
    EQUIV_PROB,
    NOISE_STD,
    PERTURB_PROB,
    T_SHIFT_SUPPORT,
    derive_sample,
    draw_transformation,
    lt_sample,
    perturb_sample,
    rc_sample,
)
from l2b_bench.methods.dsccl.model import DSCCLGNN
from l2b_bench.methods.dsccl.policy import (
    TOP_K_HYBRID,
    DSCCLPolicy,
    dsccl_method,
)
from l2b_bench.methods.dsccl.train import (
    LAMBDA_INIT,
    TAU,
    StratifiedContrast,
    assign_groups,
    collate_batch,
    load_checkpoint,
    process,
    sample_batches,
    train,
)
from l2b_bench.methods.gasse.collect import BranchingSample, collect

VAR_IDX = {name: i for i, name in enumerate(ECOLE_COL_FEATURE_NAMES)}
ROW_IDX = {name: i for i, name in enumerate(ECOLE_ROW_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sample(n_vars: int = 6, n_rows: int = 4, seed: int = 0) -> BranchingSample:
    """A synthetic sample: mixed binary/integer/continuous variables and a
    hand-built edge structure, for exercising the derivation rules."""
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
# The LT-MILP transformation (Table 5, unit rules)
# ---------------------------------------------------------------------------


class TestLTSample:
    def test_objective_and_dual_features_scale_by_T(self):
        sample = make_sample()
        T = np.array([1, -1, 1, -1, 1, -1])
        t = np.zeros(6, dtype=int)
        var = lt_sample(sample, T, t).observation.variable_features
        original = sample.observation.variable_features
        for name in ("obj_coef", "red_cost"):
            assert np.allclose(var[:, VAR_IDX[name]], T * original[:, VAR_IDX[name]]), (
                name
            )

    def test_primal_features_map_to_TV_plus_t(self):
        sample = make_sample()
        T = np.array([1, -1, 1, -1, 1, -1])
        t = np.array([2, -1, 0, 1, -2, 3])
        var = lt_sample(sample, T, t).observation.variable_features
        original = sample.observation.variable_features
        for name in ("sol_val", "best_incumbent_val", "avg_incumbent_val"):
            assert np.allclose(
                var[:, VAR_IDX[name]], T * original[:, VAR_IDX[name]] + t
            ), name

    def test_bound_and_basis_pairs_swap_where_T_flips(self):
        sample = make_sample()
        T = np.array([1, -1, 1, -1, 1, -1])
        t = np.zeros(6, dtype=int)
        var = lt_sample(sample, T, t).observation.variable_features
        original = sample.observation.variable_features
        for a, b in (
            ("has_lb", "has_ub"),
            ("sol_at_lb", "sol_at_ub"),
            ("basis_lower", "basis_upper"),
        ):
            # Unflipped rows (T = +1) unchanged, flipped rows (T = -1) swapped.
            assert np.allclose(var[T == 1, VAR_IDX[a]], original[T == 1, VAR_IDX[a]])
            assert np.allclose(var[T == 1, VAR_IDX[b]], original[T == 1, VAR_IDX[b]])
            assert np.allclose(var[T == -1, VAR_IDX[a]], original[T == -1, VAR_IDX[b]])
            assert np.allclose(var[T == -1, VAR_IDX[b]], original[T == -1, VAR_IDX[a]])

    def test_sol_frac_reflects_modulo_one(self):
        sample = make_sample()
        original = sample.observation.variable_features.copy()
        original[:, VAR_IDX["sol_frac"]] = [0.0, 0.25, 0.5, 0.75, 1.0, 0.3]
        sample.observation.variable_features = original
        T = np.array([1, -1, -1, -1, -1, 1])
        t = np.zeros(6, dtype=int)
        var = lt_sample(sample, T, t).observation.variable_features
        expected = [0.0, 0.75, 0.5, 0.25, 0.0, 0.3]
        assert np.allclose(var[:, VAR_IDX["sol_frac"]], expected)

    def test_binary_conversion_follows_the_image_of_zero_one(self):
        sample = make_sample()
        # Vars 0-1 are binary. (T, t) = (+1, 0) keeps var 0 binary; (+1, 2)
        # makes var 1 integer; (-1, 1) would keep a binary binary; (-1, 0)
        # makes it integer.
        T = np.array([1, 1, 1, 1, 1, 1])
        t = np.array([0, 2, 0, 0, 0, 0])
        var = lt_sample(sample, T, t).observation.variable_features
        assert var[0, VAR_IDX["binary"]] == 1.0
        assert var[1, VAR_IDX["binary"]] == 0.0
        assert var[1, VAR_IDX["integer"]] == 1.0
        T = np.array([-1, -1, 1, 1, 1, 1])
        t = np.array([1, 0, 0, 0, 0, 0])
        var = lt_sample(sample, T, t).observation.variable_features
        # x_hat = 1 - x maps {0, 1} onto {0, 1}: still binary.
        assert var[0, VAR_IDX["binary"]] == 1.0
        # x_hat = -x maps {0, 1} onto {-1, 0}: integer.
        assert var[1, VAR_IDX["binary"]] == 0.0
        assert var[1, VAR_IDX["integer"]] == 1.0

    def test_edges_scale_and_bias_shifts_by_the_row_weighted_sum(self):
        sample = make_sample()
        T = np.array([1, -1, 1, -1, 1, -1])
        t = np.array([2, -1, 0, 1, -2, 3])
        derived = lt_sample(sample, T, t).observation
        edge = sample.observation.edge_features
        rows, cols = np.asarray(edge.indices)[0], np.asarray(edge.indices)[1]
        assert np.allclose(
            np.asarray(derived.edge_features.values),
            np.asarray(edge.values) * T[cols],
        )
        # bias shift = a_i^T T t / ||a_i|| — the row-wise edge-weighted sum
        # of the scaled shifts.
        expected = np.bincount(
            rows,
            weights=np.asarray(edge.values) * (T * t)[cols],
            minlength=sample.observation.row_features.shape[0],
        )
        assert np.allclose(
            derived.row_features[:, ROW_IDX["bias"]],
            sample.observation.row_features[:, ROW_IDX["bias"]] + expected,
        )

    def test_everything_else_is_invariant(self):
        sample = make_sample()
        T = np.array([1, -1, 1, -1, 1, -1])
        t = np.array([2, -1, 0, 1, -2, 3])
        derived = lt_sample(sample, T, t)
        var = derived.observation.variable_features
        original = sample.observation.variable_features
        assert np.allclose(var[:, VAR_IDX["age"]], original[:, VAR_IDX["age"]])
        assert np.allclose(
            var[:, VAR_IDX["basis_basic"]], original[:, VAR_IDX["basis_basic"]]
        )
        for name in ("obj_cosine", "is_tight", "dual_sol", "age"):
            assert np.allclose(
                derived.observation.row_features[:, ROW_IDX[name]],
                sample.observation.row_features[:, ROW_IDX[name]],
            ), name
        # The labels carry over (Thm 4.3) — shared, not copied.
        assert derived.action_set is sample.action_set
        assert derived.scores is sample.scores

    def test_draw_transformation_leaves_continuous_variables_alone(self):
        sample = make_sample()
        rng = np.random.default_rng(0)
        for _ in range(20):
            T, t = draw_transformation(sample.observation.variable_features, rng)
            assert np.all(T[[4, 5]] == 1)
            assert np.all(t[[4, 5]] == 0)
            assert set(np.unique(T[:4])).issubset({-1, 1})
            assert all(ti in T_SHIFT_SUPPORT for ti in t[:4])


# ---------------------------------------------------------------------------
# The RC-MILP derivation (App. D.3, unit rules)
# ---------------------------------------------------------------------------


class TestRCSample:
    def test_appends_one_row_and_its_edges(self):
        sample = make_sample()
        derived = rc_sample(sample, np.random.default_rng(0))
        obs = derived.observation
        assert obs.row_features.shape[0] == sample.observation.row_features.shape[0] + 1
        n_edges = len(np.asarray(sample.observation.edge_features.values))
        assert len(np.asarray(obs.edge_features.values)) > n_edges
        # The pre-existing rows and edges are untouched (shared arrays).
        assert np.allclose(obs.row_features[:-1], sample.observation.row_features)
        assert (
            derived.observation.variable_features
            is sample.observation.variable_features
        )

    def test_new_row_features_follow_the_normalized_sum(self):
        sample = make_sample()
        rng = np.random.default_rng(3)
        derived = rc_sample(sample, rng).observation
        edge = sample.observation.edge_features
        rows = np.asarray(edge.indices)[0]
        cols = np.asarray(edge.indices)[1]
        vals = np.asarray(edge.values)
        new_row = derived.row_features.shape[0] - 1
        new_cols = np.asarray(derived.edge_features.indices)[1][
            np.asarray(derived.edge_features.indices)[0] == new_row
        ]
        new_vals = np.asarray(derived.edge_features.values)[
            np.asarray(derived.edge_features.indices)[0] == new_row
        ]
        # Reconstruct: the summed pair is the one whose supports' union matches.
        candidates = []
        for i in range(sample.observation.row_features.shape[0]):
            for j in range(i + 1, sample.observation.row_features.shape[0]):
                union = sorted(set(cols[rows == i]) | set(cols[rows == j]))
                if union == sorted(new_cols.tolist()):
                    candidates.append((i, j))
        assert len(candidates) == 1
        i, j = candidates[0]
        coef = {
            c: vals[(rows == i) & (cols == c)].sum()
            + vals[(rows == j) & (cols == c)].sum()
            for c in new_cols
        }
        norm = float(np.sqrt(sum(v * v for v in coef.values())))
        assert np.allclose(new_vals, [coef[c] / norm for c in new_cols])
        bias = derived.row_features[new_row, ROW_IDX["bias"]]
        assert bias == pytest.approx(
            (
                sample.observation.row_features[i, ROW_IDX["bias"]]
                + sample.observation.row_features[j, ROW_IDX["bias"]]
            )
            / norm
        )
        cosine = derived.row_features[new_row, ROW_IDX["obj_cosine"]]
        assert cosine == pytest.approx(
            (
                sample.observation.row_features[i, ROW_IDX["obj_cosine"]]
                + sample.observation.row_features[j, ROW_IDX["obj_cosine"]]
            )
            / norm
        )
        # App. D.3: the new row's dynamic features are zero.
        for name in ("is_tight", "dual_sol", "age"):
            assert derived.row_features[new_row, ROW_IDX[name]] == 0.0, name

    def test_summed_rows_have_distinct_supports(self):
        # Linearly independent rows: the fixture's rows i and i+1 share one
        # column, so any pair the draw picks has distinct supports.
        sample = make_sample()
        supports = _supports_of(sample)
        for seed in range(10):
            derived = rc_sample(sample, np.random.default_rng(seed))
            new_row = derived.observation.row_features.shape[0] - 1
            new_support = frozenset(
                np.asarray(derived.observation.edge_features.indices)[1][
                    np.asarray(derived.observation.edge_features.indices)[0] == new_row
                ].tolist()
            )
            assert new_support in {a | b for a in supports for b in supports if a != b}

    def test_labels_carry_over(self):
        sample = make_sample()
        derived = rc_sample(sample, np.random.default_rng(0))
        assert derived.action_set is sample.action_set
        assert derived.scores is sample.scores


def _supports_of(sample):
    edge = sample.observation.edge_features
    rows = np.asarray(edge.indices)[0]
    cols = np.asarray(edge.indices)[1]
    return [
        frozenset(cols[rows == r].tolist())
        for r in range(sample.observation.row_features.shape[0])
    ]


# ---------------------------------------------------------------------------
# The perturbed derivations (Defs 4.6-4.7)
# ---------------------------------------------------------------------------


class TestPerturbSample:
    def test_objective_perturbation_touches_only_obj_coef(self):
        sample = make_sample()
        derived = perturb_sample(sample, np.random.default_rng(0), "objective")
        var = derived.observation.variable_features
        original = sample.observation.variable_features
        changed = ~np.isclose(var, original)
        assert changed[:, VAR_IDX["obj_coef"]].any()
        assert np.allclose(
            var[:, VAR_IDX["obj_coef"]], original[:, VAR_IDX["obj_coef"]], atol=1
        )
        assert not changed[:, : VAR_IDX["obj_coef"]].any()
        assert not changed[:, VAR_IDX["obj_coef"] + 1 :].any()
        assert derived.observation.row_features is sample.observation.row_features
        assert derived.observation.edge_features is sample.observation.edge_features

    def test_constraint_perturbation_touches_edges_and_bias(self):
        sample = make_sample()
        derived = perturb_sample(sample, np.random.default_rng(0), "constraint")
        assert not np.allclose(
            np.asarray(derived.observation.edge_features.values),
            np.asarray(sample.observation.edge_features.values),
        )
        assert not np.allclose(
            derived.observation.row_features[:, ROW_IDX["bias"]],
            sample.observation.row_features[:, ROW_IDX["bias"]],
        )
        # Only the bias moves among the row features.
        for name in ("obj_cosine", "is_tight", "dual_sol", "age"):
            assert np.allclose(
                derived.observation.row_features[:, ROW_IDX[name]],
                sample.observation.row_features[:, ROW_IDX[name]],
            ), name
        assert (
            derived.observation.variable_features
            is sample.observation.variable_features
        )

    def test_dual_perturbation_touches_only_dual_sol(self):
        sample = make_sample()
        derived = perturb_sample(sample, np.random.default_rng(0), "dual")
        assert not np.allclose(
            derived.observation.row_features[:, ROW_IDX["dual_sol"]],
            sample.observation.row_features[:, ROW_IDX["dual_sol"]],
        )
        for name in ("bias", "obj_cosine", "is_tight", "age"):
            assert np.allclose(
                derived.observation.row_features[:, ROW_IDX[name]],
                sample.observation.row_features[:, ROW_IDX[name]],
            ), name

    def test_noise_magnitude_is_the_pilot_choice(self):
        # A single variable feature column, measured over many rows.
        sample2 = make_sample(n_vars=6, n_rows=2000, seed=0)
        derived = perturb_sample(sample2, np.random.default_rng(0), "dual")
        delta = (
            derived.observation.row_features[:, ROW_IDX["dual_sol"]]
            - sample2.observation.row_features[:, ROW_IDX["dual_sol"]]
        )
        assert abs(delta.std() - NOISE_STD) < 0.1 * NOISE_STD

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown perturbation"):
            perturb_sample(make_sample(), np.random.default_rng(0), "objective*s")


# ---------------------------------------------------------------------------
# The derivation dispatch (35% / 10%)
# ---------------------------------------------------------------------------


class TestDeriveSample:
    def test_probabilities_match_the_paper(self, monkeypatch):
        import l2b_bench.methods.dsccl.augment as aug

        calls = {"lt": 0, "rc": 0, "perturb": 0, "original": 0}

        def fake_lt(sample, T, t):
            calls["lt"] += 1
            return sample

        def fake_rc(sample, rng):
            calls["rc"] += 1
            return sample

        def fake_perturb(sample, rng, kind):
            calls["perturb"] += 1
            return sample

        monkeypatch.setattr(aug, "lt_sample", fake_lt)
        monkeypatch.setattr(aug, "rc_sample", fake_rc)
        monkeypatch.setattr(aug, "perturb_sample", fake_perturb)
        sample = make_sample()
        rng = np.random.default_rng(0)
        n = 20_000
        for _ in range(n):
            derive_sample(sample, rng)
        derived_total = calls["lt"] + calls["rc"] + calls["perturb"]
        # The rest are originals: 55%.
        assert (n - derived_total) / n == pytest.approx(
            1 - EQUIV_PROB - PERTURB_PROB, abs=0.01
        )
        # 35% equivalent (LT and RC splitting it evenly), 10% perturbed.
        assert (calls["lt"] + calls["rc"]) / n == pytest.approx(EQUIV_PROB, abs=0.01)
        assert calls["lt"] / (calls["lt"] + calls["rc"]) == pytest.approx(0.5, abs=0.02)
        assert calls["perturb"] / n == pytest.approx(PERTURB_PROB, abs=0.005)

    def test_derived_samples_keep_the_labels(self):
        sample = make_sample()
        rng = np.random.default_rng(1)
        for _ in range(50):
            derived = derive_sample(sample, rng)
            assert derived.action == sample.action
            assert np.allclose(np.asarray(derived.scores), np.asarray(sample.scores))


# ---------------------------------------------------------------------------
# The derivations against SCIP itself
# ---------------------------------------------------------------------------


class TestEmpiricalLT:
    def test_transformed_features_match_a_resolved_lt_milp(self):
        """Table 5, validated against SCIP: build a set cover A and its LT-MILP
        B (x_hat = T x + t) by hand, extract both root observations, and check
        that ``lt_sample`` on A's observation reproduces B's features.

        Integer [0, 1] variables on both sides (SCIP upgrades rows over binary
        variables, which would diverge the row structures), and root
        heuristics off on both sides: the root heuristics fix variables and
        delete the constraints their fixings expose, and they do so
        asymmetrically on the reflected model — the mapping is validated
        against the pure root LP.

        The ``age`` and incumbent columns are exempt (re-solve artifacts the
        method never sees), and so are ``red_cost``, ``basis_lower`` and
        ``basis_basic``: the reflected model's dual LP optimum is degenerate,
        so SCIP's re-solve may land on a different vertex of the optimal face
        than the bijection's. Every primal feature — including the
        bound-status and the remaining basis columns — matches exactly. The
        dual rules (red_cost x T, the basis swaps) are validated by the unit
        tests on the mathematical mapping.
        """
        rng = np.random.default_rng(11)
        n_rows, n_cols = 150, 150
        matrix = (rng.random((n_rows, n_cols)) < 0.05).astype(float)
        for i in range(n_rows):
            if not matrix[i].any():
                matrix[i, rng.integers(n_cols)] = 1.0
        cost = rng.integers(1, 101, size=n_cols).astype(float)
        T = rng.choice([-1, 1], size=n_cols)
        t = rng.integers(-2, 3, size=n_cols)

        def build(transformed):
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
            model.setHeuristics(SCIP_PARAMSETTING.OFF)
            variables = []
            for j in range(n_cols):
                if transformed:
                    # x_hat_j = T_jj x_j + t_j over x_j in [0, 1].
                    lb = min(t[j], T[j] + t[j])
                    ub = max(t[j], T[j] + t[j])
                    obj = float(cost[j] * T[j])
                else:
                    lb, ub, obj = 0.0, 1.0, float(cost[j])
                variables.append(
                    model.addVar(vtype="I", lb=lb, ub=ub, name=f"x_{j}", obj=obj)
                )
            for i in range(n_rows):
                if transformed:
                    # sum_j a_ij T_jj x_hat_j >= 1 + a_i^T T t.
                    lhs = quicksum(
                        matrix[i, j] * T[j] * variables[j] for j in range(n_cols)
                    )
                    model.addCons(lhs >= 1.0 + float(matrix[i] @ (T * t)))
                else:
                    lhs = quicksum(matrix[i, j] * variables[j] for j in range(n_cols))
                    model.addCons(lhs >= 1.0)
            return model

        def root_observation(model):
            dynamics = BranchingDynamics()
            try:
                done, action_set = dynamics.reset(model)
                assert not done and len(action_set) > 0
                return NodeBipartite().extract(model, done=False)
            finally:
                # Close even on assertion failure: a leaked solving state
                # segfaults the interpreter at GC time.
                dynamics.close()

        original, transformed_model = build(False), build(True)
        obs_a, obs_b = root_observation(original), root_observation(transformed_model)

        # SCIP permutes the transformed variables its own way — align B onto A.
        names_a = [v.name for v in original.getVars(transformed=True)]
        names_b = [v.name for v in transformed_model.getVars(transformed=True)]
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
        derived = lt_sample(sample, T, t)

        exempt = {
            "age",
            "best_incumbent_val",
            "avg_incumbent_val",
            "red_cost",
            "basis_lower",
            "basis_basic",
        }
        for name in ECOLE_COL_FEATURE_NAMES:
            if name in exempt:
                continue
            assert np.allclose(
                derived.observation.variable_features[:, VAR_IDX[name]],
                var_b[:, VAR_IDX[name]],
            ), f"variable feature {name}"
        for name in ECOLE_ROW_FEATURE_NAMES:
            if name in ("age", "dual_sol"):
                continue
            assert np.allclose(
                derived.observation.row_features[:, ROW_IDX[name]],
                obs_b.row_features[:, ROW_IDX[name]],
            ), f"row feature {name}"

        def edge_triples(indices, values):
            return sorted(
                zip(
                    indices[0].tolist(),
                    indices[1].tolist(),
                    np.round(values, 10).tolist(),
                )
            )

        assert edge_triples(
            np.asarray(derived.observation.edge_features.indices),
            np.asarray(derived.observation.edge_features.values),
        ) == edge_triples(edge_indices_b, np.asarray(obs_b.edge_features.values))


class TestEmpiricalRC:
    def test_new_row_matches_an_appended_redundant_constraint(self):
        """The normalized-sum construction against SCIP: append
        (e_i + e_j) x <= bias_i + bias_j (in true constraint space, the
        positive row scaling of the paper's a_r = a_i + a_j) to a set cover,
        extract, and check the new row's features and edges.

        The pre-existing rows' ``dual_sol`` and the variables' ``red_cost`` /
        basis columns are exempt: the appended row is tight at the LP optimum
        whenever both parents are, and SCIP may then redistribute duals among
        degenerate optima — exactly why the paper zeroes the new row's dual
        features instead of extracting them. Root heuristics are off on both
        sides (they would delete constraints and do so asymmetrically).
        """
        rng = np.random.default_rng(13)
        n_rows, n_cols = 150, 150
        matrix = (rng.random((n_rows, n_cols)) < 0.05).astype(float)
        for i in range(n_rows):
            if not matrix[i].any():
                matrix[i, rng.integers(n_cols)] = 1.0
        cost = rng.integers(1, 101, size=n_cols).astype(float)

        def build(augmented):
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
            model.setHeuristics(SCIP_PARAMSETTING.OFF)
            variables = [
                model.addVar(
                    vtype="I", lb=0.0, ub=1.0, name=f"x_{j}", obj=float(cost[j])
                )
                for j in range(n_cols)
            ]
            rows = [
                quicksum(matrix[i, j] * variables[j] for j in range(n_cols))
                for i in range(n_rows)
            ]
            for i in range(n_rows):
                model.addCons(rows[i] >= 1.0, name=f"cover_{i}")
            if augmented:
                # rows 0 and 1 (distinct supports, hence independent).
                i, j = 0, 1
                model.addCons(
                    rows[i] / np.linalg.norm(matrix[i])
                    + rows[j] / np.linalg.norm(matrix[j])
                    >= 1.0 / np.linalg.norm(matrix[i])
                    + 1.0 / np.linalg.norm(matrix[j]),
                    name="redundant",
                )
            return model

        def root_observation(model):
            dynamics = BranchingDynamics()
            try:
                done, action_set = dynamics.reset(model)
                assert not done and len(action_set) > 0
                return NodeBipartite().extract(model, done=False)
            finally:
                # Close even on assertion failure: a leaked solving state
                # segfaults the interpreter at GC time.
                dynamics.close()

        original, augmented = build(False), build(True)
        obs_a, obs_b = root_observation(original), root_observation(augmented)

        # Align B's rows onto A's by their edge signature (variable name +
        # rounded value pairs): the LP rows expose no names through
        # pyscipopt, but each cover row's support identifies it uniquely, and
        # the appended row is the only signature B has and A lacks.
        def row_signatures(model, observation):
            names = [v.name for v in model.getVars(transformed=True)]
            indices = np.asarray(observation.edge_features.indices)
            values = np.round(np.asarray(observation.edge_features.values), 10)
            signatures = {}
            for row in range(observation.row_features.shape[0]):
                mask = indices[0] == row
                signature = tuple(
                    sorted(
                        zip(
                            (names[col] for col in indices[1][mask]),
                            values[mask].tolist(),
                        )
                    )
                )
                signatures[signature] = row
            return signatures

        signatures_a = row_signatures(original, obs_a)
        signatures_b = row_signatures(augmented, obs_b)
        extra = set(signatures_b) - set(signatures_a)
        assert len(extra) == 1, "the appended row should be B's only extra row"
        order = list(signatures_a) + list(extra)
        row_perm = np.array([signatures_b[signature] for signature in order])
        row_b = obs_b.row_features[row_perm]

        # The derivation, drawn from A's observation: the sample's rows are in
        # A's row order; force the same pair (0, 1) the augmented model uses.
        sample = BranchingSample(
            observation=obs_a,
            action_set=np.arange(n_cols),
            action=0,
            scores=np.zeros(n_cols),
        )
        derived = rc_sample_with_pair(sample, 0, 1).observation
        new_row = derived.row_features.shape[0] - 1

        # The new row's static features against SCIP's extraction.
        for name in ("bias", "obj_cosine"):
            assert np.allclose(
                derived.row_features[new_row, ROW_IDX[name]],
                row_b[n_rows, ROW_IDX[name]],
            ), f"new row feature {name}"
        # The new row's dynamic features: the paper's zeros.
        for name in ("is_tight", "dual_sol", "age"):
            assert derived.row_features[new_row, ROW_IDX[name]] == 0.0, name
        # The new row's edges: same support, same normalized values (B's edge
        # columns need the variable permutation undone).
        names_va = [v.name for v in original.getVars(transformed=True)]
        names_vb = [v.name for v in augmented.getVars(transformed=True)]
        var_perm = np.array([names_vb.index(n) for n in names_va])
        var_inverse = np.empty(n_cols, dtype=int)
        var_inverse[var_perm] = np.arange(n_cols)
        edge_b = np.asarray(obs_b.edge_features.indices).copy()
        edge_b[1, :] = var_inverse[edge_b[1, :]]

        def new_row_edges(indices, values, row):
            mask = indices[0] == row
            return dict(
                zip(indices[1][mask].tolist(), np.round(values[mask], 10).tolist())
            )

        derived_edges = new_row_edges(
            np.asarray(derived.edge_features.indices),
            np.asarray(derived.edge_features.values),
            new_row,
        )
        scip_edges = new_row_edges(
            edge_b, np.asarray(obs_b.edge_features.values), n_rows
        )
        assert set(derived_edges) == set(scip_edges)
        for col in scip_edges:
            assert derived_edges[col] == pytest.approx(scip_edges[col], abs=1e-9)

        # The pre-existing rows: static features and tightness unchanged.
        for name in ("bias", "obj_cosine", "is_tight"):
            assert np.allclose(
                derived.row_features[:n_rows, ROW_IDX[name]],
                row_b[:n_rows, ROW_IDX[name]],
            ), f"row feature {name}"


def rc_sample_with_pair(sample: BranchingSample, i: int, j: int) -> BranchingSample:
    """``rc_sample`` forced onto a given row pair (the empirical test's
    counterpart of the random draw)."""
    from l2b_bench.methods.dsccl.augment import (
        _row_supports,
    )

    edge = sample.observation.edge_features
    edge_rows = np.asarray(edge.indices)[0]
    edge_cols = np.asarray(edge.indices)[1]
    edge_vals = np.asarray(edge.values, dtype=float)
    n_rows = sample.observation.row_features.shape[0]
    supports = _row_supports(edge_rows, edge_cols, n_rows)

    union = sorted(supports[i] | supports[j])
    coef = {
        c: edge_vals[(edge_rows == i) & (edge_cols == c)].sum()
        + edge_vals[(edge_rows == j) & (edge_cols == c)].sum()
        for c in union
    }
    norm = float(np.sqrt(sum(v * v for v in coef.values())))
    row_features = np.vstack(
        [
            sample.observation.row_features,
            np.zeros((1, sample.observation.row_features.shape[1])),
        ]
    )
    from l2b_bench.methods.dsccl.augment import _IDX_BIAS, _IDX_OBJ_COS

    row_features[n_rows, _IDX_BIAS] = (
        sample.observation.row_features[i, _IDX_BIAS]
        + sample.observation.row_features[j, _IDX_BIAS]
    ) / norm
    row_features[n_rows, _IDX_OBJ_COS] = (
        sample.observation.row_features[i, _IDX_OBJ_COS]
        + sample.observation.row_features[j, _IDX_OBJ_COS]
    ) / norm
    new_indices = np.vstack(
        [
            np.full(len(union), n_rows, dtype=edge.indices.dtype),
            np.asarray(union, dtype=edge.indices.dtype),
        ]
    )
    edge_features = EdgeFeatures(
        indices=np.hstack([np.asarray(edge.indices), new_indices]),
        values=np.concatenate(
            [edge_vals, np.asarray([coef[c] / norm for c in union], dtype=float)]
        ),
    )
    return BranchingSample(
        observation=BipartiteGraph(
            variable_features=sample.observation.variable_features,
            row_features=row_features,
            edge_features=edge_features,
        ),
        action_set=sample.action_set,
        action=sample.action,
        scores=sample.scores,
    )


# ---------------------------------------------------------------------------
# The stratified contrastive machinery
# ---------------------------------------------------------------------------


class TestGroups:
    def test_kmeans_recovers_well_separated_strata(self):
        samples = []
        for k in range(2):
            sample = make_sample(seed=k)
            # Push the mean-pooled features far apart between the two groups.
            sample.observation.variable_features[:, VAR_IDX["sol_val"]] = 10.0 * k
            samples.append(sample)
        groups = assign_groups(samples, m=2, seed=0)
        assert len(groups) == 2
        assert len(set(groups.tolist())) == 2

    def test_few_samples_shrink_the_cluster_count(self):
        groups = assign_groups([make_sample()], m=4, seed=0)
        assert groups.tolist() == [0]


class TestStratifiedContrast:
    def test_weight_matrix_starts_at_the_paper_s_anchor(self):
        contrast = StratifiedContrast(m=4)
        with torch.no_grad():
            contrast.theta.zero_()
        W = contrast.weight_matrix()
        assert W[0, 0].item() == pytest.approx(1 / (1 + math.exp(-1.0)))
        # alpha_1 = 1 + softplus(0) = 1 + ln 2 with theta at zero.
        assert W[0, 1].item() == pytest.approx(1 / (1 + math.exp(-(1 + math.log(2)))))
        assert W[1, 0].item() == W[0, 1].item()
        # Symmetric, diagonal = sigma(alpha_0).
        assert torch.allclose(W, W.t())

    def test_loss_matches_the_hand_computed_value(self):
        # Two identical embeddings in one cluster, a third orthogonal one in
        # the other: for i = 0 the positive's weighted similarity is
        # sigma(alpha_0) and the denominator adds the third sample's zero
        # similarity — loss = log(1 + exp(-sigma(1)/tau)).
        contrast = StratifiedContrast(m=2)
        Z = torch.eye(3)
        Z = torch.stack([Z[0], Z[0], Z[1]])
        groups = torch.tensor([0, 0, 1])
        loss = contrast(torch.nn.functional.normalize(Z, dim=1), groups)
        # i = 0: positive 1 with sim 1; denominator also holds j = 2 (sim 0).
        expected = math.log(1 + math.exp(-(1 / (1 + math.exp(-1.0))) / TAU))
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_batches_without_positives_contribute_nothing(self):
        contrast = StratifiedContrast(m=2)
        Z = torch.eye(2)
        groups = torch.tensor([0, 1])  # no two samples share a cluster
        loss = contrast(Z, groups)
        assert loss.item() == 0.0

    def test_theta_and_lambda_are_learnable(self):
        contrast = StratifiedContrast(m=3)
        assert contrast.lam.item() == pytest.approx(LAMBDA_INIT)
        Z = torch.nn.functional.normalize(torch.eye(4), dim=1)
        groups = torch.tensor([0, 0, 1, 1])
        lcons = contrast(Z, groups)
        # lambda enters the training loss as the balancing coefficient, so its
        # gradient appears exactly as in process(): ce + lambda * lcons.
        (contrast.lam * lcons).backward()
        assert contrast.theta.grad is not None
        assert contrast.lam.grad is not None
        assert contrast.lam.grad.item() == pytest.approx(lcons.item())


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


class TestTrain:
    def test_collate_batch_carries_the_groups(self, collected):
        samples = collected.samples[:3]
        groups = [0, 1, 0]
        sb = collate_batch(samples, groups)
        assert sb.n_vars == [s.observation.variable_features.shape[0] for s in samples]
        assert sb.groups == groups

    def test_process_runs_the_joint_loss(self, collected):
        model = DSCCLGNN()
        contrast = StratifiedContrast(m=2)
        batches = sample_batches(
            collected.samples,
            assign_groups(collected.samples, m=2, seed=0),
            torch.device("cpu"),
            batch_size=10_000,
            shuffle=np.random.default_rng(0),
        )
        stats = process(model, contrast, batches)
        for key in ("loss", "ce", "lcons"):
            assert np.isfinite(stats[key]), key
        assert stats["kacc"].shape == (4,)
        assert stats["loss"] == pytest.approx(
            stats["ce"] + stats["lambda"] * stats["lcons"], rel=1e-4
        )

    def test_process_handles_samples_of_different_sizes(self):
        samples = [
            make_sample(n_vars=6, n_rows=4, seed=0),
            make_sample(n_vars=8, n_rows=5, seed=1),
        ]
        model = DSCCLGNN()
        contrast = StratifiedContrast(m=2)
        stats = process(model, contrast, [collate_batch(samples, [0, 1])])
        for key in ("loss", "ce", "lcons"):
            assert np.isfinite(stats[key]), key

    def test_process_steps_the_optimizer(self, collected):
        model = DSCCLGNN()
        contrast = StratifiedContrast(m=2)
        batches = sample_batches(
            collected.samples[:4],
            assign_groups(collected.samples[:4], m=2, seed=0),
            torch.device("cpu"),
            batch_size=4,
            shuffle=np.random.default_rng(0),
        )
        optimizer = torch.optim.Adam(
            [*model.parameters(), *contrast.parameters()], lr=1e-3
        )
        before = model.output_module[0].weight.detach().clone()
        process(model, contrast, batches, optimizer=optimizer)
        assert not torch.allclose(model.output_module[0].weight.detach(), before)

    def test_train_end_to_end_and_checkpoint_roundtrip(self, collected, tmp_path):
        checkpoint = tmp_path / "dsccl.pt"
        train(
            collected.samples,
            checkpoint,
            epochs=2,
            m=2,
            batch_size=8,
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


# ---------------------------------------------------------------------------
# The hybrid policy
# ---------------------------------------------------------------------------


class FixedLogits(torch.nn.Module):
    """A stand-in model returning fixed per-variable logits."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.as_tensor(logits, dtype=torch.float32), requires_grad=False
        )

    def forward(self, *args):
        return self.logits


class TestPolicy:
    def _observation(self, sample):
        return {"bipartite": sample.observation, "sb_scores": None}

    def test_root_count_is_set_on_the_first_decision(self):
        sample = make_sample()
        policy = DSCCLPolicy(FixedLogits(np.arange(6.0)))
        assert policy._root_count is None
        policy.select_action(self._observation(sample), np.arange(6))
        assert policy._root_count == 6

    def test_upstream_uses_full_sb_over_the_top_k_shortlist(self):
        sample = make_sample()
        # Logits rank the variables 0..5 (var 5 best); among the top-5
        # shortlist {1..5}, SB prefers var 2.
        policy = DSCCLPolicy(FixedLogits(np.arange(6.0)))
        sb_scores = np.full(6, 1.0)
        sb_scores[2] = 5.0
        observation = {"bipartite": sample.observation, "sb_scores": sb_scores}
        # 6 candidates at the root: upstream (n_cand > 0.8 * 6 is false for
        # 6 > 4.8 — true).
        action = policy.select_action(observation, np.arange(6))
        assert action == 2

    def test_sb_best_outside_the_shortlist_is_ignored(self):
        sample = make_sample()
        policy = DSCCLPolicy(FixedLogits(np.arange(6.0)))
        sb_scores = np.full(6, 1.0)
        sb_scores[0] = 100.0  # SB's best, but the model ranks it last
        observation = {"bipartite": sample.observation, "sb_scores": sb_scores}
        action = policy.select_action(observation, np.arange(6))
        # The shortlist is {1..5} in the model's descending order; SB's
        # scores tie at 1.0 there, so the argmax lands on the shortlist's
        # head — the model's own best, var 5.
        assert action == 5

    def test_downstream_nodes_branch_directly(self):
        sample = make_sample()
        policy = DSCCLPolicy(FixedLogits(np.arange(6.0)))
        sb_scores = np.full(6, 1.0)
        sb_scores[0] = 100.0  # SB would pick var 0
        observation = {"bipartite": sample.observation, "sb_scores": sb_scores}
        policy.select_action(observation, np.arange(6))  # the root
        # Downstream: 4 candidates <= rho * 6 = 4.8 — the model alone.
        action = policy.select_action(observation, np.arange(4))
        assert action == 3  # the model's best (var 5 not in the action set)

    def test_shortlist_is_capped_at_top_k(self):
        sample = make_sample()
        policy = DSCCLPolicy(FixedLogits(np.arange(6.0)))
        sb_scores = np.array([0.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        observation = {"bipartite": sample.observation, "sb_scores": sb_scores}
        action = policy.select_action(observation, np.arange(6))
        # Shortlist {1, 2, 3, 4, 5}: SB's best is var 1 (score 5).
        assert action == 1
        assert TOP_K_HYBRID == 5

    def test_missing_sb_scores_fall_back_to_the_model(self):
        sample = make_sample()
        policy = DSCCLPolicy(FixedLogits(np.arange(6.0)))
        observation = {"bipartite": sample.observation, "sb_scores": None}
        assert policy.select_action(observation, np.arange(6)) == 5

    def test_trained_policy_selects_within_the_action_set(self, collected, tmp_path):
        checkpoint = tmp_path / "dsccl.pt"
        train(collected.samples, checkpoint, epochs=1, m=2, batch_size=8)
        policy = DSCCLPolicy.from_checkpoint(checkpoint)
        for sample in collected.samples:
            action = policy.select_action(
                {"bipartite": sample.observation, "sb_scores": None},
                sample.action_set,
            )
            assert action in sample.action_set

    def test_dsccl_method_is_runner_ready(self, collected, tmp_path):
        from gyozas.fidelity import get_fidelity_profile
        from gyozas.observations import MetaObservation

        checkpoint = tmp_path / "dsccl.pt"
        train(collected.samples, checkpoint, epochs=1, m=2, batch_size=8)
        method = dsccl_method(checkpoint)
        assert method.name == "dsccl"
        assert method.fidelity == "dsccl"
        # The composite observation comes from the fidelity profile: the
        # hybrid policy needs the SB scores the runner extracts.
        assert method.observation is None
        observation = get_fidelity_profile("dsccl").observation()
        assert isinstance(observation, MetaObservation)
        assert set(observation.observations) == {"bipartite", "sb_scores"}
        policy = method.policy()
        sample = collected.samples[0]
        assert (
            policy.select_action(
                {"bipartite": sample.observation, "sb_scores": None},
                sample.action_set,
            )
            in sample.action_set
        )
