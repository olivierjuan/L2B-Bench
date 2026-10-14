"""DSC-CL 2025 — the upstream-augmented MILP derivations.

Paper: "SC-MILP: ... Stratified Contrastive Learning" (Lu, Ma & Tao,
arXiv 2511.21107, preprint — no official repository, verified Oct 2026), so
this module is ported from the paper alone. The derivations are feature-level
(no re-solve) transformations of a collected sample:

- **LT-MILP** (Def 4.4, Thms 4.2-4.3): the linear transformation
  x_hat = T x + t with T diagonal +/-1 and t integer on the integer
  variables. The LP relaxation is mapped bijectively and the FSB branching
  results are identical, so the derived sample keeps the original's labels.
  Table 5 gives the exact feature mapping, reproduced below.
- **RC-MILP** (Def 4.5): append the redundant constraint a_r = a_i + a_j,
  b_r = b_i + b_j of two linearly independent rows. The LP feasible region is
  unchanged (the sum of two valid inequalities), so again the labels carry
  over; the new row's is_tight / dual_sol / age are zero (App. D.3).
- **Perturbed derivations** (Defs 4.6-4.7): small Gaussian noise on the
  objective, the constraint system, or the dual variables. The labels are
  *retained* despite the perturbation — the point is representation
  resilience, not new supervision.

The paper applies the two equivalent derivations with 35% probability and the
three perturbed ones with 10%; the magnitudes of the Gaussian noises and the
distributions of T and t are not specified — the pilot draws T_jj uniform over
{+1, -1} and t_j i.i.d. integer over {-2..2} on the discrete variables
(:data:`T_SUPPORT`, :data:`T_SHIFT_SUPPORT`), and N(0, 0.05) noise on the
normalized features (:data:`NOISE_STD`) — declared pilot choices.

A note on RC-MILP: the paper builds a_r on the true constraint rows, but a
frozen sample only carries the row-normalized representation (edge values
a_ij / ||a_i||, bias b_i / ||a_i||) — the row norms are not recoverable. The
pilot therefore applies the construction to the *normalized* rows: appending
(e_i + e_j) x <= bias_i + bias_j, itself a positive row scaling of the
paper's derivation, hence still an equivalent derivation with an identical
LP relaxation. Validated empirically against SCIP in the tests.
"""

import numpy as np
from gyozas.observations import ECOLE_COL_FEATURE_NAMES, ECOLE_ROW_FEATURE_NAMES
from gyozas.observations.structs import BipartiteGraph, EdgeFeatures

from ..gasse.collect import BranchingSample

#: The paper's derivation probabilities: an equivalent MILP (LT or RC) with
#: 35%, a perturbed one with 10%, the original otherwise.
EQUIV_PROB = 0.35
PERTURB_PROB = 0.10

#: The sign pattern is not specified in the paper — the pilot draws each
#: discrete variable's T_jj uniformly over {+1, -1}.
T_SUPPORT = np.array([-1, 1], dtype=np.int64)

#: The integer shifts are not specified either — the pilot draws i.i.d.
#: integers over the same support as CAMBranch's shifts.
T_SHIFT_SUPPORT = np.arange(-2, 3, dtype=np.int64)

#: The perturbation magnitudes are not specified — the pilot draws N(0, 0.05)
#: on the normalized features (a declared pilot choice).
NOISE_STD = 0.05

_IDX = {name: i for i, name in enumerate(ECOLE_COL_FEATURE_NAMES)}
_IDX_BINARY = _IDX["binary"]
_IDX_INTEGER = _IDX["integer"]
_IDX_IMPLICIT = _IDX["implicit_integer"]
_IDX_OBJ_COEF = _IDX["obj_coef"]
_IDX_RED_COST = _IDX["red_cost"]
_IDX_SOL_VAL = _IDX["sol_val"]
_IDX_SOL_FRAC = _IDX["sol_frac"]
_IDX_INCUMBENTS = tuple(
    _IDX[name] for name in ("best_incumbent_val", "avg_incumbent_val")
)
#: The pairs swapped by a sign flip T_jj = -1 (Table 5).
_IDX_SWAPPED_PAIRS = tuple(
    (_IDX[a], _IDX[b])
    for a, b in (
        ("has_lb", "has_ub"),
        ("sol_at_lb", "sol_at_ub"),
        ("basis_lower", "basis_upper"),
    )
)
_IDX_BIAS = ECOLE_ROW_FEATURE_NAMES.index("bias")
_IDX_OBJ_COS = ECOLE_ROW_FEATURE_NAMES.index("obj_cosine")
_IDX_IS_TIGHT = ECOLE_ROW_FEATURE_NAMES.index("is_tight")
_IDX_DUAL = ECOLE_ROW_FEATURE_NAMES.index("dual_sol")
_IDX_AGE_ROW = ECOLE_ROW_FEATURE_NAMES.index("age")


def draw_transformation(
    variable_features: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One (T, t) pair: signs and integer shifts on the discrete variables,
    identity and zero on the continuous ones (the paper's t_I in Z^k)."""
    n_vars = variable_features.shape[0]
    discrete = (
        (variable_features[:, _IDX_BINARY] > 0)
        | (variable_features[:, _IDX_INTEGER] > 0)
        | (variable_features[:, _IDX_IMPLICIT] > 0)
    )
    T = np.where(discrete, rng.choice(T_SUPPORT, size=n_vars), 1).astype(np.int64)
    t = np.where(discrete, rng.choice(T_SHIFT_SUPPORT, size=n_vars), 0).astype(np.int64)
    return T, t


def lt_sample(sample: BranchingSample, T: np.ndarray, t: np.ndarray) -> BranchingSample:
    """One LT-MILP derivation of ``sample`` under x_hat = T x + t, by the
    feature transformation of Table 5.

    Variable features — ``obj_coef`` and ``red_cost`` scale by T_jj (the
    objective and duals reflect), the primal values (``sol_val`` and the
    incumbents) map to T_jj * V + t_j, ``sol_frac`` goes to 1 - V (mod 1)
    where T_jj = -1, the (has_lb, has_ub), (sol_at_lb, sol_at_ub) and
    (basis_lower, basis_upper) pairs swap where T_jj = -1 (the bounds reflect
    — the paper's table lists basis_status as invariant, but the extractor's
    at-lower/at-upper statuses demonstrably swap under a reflection, so the
    pilot follows the extractor), a binary becomes integer unless the image
    of {0, 1} is still {0, 1} (T = +1, t = 0 or T = -1, t = 1), and ``age``
    is invariant. Edge values scale by T_jj; the row ``bias`` shifts by
    a_i^T T t / ||a_i|| (the row norms are invariant under a sign pattern);
    every other row feature is invariant (obj_cos_sim: both sides reflect;
    is_tight, dual_sol, age).
    """
    obs = sample.observation
    variable_features = np.array(obs.variable_features, copy=True)
    flip = T == -1

    variable_features[:, _IDX_OBJ_COEF] *= T
    variable_features[:, _IDX_RED_COST] *= T
    for idx in (_IDX_SOL_VAL, *_IDX_INCUMBENTS):
        variable_features[:, idx] = T * variable_features[:, idx] + t
    variable_features[flip, _IDX_SOL_FRAC] = (
        1.0 - variable_features[flip, _IDX_SOL_FRAC]
    ) % 1.0
    for idx_a, idx_b in _IDX_SWAPPED_PAIRS:
        a = variable_features[flip, idx_a].copy()
        variable_features[flip, idx_a] = variable_features[flip, idx_b]
        variable_features[flip, idx_b] = a
    # A binary stays binary iff the image of {0, 1} is {0, 1}: identity, or
    # the reflection x_hat = 1 - x. Any other (T, t) makes it integer.
    stays_binary = ((T == 1) & (t == 0)) | ((T == -1) & (t == 1))
    converted = (variable_features[:, _IDX_BINARY] == 1) & ~stays_binary
    variable_features[converted, _IDX_BINARY] = 0.0
    variable_features[converted, _IDX_INTEGER] = 1.0

    edge = obs.edge_features
    edge_rows = np.asarray(edge.indices)[0]
    edge_cols = np.asarray(edge.indices)[1]
    edge_vals = np.asarray(edge.values, dtype=float)
    edge_values = edge_vals * T[edge_cols]

    # bias shift = a_i^T T t / ||a_i||: the row-wise edge-weighted sum of
    # T_jj * t_j (each edge value is a_ij / ||a_i||).
    bias_shift = np.bincount(
        edge_rows,
        weights=edge_values * t[edge_cols],
        minlength=obs.row_features.shape[0],
    )
    row_features = np.array(obs.row_features, copy=True)
    row_features[:, _IDX_BIAS] += bias_shift

    return BranchingSample(
        observation=BipartiteGraph(
            variable_features=variable_features,
            row_features=row_features,
            edge_features=EdgeFeatures(
                indices=np.asarray(edge.indices), values=edge_values
            ),
        ),
        action_set=sample.action_set,  # invariant (Thm 4.3) — shared
        action=sample.action,
        scores=sample.scores,  # invariant — shared
    )


def _row_supports(edge_rows: np.ndarray, edge_cols: np.ndarray, n_rows: int):
    """Per-row sorted column supports, as frozensets keyed by row."""
    supports: list[set] = [set() for _ in range(n_rows)]
    for row, col in zip(edge_rows, edge_cols):
        supports[row].add(col)
    return [frozenset(s) for s in supports]


def rc_sample(sample: BranchingSample, rng: np.random.Generator) -> BranchingSample:
    """One RC-MILP derivation of ``sample``: append the normalized sum of two
    linearly independent rows (distinct supports — a scalar multiple has the
    same support, so distinct supports imply independence).

    The new row r (App. D.3): edge values (e_ik + e_jk) / ||e_i + e_j||, bias
    (bias_i + bias_j) / ||e_i + e_j||, obj_cosine likewise, and is_tight =
    dual_sol = age = 0. Everything else is untouched; the labels carry over.
    """
    obs = sample.observation
    edge = obs.edge_features
    edge_rows = np.asarray(edge.indices)[0]
    edge_cols = np.asarray(edge.indices)[1]
    edge_vals = np.asarray(edge.values, dtype=float)
    n_rows = obs.row_features.shape[0]

    supports = _row_supports(edge_rows, edge_cols, n_rows)
    nonempty = [i for i in range(n_rows) if len(supports[i]) > 0]
    # Two distinct non-empty supports, drawn until they differ.
    while True:
        i, j = rng.choice(nonempty, size=2, replace=False)
        if supports[i] != supports[j]:
            break

    # The normalized sum, over the union of the supports.
    union = sorted(supports[i] | supports[j])
    coef = {
        c: edge_vals[(edge_rows == i) & (edge_cols == c)].sum()
        + edge_vals[(edge_rows == j) & (edge_cols == c)].sum()
        for c in union
    }
    norm = float(np.sqrt(sum(v * v for v in coef.values())))
    if norm <= 0.0:  # pragma: no cover - distinct supports cannot cancel out
        raise ValueError("degenerate redundant constraint (zero norm)")

    row_features = np.vstack(
        [
            obs.row_features,
            np.zeros((1, obs.row_features.shape[1]), dtype=float),
        ]
    )
    row_features[n_rows, _IDX_BIAS] = (
        obs.row_features[i, _IDX_BIAS] + obs.row_features[j, _IDX_BIAS]
    ) / norm
    row_features[n_rows, _IDX_OBJ_COS] = (
        obs.row_features[i, _IDX_OBJ_COS] + obs.row_features[j, _IDX_OBJ_COS]
    ) / norm
    # is_tight, dual_sol, age stay zero (App. D.3).

    new_indices = np.vstack(
        [
            np.full(len(union), n_rows, dtype=edge.indices.dtype),
            np.asarray(union, dtype=edge.indices.dtype),
        ]
    )
    new_values = np.asarray([coef[c] / norm for c in union], dtype=float)
    edge_features = EdgeFeatures(
        indices=np.hstack([np.asarray(edge.indices), new_indices]),
        values=np.concatenate([edge_vals, new_values]),
    )

    return BranchingSample(
        observation=BipartiteGraph(
            variable_features=obs.variable_features,  # untouched — shared
            row_features=row_features,
            edge_features=edge_features,
        ),
        action_set=sample.action_set,  # invariant — shared
        action=sample.action,
        scores=sample.scores,  # invariant — shared
    )


def perturb_sample(
    sample: BranchingSample, rng: np.random.Generator, kind: str
) -> BranchingSample:
    """One perturbed derivation of ``sample`` (Defs 4.6-4.7): Gaussian noise
    on the objective coefficients, the constraint coefficients (edges and
    bias), or the dual variables — feature-level, the labels retained."""
    obs = sample.observation
    if kind == "objective":
        variable_features = np.array(obs.variable_features, copy=True)
        variable_features[:, _IDX_OBJ_COEF] += rng.normal(
            0.0, NOISE_STD, size=variable_features.shape[0]
        )
        return BranchingSample(
            observation=BipartiteGraph(
                variable_features=variable_features,
                row_features=obs.row_features,  # untouched — shared
                edge_features=obs.edge_features,  # untouched — shared
            ),
            action_set=sample.action_set,
            action=sample.action,
            scores=sample.scores,
        )
    if kind == "constraint":
        edge_values = np.asarray(obs.edge_features.values, dtype=float) + rng.normal(
            0.0, NOISE_STD, size=len(np.asarray(obs.edge_features.values))
        )
        row_features = np.array(obs.row_features, copy=True)
        row_features[:, _IDX_BIAS] += rng.normal(
            0.0, NOISE_STD, size=row_features.shape[0]
        )
        return BranchingSample(
            observation=BipartiteGraph(
                variable_features=obs.variable_features,  # untouched — shared
                row_features=row_features,
                edge_features=EdgeFeatures(
                    indices=np.asarray(obs.edge_features.indices),
                    values=edge_values,
                ),
            ),
            action_set=sample.action_set,
            action=sample.action,
            scores=sample.scores,
        )
    if kind == "dual":
        row_features = np.array(obs.row_features, copy=True)
        row_features[:, _IDX_DUAL] += rng.normal(
            0.0, NOISE_STD, size=row_features.shape[0]
        )
        return BranchingSample(
            observation=BipartiteGraph(
                variable_features=obs.variable_features,  # untouched — shared
                row_features=row_features,
                edge_features=obs.edge_features,  # untouched — shared
            ),
            action_set=sample.action_set,
            action=sample.action,
            scores=sample.scores,
        )
    raise ValueError(f"unknown perturbation kind {kind!r}")


PERTURB_KINDS = ("objective", "constraint", "dual")


def derive_sample(sample: BranchingSample, rng: np.random.Generator) -> BranchingSample:
    """The paper's upstream augmentation draw for one sample: with 35% an
    equivalent derivation (LT or RC, uniformly), with 10% a perturbed one
    (uniformly among the three), the original otherwise. The derived sample
    always keeps the original's labels."""
    u = rng.random()
    if u < EQUIV_PROB:
        if rng.random() < 0.5:
            T, t = draw_transformation(sample.observation.variable_features, rng)
            return lt_sample(sample, T, t)
        return rc_sample(sample, rng)
    if u < EQUIV_PROB + PERTURB_PROB:
        return perturb_sample(sample, rng, PERTURB_KINDS[rng.integers(3)])
    return sample
