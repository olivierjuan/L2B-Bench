"""CAMBranch 2024 — the variable-shifting augmentation (the AMILPs).

Paper: "Contrastive Learning with Augmented MILPs for Learning to Branch"
(Lin et al., ICLR 2024 poster) — no official repository (verified Oct 2026),
so this module is ported from the paper alone, with the feature-transformation
rules derived from gyozas' extractor semantics and validated empirically
against a hand-built shifted MILP re-solved by SCIP.

Variable shifting (the paper's Definition 3.1): x̂ = x + s with an integer
shift vector s. The AMILP is

    min c^T x̂ - c^T s   s.t.  A x̂ <= b + A s,  l + s <= x̂ <= u + s,

and the correspondence is exact: LP optima translate (Lemma 3.1), the FSB
branching decisions are identical (Theorem 3.1 — the expert label carries
over untouched), the duals and reduced costs too (Theorems 3.2-3.3). So each
collected sample yields 9 free AMILP samples by *transforming the collected
features* — no re-solve, which is the whole point (the paper's 10k originals
+ 90k AMILPs for a 10% strong-branching budget).

The transformation on gyozas' 19/5 ecole-parity layout:

- shifted by ``s_j``: ``sol_val``, ``best_incumbent_val``, ``avg_incumbent_val``
  (the primal values translate);
- the constraint ``bias`` by ``a_i^T s / ||a_i||`` — the edge-weighted sum,
  since the edge values already carry ``a_ij / ||a_i||``; every other row
  feature (``obj_cosine``, ``is_tight``, ``dual_sol``, ``age``) is invariant;
- ``binary`` flips to ``integer`` where an originally binary variable is
  shifted off zero (an AMILP binary is integer over [s_j, 1+s_j]);
- invariant: ``obj_coef``, ``has_lb``, ``has_ub``, ``red_cost``, ``sol_frac``,
  ``sol_at_lb``/``sol_at_ub`` (the bounds translate with the solution),
  ``basis_*``, the variable ``age``, the edge values and indices, and the
  action set / action / expert scores (Theorem 3.1).
"""

import numpy as np
from gyozas.observations import ECOLE_COL_FEATURE_NAMES, ECOLE_ROW_FEATURE_NAMES
from gyozas.observations.structs import BipartiteGraph

from ..gasse.collect import BranchingSample

#: The shift distribution is not specified in the paper — the pilot draws
#: i.i.d. integer shifts over this support (a declared pilot choice).
SHIFT_SUPPORT = np.arange(-2, 3, dtype=np.int64)

#: 10k originals + 90k AMILPs in the paper — 9 augmented copies per sample.
N_AUGMENTED = 9

_IDX = {name: i for i, name in enumerate(ECOLE_COL_FEATURE_NAMES)}
_IDX_BINARY = _IDX["binary"]
_IDX_INTEGER = _IDX["integer"]
_IDX_SHIFTED_COLS = tuple(
    _IDX[name] for name in ("sol_val", "best_incumbent_val", "avg_incumbent_val")
)
_IDX_BIAS = ECOLE_ROW_FEATURE_NAMES.index("bias")


def draw_shift(variable_features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One shift vector: i.i.d. integers over :data:`SHIFT_SUPPORT` on the
    discrete variables, zero on the continuous ones (the paper shifts integer
    variables; the pilot's families have none continuous)."""
    n_vars = variable_features.shape[0]
    shift = rng.choice(SHIFT_SUPPORT, size=n_vars)
    discrete = (
        (variable_features[:, _IDX_BINARY] > 0)
        | (variable_features[:, _IDX_INTEGER] > 0)
        | (variable_features[:, _IDX["implicit_integer"]] > 0)
    )
    shift[~discrete] = 0
    return shift


def shift_sample(sample: BranchingSample, shift: np.ndarray) -> BranchingSample:
    """One AMILP copy of ``sample`` under ``shift``, by feature transformation.

    The original observation, action set and scores are never mutated; the
    invariant edge structure is shared (never written to).
    """
    obs = sample.observation
    variable_features = np.array(obs.variable_features, copy=True)
    for idx in _IDX_SHIFTED_COLS:
        variable_features[:, idx] += shift
    # An originally binary variable shifted off zero becomes integer.
    flipped = (variable_features[:, _IDX_BINARY] == 1) & (shift != 0)
    variable_features[flipped, _IDX_BINARY] = 0.0
    variable_features[flipped, _IDX_INTEGER] = 1.0

    # The bias shift is the row-wise edge-weighted sum of the shift (each
    # edge value is a_ij / ||a_i||, so sum_e value_e * s[col(e)] = a_i^T s / ||a_i||).
    edge = obs.edge_features
    edge_rows = np.asarray(edge.indices)[0]
    edge_cols = np.asarray(edge.indices)[1]
    edge_vals = np.asarray(edge.values)
    bias_shift = np.bincount(
        edge_rows,
        weights=edge_vals * shift[edge_cols],
        minlength=obs.row_features.shape[0],
    )
    row_features = np.array(obs.row_features, copy=True)
    row_features[:, _IDX_BIAS] += bias_shift

    return BranchingSample(
        observation=BipartiteGraph(
            variable_features=variable_features,
            row_features=row_features,
            edge_features=edge,  # invariant — shared, never mutated
        ),
        action_set=sample.action_set,  # invariant (Thm 3.1) — shared
        action=sample.action,
        scores=sample.scores,  # invariant — shared
    )


def augment_pairs(
    samples: list[BranchingSample],
    rng: np.random.Generator,
    n_augmented: int = N_AUGMENTED,
) -> list[tuple[BranchingSample, BranchingSample]]:
    """(original, AMILP) pairs — every sample paired with each of its
    ``n_augmented`` shifted copies, the training pairs of the joint loss."""
    pairs = []
    for sample in samples:
        for _ in range(n_augmented):
            shift = draw_shift(sample.observation.variable_features, rng)
            pairs.append((sample, shift_sample(sample, shift)))
    return pairs
