"""PlanB&B (Strang et al., AAAI 2026) — MuZero-style model-based RL for
branching: a tree MDP over B&B open-node states, Gumbel MCTS self-play,
SimSiam consistency, and HL-Gauss value bins.

Paper: "Branch & Bound with a Tree MDP and Gumbel MCTS" (arXiv 2511.09219),
by the BBMDP authors. Official implementation: github.com/abfariah/planbb.
"""

from .collect import (
    BREWARD_THRESHOLD,
    NODE_LIMIT,
    PLANBB_SCIP_PARAMS,
    TD_STEPS,
    UNROLL_STEPS,
    PlanBBTrajectory,
    make_env,
    run_selfplay_episode,
    select_action,
)
from .model import N_VAR_FEATURES, NUM_BINS, PlanBBNet, TreeHLGaussLoss
from .policy import PlanBBPolicy, load_planbb, planbb_method
from .search import MCTS
from .train import train

__all__ = [
    "BREWARD_THRESHOLD",
    "NODE_LIMIT",
    "N_VAR_FEATURES",
    "NUM_BINS",
    "MCTS",
    "PLANBB_SCIP_PARAMS",
    "TD_STEPS",
    "UNROLL_STEPS",
    "PlanBBNet",
    "PlanBBPolicy",
    "PlanBBTrajectory",
    "TreeHLGaussLoss",
    "load_planbb",
    "make_env",
    "planbb_method",
    "run_selfplay_episode",
    "select_action",
    "train",
]
