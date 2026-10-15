"""BBMDP 2025 (Strang et al., NeurIPS 2025) — DQN from scratch on the
corrected vanilla MDP (Prop. 3.1), with k-step returns over the visited
subtree and an HL-Gauss categorical value head.

Paper: "A Markov Decision Process for Variable Selection in Branch & Bound"
(arXiv 2510.19348), Paul Strang, Zacharie Alès, Côme Bissuel, Olivier Juan,
Safia Kedad-Sidhoum, Emmanuel Rachelson. Official implementation:
github.com/abfariah/bbmdp."""

from .collect import (
    BBMDP_SCIP_PARAMS,
    GAMMA,
    N_STEP,
    BBMDPTransition,
    bbmdp_returns,
    make_env,
    process_episode,
    run_episode,
)
from .policy import BBMDPPolicy, bbmdp_method
from .train import TreeDQNAgent, load_dqn, train

__all__ = [
    "BBMDP_SCIP_PARAMS",
    "GAMMA",
    "N_STEP",
    "BBMDPPolicy",
    "BBMDPTransition",
    "TreeDQNAgent",
    "bbmdp_method",
    "bbmdp_returns",
    "load_dqn",
    "make_env",
    "process_episode",
    "run_episode",
    "train",
]
