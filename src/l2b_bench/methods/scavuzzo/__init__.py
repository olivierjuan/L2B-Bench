"""Scavuzzo et al. 2022 — RL fine-tuning of an IL-pretrained branching policy.

Paper: "Learning to Branch with Tree MDPs" (AAAI 2022), Gabriele Scavuzzo,
Alessandro Zacchia, Thiago Serra, Kevin Leyton-Brown. Official
implementation: github.com/lascavana/rl2branch (verified; see GYOZAS
docs/references.md).

Fidelity profile ``scavuzzo``: Gasse's ``NodeBipartite`` observation and
GNNPolicy, first imitation-learned exactly like gasse (expert_probability
0.05), then fine-tuned with REINFORCE under one of the paper's three credit
assignments — flat MDP returns from the cumulative NNodes reward (``mdp``),
or tree-MDP returns ``-(subtree_size) - 1`` (harness reward
``SubtreeSizes``) under depth-first node selection (``tmdp+DFS``) or an
objective limit at the training instances' optimum (``tmdp+ObjLim``).
"""

from .policy import scavuzzo_method
from .train import (
    BATCH_SIZE,
    ENTROPY_BONUS,
    EPISODES_PER_EPOCH,
    LR,
    MAX_EPISODE_ITERATIONS,
    SAMPLE_RATE,
    VARIANTS,
    Transition,
    TransitionBatch,
    assign_returns,
    load_trained_checkpoint,
    make_env,
    reinforce_update,
    run_episode,
    sample_action,
    save_checkpoint,
    train,
    with_objlimit,
)

__all__ = [
    "BATCH_SIZE",
    "ENTROPY_BONUS",
    "EPISODES_PER_EPOCH",
    "LR",
    "MAX_EPISODE_ITERATIONS",
    "SAMPLE_RATE",
    "VARIANTS",
    "Transition",
    "TransitionBatch",
    "assign_returns",
    "load_trained_checkpoint",
    "make_env",
    "reinforce_update",
    "run_episode",
    "sample_action",
    "save_checkpoint",
    "scavuzzo_method",
    "train",
    "with_objlimit",
]
