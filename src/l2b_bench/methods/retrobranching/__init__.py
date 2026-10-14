"""RetroBranching (Parsonson & Deisenroth 2022) — retrospective RL branching.

Paper: "Boosting tree search strategies for MIPs via retrospective
reinforcement learning" (AAAI 2022), Charlie Parsonson, Michael C. W.
Deisenroth. Official implementation: github.com/ds4dm/retro_branching
(verified; see GYOZAS docs/references.md).

Fidelity profile ``retrobranching``: the ``NodeBipartiteRetro`` observation
(19 bipartite-LP + 24 retro variable features) under the paper's
``retro_binary_fathomed`` reward — binary scores assigned retrospectively
over sub-tree episodes reconstructed from the finished search tree (the
harness provides both in ``gyozas.rewards``) — and the DQN training
protocol: uniform replay, 1-step Double-DQN targets, soft target updates,
epsilon 0.025, gasse_2019 solver parameters.
"""

from .model import RetroGCN
from .policy import RetroPolicy, retrobranching_method
from .replay import GraphState, ReplayBuffer, Transition, graph_state
from .train import (
    EPSILON,
    GAMMA,
    LR,
    SOFT_UPDATE_TAU,
    add_episode_transitions,
    dqn_update,
    load_checkpoint,
    make_env,
    run_episode,
    save_checkpoint,
    train,
)

__all__ = [
    "EPSILON",
    "GAMMA",
    "LR",
    "SOFT_UPDATE_TAU",
    "GraphState",
    "ReplayBuffer",
    "RetroGCN",
    "RetroPolicy",
    "Transition",
    "add_episode_transitions",
    "dqn_update",
    "graph_state",
    "load_checkpoint",
    "make_env",
    "retrobranching_method",
    "run_episode",
    "save_checkpoint",
    "train",
]
