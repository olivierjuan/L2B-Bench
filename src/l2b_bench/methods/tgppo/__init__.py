"""TGPPO 2025 (Tree-Gate PPO) — PPO on solver-statistics states.

Paper: "Learning Branching Policies for MILPs with Proximal Policy
Optimization" (arXiv 2511.12986, AAAI 2026), Abdelouahed Ben Mhamed,
Assia Kamal-Idrissi, Amal El Fallah Seghrouchni — the method is the
tree-gate PPO ("Tree-Gate PPO") of that paper. Official implementation:
github.com/benmhamed-a/tgppo-aaai-26 (verified; see GYOZAS
docs/references.md).

Fidelity profile ``tgppo``: the 25+8+53 ``BranchingStateObservation``
triple (Zarpellon's parameterization, the same slots as TreeGate and
T-BranT), an actor-critic transformer with BiMatchingNet refinement and a
TreeGate gated head, trained from scratch with PPO under the paper's H3
difficulty-adaptive reward (harness ``DifficultyAdaptive``) and 'sandbox'
SCIP setting (all primal heuristics off, cutoff at the optimum).
"""

from .model import Actor, BiMatchingNet, Critic, TreeGateBranchingNet
from .policy import TGPPOPolicy, tgppo_method
from .train import (
    ACTOR_LR,
    BATCH_SIZE,
    CRITIC_LR,
    ENTROPY_WEIGHT,
    EPISODES_PER_UPDATE,
    GAE_LAMBDA,
    GAMMA,
    GRAD_CLIP,
    N_EPOCHS,
    POLICY_CLIP,
    SANDBOX_PARAMS,
    WEIGHT_DECAY,
    BaselineStats,
    Transition,
    baseline_solve,
    choose_action,
    collate,
    compute_gae,
    get_device,
    load_checkpoint,
    make_env,
    ppo_update,
    run_episode,
    save_checkpoint,
    train,
    with_baselines,
)

__all__ = [
    "ACTOR_LR",
    "BATCH_SIZE",
    "CRITIC_LR",
    "ENTROPY_WEIGHT",
    "EPISODES_PER_UPDATE",
    "GAE_LAMBDA",
    "GAMMA",
    "GRAD_CLIP",
    "N_EPOCHS",
    "POLICY_CLIP",
    "SANDBOX_PARAMS",
    "WEIGHT_DECAY",
    "Actor",
    "BaselineStats",
    "BiMatchingNet",
    "Critic",
    "TGPPOPolicy",
    "Transition",
    "TreeGateBranchingNet",
    "baseline_solve",
    "choose_action",
    "collate",
    "compute_gae",
    "get_device",
    "load_checkpoint",
    "make_env",
    "ppo_update",
    "run_episode",
    "save_checkpoint",
    "tgppo_method",
    "train",
    "with_baselines",
]
