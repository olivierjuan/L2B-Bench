"""TreeGate (Zarpellon et al., 2021) — feature gating for neural branching.

Fidelity profile ``treegate``: solver-statistics state (``BranchingState``:
the original 25 + 8 + 53 parameterization, reused later by T-BranT and TGPPO),
relpscost expert under the Zarpellon (s, k) augmentation, per-node
cross-entropy over the feature-gated feedforward scores.
"""

from .collect import TreeGateSample, collect, collect_episode
from .model import NoTreePolicy, TreeGatePolicy
from .policy import TreeGatePolicyModel, treegate_method
from .train import load_checkpoint, process, save_checkpoint, sample_to_tensors

__all__ = [
    "NoTreePolicy",
    "TreeGatePolicy",
    "TreeGatePolicyModel",
    "TreeGateSample",
    "collect",
    "collect_episode",
    "load_checkpoint",
    "process",
    "save_checkpoint",
    "sample_to_tensors",
    "treegate_method",
]
