"""T-BranT (Lin et al., 2022) — tree-aware branching transformers.

Fidelity profile ``tbrant``: solver-statistics state (``BranchingState``:
25 candidate + 8 node + 53 tree features), relpscost expert, CE over
candidate logits with the BiMatching head. This stage ports the LTBranT
variant (no tree GAT); the full T-BranT tree aggregation is future work.
"""

from .collect import SolverStatsSample, collect, collect_episode
from .model import BiMatchingNet, BranchFormer
from .policy import TBranTPolicy, tbrant_method
from .train import NoamLR, collate_stats, load_checkpoint, process, save_checkpoint

__all__ = [
    "BiMatchingNet",
    "BranchFormer",
    "NoamLR",
    "SolverStatsSample",
    "TBranTPolicy",
    "collate_stats",
    "collect",
    "collect_episode",
    "load_checkpoint",
    "process",
    "save_checkpoint",
    "tbrant_method",
]
