"""Mamba-Branching 2026 — relpscost-trajectory IL with a selective SSM path model.

Official implementation: github.com/doctor-watson626/Mamba-Branching
(verified; see GYOZAS docs/references.md).

Fidelity profile ``mamba``: the 25+8+53 ``BranchingStateObservation`` triple
(Zarpellon's parameterization, the same slots as TreeGate, T-BranT and
TGPPO), the relpscost expert followed along whole solve trajectories under
the (s, k) Zarpellon augmentation, and a two-stage model — a contrastively
pretrained TreeGateNet candidate embedder, then a Mamba selective SSM over
the sequentialized path — evaluated deterministically as a streaming reader
over the last 49 blocks.
"""

from .collect import MAMBA_K, MAMBA_S, MambaSample, collect_episode
from .model import GroupedMamba, Mamba, TreeGateNet
from .policy import MambaPolicyModel, mamba_method
from .train import (
    CLIP_EPOCHS,
    CLIP_LR,
    EPOCHS,
    EVAL_MAX_SEQ_LEN,
    GRAD_CLIP,
    LR,
    MAX_SEQ_LENGTH,
    WEIGHT_DECAY,
    chunk_trajectories,
    clip_batch,
    clip_loss,
    load_checkpoint,
    save_checkpoint,
    sequence_batch,
    sequence_loss,
    train,
    train_clip,
    train_sequence,
)

__all__ = [
    "CLIP_EPOCHS",
    "CLIP_LR",
    "EPOCHS",
    "EVAL_MAX_SEQ_LEN",
    "GRAD_CLIP",
    "LR",
    "MAMBA_K",
    "MAMBA_S",
    "MAX_SEQ_LENGTH",
    "WEIGHT_DECAY",
    "GroupedMamba",
    "Mamba",
    "MambaPolicyModel",
    "MambaSample",
    "TreeGateNet",
    "chunk_trajectories",
    "clip_batch",
    "clip_loss",
    "collect_episode",
    "load_checkpoint",
    "mamba_method",
    "save_checkpoint",
    "sequence_batch",
    "sequence_loss",
    "train",
    "train_clip",
    "train_sequence",
]
