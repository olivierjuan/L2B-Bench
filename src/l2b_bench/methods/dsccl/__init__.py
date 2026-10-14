"""DSC-CL / SC-MILP 2025 (Lu, Ma & Tao, arXiv 2511.21107) — paper-only port.

No official repository exists (verified Oct 2026), so every module is ported
from the preprint. Collection reuses gasse verbatim (the paper trains on the
Easy level of the 4 Ecole families with a full-strong-branching expert, i.e.
Gasse's regime); the method diverges at training (the stratified contrastive
loss over K-means node groups, the upstream-augmented derivations) and at
inference (the hybrid top-5 + full-SB branching at upstream nodes).
"""
