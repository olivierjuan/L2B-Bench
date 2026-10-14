"""DSC-CL 2025 — Gasse's GCNN plus the graph embedding of the contrastive space.

The paper describes "a GCNN on the bipartite graph with an MLP scoring head"
whose node (graph) embeddings feed the stratified contrastive loss — the same
shape as CAMBranch's port: Gasse's ``GNNPolicy`` backbone verbatim (PreNorm
embeddings, the two message-passing passes, the per-variable output head) plus
per-node-type [MaxPool; MeanPool] pooling into an L2-normalized graph
embedding. The class is inherited rather than duplicated; the projection-head
sizes are the same declared pilot choice (2*64 -> 64 -> 64, ReLU).
"""

from ..cambranch.model import CAMBranchGNN


class DSCCLGNN(CAMBranchGNN):
    """Gasse's GCNN with the graph embedding the stratified loss pools over."""
