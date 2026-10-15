"""Seq-BnB 2026 — sequence model for joint branch-and-bound control.

Paper: "Seq-BnB: Sequence Model for Branch-and-Bound Node Selection and
Branching" (arXiv 2610.14835), Liu, Liu, Yan, Shi, Zhao. No official
repository — the paper promises the code upon publication — so this port
is reconstructed from the paper's description, with every reconstruction
choice documented in the modules (see ``collect.py`` for the empirically
pinned deviations).

1. ``collect.py``  — expert trajectories over the harness's joint
   branching + node-selection dynamics: typed token sequences, the
   x*-compatibility node-selection oracle, strong-branching labels.
2. ``model.py``    — the sequence model: GCNN over the current LP,
   Transformer over the typed tokens, cross-attention branching head and
   MLP node-selection head.
3. ``train.py``    — the DAgger loop with the joint loss
   (``L_branch + 0.1 * L_ns``).
4. ``policy.py``   — the trained model as a gyozas policy, attached to
   the environment (the joint regime's action sets are the dynamics' own
   decision objects).
"""
