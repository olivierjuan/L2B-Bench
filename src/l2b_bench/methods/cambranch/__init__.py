"""CAMBranch 2024 — the pilot method (augment, train, evaluate).

Paper: "Contrastive Learning with Augmented MILPs for Learning to Branch"
(Tao Lin, Wanxin Xiu, Keyou You, Xiong-Hui Chen, Wenjun Cai, ICLR 2024
poster). No official repository (verified Oct 2026) — a paper-only port; see
GYOZAS docs/references.md.

1. ``collect`` — reuses gasse's collection verbatim: the same explore-then-
   strong-branch protocol (the paper's SCIP 6.0.1 root-cuts-only, no-restart
   setting coincides with Gasse's collection parameters, so the pilot drives
   the same ``collect`` and shares the gasse-style cache).
2. ``augment.py`` — the variable-shifting AMILPs: 9 free samples per
   collected sample by feature transformation (Thm 3.1), no re-solve.
3. ``model.py``    — Gasse's GCNN plus the contrastive projection head.
4. ``train.py``    — the joint loss: supervised CE + 0.05 * InfoNCE +
   0.01 * auxiliary, batch 64 (32 pairs), Adam.
5. ``policy.py``   — the trained model wrapped in a gyozas ``Policy``.

The trained policy then runs through the standard benchmark grid like any
other method (fidelity profile ``cambranch``), against the baselines of
``l2b_bench.methods.BASELINE_METHODS``.
"""
