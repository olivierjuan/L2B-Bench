"""Gasse 2019 — the pilot method (obstacle-course: collect, train, evaluate).

Paper: "Exact Combinatorial Optimization with Graph Convolutional Neural
Networks" (NeurIPS 2019), Maxime Gasse, Didier Chételat, Nicola Ferroni,
Louis-Marchand Ferron, Luce Létourneau-Pinto, Andrea Lodi.
Official implementation: github.com/ds4dm/learn2branch-ecole (verified; see
GYOZAS docs/references.md).

The pilot ports this method end to end on the harness, validating the full
loop every other method will follow:

1. ``collect.py``  — strong-branching expert trajectories on the synthetic
   Ecole families (the ``gasse`` fidelity profile's native regime: root cuts
   only, no restarts, 45 min limit — SCIP-10 equivalents recorded in the
   profile metadata).
2. ``model.py``    — the GCNN on the bipartite LP graph, as in the paper
   (ported from the official repository, not reinvented).
3. ``train.py``    — the imitation-learning loop (cross-entropy on the expert's
   scores).
4. ``policy.py``   — the trained model wrapped in a gyozas ``Policy``.

The trained policy then runs through the standard benchmark grid like any
other method (fidelity profile ``gasse``), against the baselines of
``l2b_bench.methods.BASELINE_METHODS``.
"""
