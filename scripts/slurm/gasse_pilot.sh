#!/bin/bash
#SBATCH --job-name=gasse-pilot
#SBATCH --partition=vn
#SBATCH --wckey=p12es:turing
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# The Gasse 2019 pilot on the lab cluster's A5000 (partition vn; the GPUs are
# not declared to SLURM — no --gres, the card is just there). The node is
# reserved exclusively: collection fans out over the CPUs (one SCIP per
# worker), training runs on the GPU, evaluation is CPU-bound again.
#
# Everything is cached and idempotent: training instances, expert samples,
# the checkpoint, and the runner's records all survive a restart — a
# re-submission skips whatever is already done.
#
# Usage: sbatch scripts/slurm/gasse_pilot.sh

set -euo pipefail

REPO=/fsmedium/work/rd/turing/L2B-Bench
# sbatch --export=METHOD=lookback,RESULTS=$REPO/results/lookback_gpu_v1
# drives the Lookback port through the same campaign (METHOD_SPECS in
# scripts/pilot_gasse.py); the default is the Gasse campaign. EPISODES and
# EPOCHS are overridable the same way for smoke tests.
METHOD=${METHOD:-gasse}
RESULTS=${RESULTS:-$REPO/results/gasse_gpu_v2}
EPISODES=${EPISODES:-100}
EPOCHS=${EPOCHS:-100}
# The paper trains AND tests on the same set-cover family — it never claims
# cross-family transfer, and v1 (results/gasse_gpu_v1, trained 500x1000x0.05,
# evaluated 500x600x0.10) confirmed it: top-1 agreement with the SB expert
# collapses from 0.40 in-distribution to 0.07 out-of-distribution, and the
# policy ends up 27x worse than the default solver. The pilot therefore
# aligns the training family with the evaluation family.
TRAIN_FAMILY=500x600x0.10
EVAL_FAMILY=500x600x0.10     # the pilot's held-out evaluation family
COLLECT_TIME_LIMIT=300       # per-episode budget during collection
COLLECT_WORKERS=32           # parallel SCIP episodes (node has 64 cpus)

cd "$REPO"
mkdir -p logs "$RESULTS"

echo "=== $(date) — job $SLURM_JOB_ID on $(hostname) ==="
echo "cpus: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no GPU visible!"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1      # parallelism is at the episode level, one thread per SCIP

# The CUDA venv (torch cu129); falls back to CPU code paths if no GPU.
$REPO/.venv-gpu/bin/python scripts/pilot_gasse.py \
    --method "$METHOD" \
    --episodes "$EPISODES" \
    --epochs "$EPOCHS" \
    --train-family "$TRAIN_FAMILY" \
    --eval-family "$EVAL_FAMILY" \
    --collect-time-limit "$COLLECT_TIME_LIMIT" \
    --collect-workers "$COLLECT_WORKERS" \
    --results-dir "$RESULTS"

echo "=== $(date) — done ==="
