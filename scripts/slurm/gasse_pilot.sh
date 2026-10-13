#!/bin/bash
#SBATCH --job-name=gasse-pilot
#SBATCH --partition=vn
#SBATCH --wckey=p12es:turing
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# The Gasse 2019 pilot on the lab cluster's A5000 (partition vn; the GPUs are
# not declared to SLURM — no --gres, the card is just there).
#
# Collection is CPU-bound (SCIP strong branching), training runs on the GPU,
# evaluation (default/pscost/relpscost/gasse x 5 seeds x 6 instances) is
# CPU-bound again. The run is idempotent: a re-submission reuses the
# checkpoint and skips the runner's completed records.
#
# Usage: sbatch scripts/slurm/gasse_pilot.sh

set -euo pipefail

REPO=/fsmedium/work/rd/turing/L2B-Bench
RESULTS=$REPO/results/gasse_gpu_v1
EPISODES=100
EPOCHS=100
TRAIN_FAMILY=500x1000x0.05   # the paper's training set covers
EVAL_FAMILY=500x600x0.10     # the pilot's held-out evaluation family
COLLECT_TIME_LIMIT=300       # per-episode budget during collection

cd "$REPO"
mkdir -p logs "$RESULTS"

echo "=== $(date) — job $SLURM_JOB_ID on $(hostname) ==="
echo "cpus: $SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no GPU visible!"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# The CUDA venv (torch cu126); falls back to CPU code paths if no GPU.
$REPO/.venv-gpu/bin/python scripts/pilot_gasse.py \
    --episodes "$EPISODES" \
    --epochs "$EPOCHS" \
    --train-family "$TRAIN_FAMILY" \
    --eval-family "$EVAL_FAMILY" \
    --collect-time-limit "$COLLECT_TIME_LIMIT" \
    --results-dir "$RESULTS"

echo "=== $(date) — done ==="
