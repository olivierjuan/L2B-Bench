#!/bin/bash
#SBATCH --job-name=eval-snapshots
#SBATCH --partition=vn
#SBATCH --wckey=p12es:turing
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# SGM vs training budget: evaluate every numbered snapshot of a convergence
# campaign on the pilot grid.
#
#   sbatch --export=ALL,METHOD=planbb,CAMPAIGN=results/planbb_gpu_v2 \
#       scripts/slurm/eval_snapshots.sh
#
# Each snapshot <method>.pt.<tag> gets its own results dir under
# <CAMPAIGN>_curve/<tag>, seeded with hard links to the campaign's eval
# instances and baseline runs — the runner skips those cached runs, so only
# the method's 30 runs execute per snapshot. A final curve.json collects the
# per-snapshot SGMs and timelimit counts. BASELINES_FROM (default: the
# campaign) names the results dir to take the instances and baseline runs
# from — any campaign on the same eval family works (the baselines are
# bit-identical across campaigns), which lets a still-running campaign's
# snapshots be evaluated before it writes its own.
set -euo pipefail

REPO=/fsmedium/work/rd/turing/L2B-Bench
METHOD=${METHOD:?METHOD is required}
CAMPAIGN=${CAMPAIGN:?CAMPAIGN is required}
EVAL_FAMILY=${EVAL_FAMILY:-500x600x0.10}
EVAL_TIME_LIMIT=${EVAL_TIME_LIMIT:-120}
BASELINES_FROM=${BASELINES_FROM:-$CAMPAIGN}

cd "$REPO"
CURVE="${CAMPAIGN}_curve"
mkdir -p logs "$CURVE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "=== $(date) — job ${SLURM_JOB_ID:-local} on $(hostname) ==="
for snapshot in $(ls "$CAMPAIGN/$METHOD.pt."* | grep -v '\.train\.pt$' | sort); do
    tag=${snapshot##*.pt.}
    dir="$CURVE/$tag"
    if [ -f "$dir/summary.json" ]; then
        echo "[curve] $tag: cached"
        continue
    fi
    mkdir -p "$dir/runs"
    [ -d "$dir/instances" ] || cp -al "$BASELINES_FROM/instances" "$dir/instances"
    for baseline in default pscost relpscost; do
        [ -d "$dir/runs/$baseline" ] || cp -al "$BASELINES_FROM/runs/$baseline" "$dir/runs/$baseline"
    done
    ln -sf "$(realpath "$snapshot")" "$dir/$METHOD.pt"
    echo "[curve] $tag: evaluating $snapshot"
    $REPO/.venv-gpu/bin/python scripts/pilot_gasse.py \
        --method "$METHOD" \
        --eval-family "$EVAL_FAMILY" \
        --eval-time-limit "$EVAL_TIME_LIMIT" \
        --results-dir "$dir" | grep -E "^\[runner\]|^  $METHOD "
done

$REPO/.venv-gpu/bin/python - "$CURVE" "$METHOD" <<'EOF'
import json, sys
from pathlib import Path

curve, method = Path(sys.argv[1]), sys.argv[2]
points = []
for summary in sorted(curve.glob("*/summary.json")):
    s = json.loads(summary.read_text())
    tl = s.get("n_timelimit") or {
        m: sum(v != "optimal" for v in st.values()) for m, st in s["statuses"].items()
    }
    points.append({"tag": summary.parent.name, "sgm": s["sgm"][method], "timelimit": tl[method]})
(curve / "curve.json").write_text(json.dumps(points, indent=2))
for p in points:
    print(f"[curve] {p['tag']:>10}  {p['sgm']:.3f}x  ({p['timelimit']}/30 timelimit)")
EOF
echo "=== $(date) — done ==="
