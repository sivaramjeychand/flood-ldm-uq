#!/bin/bash
# Diagnostics A + B from one queue wait, via scripts/cond_scale_probe.py. Submit from the repo root.
#
#   sbatch scripts/slurm/probe_diagnostics.sh <run_tag>
#
#   A: backward_start sweep at s=1 -- is the collapse the truncated start (sampler) or the network?
#   B: narrow s sweep around 1, separately on the noisy / coarse / DEM stem slices -- which input it leans on
#
# Each sub-run gets its own output_dir (they all write results/cond_scale_probe.json),
# under experiments/probe_diag/<run_tag>/. SLURM log goes to logs/slurm/.
#
# Env overrides: CONFIG (default config/wollombi-trnf.json), VENV (default ./venv),
#                N_SCENES (default 10), N_SAMPLES (default 50)
#
#SBATCH --job-name=probediag
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --partition=gpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm/probe_diagnostics.sh <run_tag>" >&2
    exit 1
fi
RUN_TAG=$1

# The configs' ../checkpoints and ../data paths are relative to the working
# directory, so this has to run from the repo root.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -f sr.py ]; then
    echo "run sbatch from the flood-ldm-uq repo root (no sr.py in $(pwd))" >&2
    exit 1
fi

VENV=${VENV:-venv}
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "warning: no venv at $VENV, using $(command -v python)" >&2
fi

CONFIG=${CONFIG:-config/wollombi-trnf.json}
N_SCENES=${N_SCENES:-10}
N_SAMPLES=${N_SAMPLES:-50}
OUT=experiments/probe_diag/$RUN_TAG

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} tag=$RUN_TAG config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Fail fast on a CPU-only torch instead of after the queue wait + dataset load
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'torch sees no CUDA device')"

probe() {
    python scripts/cond_scale_probe.py -c "$CONFIG" \
        --n_scenes "$N_SCENES" --n_samples "$N_SAMPLES" "$@"
}

# A: s=1 only, vary the truncated-start point. 1000 = start from pure noise (no SR-latent init).
for BS in 50 100 200 500 1000; do
    echo "=== A: backward_start=$BS ==="
    probe -output_dir "$OUT/A_bs$BS" --scales 1.0 --backward_start "$BS"
done

# B: small perturbations of each stem input slice at the checkpoint's own backward_start.
for TARGET in noisy lr dem; do
    echo "=== B: target=$TARGET ==="
    probe -output_dir "$OUT/B_$TARGET" --target "$TARGET" --scales 0.95 0.97 1.0 1.03 1.05
done
