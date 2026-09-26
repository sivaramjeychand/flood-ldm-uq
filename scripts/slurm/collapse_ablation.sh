#!/bin/bash
# SLURM wrapper for scripts/collapse_ablation.py. Submit from the repo root.
#
#   sbatch [--time=HH:MM:SS] scripts/slurm/collapse_ablation.sh <run_tag> [collapse_ablation.py args...]
#
# Output goes to experiments/collapse/<run_tag>/ and the SLURM log to logs/slurm/.
# Give every job its own run_tag: each run writes results/collapse_ablation.json
# inside its output_dir, so two jobs sharing a tag would overwrite each other.
#
# Env overrides: CONFIG (default config/wollombi-trnf.json), VENV (default ./venv)
#
#SBATCH --job-name=collapse
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
# #SBATCH --partition=<fill in>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm/collapse_ablation.sh <run_tag> [collapse_ablation.py args...]" >&2
    exit 1
fi
RUN_TAG=$1; shift

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
OUT=experiments/collapse/$RUN_TAG

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} tag=$RUN_TAG config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python scripts/collapse_ablation.py -c "$CONFIG" -output_dir "$OUT" "$@"
