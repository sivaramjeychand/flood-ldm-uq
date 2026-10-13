#!/bin/bash
# SLURM wrapper for scripts/decoder_uq.py (SSF deep ensemble on the VAE decoder vs split conformal).
# Submit from the repo root.
#
#   sbatch [--time=HH:MM:SS] scripts/slurm/decoder_uq.sh <run_tag> [decoder_uq.py args...]
#
# Output goes to experiments/decoder_uq/<run_tag>/ and the SLURM log to logs/slurm/.
# Give every job its own run_tag: each run writes results/decoder_uq.json and
# results/ssf_member*.pt inside its output_dir, so two jobs sharing a tag would overwrite each other.
#
# Env overrides: CONFIG (default config/wollombi-trnf.json), VENV (default ./venv)
#
#SBATCH --job-name=decuq
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --partition=gpu

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm/decoder_uq.sh <run_tag> [decoder_uq.py args...]" >&2
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
OUT=experiments/decoder_uq/$RUN_TAG

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} tag=$RUN_TAG config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Fail fast on a CPU-only torch instead of after the queue wait + dataset load
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'torch sees no CUDA device')"

python scripts/decoder_uq.py -c "$CONFIG" -output_dir "$OUT" "$@"
