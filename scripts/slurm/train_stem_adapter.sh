#!/bin/bash
# SLURM wrapper for scripts/train_stem_adapter.py (trained stem adapter, CRPS loss through the reverse chain).
# Submit from the repo root.
#
#   sbatch [--time=HH:MM:SS] scripts/slurm/train_stem_adapter.sh <run_tag> [train_stem_adapter.py args...]
#
# Output goes to experiments/stem_adapter/<run_tag>/ and the SLURM log to logs/slurm/.
# Give every job its own run_tag, or results/train_stem_adapter.json gets overwritten.
#
# Env overrides: CONFIG (default config/wollombi-trnf.json), VENV (default ./venv)
#
#SBATCH --job-name=stemadapt
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --partition=gpu
# xgpj0: torch CUDA library preload fails there (job 939926)
#SBATCH --exclude=xgpj0

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch scripts/slurm/train_stem_adapter.sh <run_tag> [train_stem_adapter.py args...]" >&2
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
OUT=experiments/stem_adapter/$RUN_TAG

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} tag=$RUN_TAG config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Fail fast on a CPU-only torch instead of after the queue wait + dataset load
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'torch sees no CUDA device')"

python scripts/train_stem_adapter.py -c "$CONFIG" -output_dir "$OUT" "$@"
