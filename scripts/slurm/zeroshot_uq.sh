#!/bin/bash
# SLURM wrapper for scripts/zeroshot_uq.py (zero-shot UQ transfer of the frozen LDM, base / stem gain / trained adapter).
# Submit from the repo root.
#
#   sbatch [--time=HH:MM:SS] scripts/slurm/zeroshot_uq.sh <run_tag> [zeroshot_uq.py args...]
#
# Output goes to experiments/zeroshot_uq/<run_tag>/ and the SLURM log to logs/slurm/.
# Give every job its own run_tag, or results/zeroshot_uq.json gets overwritten.
#
# Env overrides: CONFIG (default config/wollombi-zeroshot-burnett.json), VENV (default ./venv)
#
#SBATCH --job-name=zeroshot
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
    echo "usage: sbatch scripts/slurm/zeroshot_uq.sh <run_tag> [zeroshot_uq.py args...]" >&2
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

CONFIG=${CONFIG:-config/wollombi-zeroshot-burnett.json}
OUT=experiments/zeroshot_uq/$RUN_TAG

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} tag=$RUN_TAG config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Fail fast on a CPU-only torch instead of after the queue wait + dataset load
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'torch sees no CUDA device')"

python scripts/zeroshot_uq.py -c "$CONFIG" -output_dir "$OUT" "$@"
