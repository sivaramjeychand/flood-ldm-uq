#!/bin/bash
#SBATCH --job-name=condscale
#SBATCH --partition=gpu
# xgpj0: torch CUDA library preload fails there (job 939926)
#SBATCH --exclude=xgpj0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
source venv/bin/activate

python scripts/cond_scale_probe.py \
  -c config/wollombi-trnf.json \
  -output_dir experiments/cond_scale_probe/baseline \
  --n_scenes 10 --n_samples 20 \
  --scales 0 0.25 0.5 0.75 0.9 1.0 1.1 1.25 1.5
