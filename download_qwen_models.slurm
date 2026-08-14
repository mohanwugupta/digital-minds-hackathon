#!/bin/bash
#SBATCH --job-name=download_qwen_models
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_qwen_models_%j.out
#SBATCH --error=logs/download_qwen_models_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/gpfs/JORDANAT/mg9965/models}"
MODEL_SELECTION="${MODEL_SELECTION:-all}"
HF_REVISION="${HF_REVISION:-main}"
CONDA_ENV="${CONDA_ENV:-value-steering-bandit}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/scratch/gpfs/JORDANAT/mg9965/hf_cache}"

cd "$PROJECT_DIR"
mkdir -p logs "$MODEL_ROOT" "$HF_CACHE_ROOT"

module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$HF_CACHE_ROOT"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

python -m scripts.download_qwen_models \
    --model-root "$MODEL_ROOT" \
    --selection "$MODEL_SELECTION" \
    --revision "$HF_REVISION" \
    --max-workers "${SLURM_CPUS_PER_TASK:-4}"
