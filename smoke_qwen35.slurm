#!/bin/bash
#SBATCH --job-name=value_bandit_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --output=logs/value_bandit_smoke_%j.out
#SBATCH --error=logs/value_bandit_smoke_%j.err

# Conda's hook reads optional _CE_* variables. Enable nounset after activation.
set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
SMOKE_BASE="/scratch/gpfs/JORDANAT/${USER}/value_steering_bandit"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
FALLBACK_MODEL_PATH="${FALLBACK_MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3-4B-Instruct-2507}"
CONDA_ENV="${CONDA_ENV:-value-steering-bandit}"

cd "$PROJECT_DIR"

module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$SMOKE_BASE/hf_cache"
export HF_DATASETS_CACHE="$SMOKE_BASE/hf_cache/datasets"
export TRANSFORMERS_CACHE="$SMOKE_BASE/hf_cache"
export TORCH_HOME="$SMOKE_BASE/torch_cache"
export XDG_CACHE_HOME="$SMOKE_BASE/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p artifacts logs "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    echo "Submit with MODEL_PATH=/absolute/model/path sbatch smoke_qwen35.slurm" >&2
    exit 1
fi

echo "Running CPU/unit gate, including ridge, counterfactual-rollout, matched-history, and supervised-probe regressions"
python -m pytest -q

echo "Running real-model compatibility gate"
python -m experiments.check_qwen_compatibility \
    --model "$MODEL_PATH" \
    --fallback-model "$FALLBACK_MODEL_PATH" \
    --output artifacts/qwen_compatibility_smoke.json

echo "Running two short end-to-end bandit episodes"
SMOKE_RUN_ID="${SLURM_JOB_ID:-manual}"
python -m experiments.run_bandit_baseline \
    --model "$MODEL_PATH" \
    --episodes 2 \
    --max-decisions 3 \
    --seed 90210 \
    --output "artifacts/bandit_smoke_${SMOKE_RUN_ID}.csv"

echo "Collecting two tiny real-model activation shards"
python -m experiments.collect_bandit_activations \
    --model "$MODEL_PATH" \
    --episodes 2 \
    --max-decisions 3 \
    --seed 90310 \
    --output-dir "artifacts/activation_smoke_${SMOKE_RUN_ID}"

echo "Running one truncated paired forced-A/B rollout gate"
python -m experiments.check_advantage_rollout \
    --model "$MODEL_PATH" \
    --activation-dir "artifacts/activation_smoke_${SMOKE_RUN_ID}" \
    --output "artifacts/advantage_rollout_smoke_${SMOKE_RUN_ID}.json"

echo "Smoke test passed"
