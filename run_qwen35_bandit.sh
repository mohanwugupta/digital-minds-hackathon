#!/bin/bash
#SBATCH --job-name=value_bandit_qwen35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1:30:00
#SBATCH --output=logs/value_bandit_qwen35_%j.out
#SBATCH --error=logs/value_bandit_qwen35_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CLUSTER_BASE="/scratch/gpfs/JORDANAT/${USER}/value_steering_bandit"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
CONDA_ENV="${CONDA_ENV:-value-steering-bandit}"
PHASE="${PHASE:-compatibility}"

cd "$PROJECT_DIR"
mkdir -p logs artifacts "$CLUSTER_BASE/hf_cache" "$CLUSTER_BASE/torch_cache" "$CLUSTER_BASE/cache"

module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$CLUSTER_BASE/hf_cache"
export HF_DATASETS_CACHE="$CLUSTER_BASE/hf_cache/datasets"
export TRANSFORMERS_CACHE="$CLUSTER_BASE/hf_cache"
export TORCH_HOME="$CLUSTER_BASE/torch_cache"
export XDG_CACHE_HOME="$CLUSTER_BASE/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    exit 1
fi

case "$PHASE" in
  compatibility)
    python -m experiments.check_qwen_compatibility --model "$MODEL_PATH"
    ;;
  pilot)
    python -m experiments.run_bandit_baseline --model "$MODEL_PATH" --episodes 200
    python -m analysis.analyze_pilot
    ;;
  collect_probe)
    python -m experiments.collect_bandit_activations --model "$MODEL_PATH" --episodes 2000
    ;;
  train_probe)
    python -m experiments.train_value_probe
    python -m experiments.calibrate_steering
    ;;
  collect_confirmatory)
    python -m experiments.collect_bandit_activations \
      --model "$MODEL_PATH" --episodes 200 --seed 52026 \
      --output-dir artifacts/confirmatory_state_bank
    ;;
  matched)
    python -m experiments.run_bandit_intervention \
      --model "$MODEL_PATH" --state-bank artifacts/confirmatory_state_bank
    python -m analysis.analyze_persistence
    ;;
  sequential)
    python -m experiments.run_bandit_sequential \
      --model "$MODEL_PATH" --matched-analysis artifacts/matched_analysis.json
    python -m analysis.analyze_sequential
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac
