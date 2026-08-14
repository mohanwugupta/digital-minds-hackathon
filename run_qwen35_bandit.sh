#!/bin/bash
#SBATCH --job-name=value_bandit_qwen35
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=4:30:00
#SBATCH --output=logs/value_bandit_qwen35_%j.out
#SBATCH --error=logs/value_bandit_qwen35_%j.err

# Conda's hook reads optional _CE_* variables. Enable nounset after activation.
set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
CLUSTER_BASE="/scratch/gpfs/JORDANAT/${USER}/value_steering_bandit"
MODEL_PATH="${MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3.5-4B}"
FALLBACK_MODEL_PATH="${FALLBACK_MODEL_PATH:-/scratch/gpfs/JORDANAT/${USER}/models/Qwen--Qwen3-4B-Instruct-2507}"
CONDA_ENV="${CONDA_ENV:-value-steering-bandit}"
PHASE="${PHASE:-compatibility}"
PROBE_EPISODES="${PROBE_EPISODES:-512}"
CONFIRMATORY_EPISODES="${CONFIRMATORY_EPISODES:-48}"
MATCHED_RANDOM_SETS="${MATCHED_RANDOM_SETS:-20}"
ADVANTAGE_ROLLOUTS="${ADVANTAGE_ROLLOUTS:-20}"
ADVANTAGE_STATES_PER_SPLIT="${ADVANTAGE_STATES_PER_SPLIT:-128}"
ADVANTAGE_NUM_SHARDS="${ADVANTAGE_NUM_SHARDS:-1}"
ADVANTAGE_SHARD_INDEX="${ADVANTAGE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ADVANTAGE_TARGET_DIR="${ADVANTAGE_TARGET_DIR:-artifacts/advantage_targets}"
ADVANTAGE_PROBE_DIR="${ADVANTAGE_PROBE_DIR:-artifacts/advantage_probes}"

cd "$PROJECT_DIR"
mkdir -p logs artifacts "$CLUSTER_BASE/hf_cache" "$CLUSTER_BASE/torch_cache" "$CLUSTER_BASE/cache"

module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u

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

collect_probe_phase() {
  echo "Starting probe activation collection (${PROBE_EPISODES} episodes)"
  python -m experiments.collect_bandit_activations \
    --model "$MODEL_PATH" --episodes "$PROBE_EPISODES"
}

train_probe_phase() {
  echo "Starting probe training, mechanism analysis, and calibration"
  python -m experiments.train_value_probe
  python -m experiments.calibrate_steering
}

train_mc_probe_phase() {
  echo "Starting supervised Monte Carlo future-return probe analysis"
  python -m experiments.train_monte_carlo_probe
  python -m analysis.analyze_monte_carlo_probe
}

calibrate_mc_probe_phase() {
  echo "Calibrating the inspected supervised Monte Carlo probe"
  python -m experiments.calibrate_steering \
    --probe artifacts/mc_value_probes/frozen_best.pt \
    --split artifacts/value_probes/episode_split.json \
    --output artifacts/mc_value_probes/steering_calibration.json
}

linear_probe_phase() {
  echo "Training ridge-linear future-return and direct-persistence probes"
  python -m experiments.train_linear_probes
  python -m analysis.analyze_linear_probes
}

collect_advantage_phase() {
  local output
  if [ "$ADVANTAGE_NUM_SHARDS" -eq 1 ]; then
    output="${ADVANTAGE_TARGET_DIR}/targets.csv"
  else
    printf -v output "%s/targets_shard_%03d.csv" "$ADVANTAGE_TARGET_DIR" "$ADVANTAGE_SHARD_INDEX"
  fi
  echo "Collecting continuation advantage: ${ADVANTAGE_ROLLOUTS} paired rollouts/state; shard ${ADVANTAGE_SHARD_INDEX}/${ADVANTAGE_NUM_SHARDS}"
  python -m experiments.collect_continuation_advantage \
    --model "$MODEL_PATH" \
    --rollouts "$ADVANTAGE_ROLLOUTS" \
    --states-per-split "$ADVANTAGE_STATES_PER_SPLIT" \
    --num-shards "$ADVANTAGE_NUM_SHARDS" \
    --shard-index "$ADVANTAGE_SHARD_INDEX" \
    --output "$output"
}

train_advantage_phase() {
  echo "Training and analyzing ridge continuation-advantage probes"
  python -m experiments.train_advantage_probe \
    --targets "${ADVANTAGE_TARGET_DIR}/targets*.csv" \
    --output-dir "$ADVANTAGE_PROBE_DIR" \
    --minimum-states-per-split "$ADVANTAGE_STATES_PER_SPLIT"
  python -m analysis.analyze_advantage_probe \
    --probe-dir "$ADVANTAGE_PROBE_DIR" \
    --output-dir "$ADVANTAGE_PROBE_DIR/publication"
}

collect_confirmatory_phase() {
  echo "Starting held-out confirmatory collection (${CONFIRMATORY_EPISODES} episodes)"
  python -m experiments.collect_bandit_activations \
    --model "$MODEL_PATH" --episodes "$CONFIRMATORY_EPISODES" --seed 52026 \
    --output-dir artifacts/confirmatory_state_bank
}

matched_phase() {
  echo "Starting matched causal replay (${MATCHED_RANDOM_SETS} random-neuron sets)"
  python -m experiments.run_bandit_intervention \
    --model "$MODEL_PATH" \
    --state-bank artifacts/confirmatory_state_bank \
    --random-sets "$MATCHED_RANDOM_SETS"
  python -m analysis.analyze_persistence
}

case "$PHASE" in
  compatibility)
    python -m experiments.check_qwen_compatibility \
      --model "$MODEL_PATH" \
      --fallback-model "$FALLBACK_MODEL_PATH"
    ;;
  pilot)
    python -m experiments.run_bandit_baseline --model "$MODEL_PATH" --episodes 200
    python -m analysis.analyze_pilot
    ;;
  collect_probe)
    collect_probe_phase
    ;;
  train_probe)
    train_probe_phase
    ;;
  probe_pipeline)
    collect_probe_phase
    train_probe_phase
    ;;
  train_mc_probe)
    train_mc_probe_phase
    ;;
  calibrate_mc_probe)
    calibrate_mc_probe_phase
    ;;
  linear_probes)
    linear_probe_phase
    ;;
  collect_advantage)
    collect_advantage_phase
    ;;
  train_advantage)
    train_advantage_phase
    ;;
  advantage_pipeline)
    if [ "$ADVANTAGE_NUM_SHARDS" -ne 1 ]; then
      echo "advantage_pipeline requires ADVANTAGE_NUM_SHARDS=1; run sharded collection and training as separate jobs" >&2
      exit 2
    fi
    collect_advantage_phase
    train_advantage_phase
    ;;
  collect_confirmatory)
    collect_confirmatory_phase
    ;;
  matched)
    matched_phase
    ;;
  causal_pipeline)
    collect_confirmatory_phase
    matched_phase
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
