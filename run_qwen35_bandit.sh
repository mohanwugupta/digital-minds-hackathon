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
ADVANTAGE_ROLLOUTS="${ADVANTAGE_ROLLOUTS:-20}"
ADVANTAGE_STATES_PER_SPLIT="${ADVANTAGE_STATES_PER_SPLIT:-128}"
ADVANTAGE_NUM_SHARDS="${ADVANTAGE_NUM_SHARDS:-1}"
ADVANTAGE_SHARD_INDEX="${ADVANTAGE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ADVANTAGE_TARGET_DIR="${ADVANTAGE_TARGET_DIR:-artifacts/advantage_targets}"
ADVANTAGE_PROBE_DIR="${ADVANTAGE_PROBE_DIR:-artifacts/advantage_probes}"
CAUSAL_NUM_SHARDS="${CAUSAL_NUM_SHARDS:-1}"
CAUSAL_SHARD_INDEX="${CAUSAL_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
CAUSAL_MAXIMUM_STATES="${CAUSAL_MAXIMUM_STATES:-0}"
CAUSAL_RANDOM_DIRECTIONS="${CAUSAL_RANDOM_DIRECTIONS:-20}"
DISSOCIATION_NUM_SHARDS="${DISSOCIATION_NUM_SHARDS:-1}"
DISSOCIATION_SHARD_INDEX="${DISSOCIATION_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
DISSOCIATION_MAXIMUM_STATES="${DISSOCIATION_MAXIMUM_STATES:-0}"
CROSS_TASK_NUM_SHARDS="${CROSS_TASK_NUM_SHARDS:-1}"
CROSS_TASK_SHARD_INDEX="${CROSS_TASK_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
CROSS_TASK_MAXIMUM_STATES="${CROSS_TASK_MAXIMUM_STATES:-0}"
LAYERWISE_NUM_SHARDS="${LAYERWISE_NUM_SHARDS:-1}"
LAYERWISE_SHARD_INDEX="${LAYERWISE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
TRACK_A_RUN_ID="${TRACK_A_RUN_ID:-track_a_v1}"
TRACK_B_RUN_ID="${TRACK_B_RUN_ID:-track_b_shared_v3}"
TRACK_B_ROOT="artifacts/cross_task/${TRACK_B_RUN_ID}"

if [[ ! "$TRACK_A_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "TRACK_A_RUN_ID may contain only letters, digits, underscores, and hyphens" >&2
    exit 2
fi
if [[ ! "$TRACK_B_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "TRACK_B_RUN_ID may contain only letters, digits, underscores, and hyphens" >&2
    exit 2
fi

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
  echo "Starting initial TD probe training and mechanism analysis"
  python -m experiments.train_value_probe
}

train_mc_probe_phase() {
  echo "Starting supervised Monte Carlo future-return probe analysis"
  python -m experiments.train_monte_carlo_probe
  python -m analysis.analyze_monte_carlo_probe
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

calibrate_causal_directions_phase() {
  echo "Calibrating frozen ridge directions on validation episodes only"
  python -m experiments.calibrate_causal_directions
}

causal_steering_collect_phase() {
  local output
  if [ "$CAUSAL_NUM_SHARDS" -eq 1 ]; then
    output="artifacts/causal_steering/replays.csv"
  else
    printf -v output "artifacts/causal_steering/replays_shard_%03d.csv" "$CAUSAL_SHARD_INDEX"
  fi
  echo "Collecting ridge causal replays; shard ${CAUSAL_SHARD_INDEX}/${CAUSAL_NUM_SHARDS}"
  python -m experiments.run_causal_steering \
    --model "$MODEL_PATH" \
    --random-directions "$CAUSAL_RANDOM_DIRECTIONS" \
    --maximum-states "$CAUSAL_MAXIMUM_STATES" \
    --num-shards "$CAUSAL_NUM_SHARDS" \
    --shard-index "$CAUSAL_SHARD_INDEX" \
    --output "$output"
}

causal_steering_analyze_phase() {
  echo "Analyzing ridge causal steering"
  python -m analysis.analyze_causal_steering
}

value_dissociation_collect_phase() {
  local output
  if [ "$DISSOCIATION_NUM_SHARDS" -eq 1 ]; then
    output="artifacts/value_dissociation/factorial.csv"
  else
    printf -v output "artifacts/value_dissociation/factorial_shard_%03d.csv" "$DISSOCIATION_SHARD_INDEX"
  fi
  echo "Collecting STOP x CONTINUE payoff factorial; shard ${DISSOCIATION_SHARD_INDEX}/${DISSOCIATION_NUM_SHARDS}"
  python -m experiments.run_value_dissociation \
    --model "$MODEL_PATH" \
    --maximum-states "$DISSOCIATION_MAXIMUM_STATES" \
    --num-shards "$DISSOCIATION_NUM_SHARDS" \
    --shard-index "$DISSOCIATION_SHARD_INDEX" \
    --output "$output"
}

value_dissociation_analyze_phase() {
  echo "Analyzing STOP x CONTINUE value dissociation"
  python -m analysis.analyze_value_dissociation
}

collect_confirmatory_phase() {
  echo "Starting held-out confirmatory collection (${CONFIRMATORY_EPISODES} episodes)"
  python -m experiments.collect_bandit_activations \
    --model "$MODEL_PATH" --episodes "$CONFIRMATORY_EPISODES" --seed 52026 \
    --output-dir artifacts/confirmatory_state_bank
}

factorial_layerwise_project_phase() {
  local output
  if [ "$LAYERWISE_NUM_SHARDS" -eq 1 ]; then
    output="artifacts/value_dissociation/layerwise_projections_${TRACK_A_RUN_ID}.csv"
  else
    printf -v output "artifacts/value_dissociation/layerwise_projections_%s_shard_%03d.csv" "$TRACK_A_RUN_ID" "$LAYERWISE_SHARD_INDEX"
  fi
  echo "Projecting the existing factorial through all frozen persistence probes; shard ${LAYERWISE_SHARD_INDEX}/${LAYERWISE_NUM_SHARDS}"
  python -m experiments.project_factorial_layers \
    --model "$MODEL_PATH" \
    --save-activations \
    --num-shards "$LAYERWISE_NUM_SHARDS" \
    --shard-index "$LAYERWISE_SHARD_INDEX" \
    --output "$output"
}

factorial_layerwise_analyze_phase() {
  echo "Analyzing complete Track A replay ${TRACK_A_RUN_ID}"
  python -m analysis.analyze_factorial_layerwise \
    --input "artifacts/value_dissociation/layerwise_projections_${TRACK_A_RUN_ID}*.csv" \
    --output-dir "artifacts/value_dissociation/layerwise_publication_${TRACK_A_RUN_ID}"
}

track_a_tests_phase() {
  echo "Running Track A unit, integration, and frozen-baseline gates"
  python -m pytest -q \
    tests/test_baseline_regression_manifest.py \
    tests/test_value_dissociation.py \
    tests/test_value_dissociation_analysis.py \
    tests/test_factorial_layerwise_analysis.py \
    tests/test_factorial_source_coverage.py \
    tests/test_layerwise_detection.py
  python -m analysis.check_baseline_regression
}

track_a_smoke_phase() {
  local smoke_id="${SLURM_JOB_ID:-manual}"
  local smoke_root="artifacts/track_a_smoke_${smoke_id}"
  echo "Running one-state Track A replay smoke into ${smoke_root}"
  python -m experiments.project_factorial_layers \
    --model "$MODEL_PATH" \
    --maximum-states 1 \
    --save-activations \
    --activation-output-dir "${smoke_root}/activations" \
    --output "${smoke_root}/layerwise_projections.csv"
  python -m analysis.analyze_factorial_layerwise \
    --input "${smoke_root}/layerwise_projections.csv" \
    --output-dir "${smoke_root}/analysis" \
    --allow-partial
  test -s "${smoke_root}/analysis/factorial_layerwise_summary.json"
  test -s "${smoke_root}/analysis/factorial_layerwise_effects.csv"
  test -s "${smoke_root}/analysis/factorial_layerwise_effects.svg"
  test -s "${smoke_root}/analysis/factorial_layerwise_trajectory.svg"
  test -s "${smoke_root}/analysis/factorial_layerwise_report.md"
  echo "Track A replay smoke passed"
}

cross_task_collect_phase() {
  local task="$1"
  local output_dir="${TRACK_B_ROOT}/${task}_activation_bank"
  echo "Collecting counterbalanced ${task} activation bank; shard ${CROSS_TASK_SHARD_INDEX}/${CROSS_TASK_NUM_SHARDS}"
  python -m experiments.collect_cross_task_activations \
    --task "$task" \
    --model "$MODEL_PATH" \
    --output-dir "$output_dir" \
    --num-shards "$CROSS_TASK_NUM_SHARDS" \
    --shard-index "$CROSS_TASK_SHARD_INDEX"
}

cross_task_power_phase() {
  echo "Running prospective independent-pair power check before Track B collection"
  python -m analysis.cross_task_power \
    --output "${TRACK_B_ROOT}/prospective_power.json"
}

cross_task_matched_label_phase() {
  local task="$1"
  echo "Replaying exact matched ${task} semantic histories under both label mappings; shard ${CROSS_TASK_SHARD_INDEX}/${CROSS_TASK_NUM_SHARDS}"
  python -m experiments.collect_matched_label_replays \
    --task "$task" \
    --model "$MODEL_PATH" \
    --activation-dir "${TRACK_B_ROOT}/${task}_activation_bank" \
    --split "${TRACK_B_ROOT}/${task}_episode_split.json" \
    --output-dir "${TRACK_B_ROOT}/${task}_matched_label_bank" \
    --maximum-states "$CROSS_TASK_MAXIMUM_STATES" \
    --num-shards "$CROSS_TASK_NUM_SHARDS" \
    --shard-index "$CROSS_TASK_SHARD_INDEX"
}

cross_task_behavioral_validate_phase() {
  echo "Auditing all tasks and validating Foraging + Solvability behavior on development episodes only"
  python -m analysis.validate_cross_task_behavior \
    --foraging-bank "${TRACK_B_ROOT}/foraging_activation_bank" \
    --solvability-bank "${TRACK_B_ROOT}/solvability_activation_bank" \
    --control-bank "${TRACK_B_ROOT}/control_activation_bank" \
    --terminality-bank "${TRACK_B_ROOT}/terminality_activation_bank" \
    --foraging-split "${TRACK_B_ROOT}/foraging_episode_split.json" \
    --solvability-split "${TRACK_B_ROOT}/solvability_episode_split.json" \
    --output-dir "${TRACK_B_ROOT}/behavioral"
}

cross_task_train_ceiling_phase() {
  local task="$1"
  python -m experiments.train_task_persistence_probes \
    --task "$task" \
    --activation-dir "${TRACK_B_ROOT}/${task}_activation_bank" \
    --split "${TRACK_B_ROOT}/${task}_episode_split.json" \
    --behavioral-gate "${TRACK_B_ROOT}/behavioral/behavioral_validation_summary.json" \
    --output-dir "${TRACK_B_ROOT}/${task}_probes" \
    --defer-test
}

cross_task_train_shared_phase() {
  python -m experiments.train_shared_persistence \
    --foraging-bank "${TRACK_B_ROOT}/foraging_activation_bank" \
    --foraging-split "${TRACK_B_ROOT}/foraging_episode_split.json" \
    --solvability-bank "${TRACK_B_ROOT}/solvability_activation_bank" \
    --solvability-split "${TRACK_B_ROOT}/solvability_episode_split.json" \
    --behavioral-gate "${TRACK_B_ROOT}/behavioral/behavioral_validation_summary.json" \
    --output-dir "${TRACK_B_ROOT}/shared_probes"
}

cross_task_shared_representational_phase() {
  python -m analysis.analyze_shared_persistence_transfer \
    --shared-probe-dir "${TRACK_B_ROOT}/shared_probes" \
    --foraging-bank "${TRACK_B_ROOT}/foraging_activation_bank" \
    --foraging-split "${TRACK_B_ROOT}/foraging_episode_split.json" \
    --solvability-bank "${TRACK_B_ROOT}/solvability_activation_bank" \
    --solvability-split "${TRACK_B_ROOT}/solvability_episode_split.json" \
    --control-bank "${TRACK_B_ROOT}/control_activation_bank" \
    --control-split "${TRACK_B_ROOT}/control_episode_split.json" \
    --terminality-bank "${TRACK_B_ROOT}/terminality_activation_bank" \
    --terminality-split "${TRACK_B_ROOT}/terminality_episode_split.json" \
    --foraging-label-bank "${TRACK_B_ROOT}/foraging_matched_label_bank" \
    --solvability-label-bank "${TRACK_B_ROOT}/solvability_matched_label_bank" \
    --foraging-probe-dir "${TRACK_B_ROOT}/foraging_probes" \
    --solvability-probe-dir "${TRACK_B_ROOT}/solvability_probes" \
    --behavioral-gate "${TRACK_B_ROOT}/behavioral/behavioral_validation_summary.json" \
    --output-dir "${TRACK_B_ROOT}/shared_transfer"
}

cross_task_bandit_diagnostic_phase() {
  python -m analysis.analyze_cross_task_transfer \
    --foraging-bank "${TRACK_B_ROOT}/foraging_activation_bank" \
    --control-bank "${TRACK_B_ROOT}/control_activation_bank" \
    --foraging-split "${TRACK_B_ROOT}/foraging_episode_split.json" \
    --control-split "${TRACK_B_ROOT}/control_episode_split.json" \
    --foraging-probe "${TRACK_B_ROOT}/foraging_probes/frozen_best_persistence.pt" \
    --behavioral-gate "${TRACK_B_ROOT}/behavioral/behavioral_validation_summary.json" \
    --output-dir "${TRACK_B_ROOT}/transfer"
}

cross_task_causal_calibrate_phase() {
  python -m experiments.calibrate_shared_persistence_steering \
    --activation-dir "${TRACK_B_ROOT}/solvability_activation_bank" \
    --split "${TRACK_B_ROOT}/solvability_episode_split.json" \
    --probe "${TRACK_B_ROOT}/shared_probes/frozen_primary.pt" \
    --representational-summary "${TRACK_B_ROOT}/shared_transfer/shared_persistence_transfer_summary.json" \
    --output "${TRACK_B_ROOT}/causal/calibration.json"
}

cross_task_causal_collect_phase() {
  local task="$1"
  local output
  if [ "$CROSS_TASK_NUM_SHARDS" -eq 1 ]; then
    output="${TRACK_B_ROOT}/causal/${task}_replays.csv"
  else
    printf -v output "%s/causal/%s_replays_shard_%03d.csv" "$TRACK_B_ROOT" "$task" "$CROSS_TASK_SHARD_INDEX"
  fi
  echo "Collecting ${task} cross-task causal replays; shard ${CROSS_TASK_SHARD_INDEX}/${CROSS_TASK_NUM_SHARDS}"
  python -m experiments.run_cross_task_steering \
    --task "$task" \
    --model "$MODEL_PATH" \
    --activation-dir "${TRACK_B_ROOT}/${task}_activation_bank" \
    --split "${TRACK_B_ROOT}/${task}_episode_split.json" \
    --calibration "${TRACK_B_ROOT}/causal/calibration.json" \
    --maximum-states "$CROSS_TASK_MAXIMUM_STATES" \
    --num-shards "$CROSS_TASK_NUM_SHARDS" \
    --shard-index "$CROSS_TASK_SHARD_INDEX" \
    --output "$output"
}

cross_task_causal_analyze_phase() {
  python -m analysis.analyze_shared_persistence_causal \
    --solvability-input "${TRACK_B_ROOT}/causal/solvability_replays*.csv" \
    --control-input "${TRACK_B_ROOT}/causal/control_replays*.csv" \
    --terminality-input "${TRACK_B_ROOT}/causal/terminality_replays*.csv" \
    --solvability-bank "${TRACK_B_ROOT}/solvability_activation_bank" \
    --control-bank "${TRACK_B_ROOT}/control_activation_bank" \
    --terminality-bank "${TRACK_B_ROOT}/terminality_activation_bank" \
    --solvability-split "${TRACK_B_ROOT}/solvability_episode_split.json" \
    --control-split "${TRACK_B_ROOT}/control_episode_split.json" \
    --terminality-split "${TRACK_B_ROOT}/terminality_episode_split.json" \
    --calibration "${TRACK_B_ROOT}/causal/calibration.json" \
    --representational-summary "${TRACK_B_ROOT}/shared_transfer/shared_persistence_transfer_summary.json" \
    --output-dir "${TRACK_B_ROOT}/causal/publication"
}

track_b_tests_phase() {
  echo "Running Track B unit, integration, and frozen-baseline gates"
  python -m pytest -q \
    tests/test_baseline_regression_manifest.py \
    tests/test_binary_action_tokens.py \
    tests/test_causal_ridge_steering.py \
    tests/test_replay_matching.py \
    tests/test_cross_task_environment.py \
    tests/test_cross_task_collection.py \
    tests/test_cross_task_transfer.py \
    tests/test_cross_task_causal.py \
    tests/test_track_b_critical.py \
    tests/test_shared_persistence_critical.py \
    tests/test_new_direction_controls.py \
    tests/test_shared_ridge_probe.py
  python -m analysis.check_baseline_regression
}

track_b_smoke_phase() {
  local smoke_id="${SLURM_JOB_ID:-manual}"
  local smoke_root="artifacts/track_b_smoke_${smoke_id}"
  echo "Running counterbalanced real-model Track B smoke into ${smoke_root}"
  python -m experiments.check_cross_task_compatibility \
    --model "$MODEL_PATH" \
    --output "${smoke_root}/compatibility.json"
  python -m experiments.collect_cross_task_activations \
    --task foraging --model "$MODEL_PATH" --episodes 16 --max-decisions 2 \
    --output-dir "${smoke_root}/foraging_activation_bank"
  python -m experiments.collect_cross_task_activations \
    --task solvability --model "$MODEL_PATH" --episodes 16 --max-attempts 2 \
    --output-dir "${smoke_root}/solvability_activation_bank"
  python -m experiments.collect_cross_task_activations \
    --task control --model "$MODEL_PATH" --episodes 16 \
    --output-dir "${smoke_root}/control_activation_bank"
  python -m experiments.collect_cross_task_activations \
    --task terminality --model "$MODEL_PATH" --episodes 16 \
    --output-dir "${smoke_root}/terminality_activation_bank"
  python -m analysis.validate_cross_task_behavior \
    --foraging-bank "${smoke_root}/foraging_activation_bank" \
    --solvability-bank "${smoke_root}/solvability_activation_bank" \
    --control-bank "${smoke_root}/control_activation_bank" \
    --terminality-bank "${smoke_root}/terminality_activation_bank" \
    --output-dir "${smoke_root}/audit" \
    --integrity-only
  python -m experiments.smoke_shared_persistence \
    --foraging-bank "${smoke_root}/foraging_activation_bank" \
    --solvability-bank "${smoke_root}/solvability_activation_bank" \
    --control-bank "${smoke_root}/control_activation_bank" \
    --terminality-bank "${smoke_root}/terminality_activation_bank" \
    --output-dir "${smoke_root}/shared"
  test -s "${smoke_root}/compatibility.json"
  test -s "${smoke_root}/audit/cross_task_integrity_summary.json"
  test -s "${smoke_root}/shared/shared_pipeline_smoke_summary.json"
  echo "Track B real-model smoke passed"
}

case "$PHASE" in
  compatibility)
    python -m experiments.check_qwen_compatibility \
      --model "$MODEL_PATH" \
      --fallback-model "$FALLBACK_MODEL_PATH"
    ;;
  pilot)
    python -m experiments.run_bandit_baseline --model "$MODEL_PATH" --episodes 200
    python -m analysis.analyze_pilot_detailed
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
  causal_calibrate)
    calibrate_causal_directions_phase
    ;;
  causal_steering_collect)
    causal_steering_collect_phase
    ;;
  causal_steering_analyze)
    causal_steering_analyze_phase
    ;;
  value_dissociation_collect)
    value_dissociation_collect_phase
    ;;
  value_dissociation_analyze)
    value_dissociation_analyze_phase
    ;;
  value_dissociation_pipeline)
    if [ "$DISSOCIATION_NUM_SHARDS" -ne 1 ]; then
      echo "value_dissociation_pipeline requires DISSOCIATION_NUM_SHARDS=1" >&2
      exit 2
    fi
    value_dissociation_collect_phase
    value_dissociation_analyze_phase
    ;;
  collect_confirmatory)
    collect_confirmatory_phase
    ;;
  baseline_regression)
    python -m analysis.check_baseline_regression
    ;;
  factorial_layerwise_project)
    factorial_layerwise_project_phase
    ;;
  factorial_layerwise_analyze)
    factorial_layerwise_analyze_phase
    ;;
  track_a_tests)
    track_a_tests_phase
    ;;
  track_a_smoke)
    track_a_smoke_phase
    ;;
  cross_task_compatibility)
    python -m experiments.check_cross_task_compatibility \
      --model "$MODEL_PATH" \
      --output "${TRACK_B_ROOT}/compatibility.json"
    ;;
  cross_task_power)
    cross_task_power_phase
    ;;
  cross_task_collect_foraging)
    cross_task_collect_phase foraging
    ;;
  cross_task_collect_solvability)
    cross_task_collect_phase solvability
    ;;
  cross_task_collect_control)
    cross_task_collect_phase control
    ;;
  cross_task_collect_terminality)
    cross_task_collect_phase terminality
    ;;
  cross_task_matched_label_foraging)
    cross_task_matched_label_phase foraging
    ;;
  cross_task_matched_label_solvability)
    cross_task_matched_label_phase solvability
    ;;
  cross_task_behavioral_validate)
    cross_task_behavioral_validate_phase
    ;;
  cross_task_train_ceiling)
    cross_task_train_ceiling_phase foraging
    ;;
  cross_task_train_foraging_ceiling)
    cross_task_train_ceiling_phase foraging
    ;;
  cross_task_train_solvability_ceiling)
    cross_task_train_ceiling_phase solvability
    ;;
  cross_task_train_shared)
    cross_task_train_shared_phase
    ;;
  cross_task_shared_representational)
    cross_task_shared_representational_phase
    ;;
  cross_task_representational)
    cross_task_bandit_diagnostic_phase
    ;;
  cross_task_causal_calibrate)
    cross_task_causal_calibrate_phase
    ;;
  cross_task_causal_solvability)
    cross_task_causal_collect_phase solvability
    ;;
  cross_task_causal_control)
    cross_task_causal_collect_phase control
    ;;
  cross_task_causal_terminality)
    cross_task_causal_collect_phase terminality
    ;;
  cross_task_causal_analyze)
    cross_task_causal_analyze_phase
    ;;
  track_b_tests)
    track_b_tests_phase
    ;;
  track_b_smoke)
    track_b_smoke_phase
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac
