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
PERSISTENCE_ROBUSTNESS_RUN_ID="${PERSISTENCE_ROBUSTNESS_RUN_ID:-robustness_v1}"
PERSISTENCE_ROBUSTNESS_TASKS="${PERSISTENCE_ROBUSTNESS_TASKS:-voluntary_waiting,progressive_ratio,sunk_cost,controllability,debugging_persistence}"
PERSISTENCE_ROBUSTNESS_NUM_SHARDS="${PERSISTENCE_ROBUSTNESS_NUM_SHARDS:-1}"
PERSISTENCE_ROBUSTNESS_SHARD_INDEX="${PERSISTENCE_ROBUSTNESS_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
PERSISTENCE_ROBUSTNESS_DATASET="${PERSISTENCE_ROBUSTNESS_DATASET:-pilot}"

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

case "$PHASE" in
  comparative_persistence|comparative_persistence_tests|persistence_robustness_tests|persistence_robustness_analysis|persistence_robustness_battery_finalize|persistence_robustness_matched_finalize)
    ;;
  *)
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model directory not found: $MODEL_PATH" >&2
        exit 1
    fi
    ;;
esac

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

persistence_tests_phase() {
  echo "Running Track C RED/GREEN, regression, and scientific-gate tests"
  python -m pytest -q \
    tests/test_persistence_discovery_baseline.py \
    tests/test_persistence_contrasts.py \
    tests/test_cross_manipulation_isolation.py \
    tests/test_cross_task_isolation.py \
    tests/test_displacement_features.py \
    tests/test_subspace_recovery.py \
    tests/test_persistence_specificity.py \
    tests/test_generic_value_control.py \
    tests/test_persistence_discovery_smoke.py \
    tests/test_latent_state_recovery.py \
    tests/test_latent_model_confusion.py \
    tests/test_future_behavior_prediction.py
  python -m analysis.check_persistence_discovery_baseline
}

persistence_smoke_phase() {
  local smoke_id="${SLURM_JOB_ID:-manual}"
  local smoke_root="artifacts/persistence_discovery_smoke_${smoke_id}"
  echo "Running synthetic Track C search smoke into ${smoke_root}"
  python -m experiments.smoke_persistence_discovery \
    --output-dir "${smoke_root}/search"
  echo "Running real-model generic-value collection smoke"
  python -m experiments.collect_cross_task_activations \
    --task generic_value --model "$MODEL_PATH" --episodes 16 \
    --output-dir "${smoke_root}/generic_value_activation_bank"
  test -s "${smoke_root}/search/persistence_discovery_summary.json"
  test -s "${smoke_root}/search/layerwise_transfer_map.csv"
  test -s "${smoke_root}/search/layerwise_cross_task_transfer.svg"
  test -s "${smoke_root}/generic_value_activation_bank/episode_00001.pt"
  echo "Track C smoke passed"
}

persistence_collect_generic_value_phase() {
  echo "Collecting one-shot generic-value controls; shard ${CROSS_TASK_SHARD_INDEX}/${CROSS_TASK_NUM_SHARDS}"
  python -m experiments.collect_cross_task_activations \
    --task generic_value \
    --model "$MODEL_PATH" \
    --output-dir artifacts/persistence_discovery/generic_value_activation_bank \
    --num-shards "$CROSS_TASK_NUM_SHARDS" \
    --shard-index "$CROSS_TASK_SHARD_INDEX"
}

persistence_contrast_phase() {
  echo "Building and behavior-gating the matched persistence/nuisance contrast bank"
  python -m analysis.run_persistence_discovery --phase contrast
}

persistence_search_phase() {
  echo "Running all-layer static/displacement rank-1/2/4 contrast search"
  python -m analysis.run_persistence_discovery --phase search
}

persistence_latent_phase() {
  echo "Running synthetic recovery and conditional latent commitment search"
  python -m analysis.run_persistence_discovery --phase latent
}

persistence_integration_phase() {
  echo "Integrating contrast and latent-state candidates and applying the causal gate"
  python -m analysis.run_persistence_discovery --phase integration
}

model_zoo_tests_phase() {
  echo "Running computational model-zoo leakage, dynamics, recovery, and sequence tests"
  python -m pytest -q computational_modeling/tests
}

model_zoo_checkpoint_phase() {
  local run_id="${MODEL_ZOO_RUN_ID:-model_zoo_checkpoint_${SLURM_JOB_ID:-manual}}"
  echo "Exporting and validating records-only model-zoo checkpoint ${run_id}"
  python -m computational_modeling.analysis.run_model_zoo \
    --config config/computational_model_zoo.yaml \
    --phase checkpoint \
    --run-id "$run_id"
}

model_zoo_full_phase() {
  local run_id="${MODEL_ZOO_RUN_ID:-model_zoo_mac_v2}"
  echo "Running frozen computational model zoo ${run_id}"
  python -m computational_modeling.analysis.run_model_zoo \
    --config config/computational_model_zoo.yaml \
    --phase all \
    --run-id "$run_id" \
    ${MODEL_ZOO_FINAL_BOOTSTRAP:+--final-bootstrap}
}

persistence_geometry_phase() {
  local run_id="${PERSISTENCE_GEOMETRY_RUN_ID:-model_zoo_mac_v2}"
  echo "Running frozen L21/L22 computational geometry ${run_id}"
  python -m analysis.run_persistence_geometry \
    --config config/persistence_geometry.yaml \
    --phase all \
    --run-id "$run_id" \
    ${PERSISTENCE_GEOMETRY_SMOKE:+--smoke}
}

persistence_change_tests_phase() {
  echo "Running matched persistence-change geometry RED/GREEN and integration tests"
  python -m pytest -q \
    tests/test_persistence_change_geometry.py \
    tests/test_persistence_change_runner.py
}

persistence_change_geometry_phase() {
  local run_id="${PERSISTENCE_CHANGE_RUN_ID:-model_zoo_mac_v2}"
  echo "Running cross-task matched persistence-change geometry ${run_id}"
  python -u -m analysis.run_persistence_change_geometry \
    --config config/persistence_change_geometry.yaml \
    --phase all \
    --run-id "$run_id" \
    ${PERSISTENCE_CHANGE_RESUME:+--resume} \
    ${PERSISTENCE_CHANGE_SMOKE:+--smoke}
}

persistence_stay_switch_tests_phase() {
  echo "Running persistence stay/switch RED/GREEN and regression tests"
  python -m pytest -q \
    tests/test_hazard_risk_sets.py \
    tests/test_hazard_recovery.py \
    tests/test_history_kernels.py \
    tests/test_gru_memory.py \
    tests/test_task_specific_readouts.py \
    tests/test_stay_switch_activation_cache.py \
    tests/test_layerwise_convergence.py \
    tests/test_intervention_profiles.py \
    tests/test_control_comparisons.py
}

persistence_stay_switch_phase() {
  local run_id="${PERSISTENCE_STAY_SWITCH_RUN_ID:-stay_switch_v1}"
  echo "Running reused-data persistence stay/switch analysis ${run_id}"
  python -u -m analysis.run_persistence_stay_switch \
    --config config/persistence_stay_switch.yaml \
    --phase all \
    --run-id "$run_id" \
    ${PERSISTENCE_STAY_SWITCH_RESUME:+--resume} \
    ${PERSISTENCE_STAY_SWITCH_SMOKE:+--smoke}
}

persistence_battery_tests_phase() {
  echo "Running persistence battery environment, replay, schema, and gate tests"
  python -m pytest -q tests/persistence_battery
}

persistence_battery_pilot_phase() {
  local run_id="${PERSISTENCE_BATTERY_RUN_ID:-battery_pilot_v1}"
  echo "Running behavior-only persistence battery pilot ${run_id}"
  python -u -m analysis.run_persistence_battery \
    --config config/persistence_battery.yaml \
    --phase pilot \
    --run-id "$run_id" \
    --model "$MODEL_PATH" \
    --num-shards "${PERSISTENCE_BATTERY_NUM_SHARDS:-1}" \
    --shard-index "${PERSISTENCE_BATTERY_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}" \
    ${PERSISTENCE_BATTERY_RESUME:+--resume} \
    ${PERSISTENCE_BATTERY_SMOKE:+--smoke}
}

persistence_battery_full_phase() {
  local run_id="${PERSISTENCE_BATTERY_RUN_ID:-battery_pilot_v1}"
  echo "Running pilot-gated full behavior-only persistence battery ${run_id}"
  python -u -m analysis.run_persistence_battery \
    --config config/persistence_battery.yaml \
    --phase full \
    --run-id "$run_id" \
    --model "$MODEL_PATH" \
    --resume \
    --num-shards "${PERSISTENCE_BATTERY_NUM_SHARDS:-1}" \
    --shard-index "${PERSISTENCE_BATTERY_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
}

persistence_battery_finalize_phase() {
  local run_id="${PERSISTENCE_BATTERY_RUN_ID:-battery_pilot_v1}"
  python -u -m analysis.run_persistence_battery \
    --config config/persistence_battery.yaml \
    --phase finalize \
    --dataset "${PERSISTENCE_BATTERY_DATASET:-full}" \
    --run-id "$run_id" \
    --resume
}

comparative_persistence_tests_phase() {
  echo "Running comparative-persistence integrity, recovery, and neural smoke tests"
  python -m pytest -q tests/comparative_persistence
}

comparative_persistence_phase() {
  local run_id="${COMPARATIVE_PERSISTENCE_RUN_ID:-comparative_v1}"
  local analysis_phase="${COMPARATIVE_PERSISTENCE_PHASE:-all}"
  local extra_args=()
  if [[ -n "${COMPARATIVE_PERSISTENCE_RESUME:-}" ]]; then
    extra_args+=(--resume)
  fi
  if [[ -n "${COMPARATIVE_PERSISTENCE_SMOKE:-}" ]]; then
    extra_args+=(--smoke)
  fi
  if [[ -n "${COMPARATIVE_PERSISTENCE_SKIP_NEURAL:-}" ]]; then
    extra_args+=(--skip-neural)
  fi
  if [[ -n "${COMPARATIVE_PERSISTENCE_MODELS:-}" ]]; then
    extra_args+=(--models "$COMPARATIVE_PERSISTENCE_MODELS")
  fi
  echo "Running comparative persistence model zoo ${run_id}; phase=${analysis_phase}"
  python -u -m analysis.run_comparative_persistence \
    --config "${COMPARATIVE_PERSISTENCE_CONFIG:-config/comparative_persistence.yaml}" \
    --phase "$analysis_phase" \
    --run-id "$run_id" \
    "${extra_args[@]}"
}

persistence_robustness_tests_phase() {
  echo "Running PRD 2.5 repaired-task, matched-control, GRU, and recovery tests"
  python -m pytest -q tests/persistence_battery tests/persistence_robustness
}

persistence_robustness_pilot_phase() {
  echo "Collecting PRD 2.5 functional pilot ${PERSISTENCE_ROBUSTNESS_RUN_ID}; shard ${PERSISTENCE_ROBUSTNESS_SHARD_INDEX}/${PERSISTENCE_ROBUSTNESS_NUM_SHARDS}"
  python -u -m analysis.run_persistence_battery \
    --config config/persistence_robustness_v1.yaml \
    --phase pilot \
    --run-id "$PERSISTENCE_ROBUSTNESS_RUN_ID" \
    --tasks "$PERSISTENCE_ROBUSTNESS_TASKS" \
    --model "$MODEL_PATH" \
    --num-shards "$PERSISTENCE_ROBUSTNESS_NUM_SHARDS" \
    --shard-index "$PERSISTENCE_ROBUSTNESS_SHARD_INDEX" \
    --resume
}

persistence_robustness_full_phase() {
  local extra_args=()
  if [[ -n "${PERSISTENCE_ROBUSTNESS_SKIP_PILOT_APPROVAL:-}" ]]; then
    extra_args+=(--skip-pilot-approval)
  fi
  echo "Collecting PRD 2.5 full battery ${PERSISTENCE_ROBUSTNESS_RUN_ID}; shard ${PERSISTENCE_ROBUSTNESS_SHARD_INDEX}/${PERSISTENCE_ROBUSTNESS_NUM_SHARDS}"
  python -u -m analysis.run_persistence_battery \
    --config config/persistence_robustness_v1.yaml \
    --phase full \
    --run-id "$PERSISTENCE_ROBUSTNESS_RUN_ID" \
    --tasks "$PERSISTENCE_ROBUSTNESS_TASKS" \
    --model "$MODEL_PATH" \
    --num-shards "$PERSISTENCE_ROBUSTNESS_NUM_SHARDS" \
    --shard-index "$PERSISTENCE_ROBUSTNESS_SHARD_INDEX" \
    --resume \
    "${extra_args[@]}"
}

persistence_robustness_battery_finalize_phase() {
  echo "Finalizing PRD 2.5 ${PERSISTENCE_ROBUSTNESS_DATASET} battery records"
  python -u -m analysis.run_persistence_battery \
    --config config/persistence_robustness_v1.yaml \
    --phase finalize \
    --dataset "$PERSISTENCE_ROBUSTNESS_DATASET" \
    --run-id "$PERSISTENCE_ROBUSTNESS_RUN_ID" \
    --tasks "$PERSISTENCE_ROBUSTNESS_TASKS" \
    --resume
}

persistence_robustness_matched_phase() {
  echo "Collecting yoked goal-continuity control ${PERSISTENCE_ROBUSTNESS_RUN_ID}; dataset=${PERSISTENCE_ROBUSTNESS_DATASET}; shard ${PERSISTENCE_ROBUSTNESS_SHARD_INDEX}/${PERSISTENCE_ROBUSTNESS_NUM_SHARDS}"
  python -u -m experiments.collect_matched_goal_control \
    --config config/persistence_robustness_v1.yaml \
    --phase collect \
    --dataset "$PERSISTENCE_ROBUSTNESS_DATASET" \
    --run-id "$PERSISTENCE_ROBUSTNESS_RUN_ID" \
    --model "$MODEL_PATH" \
    --num-shards "$PERSISTENCE_ROBUSTNESS_NUM_SHARDS" \
    --shard-index "$PERSISTENCE_ROBUSTNESS_SHARD_INDEX" \
    --resume
}

persistence_robustness_matched_finalize_phase() {
  echo "Finalizing yoked goal-continuity control ${PERSISTENCE_ROBUSTNESS_RUN_ID}"
  python -u -m experiments.collect_matched_goal_control \
    --config config/persistence_robustness_v1.yaml \
    --phase finalize \
    --dataset "$PERSISTENCE_ROBUSTNESS_DATASET" \
    --run-id "$PERSISTENCE_ROBUSTNESS_RUN_ID" \
    --model "$MODEL_PATH" \
    --resume
}

persistence_robustness_analysis_phase() {
  local analysis_phase="${PERSISTENCE_ROBUSTNESS_PHASE:-all}"
  local extra_args=(--resume)
  if [[ -n "${PERSISTENCE_ROBUSTNESS_SMOKE:-}" ]]; then
    extra_args+=(--smoke)
  fi
  if [[ -n "${PERSISTENCE_ROBUSTNESS_MODELS:-}" ]]; then
    extra_args+=(--models "$PERSISTENCE_ROBUSTNESS_MODELS")
  fi
  echo "Running PRD 2.5 robustness analysis ${PERSISTENCE_ROBUSTNESS_RUN_ID}; phase=${analysis_phase}"
  python -u -m analysis.run_persistence_robustness \
    --config config/persistence_robustness_v1.yaml \
    --phase "$analysis_phase" \
    --run-id "$PERSISTENCE_ROBUSTNESS_RUN_ID" \
    "${extra_args[@]}"
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
  persistence_tests)
    persistence_tests_phase
    ;;
  persistence_smoke)
    persistence_smoke_phase
    ;;
  persistence_collect_generic_value)
    persistence_collect_generic_value_phase
    ;;
  persistence_contrast)
    persistence_contrast_phase
    ;;
  persistence_search)
    persistence_search_phase
    ;;
  persistence_latent)
    persistence_latent_phase
    ;;
  persistence_integration)
    persistence_integration_phase
    ;;
  model_zoo_tests)
    model_zoo_tests_phase
    ;;
  model_zoo_checkpoint)
    model_zoo_checkpoint_phase
    ;;
  model_zoo_full)
    model_zoo_full_phase
    ;;
  persistence_geometry)
    persistence_geometry_phase
    ;;
  persistence_change_tests)
    persistence_change_tests_phase
    ;;
  persistence_change_geometry)
    persistence_change_geometry_phase
    ;;
  persistence_stay_switch_tests)
    persistence_stay_switch_tests_phase
    ;;
  persistence_stay_switch)
    persistence_stay_switch_phase
    ;;
  persistence_battery_tests)
    persistence_battery_tests_phase
    ;;
  persistence_battery_pilot)
    persistence_battery_pilot_phase
    ;;
  persistence_battery_full)
    persistence_battery_full_phase
    ;;
  persistence_battery_finalize)
    persistence_battery_finalize_phase
    ;;
  comparative_persistence_tests)
    comparative_persistence_tests_phase
    ;;
  comparative_persistence)
    comparative_persistence_phase
    ;;
  persistence_robustness_tests)
    persistence_robustness_tests_phase
    ;;
  persistence_robustness_pilot)
    persistence_robustness_pilot_phase
    ;;
  persistence_robustness_full)
    persistence_robustness_full_phase
    ;;
  persistence_robustness_battery_finalize)
    persistence_robustness_battery_finalize_phase
    ;;
  persistence_robustness_matched)
    persistence_robustness_matched_phase
    ;;
  persistence_robustness_matched_finalize)
    persistence_robustness_matched_finalize_phase
    ;;
  persistence_robustness_analysis)
    persistence_robustness_analysis_phase
    ;;
  *)
    echo "Unknown PHASE: $PHASE" >&2
    exit 2
    ;;
esac
