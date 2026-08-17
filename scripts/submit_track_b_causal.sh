#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${PROJECT_DIR}/run_qwen35_bandit.sh"
TRACK_B_RUN_ID="${TRACK_B_RUN_ID:-track_b_shared_v3}"
TRACK_B_CAUSAL_SHARDS="${TRACK_B_CAUSAL_SHARDS:-8}"
TRACK_B_CAUSAL_TIME="${TRACK_B_CAUSAL_TIME:-12:00:00}"
SUMMARY="${PROJECT_DIR}/artifacts/cross_task/${TRACK_B_RUN_ID}/shared_transfer/shared_persistence_transfer_summary.json"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on the cluster login node" >&2
  exit 1
fi
if [ ! -f "$RUN_SCRIPT" ] || [ ! -f "$SUMMARY" ]; then
  echo "Run from the repository root after the Track B representational job finishes" >&2
  exit 1
fi
if [[ ! "$TRACK_B_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "TRACK_B_RUN_ID may contain only letters, digits, underscores, and hyphens" >&2
  exit 2
fi
if ! [[ "$TRACK_B_CAUSAL_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRACK_B_CAUSAL_SHARDS must be a positive integer" >&2
  exit 2
fi

classification=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["classification"])' "$SUMMARY")
case "$classification" in
  strong_shared_transfer|partial_shared_transfer) ;;
  *)
    echo "Causal Track B not submitted: representational classification is ${classification}" >&2
    exit 3
    ;;
esac

last_shard=$((TRACK_B_CAUSAL_SHARDS - 1))
common_export="ALL,TRACK_B_RUN_ID=${TRACK_B_RUN_ID}"

calibration_submission=$(sbatch --parsable \
  --job-name=track_b_calibrate \
  --time=00:30:00 \
  --export="${common_export},PHASE=cross_task_causal_calibrate" \
  "$RUN_SCRIPT")
calibration_job="${calibration_submission%%;*}"

solvability_submission=$(sbatch --parsable \
  --job-name=track_b_causal_s \
  --array="0-${last_shard}" \
  --time="$TRACK_B_CAUSAL_TIME" \
  --dependency="afterok:${calibration_job}" \
  --export="${common_export},PHASE=cross_task_causal_solvability,CROSS_TASK_NUM_SHARDS=${TRACK_B_CAUSAL_SHARDS}" \
  "$RUN_SCRIPT")
solvability_job="${solvability_submission%%;*}"

control_submission=$(sbatch --parsable \
  --job-name=track_b_causal_c \
  --array="0-${last_shard}" \
  --time="$TRACK_B_CAUSAL_TIME" \
  --dependency="afterok:${calibration_job}" \
  --export="${common_export},PHASE=cross_task_causal_control,CROSS_TASK_NUM_SHARDS=${TRACK_B_CAUSAL_SHARDS}" \
  "$RUN_SCRIPT")
control_job="${control_submission%%;*}"

terminality_submission=$(sbatch --parsable \
  --job-name=track_b_causal_t \
  --array="0-${last_shard}" \
  --time="$TRACK_B_CAUSAL_TIME" \
  --dependency="afterok:${calibration_job}" \
  --export="${common_export},PHASE=cross_task_causal_terminality,CROSS_TASK_NUM_SHARDS=${TRACK_B_CAUSAL_SHARDS}" \
  "$RUN_SCRIPT")
terminality_job="${terminality_submission%%;*}"

analysis_submission=$(sbatch --parsable \
  --job-name=track_b_causal_a \
  --time=00:45:00 \
  --dependency="afterok:${solvability_job}:${control_job}:${terminality_job}" \
  --export="${common_export},PHASE=cross_task_causal_analyze" \
  "$RUN_SCRIPT")
analysis_job="${analysis_submission%%;*}"

echo "Track B causal chain submitted after ${classification} representational clearance"
echo "  calibration:      ${calibration_job}"
echo "  solvability replay:${solvability_job}_[0-${last_shard}]"
echo "  control replay:   ${control_job}_[0-${last_shard}]"
echo "  terminality replay:${terminality_job}_[0-${last_shard}]"
echo "  causal analysis:  ${analysis_job}"
echo "Final outputs: artifacts/cross_task/${TRACK_B_RUN_ID}/causal/publication/"
