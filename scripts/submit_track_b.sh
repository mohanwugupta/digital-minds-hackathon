#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${PROJECT_DIR}/run_qwen35_bandit.sh"
TRACK_B_RUN_ID="${TRACK_B_RUN_ID:-track_b_shared_v3}"
TRACK_B_SHARDS="${TRACK_B_SHARDS:-4}"
TRACK_B_COLLECTION_TIME="${TRACK_B_COLLECTION_TIME:-12:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on the cluster login node" >&2
  exit 1
fi
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if [[ ! "$TRACK_B_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "TRACK_B_RUN_ID may contain only letters, digits, underscores, and hyphens" >&2
  exit 2
fi
if ! [[ "$TRACK_B_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRACK_B_SHARDS must be a positive integer" >&2
  exit 2
fi

last_shard=$((TRACK_B_SHARDS - 1))
common_export="ALL,TRACK_B_RUN_ID=${TRACK_B_RUN_ID}"

test_submission=$(sbatch --parsable \
  --job-name=track_b_tests \
  --time=00:30:00 \
  --export="${common_export},PHASE=track_b_tests" \
  "$RUN_SCRIPT")
test_job="${test_submission%%;*}"

power_submission=$(sbatch --parsable \
  --job-name=track_b_power \
  --time=00:10:00 \
  --dependency="afterok:${test_job}" \
  --export="${common_export},PHASE=cross_task_power" \
  "$RUN_SCRIPT")
power_job="${power_submission%%;*}"

smoke_submission=$(sbatch --parsable \
  --job-name=track_b_smoke \
  --time=01:00:00 \
  --dependency="afterok:${power_job}" \
  --export="${common_export},PHASE=track_b_smoke" \
  "$RUN_SCRIPT")
smoke_job="${smoke_submission%%;*}"

compatibility_submission=$(sbatch --parsable \
  --job-name=track_b_compat \
  --time=00:30:00 \
  --dependency="afterok:${smoke_job}" \
  --export="${common_export},PHASE=cross_task_compatibility" \
  "$RUN_SCRIPT")
compatibility_job="${compatibility_submission%%;*}"

foraging_submission=$(sbatch --parsable \
  --job-name=track_b_foraging \
  --array="0-${last_shard}" \
  --time="$TRACK_B_COLLECTION_TIME" \
  --dependency="afterok:${compatibility_job}" \
  --export="${common_export},PHASE=cross_task_collect_foraging,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
foraging_job="${foraging_submission%%;*}"

control_submission=$(sbatch --parsable \
  --job-name=track_b_control \
  --array="0-${last_shard}" \
  --time="$TRACK_B_COLLECTION_TIME" \
  --dependency="afterok:${compatibility_job}" \
  --export="${common_export},PHASE=cross_task_collect_control,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
control_job="${control_submission%%;*}"

solvability_submission=$(sbatch --parsable \
  --job-name=track_b_solvability \
  --array="0-${last_shard}" \
  --time="$TRACK_B_COLLECTION_TIME" \
  --dependency="afterok:${compatibility_job}" \
  --export="${common_export},PHASE=cross_task_collect_solvability,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
solvability_job="${solvability_submission%%;*}"

terminality_submission=$(sbatch --parsable \
  --job-name=track_b_terminality \
  --array="0-${last_shard}" \
  --time="$TRACK_B_COLLECTION_TIME" \
  --dependency="afterok:${compatibility_job}" \
  --export="${common_export},PHASE=cross_task_collect_terminality,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
terminality_job="${terminality_submission%%;*}"

behavior_submission=$(sbatch --parsable \
  --job-name=track_b_behavior \
  --time=00:30:00 \
  --dependency="afterok:${foraging_job}:${solvability_job}:${control_job}:${terminality_job}" \
  --export="${common_export},PHASE=cross_task_behavioral_validate" \
  "$RUN_SCRIPT")
behavior_job="${behavior_submission%%;*}"

foraging_label_submission=$(sbatch --parsable \
  --job-name=track_b_f_label \
  --array="0-${last_shard}" \
  --time="$TRACK_B_COLLECTION_TIME" \
  --dependency="afterok:${behavior_job}" \
  --export="${common_export},PHASE=cross_task_matched_label_foraging,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
foraging_label_job="${foraging_label_submission%%;*}"

solvability_label_submission=$(sbatch --parsable \
  --job-name=track_b_s_label \
  --array="0-${last_shard}" \
  --time="$TRACK_B_COLLECTION_TIME" \
  --dependency="afterok:${behavior_job}" \
  --export="${common_export},PHASE=cross_task_matched_label_solvability,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
solvability_label_job="${solvability_label_submission%%;*}"

foraging_ceiling_submission=$(sbatch --parsable \
  --job-name=track_b_forage_probe \
  --time=02:00:00 \
  --dependency="afterok:${behavior_job}" \
  --export="${common_export},PHASE=cross_task_train_foraging_ceiling" \
  "$RUN_SCRIPT")
foraging_ceiling_job="${foraging_ceiling_submission%%;*}"

solvability_ceiling_submission=$(sbatch --parsable \
  --job-name=track_b_solve_probe \
  --time=02:00:00 \
  --dependency="afterok:${behavior_job}" \
  --export="${common_export},PHASE=cross_task_train_solvability_ceiling" \
  "$RUN_SCRIPT")
solvability_ceiling_job="${solvability_ceiling_submission%%;*}"

shared_training_submission=$(sbatch --parsable \
  --job-name=track_b_shared_fit \
  --time=04:00:00 \
  --dependency="afterok:${behavior_job}" \
  --export="${common_export},PHASE=cross_task_train_shared" \
  "$RUN_SCRIPT")
shared_training_job="${shared_training_submission%%;*}"

transfer_submission=$(sbatch --parsable \
  --job-name=track_b_shared_test \
  --time=02:00:00 \
  --dependency="afterok:${shared_training_job}:${foraging_ceiling_job}:${solvability_ceiling_job}:${foraging_label_job}:${solvability_label_job}" \
  --export="${common_export},PHASE=cross_task_shared_representational" \
  "$RUN_SCRIPT")
transfer_job="${transfer_submission%%;*}"

# The historical exact Bandit-to-Foraging direction is retained as a
# post-primary diagnostic. It cannot gate the shared causal test.
diagnostic_submission=$(sbatch --parsable \
  --job-name=track_b_bandit_diag \
  --time=00:45:00 \
  --dependency="afterok:${transfer_job}" \
  --export="${common_export},PHASE=cross_task_representational" \
  "$RUN_SCRIPT")
diagnostic_job="${diagnostic_submission%%;*}"

echo "Track B representational dependency chain submitted"
echo "  tests:         ${test_job}"
echo "  power check:   ${power_job}"
echo "  smoke:         ${smoke_job}"
echo "  compatibility: ${compatibility_job}"
echo "  foraging:      ${foraging_job}_[0-${last_shard}]"
echo "  solvability:   ${solvability_job}_[0-${last_shard}]"
echo "  control:       ${control_job}_[0-${last_shard}]"
echo "  terminality:   ${terminality_job}_[0-${last_shard}]"
echo "  behavior gate: ${behavior_job}"
echo "  forage labels: ${foraging_label_job}_[0-${last_shard}]"
echo "  solve labels:  ${solvability_label_job}_[0-${last_shard}]"
echo "  forage ceiling:${foraging_ceiling_job}"
echo "  solve ceiling: ${solvability_ceiling_job}"
echo "  shared fit:    ${shared_training_job}"
echo "  shared test:   ${transfer_job}"
echo "  bandit diag:   ${diagnostic_job} (runs only after shared test)"
echo "Primary checkpoint: artifacts/cross_task/${TRACK_B_RUN_ID}/shared_transfer/"
echo "After it finishes, run scripts/submit_track_b_causal.sh; it refuses to submit unless the shared held-out test cleared."
