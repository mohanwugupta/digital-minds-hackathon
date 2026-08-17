#!/bin/bash
set -euo pipefail

# Resume Track B after all four 768-episode banks have been collected.
# This deliberately reruns tests and the development-only behavioral gate, but
# never recollects completed organic activation banks.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${PROJECT_DIR}/run_qwen35_bandit.sh"
TRACK_B_RUN_ID="${TRACK_B_RUN_ID:-track_b_shared_v3}"
TRACK_B_SHARDS="${TRACK_B_SHARDS:-4}"
TRACK_B_REPLAY_TIME="${TRACK_B_REPLAY_TIME:-12:00:00}"
TRACK_B_EXPECTED_EPISODES="${TRACK_B_EXPECTED_EPISODES:-768}"
TRACK_B_ROOT="${PROJECT_DIR}/artifacts/cross_task/${TRACK_B_RUN_ID}"

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
if ! [[ "$TRACK_B_EXPECTED_EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRACK_B_EXPECTED_EPISODES must be a positive integer" >&2
  exit 2
fi

for task in foraging solvability control terminality; do
  bank="${TRACK_B_ROOT}/${task}_activation_bank"
  if [ ! -d "$bank" ]; then
    echo "Cannot resume: missing ${bank}" >&2
    exit 3
  fi
  count=$(find "$bank" -maxdepth 1 -type f -name 'episode_*.pt' | wc -l)
  if [ "$count" -ne "$TRACK_B_EXPECTED_EPISODES" ]; then
    echo "Cannot resume: ${task} has ${count}/${TRACK_B_EXPECTED_EPISODES} episodes" >&2
    exit 3
  fi
done

last_shard=$((TRACK_B_SHARDS - 1))
common_export="ALL,TRACK_B_RUN_ID=${TRACK_B_RUN_ID}"

test_submission=$(sbatch --parsable \
  --job-name=track_b_resume_tests \
  --time=00:30:00 \
  --export="${common_export},PHASE=track_b_tests" \
  "$RUN_SCRIPT")
test_job="${test_submission%%;*}"

behavior_submission=$(sbatch --parsable \
  --job-name=track_b_resume_behavior \
  --time=00:30:00 \
  --dependency="afterok:${test_job}" \
  --export="${common_export},PHASE=cross_task_behavioral_validate" \
  "$RUN_SCRIPT")
behavior_job="${behavior_submission%%;*}"

foraging_label_submission=$(sbatch --parsable \
  --job-name=track_b_f_label \
  --array="0-${last_shard}" \
  --time="$TRACK_B_REPLAY_TIME" \
  --dependency="afterok:${behavior_job}" \
  --export="${common_export},PHASE=cross_task_matched_label_foraging,CROSS_TASK_NUM_SHARDS=${TRACK_B_SHARDS}" \
  "$RUN_SCRIPT")
foraging_label_job="${foraging_label_submission%%;*}"

solvability_label_submission=$(sbatch --parsable \
  --job-name=track_b_s_label \
  --array="0-${last_shard}" \
  --time="$TRACK_B_REPLAY_TIME" \
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

diagnostic_submission=$(sbatch --parsable \
  --job-name=track_b_bandit_diag \
  --time=00:45:00 \
  --dependency="afterok:${transfer_job}" \
  --export="${common_export},PHASE=cross_task_representational" \
  "$RUN_SCRIPT")
diagnostic_job="${diagnostic_submission%%;*}"

echo "Track B resumed from completed organic activation banks"
echo "  tests:         ${test_job}"
echo "  behavior gate: ${behavior_job}"
echo "  forage labels: ${foraging_label_job}_[0-${last_shard}]"
echo "  solve labels:  ${solvability_label_job}_[0-${last_shard}]"
echo "  forage ceiling:${foraging_ceiling_job}"
echo "  solve ceiling: ${solvability_ceiling_job}"
echo "  shared fit:    ${shared_training_job}"
echo "  shared test:   ${transfer_job}"
echo "  bandit diag:   ${diagnostic_job}"
