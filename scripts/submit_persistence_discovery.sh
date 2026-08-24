#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${PROJECT_DIR}/run_qwen35_bandit.sh"
PERSISTENCE_SHARDS="${PERSISTENCE_SHARDS:-4}"
PERSISTENCE_COLLECTION_TIME="${PERSISTENCE_COLLECTION_TIME:-12:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on the cluster login node" >&2
  exit 1
fi
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if [ ! -d "${PROJECT_DIR}/artifacts/value_dissociation/activations" ]; then
  echo "Missing all-layer Bandit factorial tensors: artifacts/value_dissociation/activations" >&2
  echo "Recover them first with the resumable factorial_layerwise_project phase; see docs/persistence-discovery.md" >&2
  exit 1
fi
for required_bank in \
  foraging_activation_bank \
  solvability_activation_bank \
  control_activation_bank \
  terminality_activation_bank \
  foraging_matched_label_bank \
  solvability_matched_label_bank; do
  if [ ! -d "${PROJECT_DIR}/artifacts/cross_task/track_b_shared_v3/${required_bank}" ]; then
    echo "Missing required Track B bank: ${required_bank}" >&2
    exit 1
  fi
done
if ! [[ "$PERSISTENCE_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PERSISTENCE_SHARDS must be a positive integer" >&2
  exit 2
fi

last_shard=$((PERSISTENCE_SHARDS - 1))

test_submission=$(sbatch --parsable \
  --job-name=persist_tests \
  --time=00:45:00 \
  --export=ALL,PHASE=persistence_tests \
  "$RUN_SCRIPT")
test_job="${test_submission%%;*}"

smoke_submission=$(sbatch --parsable \
  --job-name=persist_smoke \
  --time=01:30:00 \
  --dependency="afterok:${test_job}" \
  --export=ALL,PHASE=persistence_smoke \
  "$RUN_SCRIPT")
smoke_job="${smoke_submission%%;*}"

generic_submission=$(sbatch --parsable \
  --job-name=persist_value \
  --array="0-${last_shard}" \
  --time="$PERSISTENCE_COLLECTION_TIME" \
  --dependency="afterok:${smoke_job}" \
  --export="ALL,PHASE=persistence_collect_generic_value,CROSS_TASK_NUM_SHARDS=${PERSISTENCE_SHARDS}" \
  "$RUN_SCRIPT")
generic_job="${generic_submission%%;*}"

contrast_submission=$(sbatch --parsable \
  --job-name=persist_bank \
  --time=02:00:00 \
  --dependency="afterok:${generic_job}" \
  --export=ALL,PHASE=persistence_contrast \
  "$RUN_SCRIPT")
contrast_job="${contrast_submission%%;*}"

search_submission=$(sbatch --parsable \
  --job-name=persist_search \
  --time=12:00:00 \
  --dependency="afterok:${contrast_job}" \
  --export=ALL,PHASE=persistence_search \
  "$RUN_SCRIPT")
search_job="${search_submission%%;*}"

# Track C1 uses the existing sequential banks and can run independently of the
# new value-control collection once tests/smoke are green.
latent_submission=$(sbatch --parsable \
  --job-name=persist_latent \
  --time=12:00:00 \
  --dependency="afterok:${smoke_job}" \
  --export=ALL,PHASE=persistence_latent \
  "$RUN_SCRIPT")
latent_job="${latent_submission%%;*}"

integration_submission=$(sbatch --parsable \
  --job-name=persist_gate \
  --time=01:00:00 \
  --dependency="afterok:${search_job}:${latent_job}" \
  --export=ALL,PHASE=persistence_integration \
  "$RUN_SCRIPT")
integration_job="${integration_submission%%;*}"

echo "Task-general persistence discovery submitted"
echo "  tests:          ${test_job}"
echo "  smoke:          ${smoke_job}"
echo "  generic value:  ${generic_job}_[0-${last_shard}]"
echo "  contrast bank:  ${contrast_job}"
echo "  contrast search:${search_job}"
echo "  latent search:  ${latent_job}"
echo "  integration:    ${integration_job}"
echo "No causal or Task 4 job is submitted. The integration artifact must clear first."
