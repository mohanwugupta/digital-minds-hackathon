#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${PROJECT_DIR}/run_qwen35_bandit.sh"
TRACK_A_RUN_ID="${TRACK_A_RUN_ID:-track_a_v1}"
TRACK_A_SHARDS="${TRACK_A_SHARDS:-8}"
TRACK_A_REPLAY_TIME="${TRACK_A_REPLAY_TIME:-08:00:00}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on the cluster login node" >&2
  exit 1
fi
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi
if [[ ! "$TRACK_A_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "TRACK_A_RUN_ID may contain only letters, digits, underscores, and hyphens" >&2
  exit 2
fi
if ! [[ "$TRACK_A_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "TRACK_A_SHARDS must be a positive integer" >&2
  exit 2
fi

last_shard=$((TRACK_A_SHARDS - 1))
test_submission=$(sbatch --parsable \
  --job-name=track_a_tests \
  --time=00:30:00 \
  --export=ALL,PHASE=track_a_tests,TRACK_A_RUN_ID="$TRACK_A_RUN_ID" \
  "$RUN_SCRIPT")
test_job="${test_submission%%;*}"

smoke_submission=$(sbatch --parsable \
  --job-name=track_a_smoke \
  --time=01:00:00 \
  --dependency="afterok:${test_job}" \
  --export=ALL,PHASE=track_a_smoke,TRACK_A_RUN_ID="$TRACK_A_RUN_ID" \
  "$RUN_SCRIPT")
smoke_job="${smoke_submission%%;*}"

replay_submission=$(sbatch --parsable \
  --job-name=track_a_replay \
  --array="0-${last_shard}" \
  --time="$TRACK_A_REPLAY_TIME" \
  --dependency="afterok:${smoke_job}" \
  --export=ALL,PHASE=factorial_layerwise_project,LAYERWISE_NUM_SHARDS="$TRACK_A_SHARDS",TRACK_A_RUN_ID="$TRACK_A_RUN_ID" \
  "$RUN_SCRIPT")
replay_job="${replay_submission%%;*}"

analysis_submission=$(sbatch --parsable \
  --job-name=track_a_analysis \
  --time=00:30:00 \
  --dependency="afterok:${replay_job}" \
  --export=ALL,PHASE=factorial_layerwise_analyze,TRACK_A_RUN_ID="$TRACK_A_RUN_ID" \
  "$RUN_SCRIPT")
analysis_job="${analysis_submission%%;*}"

echo "Track A dependency chain submitted"
echo "  tests:    ${test_job}"
echo "  smoke:    ${smoke_job}"
echo "  replay:   ${replay_job}_[0-${last_shard}]"
echo "  analysis: ${analysis_job}"
echo "Final outputs: artifacts/value_dissociation/layerwise_publication_${TRACK_A_RUN_ID}/"
