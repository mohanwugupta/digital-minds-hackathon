#!/bin/bash
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
RUN_SCRIPT="${PROJECT_DIR}/run_qwen35_bandit.sh"
MODEL_ZOO_RUN_ID="${MODEL_ZOO_RUN_ID:-model_zoo_checkpoint_v1}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is unavailable; run this helper on the cluster login node" >&2
  exit 1
fi
if [ ! -f "$RUN_SCRIPT" ]; then
  echo "Run this helper from the repository root" >&2
  exit 1
fi

test_submission=$(sbatch --parsable \
  --job-name=model_zoo_tests \
  --time=00:30:00 \
  --export=ALL,PHASE=model_zoo_tests \
  "$RUN_SCRIPT")
test_job="${test_submission%%;*}"

checkpoint_submission=$(sbatch --parsable \
  --job-name=model_zoo_checkpoint \
  --time=01:00:00 \
  --dependency="afterok:${test_job}" \
  --export="ALL,PHASE=model_zoo_checkpoint,MODEL_ZOO_RUN_ID=${MODEL_ZOO_RUN_ID}" \
  "$RUN_SCRIPT")
checkpoint_job="${checkpoint_submission%%;*}"

echo "Computational model-zoo checkpoint submitted"
echo "  tests:      ${test_job}"
echo "  checkpoint: ${checkpoint_job}"
echo "Review artifacts/computational_modeling/${MODEL_ZOO_RUN_ID}/implementation_checkpoint.json before submitting PHASE=model_zoo_full."
