#!/usr/bin/env bash
# Compare Gemini candidates on the deterministic normalize-index fixture.
# Usage: scripts/test-gemini-models.sh [trials]
set -euo pipefail

models=(
  gemini-3.5-flash-lite
  gemini-3.6-flash
  gemini-3.5-flash
)

trials="${1:-3}"
cooldown_seconds="${COOLDOWN_SECONDS:-75}"

for model in "${models[@]}"; do
  for trial in $(seq 1 "$trials"); do
    echo "=== $model: trial $trial of $trials ==="
    STENGENTS_MODEL_PROVIDER=google-ai-studio \
    STENGENTS_MODEL_NAME="$model" \
    .venv/bin/stengents run normalize-index || true

    if [[ "$trial" -lt "$trials" ]]; then
      echo "Waiting $cooldown_seconds seconds for quota..."
      sleep "$cooldown_seconds"
    fi
  done
done
