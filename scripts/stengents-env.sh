#!/usr/bin/env bash
# Resolve the *stable* connection to the development-time model on `gym` and
# print it as shell `export` lines. Load it into your shell with:
#
#     eval "$(scripts/stengents-env.sh)"
#
# This sets only the connection (base URL + API key), which never changes. The
# model is a per-run choice — do not export it here. Pick it per command with
# `stengents run <fixture-id> --model <name>` or an inline STENGENTS_MODEL_NAME.
#
# Run it with no `eval` to inspect exactly what it would set.
#
# Override the defaults from your environment before calling if `gym`'s endpoint
# or key ever differ:
#     STENGENTS_MODEL_BASE_URL=http://127.0.0.1:11500 eval "$(scripts/stengents-env.sh)"
set -euo pipefail

base_url="${STENGENTS_MODEL_BASE_URL:-http://127.0.0.1:11434}"
api_key="${STENGENTS_MODEL_API_KEY:-local}"

printf 'export STENGENTS_MODEL_BASE_URL=%q\n' "$base_url"
printf 'export STENGENTS_MODEL_API_KEY=%q\n' "$api_key"
