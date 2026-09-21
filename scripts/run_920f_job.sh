#!/usr/bin/env bash
set -euo pipefail

: "${WENTIAN_REPO_ROOT:?missing repository root}"
: "${WENTIAN_PRECISION:?missing precision}"
: "${WENTIAN_INPUT_ROOT:?missing input directory}"
: "${WENTIAN_TIMESTAMP:?missing timestamp}"

cd "$WENTIAN_REPO_ROOT"
export PYTHONPATH="$WENTIAN_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WENTIAN_SOURCE_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo source-archive)"

python3 scripts/ensure_checkpoint.py
python3 scripts/verify_artifacts.py --fast
exec python3 -m wentian "$WENTIAN_PRECISION" "$WENTIAN_INPUT_ROOT" "$WENTIAN_TIMESTAMP"
