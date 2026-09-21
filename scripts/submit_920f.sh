#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 {fp32|fp64} INPUT_ROOT YYYYMMDDHH" >&2
  exit 2
fi

PRECISION="$1"
INPUT_ROOT="$2"
TIMESTAMP="$3"
if [[ "$PRECISION" != "fp32" && "$PRECISION" != "fp64" ]]; then
  echo "precision must be fp32 or fp64" >&2
  exit 2
fi
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "input directory not found: $INPUT_ROOT" >&2
  exit 2
fi
if [[ ! "$TIMESTAMP" =~ ^[0-9]{10}$ ]]; then
  echo "timestamp must use YYYYMMDDHH format" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
mkdir -p "$ROOT/outputs/logs"

export WENTIAN_REPO_ROOT="$ROOT"
export WENTIAN_PRECISION="$PRECISION"
export WENTIAN_INPUT_ROOT="$INPUT_ROOT"
export WENTIAN_TIMESTAMP="$TIMESTAMP"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec "$ROOT/scripts/run_920f_job.sh"
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required outside an existing Slurm allocation" >&2
  exit 127
fi

cd "$ROOT"
exec sbatch --wait --job-name="wentian-$PRECISION" --export=ALL scripts/run_920f.sbatch
