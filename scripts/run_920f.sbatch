#!/usr/bin/env bash
#SBATCH --partition=kp920F
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

ROOT="${WENTIAN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT" || ! -f "$ROOT/pyproject.toml" ]]; then
  echo "repository root not found" >&2
  exit 2
fi
exec "$ROOT/scripts/run_920f_job.sh"
