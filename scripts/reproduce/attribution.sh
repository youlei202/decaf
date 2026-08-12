#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$REPOSITORY_ROOT"
if [[ "${DECAF_B200_VERIFY:-0}" == "1" ]]; then
  GPU_PYTHON="${DECAF_GPU_PYTHON:-python}"
  export PYTHONPATH="$REPOSITORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$GPU_PYTHON" -m decaf.experiments.attribution.cli "$@"
fi
exec uv run python -m decaf.experiments.attribution.cli "$@"
