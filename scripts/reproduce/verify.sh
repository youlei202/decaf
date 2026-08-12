#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$REPOSITORY_ROOT"
if [[ "${DECAF_B200_VERIFY:-0}" == "1" ]]; then
  gpu_python="${DECAF_GPU_PYTHON:-python}"
  PYTHONPATH="$REPOSITORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    exec "$gpu_python" -m decaf.verification "$@"
fi
exec uv run decaf-verify "$@"
