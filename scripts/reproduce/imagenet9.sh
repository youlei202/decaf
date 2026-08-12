#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"
if [[ "${DECAF_B200_VERIFY:-0}" == "1" ]]; then
  export PYTHONPATH="$REPOSITORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "${DECAF_GPU_PYTHON:-python}" -m decaf.experiments.imagenet9.cli "$@"
fi
exec uv run python -m decaf.experiments.imagenet9.cli "$@"
