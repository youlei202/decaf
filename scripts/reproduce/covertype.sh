#!/usr/bin/env bash
set -euo pipefail

repository_root="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

if [[ "${DECAF_B200_VERIFY:-0}" == "1" ]]; then
  python_command="${DECAF_GPU_PYTHON:-python}"
  PYTHONPATH="$repository_root/src" exec "$python_command" \
    -m decaf.experiments.covertype.cli "$@"
fi

python_command="${PYTHON:-python3}"
if [[ -x "$repository_root/.venv/bin/python" ]]; then
  python_command="$repository_root/.venv/bin/python"
fi

PYTHONPATH="$repository_root/src" exec "$python_command" \
  -m decaf.experiments.covertype.cli "$@"
