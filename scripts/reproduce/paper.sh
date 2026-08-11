#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
cd "$REPOSITORY_ROOT"
exec uv run decaf-paper "$@"
