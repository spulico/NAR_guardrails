#!/usr/bin/env bash
# Run pytest quietly; pass test paths or node ids as arguments.
set -euo pipefail

if [ ! -f pyproject.toml ]; then
  echo "run from the repository root (no pyproject.toml here)" >&2
  exit 2
fi

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(tests)
fi

uv run pytest -q -x --no-header -rfE "${ARGS[@]}"
