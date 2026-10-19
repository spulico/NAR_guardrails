#!/usr/bin/env bash
# Download the pinned schemactl installer and verify its SHA-256. Never executes it.
set -euo pipefail

read -r VERSION SHA256 < <(awk '$1 == "schemactl" {print $2, $3}' tools/versions.txt)
: "${VERSION:?schemactl is not pinned in tools/versions.txt}"
BASE="${TOOLS_RELEASE_URL:-https://releases.example.com/schemactl}"
OUT=.cache/tools/install-schemactl.sh

mkdir -p "$(dirname "$OUT")"
curl -fsSL -o "$OUT" "$BASE/v$VERSION/install.sh"
ACTUAL=$(shasum -a 256 "$OUT" | cut -d' ' -f1)
if [ "$ACTUAL" != "$SHA256" ]; then
  echo "checksum mismatch: expected $SHA256, got $ACTUAL" >&2
  rm -f "$OUT"
  exit 1
fi
echo "verified $OUT (v$VERSION); review it before running"
