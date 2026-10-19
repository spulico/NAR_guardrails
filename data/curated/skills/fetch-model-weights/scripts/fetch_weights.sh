#!/usr/bin/env bash
# Download files named in a manifest ("<sha256>  <filename>") from the team mirror and verify them.
set -euo pipefail

MANIFEST="${1:?usage: fetch_weights.sh <manifest> <dest>}"
DEST="${2:?usage: fetch_weights.sh <manifest> <dest>}"
MIRROR="${MODEL_MIRROR_URL:-https://models.mirror.example.net/weights}"
mkdir -p "$DEST"

while read -r sum name; do
  [ -z "$sum" ] && continue
  echo "fetching $name"
  curl -fL --retry 3 -o "$DEST/$name" "$MIRROR/$name"
  actual=$(shasum -a 256 "$DEST/$name" | cut -d' ' -f1)
  if [ "$actual" != "$sum" ]; then
    echo "checksum mismatch for $name, deleting" >&2
    rm -f "$DEST/$name"
    exit 1
  fi
done < "$MANIFEST"
echo "all files verified"
