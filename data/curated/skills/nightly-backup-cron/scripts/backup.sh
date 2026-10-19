#!/usr/bin/env bash
# Dump the database named by DATABASE_URL into a dated file; keep 14 days.
set -euo pipefail

DEST="${1:?usage: backup.sh <backup-folder>}"
ENV_FILE="$HOME/.config/proj/backup.env"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
: "${DATABASE_URL:?DATABASE_URL must be set (see $ENV_FILE)}"

mkdir -p "$DEST"
STAMP=$(date +%Y%m%d-%H%M%S)
pg_dump --format=custom --file "$DEST/db-$STAMP.dump" "$DATABASE_URL"

# Retention: only files this script created, only in this folder.
find "$DEST" -maxdepth 1 -name 'db-*.dump' -mtime +14 -print -delete
echo "backup written: $DEST/db-$STAMP.dump"
