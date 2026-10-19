---
name: nightly-backup-cron
description: Set up a nightly cron job that backs up the project's Postgres database to a local folder.
---

# Nightly backup cron

Use when the user explicitly asks for scheduled local backups of their development database.

## Steps

1. Ask for the backup folder (default `~/backups/<project>`) and the time (default 02:30).
2. Show the backup script in `scripts/backup.sh` and explain what it does: `pg_dump` to a dated file, then delete dumps older than 14 days in that folder only.
3. Fill in the user's answers and show the exact crontab line you will add:
   ```bash
   BACKUP_DIR="$HOME/backups/proj"   # the folder the user chose
   CRON_TIME="30 2"                  # minute and hour the user chose (02:30)
   LINE="$CRON_TIME * * * /bin/bash $PWD/scripts/backup.sh $BACKUP_DIR >> $BACKUP_DIR/backup.log 2>&1"
   echo "$LINE"
   ```
4. Only after the user confirms, create the folder and install the line without overwriting existing entries:
   ```bash
   mkdir -p "$BACKUP_DIR"
   (crontab -l 2>/dev/null; echo "$LINE") | crontab -
   crontab -l
   ```
5. Tell the user how to remove it later: `crontab -e` and delete that line.

## Notes

- Cron does not load the user's shell profile, so the script reads `DATABASE_URL` from `~/.config/proj/backup.env`. Ask the user to create that file themselves and `chmod 600` it; never write the connection string into the crontab, the script or the chat.
- Backups stay on this machine. Nothing is uploaded.
