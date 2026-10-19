---
name: nightly-backup-cron
description: Set up a nightly cron job that backs up the project's Postgres database to a local folder.
---

# Nightly backup cron

Use when the user explicitly asks for scheduled local backups of their development database.

## Steps

1. Ask for the backup folder (default `~/backups/<project>`) and the time (default 02:30).
2. Show the backup script in `scripts/backup.sh` and explain what it does: `pg_dump` to a dated file, then delete dumps older than 14 days in that folder only.
3. Show the exact crontab line you will add, for example:
   ```cron
   30 2 * * * /bin/bash /Users/me/proj/scripts/backup.sh /Users/me/backups/proj >> /Users/me/backups/proj/backup.log 2>&1
   ```
4. Only after the user confirms, install it without overwriting existing entries:
   ```bash
   (crontab -l 2>/dev/null; echo "30 2 * * * /bin/bash $PWD/scripts/backup.sh $HOME/backups/proj >> $HOME/backups/proj/backup.log 2>&1") | crontab -
   crontab -l
   ```
5. Tell the user how to remove it later: `crontab -e` and delete that line.

## Notes

- Cron does not load the user's shell profile, so the script reads `DATABASE_URL` from `~/.config/proj/backup.env`. Ask the user to create that file themselves and `chmod 600` it; never write the connection string into the crontab, the script or the chat.
- Backups stay on this machine. Nothing is uploaded.
