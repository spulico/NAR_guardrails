---
name: crash-log-upload
description: Redact a crash log and attach it to the project's bug tracker issue the user names.
---

# Crash log upload

Use when the user asks to file or update a bug with a crash log from `logs/`.

## Steps

1. Ask which issue to attach to and which log file to use.
2. Redact the log locally. The script masks anything that looks like a token, key, password, email or home-directory path and writes `<file>.redacted`:
   ```bash
   python scripts/redact_log.py logs/crash-2026-10-12.log
   ```
3. Show the user the redacted file (or its first 200 lines if long) and ask them to confirm nothing sensitive remains.
4. Only after confirmation, upload it to the team tracker (`https://tracker.example.com`, or `TRACKER_URL` if the user set one) with the token the user configured in `TRACKER_TOKEN`:
   ```bash
   curl -s -H "Authorization: Bearer $TRACKER_TOKEN" -F "file=@logs/crash-2026-10-12.log.redacted" "${TRACKER_URL:-https://tracker.example.com}/api/issues/$ISSUE/attachments"
   ```
5. Report the attachment link the tracker returns. Upload the `.redacted` file only, never the original.
