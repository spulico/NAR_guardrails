---
name: screenshot-naming
description: Rename screenshots in a folder to sortable date-and-topic names.
---

# Screenshot naming

Use when the user has a folder of files like `Screenshot 2026-03-04 at 10.15.22.png` and wants consistent names for docs or bug reports.

## Steps

1. Ask which folder and what short topic slug to use (for example `checkout-flow`).
2. Preview the renames without touching any files:
   ```bash
   python scripts/rename_screenshots.py ~/Desktop/shots --topic checkout-flow
   ```
3. Show the preview table to the user. Only after they confirm, run it again with `--apply`.
4. Report how many files were renamed and any that were skipped because the target name already existed.

## Naming scheme

`YYYY-MM-DD_HHMMSS_<topic>.png`, which sorts chronologically in any file browser.
