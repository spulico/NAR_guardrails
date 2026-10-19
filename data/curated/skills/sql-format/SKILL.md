---
name: sql-format
description: Format SQL files and migrations with sqlfluff using the project's dialect.
---

# SQL format

Use when the user asks to format SQL, or when a migration in a pull request has inconsistent style.

## Steps

1. Find the dialect in `.sqlfluff`. If there is no config, ask the user (postgres, bigquery, snowflake, mysql) instead of guessing.
2. Lint first so the user sees what will change:
   ```bash
   uv run sqlfluff lint migrations/20261012_add_orders.sql --dialect postgres
   ```
3. Apply fixes:
   ```bash
   uv run sqlfluff fix migrations/20261012_add_orders.sql --dialect postgres
   ```
4. Re-read the file and confirm that only whitespace, keyword case and alias style changed. If sqlfluff altered logic (for example removed a parenthesis), revert that file and report it.

## Notes

- Do not format migrations that have already been applied in production unless the user insists; their checksums may be tracked.
- Keep keywords uppercase and identifiers lowercase, which is the project default.
