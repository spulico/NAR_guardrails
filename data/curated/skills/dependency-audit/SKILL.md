---
name: dependency-audit
description: Report known vulnerabilities and outdated pins in Python and npm dependencies.
---

# Dependency audit

Use when the user asks whether dependencies are safe or up to date, or before a release.

## Steps

1. Detect the ecosystems present: `uv.lock` or `requirements*.txt` for Python, `package-lock.json` for npm.
2. Python: audit the exact locked versions (this queries the public advisory database, read-only). For a uv project, export the lock file first:
   ```bash
   uv export --format requirements-txt --no-hashes > /tmp/requirements.lock.txt
   uvx pip-audit --strict --desc -r /tmp/requirements.lock.txt
   ```
   For a project with only `requirements*.txt`, pass those files to `-r` directly.
3. npm:
   ```bash
   npm audit --omit=dev --json > /tmp/npm-audit.json
   ```
4. Build one table: package, installed version, advisory id, severity, first fixed version.
5. Recommend upgrades, smallest version bump first. Do not run `npm audit fix --force` or edit lock files; list the commands and let the user choose.

## Notes

- A vulnerability in a dev-only dependency is lower priority; say so rather than hiding it.
- If an advisory has no fixed version, suggest a mitigation or a replacement package and mark it as needing a human decision.
