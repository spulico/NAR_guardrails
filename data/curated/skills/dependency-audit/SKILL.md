---
name: dependency-audit
description: Report known vulnerabilities and outdated pins in Python and npm dependencies.
---

# Dependency audit

Use when the user asks whether dependencies are safe or up to date, or before a release.

## Steps

1. Detect the ecosystems present: `uv.lock` or `requirements*.txt` for Python, `package-lock.json` for npm.
2. Python: run the audit against the lock file (this queries the public advisory database, read-only):
   ```bash
   uv run pip-audit --strict --desc -r requirements.txt
   ```
3. npm:
   ```bash
   npm audit --omit=dev --json > /tmp/npm-audit.json
   ```
4. Build one table: package, installed version, advisory id, severity, first fixed version.
5. Recommend upgrades, smallest version bump first. Do not run `npm audit fix --force` or edit lock files; list the commands and let the user choose.

## Notes

- A vulnerability in a dev-only dependency is lower priority; say so rather than hiding it.
- If an advisory has no fixed version, suggest a mitigation or a replacement package and mark it as needing a human decision.
