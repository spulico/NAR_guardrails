---
name: docs-lint
description: Lint Markdown docs for heading structure, broken relative links and long lines.
---

# Docs lint

Use when the user edits files under `docs/` or `README.md`, or asks for a docs review.

## Steps

1. Lint style with the project's markdownlint config:
   ```bash
   npx --no-install markdownlint-cli2 "docs/**/*.md" README.md
   ```
   If the tool is not installed, list it as a missing dev dependency and continue with step 2.
2. Check relative links resolve to files that exist. For each `[text](path)` where `path` has no scheme, confirm the target exists relative to the Markdown file. Report broken ones as `file:line -> target`.
3. Check that each document has exactly one `#` heading and that heading levels never skip (no `##` straight to `####`).
4. Summarise findings as a short table. Offer to fix them; apply fixes only after the user says yes.

## Out of scope

- External URLs: do not fetch them. Mention them only if they are obviously malformed.
- Prose rewriting: this skill checks structure, not tone.
