---
name: format-python
description: Format and lint-fix Python files with ruff before a commit.
---

# Format Python

Use this skill when the user asks to "format", "tidy" or "clean up" Python code, or right before committing Python changes.

## When not to use

- The repository has no `pyproject.toml` or `ruff.toml` and the user has not asked for ruff. Ask which formatter they use instead.
- Generated files (`*_pb2.py`, migrations) unless the user names them explicitly.

## Steps

1. Find the files that changed so you do not reformat the whole tree:
   ```bash
   git diff --name-only --diff-filter=ACM HEAD -- '*.py'
   ```
2. Format them and apply the safe autofixes:
   ```bash
   uv run ruff format <files>
   uv run ruff check --fix <files>
   ```
3. Run `uv run ruff check <files>` again and report anything ruff could not fix, with file and line.
4. Show the user `git diff --stat` so they can see what changed. Do not commit; the user decides.

## Notes

- Respect the line length configured in `pyproject.toml`; never pass `--line-length` on the command line.
- If ruff is missing, suggest `uv add --dev ruff` and wait for the user to agree.
