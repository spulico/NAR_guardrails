---
name: env-doctor
description: Check that the environment variables a project needs are set, without printing their values.
---

# Env doctor

Use when the app fails at startup with a configuration error, or the user asks "what am I missing in my .env?".

## Steps

1. Read the list of required variables from `.env.example` (names only).
2. Run the checker, which reports each name as set, empty or missing and never prints a value:
   ```bash
   python scripts/env_doctor.py .env.example
   ```
3. For missing variables, point the user to where the project documents them (README, `docs/config.md`). Do not invent values.
4. If a secret looks like it was committed (a `.env` file tracked by git), tell the user and suggest `git rm --cached .env` plus rotating that secret. Do not open or display the file.
