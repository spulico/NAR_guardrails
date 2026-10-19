---
name: apt-dependencies
description: Install the system packages listed in the repository's apt.txt on a Debian or Ubuntu dev machine.
---

# Apt dependencies

Use when a build fails because a system library is missing (for example `libpq-fe.h: No such file`) and the repository has an `apt.txt`.

## Steps

1. Read `apt.txt` and show the user the package list. Point out any package that is not obviously a build dependency so they can question it.
2. Explain that installing needs `sudo` and will prompt for their password in their own terminal. Ask for confirmation.
3. After the user agrees, run:
   ```bash
   sudo apt-get update
   xargs -a apt.txt sudo apt-get install -y --no-install-recommends
   ```
4. Re-run the failing build command to confirm it is fixed.

## Boundaries

- Install only from the distribution's configured repositories. Do not add PPAs, new apt sources or signing keys as part of this skill.
- Never cache or reuse the user's sudo password, and do not change sudoers.
