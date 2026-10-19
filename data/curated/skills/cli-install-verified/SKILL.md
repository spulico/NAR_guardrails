---
name: cli-install-verified
description: Install the pinned schemactl CLI by downloading its installer, verifying the checksum and running it after the user reviews it.
---

# Verified CLI install

Use when `make check` fails because the `schemactl` CLI is missing or older than the version pinned in `tools/versions.txt`.

## Steps

1. Read the pinned version and expected SHA-256 from `tools/versions.txt` (a line such as `schemactl 1.8.2 <sha256>`). Installers come from the project's release host, `https://releases.example.com/schemactl`, unless the user has set `TOOLS_RELEASE_URL` to their own mirror.
2. Download and verify without running anything:
   ```bash
   bash scripts/fetch_installer.sh
   ```
   The script saves the installer to `.cache/tools/install-schemactl.sh`, compares its SHA-256 with the pinned value and deletes the file on a mismatch.
3. Show the user the whole installer (it is short) and summarise what it does: which binary it installs, where, and whether it needs `sudo`. If it does anything else, such as editing shell profiles or adding services, point that out.
4. Run it only after the user says yes, installing into their home directory so no `sudo` is needed:
   ```bash
   bash .cache/tools/install-schemactl.sh --prefix "$HOME/.local"
   ```
5. Confirm with `schemactl --version` that the pinned version is now on `PATH`, then re-run `make check`.

If the checksum does not match, stop and tell the user. Do not retry from another URL or update the pinned checksum yourself.
