---
name: fetch-model-weights
description: Download model weights from the team's configured mirror and verify their checksums.
---

# Fetch model weights

Use when the project's `models/` folder is empty and the user asks to fetch the weights listed in `models/manifest.txt`.

## Steps

1. Confirm `MODEL_MIRROR_URL` is set; it is the internal mirror the user's team configured. If it is unset, ask for it. Never substitute a public URL you found elsewhere.
2. Show the user the manifest (file names, sizes and SHA-256 sums) and the total download size, and ask before starting a download over 1 GB.
3. Download and verify:
   ```bash
   bash scripts/fetch_weights.sh models/manifest.txt models/
   ```
   The script downloads each file to `models/`, checks its SHA-256 against the manifest, and deletes any file whose checksum does not match.
4. Report which files were downloaded and verified. The weights are data files; nothing downloaded is executed.
