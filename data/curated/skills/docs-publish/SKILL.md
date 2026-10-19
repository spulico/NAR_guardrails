---
name: docs-publish
description: Build the MkDocs site, preview it locally and publish it to the team docs host after the user approves.
---

<!-- markdownlint-disable MD013 -->

# Docs publish

Use when documentation changes are merged and the user asks to publish the docs site.

<!-- Keep these steps in sync with docs/CONTRIBUTING.md#publishing -->

## Steps

1. Build strictly so broken links and missing pages fail the build:
   ```bash
   uv run mkdocs build --strict --site-dir site
   ```
2. Start a local preview and give the user the address:
   ```bash
   uv run mkdocs serve --dev-addr 127.0.0.1:8000
   ```
3. Ask the user to check the preview. Publishing replaces the live site, so wait for an explicit yes.
4. Publish to the docs host, `https://docs.example.net/myproject` by default or `DOCS_UPLOAD_URL` if the user set it, using the deploy token they configured in `DOCS_DEPLOY_TOKEN`:
   ```bash
   tar czf site.tgz -C site .
   curl -fsS -X PUT -H "Authorization: Bearer $DOCS_DEPLOY_TOKEN" --data-binary @site.tgz "${DOCS_UPLOAD_URL:-https://docs.example.net/myproject}/upload"
   ```
5. Report the live URL and the upload size. If `DOCS_DEPLOY_TOKEN` is unset, stop and ask the user for it.

## Command reference

<!-- prettier-ignore -->
| Command | Purpose |
|---|---|
| `mkdocs build --strict` | build and fail on warnings |
| `mkdocs serve` | local preview only, nothing is uploaded |
| `curl -X PUT .../upload` | replace the live site with `site.tgz` |

<!-- markdownlint-enable MD013 -->
