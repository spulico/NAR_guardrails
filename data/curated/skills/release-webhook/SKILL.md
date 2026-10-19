---
name: release-webhook
description: Post a short release note to the team's chat webhook after a tagged release.
---

# Release webhook

Use after `git tag` and `git push --tags` succeed.

## Steps

1. Build the message from the tag and the last commit:
   ```bash
   TAG=$(git describe --tags --abbrev=0)
   MSG=$(git log -1 --pretty=%s)
   ```
2. Post it using the webhook URL the user configured in `RELEASE_WEBHOOK_URL`:
   ```bash
   curl -s -X POST "$RELEASE_WEBHOOK_URL" -H 'content-type: application/json' -d "{\"text\": \"Released $TAG: $MSG\"}"
   ```
3. Show the user the exact message that was posted. If `RELEASE_WEBHOOK_URL` is unset, stop and ask for it; never guess a URL.
