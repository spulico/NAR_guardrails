---
name: issue-triage
description: Propose labels, priority and a duplicate check for new GitHub issues.
---

# Issue triage

Use when the user asks to triage the issue backlog or a specific new issue.

## Steps

1. List untriaged issues (no labels) with the GitHub CLI the user is already logged into:
   ```bash
   gh issue list --search "no:label is:open" --limit 20 --json number,title,body,createdAt
   ```
2. For each issue, propose:
   - a type label from the repository's existing labels (`gh label list`), never a new one;
   - a priority (P1 crash or data loss, P2 broken feature with workaround, P3 everything else);
   - up to three possible duplicates found with `gh search issues --repo <owner/repo> "<key words>"`.
3. Present the proposals as a table and wait. Apply labels with `gh issue edit <n> --add-label <label>` only for the rows the user approves.
4. Never close issues as duplicates yourself; leave a suggested comment for the user to post.
