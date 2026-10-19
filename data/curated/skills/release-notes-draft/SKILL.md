---
name: release-notes-draft
description: Draft human-readable release notes from merged commits since the last tag.
---

# Release notes draft

Use when the user is preparing a release and wants a first draft of the notes.

## Inputs

- The previous tag (default: the most recent tag reachable from `HEAD`).
- Optional: the version number of the new release.

## Steps

1. Run the helper to group commits by conventional-commit type:
   ```bash
   python scripts/draft_notes.py --since "$(git describe --tags --abbrev=0)" > /tmp/notes.md
   ```
2. Read `/tmp/notes.md` and rewrite each bullet so it explains the user-visible effect, not the implementation. Drop `chore:` and `ci:` entries unless they change behaviour.
3. Put breaking changes first under a "Breaking changes" heading, each with a one-line migration hint.
4. Present the draft in the chat. Do not create the GitHub release or push anything; the user publishes it.

## Style

- Present tense, no emoji, no internal ticket numbers unless the project already uses them publicly.
- Credit external contributors by their GitHub handle when the commit has a `Co-authored-by` trailer.
