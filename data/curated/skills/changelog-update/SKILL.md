---
name: changelog-update
description: Add an entry to CHANGELOG.md under Unreleased, following Keep a Changelog.
---

# Changelog update

Use when a pull request changes user-visible behaviour and the user asks you to "add a changelog entry".

## Format

The project follows Keep a Changelog 1.1. Entries go under `## [Unreleased]` in one of these sections: Added, Changed, Deprecated, Removed, Fixed, Security.

## Steps

1. Read `CHANGELOG.md` and find the `## [Unreleased]` heading. If it does not exist, create it directly below the introduction.
2. Pick the section from the change type (new flag -> Added, bug fix -> Fixed, behaviour change -> Changed).
3. Write one line in the past-tense style the file already uses, ending with the PR number if the user gave one, for example:
   ```markdown
   - Fixed CSV export dropping the last row when the file had no trailing newline (#482).
   ```
4. Keep sections in the canonical order and do not reorder existing entries.
5. Show the diff of `CHANGELOG.md` and stop. Releasing (moving Unreleased to a version) is a separate step the maintainer does.
