---
name: i18n-check
description: Find translation keys missing from non-English locale files.
---

# i18n check

Use when the user adds UI strings, or before a release that ships translations.

## Steps

1. Locale files live in `locales/<lang>.json`; English (`en.json`) is the source of truth.
2. Run the checker:
   ```bash
   python scripts/check_i18n.py locales
   ```
3. It prints, per language, keys that are missing and keys that exist only in that language (probably stale).
4. Do not machine-translate missing strings into the files. List them for the translators, or add the English text with a `TODO(i18n)` marker only if the user asks.
5. Also flag placeholders that differ between languages, such as `{count}` in English but `{n}` in German, because they break at runtime.
