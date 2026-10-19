"""Mask secrets and personal data in a log file; writes <file>.redacted."""
import re
import sys
from pathlib import Path

RULES = [
    (re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd)\b(\s*[:=]\s*)\S+"), r"\1\2[REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY_ID]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
    (re.compile(r"/(Users|home)/[^/\s]+"), r"/\1/[user]"),
]


def main(path: str) -> None:
    src = Path(path)
    text = src.read_text(encoding="utf-8", errors="replace")
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    out = src.with_name(src.name + ".redacted")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
