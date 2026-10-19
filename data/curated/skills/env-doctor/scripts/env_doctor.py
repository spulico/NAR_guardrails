"""Report which variables from an example env file are set, empty or missing.

Values are never printed: only names and status.
"""
import os
import sys
from pathlib import Path


def names_from_example(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            names.append(line.split("=", 1)[0].removeprefix("export ").strip())
    return names


def main(example: str) -> int:
    missing = 0
    for name in names_from_example(Path(example)):
        value = os.environ.get(name)
        status = "missing" if value is None else ("empty" if value == "" else "set")
        missing += status != "set"
        print(f"{name:<32} {status}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".env.example"))
