"""Compare locale JSON files against en.json: missing, stale and placeholder drift."""
import json
import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def main(folder: str) -> int:
    base = flatten(json.loads((Path(folder) / "en.json").read_text(encoding="utf-8")))
    problems = 0
    for path in sorted(Path(folder).glob("*.json")):
        if path.name == "en.json":
            continue
        other = flatten(json.loads(path.read_text(encoding="utf-8")))
        missing = sorted(set(base) - set(other))
        stale = sorted(set(other) - set(base))
        drift = [k for k in set(base) & set(other)
                 if set(PLACEHOLDER.findall(str(base[k]))) != set(PLACEHOLDER.findall(str(other[k])))]
        for label, keys in (("missing", missing), ("stale", stale), ("placeholder drift", sorted(drift))):
            for k in keys:
                print(f"{path.stem}\t{label}\t{k}")
        problems += len(missing) + len(drift)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "locales"))
