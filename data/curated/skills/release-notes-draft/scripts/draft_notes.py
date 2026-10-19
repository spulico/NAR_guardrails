"""Group commit subjects since a tag by conventional-commit type."""
import argparse
import re
import subprocess
from collections import defaultdict

TYPES = {"feat": "Features", "fix": "Fixes", "perf": "Performance", "docs": "Documentation"}
PATTERN = re.compile(r"^(?P<type>\w+)(\(.+\))?(?P<bang>!)?: (?P<msg>.+)$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--since", required=True)
    args = p.parse_args()
    log = subprocess.run(["git", "log", f"{args.since}..HEAD", "--pretty=%s"],
                         capture_output=True, text=True, check=True).stdout
    groups = defaultdict(list)
    for subject in log.splitlines():
        m = PATTERN.match(subject)
        if not m:
            groups["Other"].append(subject)
            continue
        key = "Breaking changes" if m["bang"] else TYPES.get(m["type"], "Other")
        groups[key].append(m["msg"])
    for heading in ["Breaking changes", *TYPES.values(), "Other"]:
        if groups.get(heading):
            print(f"## {heading}\n")
            print("\n".join(f"- {line}" for line in groups[heading]) + "\n")


if __name__ == "__main__":
    main()
