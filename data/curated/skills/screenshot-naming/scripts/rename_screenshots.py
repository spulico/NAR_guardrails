"""Preview or apply date-based renames for macOS-style screenshot names."""
import argparse
import re
from pathlib import Path

PATTERN = re.compile(r"Screen ?[Ss]hot (\d{4}-\d{2}-\d{2}) at (\d{1,2})\.(\d{2})\.(\d{2})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--topic", required=True)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    for f in sorted(args.folder.glob("*.png")):
        m = PATTERN.search(f.name)
        if not m:
            continue
        date, hh, mm, ss = m.groups()
        target = f.with_name(f"{date}_{int(hh):02d}{mm}{ss}_{args.topic}.png")
        if target.exists():
            print(f"skip   {f.name} (target exists)")
            continue
        print(f"{'rename' if args.apply else 'would '} {f.name} -> {target.name}")
        if args.apply:
            f.rename(target)


if __name__ == "__main__":
    main()
