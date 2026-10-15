import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nar", description="NAR guardrails lab")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("smoke", help="run every configured detector on 20 items per track")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # --help exits 0, bad args exit 2
        return int(e.code or 0)
    if args.cmd is None:
        parser.print_help()
        return 0
    return 0
