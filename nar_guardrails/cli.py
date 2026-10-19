import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nar", description="NAR guardrails lab")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("smoke", help="run every configured detector on 20 items per track")
    r = sub.add_parser("run", help="run detectors on a track/split")
    r.add_argument("--track", required=True, choices=["prompts", "skills"])
    r.add_argument("--split", required=True)
    r.add_argument("--detectors", default="all", help="comma-separated keys or 'all'")
    r.add_argument("--policy", default="shared")
    r.add_argument("--protocol", default=None, help="skills only: MSD or SD")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--run-id", default=None)
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--throughput", action="store_true")
    r.add_argument("--data-dir", default="data/processed")
    r.add_argument("--results-dir", default="results")
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
    if args.cmd == "run":
        from pathlib import Path
        from nar_guardrails.bench.runner import RunConfig, run
        out = run(RunConfig(track=args.track, split=args.split, detectors=args.detectors.split(","),
                            policy=args.policy, protocol=args.protocol, limit=args.limit, run_id=args.run_id, seed=args.seed,
                            allow_dirty=args.allow_dirty, throughput=args.throughput,
                            data_dir=Path(args.data_dir), results_dir=Path(args.results_dir)))
        print(out)
        return 0
    return 0
