import argparse
import json


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
    d = sub.add_parser("data", help="dataset commands")
    dsub = d.add_subparsers(dest="data_cmd")
    b = dsub.add_parser("build")
    b.add_argument("--track", required=True, choices=["prompts", "skills"])
    b.add_argument("--raw-dir", default="data/raw")
    b.add_argument("--out-dir", default="data/processed")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-strict", action="store_true")
    rp = sub.add_parser("report", help="build report.md for a run")
    rp.add_argument("--run", default="latest")
    rp.add_argument("--dev-run", default=None)
    rp.add_argument("--results-dir", default="results")
    rp.add_argument("--data-dir", default="data/processed")
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
    if args.cmd == "report":
        from pathlib import Path
        from nar_guardrails.report.build import build_report
        print(build_report(Path(args.results_dir), args.run, args.dev_run, data_dir=Path(args.data_dir)))
        return 0
    if args.cmd == "data" and args.data_cmd == "build":
        import os
        import sys
        from pathlib import Path
        code = 0
        try:
            if args.track == "prompts":
                from nar_guardrails.datasets.build_prompts import build_prompts
                m = build_prompts(Path(args.raw_dir), Path(args.out_dir) / "prompts.jsonl", seed=args.seed, strict=not args.no_strict)
            else:
                from nar_guardrails.datasets.build_skills import build_skills
                m = build_skills(Path(args.raw_dir), Path(args.out_dir) / "skills.jsonl", seed=args.seed, strict=not args.no_strict)
            print(json.dumps(m, indent=2))
        except BaseException as e:  # noqa: BLE001 - backstop: HF streaming threads can keep the process alive
            code = 1
            print(f"error: {e}", file=sys.stderr)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    return 0
