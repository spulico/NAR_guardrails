# NAR Guardrails Lab

Benchmark harness comparing non-autoregressive guardrail models (Laya, GLiNER2, encoder classifiers, ...) against Claude on two tracks: prompt injection (`prompts`) and malicious agent skills (`skills`).
Design: `docs/superpowers/specs/2026-10-14-nar-guardrails-lab-design.md`.

## Setup

```bash
make setup                      # Python 3.12 + dependencies via uv
hf auth login                   # needed for gated models/datasets (older CLI: huggingface-cli login)
export ANTHROPIC_API_KEY=...    # needed for the claude-* detectors
```

Without credentials the affected detectors are skipped (with a reason in the manifest), not failed.

Gated models (accept each gate on its Hugging Face page after logging in): `promptguard2-86m` and `promptguard2-22m` (Meta Llama Prompt Guard 2), `sentinel` (qualifire/prompt-injection-sentinel).
Gated datasets: `prodnull`, `bells`, `pliny`, `mindgard`. `make data` fails in strict mode if their counts are out of range; use `nar data build ... --no-strict` to build without them.

Optional HTTP detectors `openjev-2b` and `basal-2.0-mini` are skipped until their servers run; see `scripts/systemone_servers.md`.

## Commands

- `make test` - run the test suite
- `make smoke` - run every configured detector on 20 items per track (`nar smoke [--n N] [--detectors a,b] [--results-dir DIR]`), prints one row per detector per track
- `make data` - build `data/processed/prompts.jsonl` and `skills.jsonl`
- `make bench` - data build, then the full benchmark (see below)

Examples:

```bash
uv run nar run --track prompts --split dev  --detectors all
uv run nar run --track prompts --split test --detectors all --throughput
uv run nar run --track prompts --split test --detectors claude-haiku-5-5,claude-sonnet-5-5 --policy 8shot
uv run nar run --track skills  --split test --detectors all --protocol MSD --throughput
uv run nar run --track skills  --split test --detectors all --protocol SD
uv run nar report --run latest
uv run nar report --run <id> --dev-run <dev-id>
```

`make bench` ends with `nar report --run latest`, which reports only the last run. Report the others explicitly, one call per run:

```bash
uv run nar report --run <run-id>            # dev/val runs
uv run nar report --run <test-run-id> --dev-run <dev-run-id>   # test runs use thresholds selected on the dev/val run
```

Run ids are the directory names under `results/` (the newest is in `results/LATEST`).

## Results

Each run writes `results/<run-id>/`:

- `manifest.json` - config, git sha, per-detector status (`ok`, `skipped`, `error`) and skip reasons
- `verdicts_<detector>.jsonl` - one verdict per item (outcome, `p_malicious`, latency, tokens)
- `thresholds.json`, `report.md`, `cost_latency.png`, `scenario_heatmap.png` - written by `nar report`

Smoke runs go to `results/smoke-<timestamp>-<track>/`.
`results/`, `data/raw/`, `data/processed/` and `models/` are gitignored; never commit them.

## Extending

- New detector: add an adapter class in `nar_guardrails/detectors/` (subclass `Detector`, decorate with `@register("<adapter>")`, import it in `_load_adapters`) and add a key to `configs/models.yaml` with `adapter`, model settings and `tracks`.
- New dataset source: write a loader in `nar_guardrails/datasets/public_prompts.py` (or `skills_bench.py`) returning `Item`s, add it to the `GIT_SOURCES`/`HF_SOURCES` registry there, and add its expected `{min, max}` item range to `configs/datasets.yaml`.

## Conventions

- Never execute anything under `data/`. Skill samples and injection prompts are untrusted; they are only read as text.
- Skill items come in variants `text` (SKILL.md only) and `text+scripts`. The `payload_in` meta flag on curated skills (`text` or `scripts`) records where the malicious payload lives.
- Commits made with Claude end with a `Co-Authored-By: Claude ... <noreply@anthropic.com>` trailer.

## Known device notes

- All local models currently run on MPS (`device: auto`); set `device: cpu` for a detector in `configs/models.yaml` if an MPS-specific error appears.
- Qwen3Guard-Stream runs under transformers 5 through local shims; its numerics are not validated against transformers 4. It is a content-safety model and labels plain injection text Safe, so low prompt-injection scores are expected.

## Resolved dependency versions

torch 2.14.1, transformers 5.20.0, gliner2 2.1.0, laya 0.4.8, anthropic 1.15.0
