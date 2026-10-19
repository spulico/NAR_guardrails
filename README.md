# NAR Guardrails Lab

Benchmark harness comparing non-autoregressive guardrail models (Laya, GLiNER2, encoder classifiers, ...) against Claude on two tracks: prompt injection (`prompts`) and malicious agent skills (`skills`).
Design: `docs/superpowers/specs/2026-10-14-nar-guardrails-lab-design.md`.

## Setup

```bash
make setup                      # Python 3.12 + dependencies via uv
hf auth login                   # needed for gated models/datasets (older CLI: huggingface-cli login)
export ANTHROPIC_API_KEY=...    # needed for the claude-* detectors
```

Without credentials the Claude detectors are skipped (a missing key is detected at construction); gated models are skipped with a reason in the manifest, not failed.

Gated models (accept each gate on its Hugging Face page after logging in): `promptguard2-86m` and `promptguard2-22m` (Meta Llama Prompt Guard 2), `sentinel` (qualifire/prompt-injection-sentinel).
Gated datasets: `prodnull`, `bells`, `pliny`, `mindgard`. `make data` fails in strict mode if their counts are out of range; use `nar data build ... --no-strict` to build without them.

Optional HTTP detectors `openjev-2b` and `basal-2.0-mini` are skipped until their servers run; see `scripts/systemone_servers.md`.

## Commands

- `make test` - run the test suite
- `make smoke` - run every configured detector on 20 items per track (`nar smoke [--n N] [--detectors a,b] [--results-dir DIR]`), prints one row per detector per track
- `make data` - build `data/processed/prompts.jsonl` and `skills.jsonl`
- `make bench` - smoke check, data build, then the full benchmark and its five reports (see below)

Examples:

```bash
uv run nar run --track prompts --split dev  --detectors all --run-id prompts-dev-shared
uv run nar run --track prompts --split test --detectors all --run-id prompts-test-shared --throughput
uv run nar run --track prompts --split test --detectors claude-haiku-5-5,claude-sonnet-5-5 --policy 8shot
uv run nar run --track skills  --split test --detectors all --protocol MSD --throughput
uv run nar run --track skills  --split test --detectors all --protocol MSD --variant text+scripts --run-id skills-test-msd-scripts
uv run nar report --run prompts-test-shared --dev-run prompts-dev-shared
uv run nar report --run latest
```

Run ids are the directory names under `results/` (the newest is in `results/LATEST`); without `--run-id` a timestamp is used.
Test runs are reported with `--dev-run <dev-or-val-run>`: thresholds are chosen on that run. Without `--dev-run`, `nar report` uses the newest text-variant dev/val run with the same track, policy and protocol, or 0.5 if there is none.

Reusing a run id: `nar run --run-id X` on an existing `results/X/` first deletes every `verdicts_*.jsonl` in it, so the directory never mixes verdicts from an earlier run (the manifest is rewritten; `nar report` rewrites the report files).

Skill items come in two variants and a run uses exactly one: `--variant text` (default, SKILL.md only) or `--variant text+scripts` (SKILL.md plus bundled scripts; only the curated skills and SkillSpector fixtures have it). The variant is recorded in the run manifest. A skill-track report adds a `text+scripts variant` section when a run of the other variant exists for the same track, split, protocol and policy: per detector, recall on the malicious skills that ship scripts, paired by item id (`<id>` vs `<id>#scripts`), at the F1 threshold. Otherwise the section says "no text+scripts run found".

### make bench

`make bench` runs `nar smoke` first (it stops there if a detector fails), then `make data`, then twelve `nar run` calls with fixed run ids, and ends with one report per track, protocol and policy:

| Report | Test run | Thresholds from |
|---|---|---|
| prompts, shared policy | `prompts-test-shared` | `prompts-dev-shared` |
| prompts, Claude 8-shot | `prompts-test-8shot` | `prompts-dev-8shot` |
| skills MSD, shared policy | `skills-test-msd` | `skills-val-msd` |
| skills MSD, Claude 8-shot | `skills-test-msd-8shot` | `skills-val-msd-8shot` |
| skills SD, shared policy | `skills-test-sd` | `skills-val-sd` |

It is a multi-hour job: every detector scores every item at batch size 1 so that latency is per item, and Claude calls run four at a time. On the current `data/processed` the Claude runs make about 26,000 calls per model, about 52,000 for Haiku and Sonnet together: 9,086 prompt dev+test items under each of the two policies, 4,672 skill MSD/SD val+test items under the shared policy, and 2,994 MSD val+test items under 8-shot (dev and test runs, and MSD and SD runs, share the verdict cache). Gated datasets, when they load, add to this.
Rough cost from `configs/prices.yaml`, assuming about 4 characters per token and about 60 output tokens per call: about 48M input tokens per model, so about $6 for Haiku and $110 for Sonnet; up to about $170 in total if every reply used the full 256 output tokens. Skill texts (about 11,000 characters on average) dominate the input. Without Anthropic credentials the Claude detectors are skipped and cost nothing.

## Results

Each run writes `results/<run-id>/`:

- `manifest.json` - config, git sha, per-detector status (`ok`, `skipped`, `error`) and skip reasons
- `verdicts_<detector>.jsonl` - one verdict per item (outcome, `p_malicious`, latency, tokens)
- `thresholds.json`, `report.md`, `cost_latency.png`, `scenario_heatmap.png` - written by `nar report`

Smoke runs go to `results/smoke-<timestamp>-<track>/`; `nar smoke` keeps its own cache there and does not update `results/LATEST`. It exits 1 if any detector is `failed` or `degraded` with no usable verdicts.
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
