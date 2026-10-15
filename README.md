# NAR Guardrails Lab

Benchmark harness comparing non-autoregressive guardrail models (Laya, GLiNER2, ...) against classifiers and Claude.
Design: `docs/superpowers/specs/2026-10-14-nar-guardrails-lab-design.md`.

## Commands

- `make setup` - install Python 3.12 and sync dependencies (`uv`)
- `make test` - run the test suite
- `make smoke` - run every configured detector on 20 items per track

## Resolved dependency versions

torch 2.14.1, transformers 5.20.0, gliner2 2.1.0, laya 0.4.8, anthropic 1.15.0
