---
name: run-tests
description: Run the smallest relevant slice of the pytest suite and summarise failures.
---

# Run tests

Use after editing code, or when the user asks "do the tests pass?".

## Steps

1. Work out which tests are relevant. Prefer the test file that mirrors the module you touched (`src/pkg/foo.py` -> `tests/test_foo.py`).
2. Run them with the helper, which stops at the first failure and keeps output short:
   ```bash
   bash scripts/run_tests.sh tests/test_foo.py
   ```
3. If they pass, run the full suite once with `bash scripts/run_tests.sh` before saying the work is done.
4. For each failure, report the test id, the assertion message and your best guess at the cause. Do not edit tests to make them pass unless the user agrees the test is wrong.

## Notes

- Never add `-p no:cacheprovider` or skip markers to hide failures.
- If the suite needs services (database, Redis) and they are not running, say so instead of retrying.
