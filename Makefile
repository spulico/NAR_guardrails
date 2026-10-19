.PHONY: setup test smoke data bench
setup:
	uv python install 3.12
	uv sync --extra dev
test:
	uv run pytest -q
smoke:
	uv run nar smoke
data:
	uv run nar data build --track prompts
	uv run nar data build --track skills
bench:
	$(MAKE) smoke
	$(MAKE) data
	uv run nar run --track prompts --split dev  --detectors all --run-id prompts-dev-shared
	uv run nar run --track prompts --split test --detectors all --run-id prompts-test-shared --throughput
	uv run nar run --track prompts --split dev  --detectors claude-haiku-5-5,claude-sonnet-5-5 --policy 8shot --run-id prompts-dev-8shot
	uv run nar run --track prompts --split test --detectors claude-haiku-5-5,claude-sonnet-5-5 --policy 8shot --run-id prompts-test-8shot
	uv run nar run --track skills --split val  --detectors all --protocol MSD --run-id skills-val-msd
	uv run nar run --track skills --split test --detectors all --protocol MSD --run-id skills-test-msd --throughput
	uv run nar run --track skills --split val  --detectors claude-haiku-5-5,claude-sonnet-5-5 --protocol MSD --policy 8shot --run-id skills-val-msd-8shot
	uv run nar run --track skills --split test --detectors claude-haiku-5-5,claude-sonnet-5-5 --protocol MSD --policy 8shot --run-id skills-test-msd-8shot
	uv run nar run --track skills --split val  --detectors all --protocol SD --run-id skills-val-sd
	uv run nar run --track skills --split test --detectors all --protocol SD --run-id skills-test-sd
	uv run nar report --run prompts-test-shared --dev-run prompts-dev-shared
	uv run nar report --run prompts-test-8shot  --dev-run prompts-dev-8shot
	uv run nar report --run skills-test-msd     --dev-run skills-val-msd
	uv run nar report --run skills-test-msd-8shot --dev-run skills-val-msd-8shot
	uv run nar report --run skills-test-sd      --dev-run skills-val-sd
