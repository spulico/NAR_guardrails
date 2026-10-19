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
	$(MAKE) data
	uv run nar run --track prompts --split dev  --detectors all
	uv run nar run --track prompts --split test --detectors all --throughput
	uv run nar run --track prompts --split dev  --detectors claude-haiku-5-5,claude-sonnet-5-5 --policy 8shot
	uv run nar run --track prompts --split test --detectors claude-haiku-5-5,claude-sonnet-5-5 --policy 8shot
	uv run nar run --track skills --split val  --detectors all --protocol MSD
	uv run nar run --track skills --split test --detectors all --protocol MSD --throughput
	uv run nar run --track skills --split val  --detectors all --protocol SD
	uv run nar run --track skills --split test --detectors all --protocol SD
	uv run nar report --run latest
