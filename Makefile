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
	uv run nar run --track prompts --split dev --detectors all
	uv run nar run --track prompts --split test --detectors all
	uv run nar run --track skills --split val --detectors all
	uv run nar run --track skills --split test --detectors all
	uv run nar report --run latest
