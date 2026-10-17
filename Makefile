.PHONY: install lint format test check

install:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

check: lint test

.PHONY: data

data:
	uv run python -m src.data.make_dataset


.PHONY: load_dataset

load_dataset:
	uv run python -m src.data.load_dataset