.PHONY: install format lint typecheck test integration eval gate up down logs

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

integration:
	CLINVERIFY_INTEGRATION=1 uv run pytest

eval:
	CLINVERIFY_EVAL=1 uv run pytest -m eval

gate: lint typecheck test

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f
