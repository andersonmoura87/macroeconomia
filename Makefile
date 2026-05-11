.PHONY: install sync lint format typecheck test cov docker-build docker-up pre-commit clean

UV ?= uv

install:
	$(UV) python install 3.11
	$(UV) sync --group dev

sync:
	$(UV) sync --group dev

lint:
	$(UV) run ruff check src tests pipelines dashboard scripts

format:
	$(UV) run ruff check src tests pipelines dashboard scripts --fix
	$(UV) run black src tests pipelines dashboard scripts

typecheck:
	$(UV) run mypy src/macroeconomia

test:
	$(UV) run pytest

demo-panel:
	$(UV) run python scripts/gen_demo_panel.py

cov: test

docker-build:
	docker compose build

docker-up:
	docker compose up --build

pre-commit:
	$(UV) run pre-commit install
	$(UV) run pre-commit run --all-files

clean:
	$(UV) run ruff clean || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
