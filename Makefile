.PHONY: install lint format typecheck test check run run-synthetic dashboard docker clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev,app,data]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

typecheck:
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint typecheck test

run:
	quantvol run --config configs/spy.yaml --out reports/spy

run-synthetic:
	quantvol run --config configs/default.yaml --out reports/synthetic

dashboard:
	quantvol dashboard

docker:
	docker build -t quantvol .
	docker run --rm -v "$(CURDIR)/reports:/home/quant/reports" quantvol

clean:
	rm -rf reports .coverage coverage.xml htmlcov .mypy_cache .ruff_cache .pytest_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
