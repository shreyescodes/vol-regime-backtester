.PHONY: setup test run web docker-build docker-run clean

# Variables
PYTHON = python
DOCKER_IMAGE = quantvol-backtester

setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	@echo "Running unit tests..."
	$(PYTHON) -m unittest discover tests/

run:
	@echo "Running backend simulation..."
	$(PYTHON) run_backtest.py

web:
	@echo "Starting Streamlit dashboard..."
	$(PYTHON) -m streamlit run app.py

docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	@echo "Running Docker container..."
	docker run --rm -v $(PWD):/app $(DOCKER_IMAGE)

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
