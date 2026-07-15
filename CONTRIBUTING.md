# Contributing to QuantVol Backtester

Thank you for your interest in contributing to this project! This is a professional portfolio project designed to showcase systematic trading logic, vectorization, and software engineering best practices. 

## Development Setup

We use a standard `Makefile` to simplify development. 

1. **Clone the repository:**
   ```bash
   git clone git@github.com:shreyescodes/vol-regime-backtester.git
   cd vol-regime-backtester
   ```

2. **Install dependencies:**
   ```bash
   make setup
   ```
   *This installs both the production requirements and the development tools (pytest, black, flake8, jupyter).*

## How to Contribute

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Run tests before committing:**
   Ensure you haven't broken the mathematical logic or lookahead protections.
   ```bash
   make test
   ```

3. **Format and Lint:**
   This project strictly adheres to PEP-8 guidelines.
   ```bash
   black .
   flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
   ```
   *Note: Our GitHub Actions CI pipeline enforces linting and will reject PRs with syntax errors.*

4. **Submit a Pull Request!**

## Reporting Bugs
If you find a lookahead bias or a bug in the strategy friction modeling, please open an issue with a reproducible CSV snippet and the exact `run_backtest.py` config parameters.
