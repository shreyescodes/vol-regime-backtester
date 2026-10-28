# Contributing

Contributions are welcome. Correctness comes first: in a backtester, a subtle look-ahead or cost
error silently produces flattering numbers.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
make install            # pip install -e ".[dev,app,data]" and pre-commit hooks
```

## Workflow

1. Branch from `main`: `git checkout -b feat/short-description`.
2. Make the change with tests. Bug fixes should include a test that fails without the fix.
3. Run everything CI runs:
   ```bash
   make check          # ruff lint + format check, mypy --strict, pytest with coverage gate
   ```
4. Open a pull request describing the change and its effect on results, if any.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).

## Standards

* **Causality.** A strategy's weight at bar *t* may use only data up to the close of *t*. New
  strategies must be added to the parametrized list in `tests/test_lookahead.py`.
* **Costs.** Do not bypass the engine's turnover costs when comparing against the benchmark.
* **Types and style.** `mypy --strict` and `ruff` must pass; public functions need docstrings that
  state assumptions (units, annualization, conventions).
* **Determinism.** Use a local `numpy.random.Generator`; never seed global state. Tests and CI
  must not use the network (mark such tests `@pytest.mark.network`).
* **Config.** New tunables belong in `config.py` with validation, not as module constants.

## Reporting bugs

Open an issue with a minimal data snippet, the YAML config, the observed and expected output,
and the `quantvol --version`. Suspected look-ahead or cost-modelling errors are the highest
priority.
