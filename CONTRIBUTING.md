# Contributing

Thanks for your interest in improving this project. The notes below should
help you get a working setup, run the checks locally, and submit a change
that has a good chance of being merged quickly.

## Development setup

```bash
git clone <repo-url>
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"            # if a dev extra is declared
pre-commit install
```

Run the application locally:

```bash
uvicorn app.main:app --reload --port 8000
```

## Running the checks

The CI pipeline runs the same commands you can run locally. If they pass on
your machine, they should pass in CI.

```bash
ruff check .                       # lint
ruff format --check .              # formatting
mypy app/                          # type check
bandit -r app/ -ll                 # security scan
pytest tests/ --cov=app --cov-fail-under=80
```

`pre-commit run --all-files` will run the full set in one go.

## Branching and commits

- Branch from `main`: `feat/<short-description>` or `fix/<short-description>`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages — `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Keep commits focused. A reviewer should be able to read the message and
  know what to look at.

## Pull requests

A good PR:

- Has a descriptive title and a brief body explaining the *why*.
- Includes tests for new behaviour and regression tests for fixes.
- Updates documentation when behaviour or interfaces change.
- Keeps the diff small. Split unrelated changes into separate PRs.
- Passes CI without exemptions or `# noqa`/`# type: ignore` unless justified
  in a comment.

## Code style

- **Line length**: 100 characters (configured in `pyproject.toml`).
- **Type hints**: required on public functions and class methods.
- **Docstrings**: Google style for public APIs; one-line summaries are fine
  for internal helpers.
- **Errors**: raise specific exceptions from `core_exceptions`; do not catch
  bare `Exception` unless you re-raise or log with `exc_info=True`.
- **Logging**: use `logger = logging.getLogger(__name__)`. No `print` in
  application code.

## Testing

- New code requires unit tests. Cover the success path, at least one error
  path, and any branching logic.
- Integration tests go under `tests/integration/` and are marked with
  `@pytest.mark.integration`.
- Tests should not require network access. Mock external services with
  `pytest-httpx`, `respx`, or fixtures.

## Reporting bugs

Open an issue using the **Bug Report** template. Include the version, the
exact error, and the smallest reproduction you can create.

## Reporting security issues

Please follow `SECURITY.md` rather than opening a public issue.

## Code of conduct

Participation in this project is governed by `CODE_OF_CONDUCT.md`.
