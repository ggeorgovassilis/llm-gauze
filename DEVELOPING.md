# Developing

This is the developer guide for llm-gauze. Public users should follow the
[README](README.md).

## Prerequisites

- Docker and Docker Compose.

## Setup

Copy `.env.example` to `.env` if you don't have one — the dev script does this
automatically:

```bash
cp .env.example .env
```

## Running locally

```bash
./scripts/dev.sh
```

This builds the `runtime` target from `docker-compose.dev.yml` and runs the
gateway with `./source` live-mounted, so edits are picked up without a rebuild.
The gateway listens on `http://localhost:9317`.

## Running the tests

Tests run inside Docker — the same environment CI uses:

```bash
./scripts/test.sh
```

This builds the `test` target of `docker-compose.dev.yml` and runs the full
check suite: `ruff check`, `ruff format --check`, `mypy`, then `pytest`. To run
a subset or pass arguments through (they are forwarded to `pytest`):

```bash
./scripts/test.sh tests/test_loop_detector.py
./scripts/test.sh -k coast
```

The test runner is `pytest` (pinned in `requirements-dev.txt`); config lives in
`pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["source"]`). Lint and
type-check config (`ruff`, `mypy`) also lives in `pyproject.toml`.

Tests use an in-process mock OpenAI-compatible upstream (`http.server` in the
`*_integration.py` files) — never a real LLM, which is non-deterministic and
cannot reproduce a stall/loop/overflow on demand.

## Regenerating the dependency lockfiles

Python dependencies are pinned twice: the direct pins in `source/requirements.txt`
and `requirements-dev.txt`, and the transitive/hashed lockfiles `requirements.lock`
(runtime) and `requirements-dev.lock` (test). The Dockerfile installs from the
lockfiles with `--require-hashes`.

When you change a direct dependency, regenerate the lockfiles with `pip-compile`
(`pip-tools`), matching Python 3.12:

```bash
pip-compile --generate-hashes --no-index --output-file=requirements.lock --strip-extras source/requirements.txt
pip-compile --generate-hashes --no-index --output-file=requirements-dev.lock --strip-extras requirements-dev.txt source/requirements.txt
```

These are the exact commands recorded in the lockfile headers, and must be used
verbatim so a regeneration reproduces the committed files byte-for-byte. In
particular:

- `--no-index` makes `pip-compile` resolve against the local pip cache only,
  rather than hitting PyPI — this is how the committed lockfiles were generated,
  and why the flag (and the `--output-file=<name>` form) must match.
- The dev lockfile lists `requirements-dev.txt` **before** `source/requirements.txt`,
  so dev pins take precedence over the runtime pins they build on.

Commit both lockfiles with the change.

## Adding a detector or remediation

Detection and remediation are separate, pluggable concerns — see
[`docs/architecture.md`](docs/architecture.md) for the design.

- **Detectors** subclass the interfaces in `source/app/remediation/base.py`
  (`Detector` for buffered requests, `ContentWatchdog` for content streams) and
  classify a failure into a `Diagnosis`.
- **Remediation** policies (`NudgePolicy`, `CoastPolicy`, `LoopRetryPolicy`, …)
  turn a diagnosis into behaviour — a re-submission or a content transform.
- Add any new configuration to `source/app/config.py`, and document it in both
  [`docs/configuration.md`](docs/configuration.md) and `.env.example`.
- Add tests in `tests/` as `test_*.py` files with module-level `test_*`
  functions so `pytest` collects them. No per-file `__main__` runners.
