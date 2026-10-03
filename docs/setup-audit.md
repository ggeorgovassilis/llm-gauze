# Project setup audit

Scope: [#38](https://github.com/ggeorgovassilis/llm-gauze/issues/38). A holistic
review of the project setup: directory layout and naming, `scripts/` wrappers and
dev/run workflows, Dockerfile and compose setup, dependency pinning and
reproducibility, documentation coverage, and editor/CI/repo conventions. The
deliverable is this document: findings, their risks, and prioritised
recommendations, where each actionable item has been filed as its own ticket.

## Strengths

The setup is in good shape overall. Worth calling out explicitly:

- **Reproducible builds** — the base image is pinned to an exact digest, every
  transitive dependency is pinned and hashed (`pip-compile --generate-hashes`,
  installed with `--require-hashes`), and GitHub Actions are pinned to full
  commit SHAs.
- **Clear dev/consume split** — `docker-compose.yml` (published image, the
  "consume" path) is cleanly separated from `docker-compose.dev.yml` (build from
  source + live mounts), each with an explanatory header comment.
- **Single, well-documented toolchain** — `ruff` (lint + format + import sort)
  and `mypy` are configured once in `pyproject.toml` and used identically
  locally (`./scripts/test.sh`) and in CI.
- **Secrets hygiene** — `.env` is gitignored, `.env.example` is the tracked
  template, and `data/` (recordings) is gitignored so no runtime data leaks into
  the repository.

## Findings

### F1 — `.env.example` omits two whole feature sections (high)

`source/app/config.py` defines the Coast-detection settings
(`coast_detection_enabled`, `coast_nudge_text`, `coast_max_attempts`) and the
Message-overflow settings (`message_overflow_enabled`, `message_overflow_threshold`,
`message_overflow_truncate`, `message_overflow_warning`), and
`docs/configuration.md` documents them — but `.env.example` has **no** section
for either. A user who copies `.env.example` cannot configure coast detection or
message-overflow protection at all, even though the features ship enabled by
default.

The same class of drift, lower severity: `.env.example` sets
`RETRY_MAX_ATTEMPTS=8` while the code default is `3`. `configuration.md` already
flags this, but the two "recommended" values remain out of step.

**Risk.** The configuration surface is split across three files that have
already drifted apart; a user cannot discover or tune two production features.

**Recommendation.** Add the two missing sections to `.env.example`; align or
annotate the `RETRY_MAX_ATTEMPTS` drift. Filed as
[#74](https://github.com/ggeorgovassilis/llm-gauze/issues/74).

### F2 — `HOST`/`PORT` settings are dead code with a confused meaning (high)

`config.py` defines `host` and `port` (defaults `0.0.0.0` / `8000`) and
`.env.example` ships `HOST` / `PORT`, but **nothing reads** `settings.host` or
`settings.port`: the Dockerfile `CMD` hardcodes
`uvicorn app.main:app --host 0.0.0.0 --port 8000`. Separately, the compose files
interpolate `PORT` as the *host* publish port (`ports: "${PORT:-9317}:8000"`).
So `PORT` has two documented meanings — the gateway listen port (inert) and the
host publish port (live) — and `HOST` has none.

**Risk.** A user setting `PORT=9999` expecting the gateway to move ports will be
surprised (the container still listens on 8000). The duplication invites
misconfiguration.

**Recommendation.** Either wire `settings.host`/`settings.port` into the
entrypoint (and drop the hardcoded `--host`/`--port`), or remove them and
document that `PORT` is the compose host-publish port only. Filed as
[#75](https://github.com/ggeorgovassilis/llm-gauze/issues/75).

### F3 — Lockfile regeneration docs diverge from the committed lockfiles (medium)

`DEVELOPING.md` documents the regeneration as:

```
pip-compile --generate-hashes --strip-extras --output-file requirements.lock source/requirements.txt
pip-compile --generate-hashes --strip-extras --output-file requirements-dev.lock source/requirements.txt requirements-dev.txt
```

The committed lockfile headers record:

```
pip-compile --generate-hashes --no-index --output-file=requirements.lock --strip-extras source/requirements.txt
pip-compile --generate-hashes --no-index --output-file=requirements-dev.lock --strip-extras requirements-dev.txt source/requirements.txt
```

Two divergences: the docs omit `--no-index`, and the dev-lock input order is
swapped. A developer following `DEVELOPING.md` will produce a differently-resolved
lockfile than the committed one, quietly breaking the reproducibility story that
the hashed lockfiles otherwise provide.

**Risk.** Lockfile churn / non-reproducible builds over time.

**Recommendation.** Decide whether `--no-index` is intended, then make the two
documented commands match the committed headers exactly. Filed as
[#76](https://github.com/ggeorgovassilis/llm-gauze/issues/76).

### F4 — The publish path skips lint/type-check (medium)

`.github/workflows/ci.yml` gates pull requests with `ruff check`,
`ruff format --check`, and `mypy`. `.github/workflows/publish.yml` (called by
`release.yml`) runs only `pytest`. A release is cut manually and does not re-run
the static gates, so it is possible to publish code that would not have passed a
PR.

**Risk.** A release can ship style/type regressions that CI would have caught.

**Recommendation.** Run the same lint/format/type-check commands in the publish
workflow before publishing. Filed as
[#77](https://github.com/ggeorgovassilis/llm-gauze/issues/77).

### F5 — `.dockerignore` omits local cache/virtualenv directories (low)

`.dockerignore` excludes `.git`, `.github`, `.env`, `data/`, `__pycache__`,
`*.pyc`, `.pytest_cache/`, `*.jsonl`, and `chat.json`, but not the cache and
virtualenv directories that accumulate locally: `.mypy_cache/`, `.ruff_cache/`,
`.venv/`, `htmlcov/`, `.coverage`, and `*.egg-info/`. These bloat the build
context sent to the daemon. No correctness impact (the Dockerfile `COPY`s
explicit paths), purely context size.

**Risk.** Slower builds; noisy context uploads.

**Recommendation.** Add the missing cache/virtualenv patterns. Filed as
[#78](https://github.com/ggeorgovassilis/llm-gauze/issues/78).

### F6 — Layout asymmetry: runtime pins under `source/`, dev pins at root (observation)

Direct runtime dependencies live in `source/requirements.txt`, while direct dev
dependencies live at the repo root in `requirements-dev.txt` — so the root has
`requirements-dev.txt` but no sibling `requirements.txt`. This is consistent and
works (the lockfiles and `DEVELOPING.md` all agree on the paths), but the naming
is slightly asymmetric for new contributors to discover.

**Recommendation.** Cosmetic; no ticket filed. Could be folded into a future
"directory layout polish" pass if desired.

## Prioritisation

| # | Finding | Severity | Ticket |
| --- | --- | --- | --- |
| F1 | `.env.example` missing Coast + Message-overflow sections | High | [#74](https://github.com/ggeorgovassilis/llm-gauze/issues/74) |
| F2 | `HOST`/`PORT` dead code + confused meaning | High | [#75](https://github.com/ggeorgovassilis/llm-gauze/issues/75) |
| F3 | Lockfile regeneration docs diverge from headers | Medium | [#76](https://github.com/ggeorgovassilis/llm-gauze/issues/76) |
| F4 | Publish path skips lint/type-check | Medium | [#77](https://github.com/ggeorgovassilis/llm-gauze/issues/77) |
| F5 | `.dockerignore` misses cache/virtualenv dirs | Low | [#78](https://github.com/ggeorgovassilis/llm-gauze/issues/78) |
| F6 | `requirements*.txt` layout asymmetry | — | none (observation) |
