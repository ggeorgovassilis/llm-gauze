# Overview

This project is an LLM gateway that detects and remediates shortcommings of local LLMs.

# Talking with the user

- Project language is UK english
- Be brief, short, to the point
- Explore one option at a time
- When the user asks you a question, just answer the question, don't undertake any modifications.
- When asked to come up with a plan, formulate the plan in self-contained phases. Each phase delivers one thing and has clear acceptance criteria.
- When the user asks you to "commit", they mean commit to github (with a concise message) and push.
- When the user asks you to list tickets, use the `gh` tool to list open repository issues. Group them by milestone (`current` first, `backlog` last). Make the ticket ID clickable with a link to the github issue.
- Everytime you reference a github ticket, mention the ticket ID and make it a link that takes you to the github issue. 

# Project tech
- Python in docker
- Pin library/framework dependencies
- Write wrapper scripts
- The github cli `gh` is installed and authenticated to the repository.

# Project structure
- The project is a github repository `bandaid`.
- `docs/` - Documentation
- `source/` - Source code
- `tests/` - Tests
- `scripts/` - Wrapper scripts

# Testing
- Tests run inside Docker — the same environment CI uses. Run them with `./scripts/test.sh` (which builds the `test` target of `docker-compose.dev.yml` and runs `pytest`).
- The test runner is `pytest` (pinned in `requirements-dev.txt`); config lives in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["source"]`).
- Add or change tests as part of the ticket that changes the behaviour they cover — never leave a behaviour change without test coverage.
- New tests go in `tests/` as `test_*.py` files with module-level `test_*` functions so `pytest` collects them; do not add per-file `__main__` runners to new tests.
- Run `./scripts/test.sh` before committing and make sure the suite is green.
- Tests use an in-process mock OpenAI-compatible upstream (`http.server` in the `*_integration.py` files) — do not depend on a real LLM; a real model is non-deterministic and cannot reproduce a stall/loop/overflow on demand.
- When a test fails, investigate the failure; do not weaken or delete a test just to make it pass unless the test itself is wrong.

# Implementing features
- Always work with a github ticket
- Create a branch for each tickcet, work in there
- Submit a pull request when done
- Have the user review the pull request
- Merge the pull request once the user approves it and close the associated ticket
