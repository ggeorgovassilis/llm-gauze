#!/usr/bin/env bash
# Run the full check suite inside the Docker test image — the same environment
# CI uses: lint + format + type-check, then the test suite.
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose -f docker-compose.dev.yml build test
docker compose -f docker-compose.dev.yml run --rm test \
    bash -c 'ruff check source tests && ruff format --check source tests && mypy source && pytest "$@"' \
    -- "$@"
