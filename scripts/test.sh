#!/usr/bin/env bash
# Run the test suite inside the Docker test image — the same environment CI uses.
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose -f docker-compose.dev.yml build test
docker compose -f docker-compose.dev.yml run --rm test pytest "$@"
