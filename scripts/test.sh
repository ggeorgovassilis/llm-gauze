#!/usr/bin/env bash
# Run the test suite inside the Docker test image — the same environment CI uses.
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose build test
docker compose run --rm test pytest "$@"
