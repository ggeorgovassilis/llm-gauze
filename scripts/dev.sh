#!/usr/bin/env bash
# Build and run the gateway locally via Docker Compose.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "Missing .env — copying from .env.example" >&2
    cp .env.example .env
fi

docker compose up --build
