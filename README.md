# llm-gauze

[![CI](https://github.com/ggeorgovassilis/llm-gauze/actions/workflows/ci.yml/badge.svg)](https://github.com/ggeorgovassilis/llm-gauze/actions/workflows/ci.yml)

llm-gauze is an HTTP gateway that sits in front of a local LLM (served via an
OpenAI-compatible API) and works around its shortcomings: transient errors
without retries, silently hung or looping models, empty or sloppy responses,
context-window overflows, runaway reasoning, and malformed tool calls. It
logs every exchange and remediates what it can before the client ever sees
it.

## Quick start

llm-gauze ships as a published container image and runs with Docker Compose.

1. Create your configuration:

   ```bash
   cp .env.example .env
   ```

2. Point `LLM_BASE_URL` in `.env` at your local LLM's OpenAI-compatible
   endpoint (default `http://host.docker.internal:14434`, which reaches a
   host-side LLM on port 14434 from inside the container).

3. Start the gateway:

   ```bash
   docker compose up
   ```

   This runs the published image `ghcr.io/ggeorgovassilis/llm-gauze:latest`
   (defined in `docker-compose.yml`). To pin a specific release, override the
   tag — for example `ghcr.io/ggeorgovassilis/llm-gauze:7`.

The gateway listens on `http://localhost:9317` and exposes an
OpenAI-compatible API (e.g. `POST /v1/chat/completions`), forwarding to the
`LLM_BASE_URL` in `.env`.

### Provider API paths

The gateway appends the client's request path to `LLM_BASE_URL`. For Hetzner
Inference, set `LLM_BASE_URL=https://inference.hetzner.com/api` and point a
chat-completions client at `http://localhost:9317` so `/v1/chat/completions`
reaches `/api/v1/chat/completions` upstream. Do not include `/v1` twice or use
the provider's website root as the API base.

After changing `.env`, recreate the service; restarting an existing container
does not reload its environment:

```bash
docker compose up -d --force-recreate gateway
```

For local development, use
`docker compose -f docker-compose.dev.yml up -d --build --force-recreate gateway`.
Check `/health` afterwards: its `upstream` must show the intended API base,
including `/api` for Hetzner. Rebuilding an image alone does not correct a
wrong value in `.env`.

For VS Code Copilot custom endpoints, explicitly select
provider-level `apiType: "chat-completions"` for both the direct provider and
gateway entries.
The direct entry's URL is `https://inference.hetzner.com/api/v1`; the gateway
entry's URL is `http://localhost:9317`. Supply the provider's bearer token in
the client's `Authorization` request header; the gateway forwards it unchanged.
Never put credentials in committed configuration.

The current Copilot `customendpoint` implementation gives an explicit
`Authorization` request header precedence over its provider API key. Without
that header, the provider API key must be the real token, not a placeholder.

An upstream HTTP 200 without completion choices is not a successful model
reply. With streaming remediation enabled, the gateway reports HTTP 502 with
`invalid_upstream_response` rather than fabricating a completion or an empty
SSE success. Check the upstream API path and streaming support first. Upstream
HTTP errors retain their status and body.

For an immediate VS Code 502, inspect `data/records.jsonl` structurally rather
than dumping recordings or private conversations. Match the client's
`x-request-id` in `request_headers` to the gateway's `request_id`, then compare
the record timestamp and attempt `duration` with container completion logs.
`GitHubCopilotChat` in the user agent distinguishes VS Code traffic from smoke
probes. A recorded `invalid_upstream_response` is a gateway protocol rejection,
not evidence that the provider returned HTTP 502: the rejected upstream status
and content type are not retained in that record. Verify the running API base
and compare a synthetic request directly when those details are needed.
The Hetzner website-root route has returned HTTP 200 with nine non-SSE bytes
and `application/octet-stream`, producing this immediate rejection even for
a valid chat-completions request with tools.

Temporary Compose overrides do not persist an `.env` correction. Before using
the ordinary start command again, privately set the correct `LLM_BASE_URL`
and recreate the gateway. A corrected route alone does not prove a model
can complete the VS Code workflow. Compare the relevant request shape,
including tools, message roles and `stream_options`, not just an OK-only curl.
Use synthetic content and bounded time/output for live probes, and report any
added token cap as a difference from the recorded request.

If authenticated generation still times out, compare the same small request
directly against the provider. Record time to response headers and first SSE
data separately from total duration: no response headers is not evidence of
ongoing reasoning. Authenticated `/models` discovery proves API reachability
and token acceptance, but not model readiness. A successful completion from
another advertised model with the same token helps isolate a model-specific
provider failure; report that comparison to the provider rather than silently
changing models or treating a gateway health check as successful generation.

### File ownership

The container runs as your host user (`UID`/`GID`, default `1000`) so the
files it writes to the bind-mounted `data/` directory are owned by you, not
root. If a `data/` directory was created earlier as root, fix ownership once
before the non-root container can write to it:

```bash
sudo chown -R "$(id -u):$(id -g)" data
```

## Health check

```bash
curl http://localhost:9317/health
```

Returns `{"status": "ok", "upstream": "<LLM_BASE_URL>"}` when the gateway is up.
Operational metrics are exposed at `/metrics` (Prometheus text format, or JSON
with `Accept: application/json`).

## Configuration

Every setting is documented in [`docs/configuration.md`](docs/configuration.md).

## How it works

llm-gauze sits between your client and the local LLM, recording every exchange:

```mermaid
flowchart LR
    Client[Your client] -->|OpenAI-compatible API| Gauze[llm-gauze]
    Gauze -->|forwards| LLM[Local LLM]
    Gauze -.->|logs every exchange| Store[(data/*.jsonl)]
```

When the model misbehaves, llm-gauze detects it and remediates what it can before
you ever see it — retrying transient failures, nudging empty replies, cleaning
leaked thinking tags, breaking loops with varied sampling:

```mermaid
sequenceDiagram
    participant C as Client
    participant B as llm-gauze
    participant L as Local LLM

    C->>B: POST /v1/chat/completions
    B->>L: forward
    L-->>B: error, hang, loop, or sloppy reply
    B->>B: detect & classify
    B->>L: remediate (retry / nudge / repair)
    L-->>B: clean completion
    B-->>C: chat.completion
```

See [`docs/architecture.md`](docs/architecture.md) for the full design and
[`docs/configuration.md`](docs/configuration.md) for every setting.

## Developing

See [`DEVELOPING.md`](DEVELOPING.md) for local development, running the tests,
and adding a detector or remediation.
