# Bandaid

An HTTP gateway that sits in front of a local LLM (served via an
OpenAI-compatible API) and works around its shortcomings — technical errors,
empty responses, sloppy output, missing retries, stuck/looping models, and so
on.

## Phase 1 — Stub

This phase delivers a minimal but working skeleton:

- `docker compose` based run (no host installs).
- A basic HTTP gateway that forwards every request verbatim to the upstream LLM.
- Upstream LLM endpoint configurable via `.env`.
- Every request/response exchange is recorded to JSONL for later analysis.
- No auth/security — this is a PoC.

### Run

```bash
./scripts/dev.sh
```

The gateway listens on `http://localhost:9317` and exposes an OpenAI-compatible
API (e.g. `POST /v1/chat/completions`), forwarding to the `LLM_BASE_URL` in
`.env`.

### Configuration

Copy `.env.example` to `.env` (done automatically by `dev.sh`) and adjust:

- `LLM_BASE_URL` — the local LLM's OpenAI-compatible endpoint. Defaults to
  `http://host.docker.internal:14434`, which reaches a host-side LLM on port
  14434 from inside the container.
- `PORT` — the port the gateway listens on.
- `REQUEST_TIMEOUT` — upstream read/write timeout in seconds (generation can be
  slow).
- `CONNECT_TIMEOUT` — upstream connection timeout in seconds; kept short so a
  dead/blackholed endpoint fails fast instead of hanging until
  `REQUEST_TIMEOUT`.

Retries (transient upstream failures are retried with exponential backoff):

- `RETRY_MAX_ATTEMPTS` — max attempts per request (1 = no retry).
- `RETRY_BACKOFF_INITIAL` — initial backoff delay in seconds.
- `RETRY_BACKOFF_BASE` — exponential base: `delay = initial * base^(attempt-1)`.
- `RETRY_BACKOFF_MAX` — upper bound on any single delay, in seconds.
- `RETRY_BACKOFF_JITTER` — apply full jitter to the computed delay.
- `RETRYABLE_STATUS_CODES` — comma-separated HTTP statuses that warrant a retry.

Loop detection (chat completions are streamed from upstream and observed for
repetitive output; a loop aborts the request with a `loop_detected` error):

- `LOOP_DETECTION_ENABLED` — master switch for loop detection.
- `LOOP_WINDOW_SENTENCES` — recent sentences compared against.
- `LOOP_JACCARD_THRESHOLD` — similarity threshold for "same sentence".
- `LOOP_MIN_LOOP_COUNT` — similar sentences in the window that constitute a loop.
- `LOOP_NGRAM_SIZE` — word n-gram size used for similarity.
- `LOOP_COMPRESSION_RATIO` — compression ratio below which output is low-entropy.
- `LOOP_COMPRESSION_MIN_CHARS` — min window length before the compression check.
- `LOOP_ABORT_STATUS` — HTTP status returned when a loop is detected.

Recording:

- `DATA_DIR` / `RECORD_FILE` — where exchanges are recorded (JSONL).

### Recorded data

Every attempt of every exchange is appended as a JSON line to
`data/records.jsonl`, containing method, path, request headers/body, response
status/headers/body, duration, any error (with traceback), the diagnosis, and
the attempt number. This is the raw material for the detection/remediation
phases to come.

## Roadmap

- ~~Retries on transient errors.~~
- Error detection & remediation pipeline (modular, pluggable).
- Sloppy-response cleanup (`<think>` tags, malformed tool calls).
- ~~Stuck/loop detection.~~
- Context-window detection.
- Streaming pass-through.
