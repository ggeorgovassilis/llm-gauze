# Architecture

Bandaid is an HTTP gateway layered in front of a local LLM. This document
captures the intended design so later phases stay aligned.

## Principles

- **Modular & extensible.** Detection and remediation are separate, pluggable
  concerns. Simple fixes should be cheap to add; complex ones should be
  possible without rearchitecting.
- **Record everything.** Every request/response is persisted so detectors can
  reason over history (past requests, responses, and outcomes).
- **No assumptions yet.** Error manifestations are unknown; the gateway first
  records, then (in later phases) reasons and remediates.

## Layers

```
client ──▶ gateway (FastAPI) ──▶ recorder (JSONL)
                │
                └──▶ proxy ──▶ remediation ──▶ upstream local LLM
```

- `app/config.py` — environment-driven settings (`.env`).
- `app/main.py` — FastAPI app + routing (OpenAI-compatible surface).
- `app/proxy.py` — forwarding seam; records each attempt and delegates
  failure handling to the remediation pipeline.
- `app/recorder.py` — append-only JSONL recording of exchanges/attempts.
- `app/remediation/` — the pluggable remediation pipeline (below).

## Separation of concerns

The remediation pipeline deliberately separates three concerns so new
capabilities can be added without rearchitecting:

1. **Data collection** — `app/recorder.py` persists every exchange/attempt.
2. **Understanding** — `Detector` classes classify a failure into a
   `Diagnosis` (retryable or not, and why).
3. **Action** — `Backoff`/`RetryPolicy` classes turn a `Diagnosis` into
   behaviour (e.g. retry with exponential backoff).

```
failure ──▶ Detector (understanding) ──▶ Diagnosis ──▶ Backoff/RetryPolicy (action)
```

- `app/remediation/base.py` — abstract `Detector`, `Backoff`, `Diagnosis`,
  `RetryPolicy`.
- `app/remediation/retry.py` — `RetryableDetector` (transient exceptions +
  retryable HTTP statuses) and `ExponentialBackoff` (with optional jitter).

Future capabilities (sloppy-response cleanup, loop detection, context-window
detection, ...) implement these interfaces rather than editing the proxy.

## Future phases

Each phase is self-contained with clear acceptance criteria. The recorder's
JSONL store is the shared substrate every subsequent phase reads from.

1. **Stub** — compose, forwarding gateway, `.env` config, logging.
2. **Retry** (current) — retry transient upstream failures with configurable
   exponential backoff.
3. **Detection** — analyse recorded exchanges, classify further failures.
4. **Remediation** — sloppy-response cleanup, loop/context detection.
5. **Streaming** — SSE pass-through and per-chunk handling.
