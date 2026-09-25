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

- `app/config.py` — environment-driven settings (`.env`). Timeouts are split:
  `connect_timeout` (fail fast on a dead endpoint) vs `request_timeout`
  (read/write once connected — generation can be slow).
- `app/main.py` — FastAPI app + routing (OpenAI-compatible surface).
- `app/proxy.py` — forwarding seam; records each attempt and delegates
  failure handling to the remediation pipeline. Uses `httpx.Timeout` with
  separate `connect`/`pool` and `read`/`write` values.
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
  `RetryPolicy`, plus `StreamDetector`/`StreamVerdict` for content streams.
- `app/remediation/retry.py` — `RetryableDetector` (transient exceptions +
  retryable HTTP statuses) and `ExponentialBackoff` (with optional jitter).
- `app/remediation/loop.py` — `ThinkingLoopDetector`, a stateful
  `StreamDetector` that flags repetitive output via n-gram recurrence and low
  compression entropy.

`RetryableDetector` classifies transport-level failures **by category**
(`httpx.RequestError`, `OSError`, `TimeoutError`) rather than enumerating every
possible exception, so any network/timeout error is retryable. Retryable HTTP
statuses are configurable. Non-transient exceptions (bugs in our own code) are
deliberately left unretryable so they surface instead of being masked.

Future capabilities (sloppy-response cleanup, context-window detection, ...)
implement these interfaces rather than editing the proxy.

## Loop detection

Reasoning models sometimes get stuck regenerating the same or near-identical
sentences. Loop detection is a **content** signal (unlike retry, which handles
transport/status failures), so the gateway observes output as it is generated:

- Chat completions are requested from upstream with `stream: true` so the
  output can be read incrementally, then reassembled into a non-streaming
  `chat.completion` for the client.
- Two independent `ThinkingLoopDetector` instances run — one over the thinking
  (reasoning) stream, one over the visible response. The windows are **never
  combined**: repeating thinking content inside the response is not a loop.
- Scope is a **single response**; a later response repeating an earlier one is
  not a loop.
- Each detector is stateful and created fresh per request (with `reset()`),
  never shared across concurrent requests.
- On detection the gateway logs it prominently, records a `loop_detected`
  diagnosis, aborts the upstream request (no retry — retrying just re-enters
  the loop), and returns a `loop_abort_status` error to the client.

Detection uses two independent signals, both configurable via `.env`:

1. **n-gram recurrence** — Jaccard similarity of word n-grams against a sliding
   window of recent sentences (`LOOP_*` settings).
2. **low compression entropy** — zlib compression ratio of the window text,
   gated by a minimum window length.


## Timeout model

A single request timeout is not enough: the same value applied to *connecting*
and to *reading a response* means a blackholed endpoint (packets dropped, no
RST) hangs for the full request timeout before the first retry fires — which
looks like an "immediate failure" to a client with a shorter timeout.

- `CONNECT_TIMEOUT` — how long to establish a TCP connection; kept short
  (default 10s) so a dead/blackholed endpoint fails fast.
- `REQUEST_TIMEOUT` — read/write once connected; can be long (default 300s)
  because generation is slow.

These map onto `httpx.Timeout(connect=…, read=…, write=…, pool=…)`.

## Phases

Each phase is self-contained with clear acceptance criteria. The recorder's
JSONL store is the shared substrate every subsequent phase reads from.

1. **Stub** — compose, forwarding gateway, `.env` config, logging. ✅
2. **Retry** — retry transient upstream failures with configurable exponential
   backoff, separated into collection/understanding/action. ✅
3. **Detection** (current) — analyse recorded exchanges, classify further
   failures.
4. **Remediation** — sloppy-response cleanup, loop/context detection.
5. **Streaming** — SSE pass-through and per-chunk handling.
