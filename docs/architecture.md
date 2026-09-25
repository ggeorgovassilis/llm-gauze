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
- `app/remediation/stall.py` — `StallDetector`, a time-based watchdog that
  flags a stream which has stopped producing content tokens (silent hang).
- `app/remediation/context.py` — `ContextWindowDetector`, which recognises the
  upstream's context-window-fill error and reclassifies it non-retryable.
- `app/remediation/think.py` — `ThinkContentGuard`, a *transform* (not a
  `Detector`) that relocates leaked thinking tags out of visible content and
  guarantees a non-empty reply.

`RetryableDetector` classifies transport-level failures **by category**
(`httpx.RequestError`, `OSError`, `TimeoutError`) rather than enumerating every
possible exception, so any network/timeout error is retryable. Retryable HTTP
statuses are configurable. Non-transient exceptions (bugs in our own code) are
deliberately left unretryable so they surface instead of being masked.

Future capabilities (sloppy-response cleanup, ...) implement these interfaces
rather than editing the proxy.

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
- On detection the gateway logs it prominently, records an `abort_kind`
  diagnosis, aborts the upstream request (no retry — retrying just re-enters
  the loop), and returns a `loop_abort_status` error to the client.

Detection uses two independent signals, both configurable via `.env`:

1. **n-gram recurrence** — Jaccard similarity of word n-grams against a sliding
   window of recent sentences (`LOOP_*` settings).
2. **low compression entropy** — zlib compression ratio of the window text,
   gated by a minimum window length.


## Stall detection

A model can return HTTP 200 headers and then emit nothing at all. Loop
detection never fires in this case (there is no text to observe), and a plain
per-read timeout is defeated by upstream keepalives — LiteLLM held the SSE
connection open with comment frames, so every read succeeded while no token
ever arrived. The result is a request that hangs indefinitely.

Stall detection is therefore a **time** signal rather than a content signal:

- Two wall-clock deadlines, anchored to *content-bearing* tokens only (thinking
  or response deltas). SSE keepalives/comment lines and empty frames never
  reset the timer.
- **time-to-first-token** (`STALL_TTFT_SECONDS`) — generous, because prefill on
  a large prompt is legitimately slow.
- **inter-token gap** (`STALL_GAP_SECONDS`) — tighter, once generation has
  begun.
- The streaming loop bounds each read with `asyncio.wait_for` using the
  detector's remaining budget, so a silent stream trips the watchdog even while
  keepalives continue to arrive.
- On detection the gateway records `abort_kind=stalled` and returns a
  `stall_abort_status` error (never retried).

Unlike `ThinkingLoopDetector`, `StallDetector` observes time, not text, so it
is driven directly by the streaming loop rather than through the
`StreamDetector.feed()` interface. It takes an injectable monotonic clock so it
is unit-testable without sleeping.


## Context-window overflow

llama.cpp rejects a request that no longer fits its context window with a
**deterministic error message** carried in the HTTP response body (or, from a
few stacks, an exception message) — never a dedicated status code. Retrying
cannot succeed: the same oversized input fails identically on every attempt.

Context-window detection is therefore a **message** signal, classified before
the retry decision:

- `ContextWindowDetector` wraps the retryable-status detector and matches a set
  of exact, case-insensitive substrings (`CONTEXT_WINDOW_MARKERS`). Exact
  matching keeps it free of false positives — no fuzzy heuristics.
- A match produces `Diagnosis(retryable=False, code="context_window_exceeded")`,
  which short-circuits the retry loop and returns a precise
  `context_window_abort_status` error (default 413 Payload Too Large) with a
  named `context_window_exceeded` body instead of a generic 502.
- The detector inspects both the buffered forward path (response body) and the
  streaming path (error body / transport exception), so both request shapes
  fail fast on a full context window.


## Think-tag cleanup

Reasoning models occasionally emit their chain-of-thought as literal
`<think>…</think>` / `<reasoning>…</reasoning>` markup inside the *visible*
`content` field instead of the dedicated `reasoning_content` stream. Two
problems follow: the visible message is polluted, and — when the model emits
only the tag and nothing else — the visible turn is empty, which aborts
Copilot's agentic flow.

Think-tag cleanup is therefore a **content transform** (not a failure
`Detector`): it rewrites the assembled completion rather than classifying it.

- `ThinkContentGuard.clean(content, reasoning)` **relocates** the inner text of
  any leaked thinking tag out of `content` and appends it to `reasoning` —
  nothing is discarded. Tag names are configurable (`THINK_TAGS`) and matched
  case-insensitively by exact tag name — no fuzzy heuristics.
- A trailing unmatched opening tag (`<think>…` with no close) is treated the
  same way: the remainder is reasoning.
- If the visible `content` is empty/whitespace after relocation but reasoning
  exists, a short configurable placeholder (`THINK_EMPTY_RESPONSE_PLACEHOLDER`)
  is emitted instead of an empty turn.
- Every change is reported back to the recorder as a `think_cleanup` list of
  `{kind, …}` records (`relocated_think`, `empty_content_placeholder`), so the
  mutation is never silent.
- It runs after streaming reconstruction, before the response is rebuilt. For a
  `stream: true` client whose output was changed, the SSE stream is rebuilt
  from the cleaned values; untouched streams are passed through verbatim.

Extracting a *usable answer* from the relocated chain-of-thought is deliberately
out of scope here (see #13): it is not deterministically fixable, and surfacing
raw reasoning as content would be a quality regression.


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
