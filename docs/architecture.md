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
  `StreamDetector` that flags repetitive output via byte-window compression
  entropy.
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
text. Loop detection is a **content** signal (unlike retry, which handles
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
- On detection the gateway logs it prominently and records an `abort_kind`
  diagnosis.

A detected **loop** is first *remediated* before giving up: the request is
re-submitted with varied sampling. Model-appropriate sampling is the client's
and endpoint's domain, so bandaid never invents values — a sampling parameter
the client already submitted is nudged up by `LOOP_RETRY_INCREMENT` (0.1); one
it did *not* submit is set to its configured fallback. The parameters affected
are `temperature`, `repeat_penalty` (llama.cpp), `presence_penalty` and
`frequency_penalty` (OpenAI). A loop is often a fixed point of the sampler:
the same deterministic reasoning cycle repeats because, given the same prompt
and the same parameters, the model re-walks the same path. Perturbing the
sampling breaks the cycle without changing the task. This is bounded by
`LOOP_RETRY_MAX_ATTEMPTS`; only when every re-submission also loops does the
gateway return a `loop_abort_status` error. A **stall** is never remediated
this way — retrying just re-waits for a silent model.

Detection is **byte-level** and **gated**, both configurable via `.env`:

1. **compression entropy** — zlib compression ratio over a sliding byte window
   (`LOOP_WINDOW_BYTES`); a ratio below `LOOP_COMPRESSION_RATIO` means the
   window is repetitive.
2. **arming gate** — detection only starts once the stream has produced
   `LOOP_MIN_OUTPUT_FRACTION` of the window. Short structured output (e.g. a
   model enumerating near-identical tool results) finishes before the gate, so
   it is never flagged; a genuine loop keeps repeating until it is caught.

A third, *terminal* signal catches drift loops (near-identical, not verbatim)
that the compression check cannot: **`finish_reason: "length"`**. When the
upstream fills the output window without the model finishing, llama.cpp
reports `truncated = 1` and LiteLLM surfaces `finish_reason: "length"`. The
gateway treats that truncation itself as a runaway loop — so it flows through
the same remediation/abort path above — rather than passing a truncated 200
through to the client.


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
It is split into two steps so the nudge rung (below) can sit between them:

- `ThinkContentGuard.relocate(content, reasoning)` **relocates** the inner text
  of any leaked thinking tag out of `content` and appends it to `reasoning` —
  nothing is discarded. Tag names are configurable (`THINK_TAGS`) and matched
  case-insensitively by exact tag name — no fuzzy heuristics.
- A trailing unmatched opening tag (`<think>…` with no close) is treated the
  same way: the remainder is reasoning.
- `ThinkContentGuard.guard_empty(content, reasoning, tool_calls)` applies the
  **placeholder floor**: if the visible `content` is empty/whitespace but
  reasoning exists (and there are no tool calls), a short configurable
  placeholder (`THINK_EMPTY_RESPONSE_PLACEHOLDER`) is emitted instead of an
  empty turn.
- `ThinkContentGuard.clean(...)` is the composition of `relocate` + `guard_empty`.
- Every change is reported back to the recorder as a `think_cleanup` list of
  `{kind, …}` records (`relocated_think`, `empty_content_placeholder`), so the
  mutation is never silent.
- It runs after streaming reconstruction, before the response is rebuilt. For a
  `stream: true` client whose output was changed, the SSE stream is rebuilt
  from the cleaned values; untouched streams are passed through verbatim.

## Nudge (re-prompt empty-text turns)

The placeholder above only guarantees a *non-empty* turn — it does not recover
the answer the model was heading toward. A turn that finished with
`finish_reason: stop`, no visible `content`, no `tool_calls`, but non-empty
`reasoning` is instead re-submitted once with a short nudge re-prompt, giving
the model a second chance to surface real output. This is the first rung of the
empty-response ladder (`nudge → extract → placeholder floor`).

- `NudgePolicy.should_nudge(finish_reason, content, tool_calls, reasoning)` is a
  pure predicate over the *relocated* turn: stop + empty content + no tool
  calls + non-empty reasoning. Deterministic — no heuristics.
- `NudgePolicy.apply(body)` appends `{"role": "user", "content": <nudge>}` to a
  copy of the request (the nudge text is `THINK_NUDGE_TEXT`) without mutating
  the input.
- The re-submitted request reuses the streaming path unchanged; the budget is
  `THINK_NUDGE_MAX_ATTEMPTS`, and each nudge pass is recorded with an
  `outcome` (`triggered`, `succeeded`, or `exhausted`) so the intervention is
  visible in `records.jsonl`.
- On exhaustion the placeholder floor (`guard_empty`) is applied as before, so
  the client still receives a non-empty turn.

Extracting a *usable answer* from the relocated chain-of-thought is deliberately
out of scope here (see #13): it is not deterministically fixable, and surfacing
raw reasoning as content would be a quality regression.

## Tool-call syntax enforcement

Local models sometimes emit malformed or truncated tool calls: unbalanced
braces, cut-off `function.arguments`, or a stray trailing comma. Forwarded
verbatim these break the client's own JSON parse. This is a **content
transform** (not a failure `Detector`) that runs on the assembled tool-call
list after streaming reconstruction:

- `ToolCallGuard.validate(tool_calls)` validates each call's
  `function.arguments` as a JSON string. Valid arguments pass through untouched.
- Truncation — the only breakage fixable *deterministically* — is repaired: a
  single forward scan tracks string/escape state and bracket nesting, closes an
  unterminated string, then closes open containers innermost-first while
  dropping a dangling trailing comma. The repaired string must re-parse as JSON
  or the call is left alone.
- Anything else (mismatched brackets, a dangling escape, missing
  `function`/`name`, non-string `arguments`) is **flagged**, never repaired:
  the request is not crashed and the malformed call is passed through.
- Every mutation/flag is reported back to the recorder as a `tool_repair` list
  of `{kind, index, …}` records (`repaired_arguments`, `flagged_malformed`), so
  the intervention is never silent.
- When the output changed (repair or think-cleanup), the `stream: true` client's
  SSE stream is rebuilt from the repaired values; untouched streams pass through.

Schema-level validation (does `arguments` match the target function's shape) is
out of scope: the gateway does not know the tool schema. This guard only ensures
`arguments` is *well-formed JSON*.


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
