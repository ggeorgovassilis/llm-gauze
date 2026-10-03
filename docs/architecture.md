# Architecture

llm-gauze is an HTTP gateway layered in front of a local LLM. This document
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
- `app/recorder.py` — append-only JSONL recording of exchanges/attempts. The
  stabilised record field contract is documented in
  [`record-schema.md`](record-schema.md).
- `app/remediation/` — the pluggable remediation pipeline (below).

## Separation of concerns

The remediation pipeline deliberately separates three concerns so new
capabilities can be added without rearchitecting:

1. **Data collection** — `app/recorder.py` persists every exchange/attempt.
2. **Understanding** — `Detector` classes classify a failure into a
   `Diagnosis` (retryable or not, and why).
3. **Action** — `Backoff`/`RetryPolicy` classes turn a `Diagnosis` into
   behaviour (e.g. retry with exponential backoff), and `Remediation` steps
   re-submit a streamed turn that failed; `Transform` classes rewrite content
   or a request body in place.

```
failure ──▶ Detector (understanding) ──▶ Diagnosis ──▶ Backoff/RetryPolicy (action)
```

- `app/remediation/base.py` — abstract `Detector`, `Backoff`, `Diagnosis`,
  `RetryPolicy`, plus `ContentWatchdog`/`StreamVerdict` for content streams,
  and the `Turn` value object with the `Remediation`/`Transform` protocols
  for the composable ladder.

### Remediation ladder

The remediation of a streamed turn is expressed as an **ordered, composable
ladder** of `Remediation` steps rather than a hard-coded `if`-chain. Each step
implements the one uniform protocol in `base.py`:

- `applies(turn, request_body) -> bool` — the step's trigger, a pure predicate
  over the assembled `Turn` (finish reason, content, reasoning, tool calls,
  and any abort verdict) plus the request body it came from.
- `apply(turn, request_body) -> dict` — returns a **copy** of the request body
  prepared for re-submission; it never mutates the input.
- `name` (a short id used for recording/telemetry) and `max_attempts` (the
  step's own re-submission budget) are declared as class attributes.

The ladder is assembled from enabled settings in `_build_remediation_ladder`
as a plain ordered list — **runaway → loop_retry → nudge → coast** — and a
disabled rung is simply left out. After each exchange the proxy walks the
ladder in order and re-submits via the first rung whose `applies` fires; when
no rung fires the outcome is final. A new remediation is therefore "implement
and register" (append a `Remediation` subclass to the ladder), not "edit the
loop". `Transform` classes (`ThinkContentGuard`, `ToolCallGuard`,
`MessageOverflowGuard`) share a sibling protocol: a canonical `apply` that
rewrites a value and reports its mutations, never re-submitting.
- `app/remediation/retry.py` — `RetryableDetector` (transient exceptions +
  retryable HTTP statuses) and `ExponentialBackoff` (with optional jitter).
- `app/remediation/loop.py` — `ThinkingLoopDetector`, a stateful
  `ContentWatchdog` that flags repetitive output via byte-window compression
  entropy.
- `app/remediation/stall.py` — `StallDetector`, a time-based watchdog that
  flags a stream which has stopped producing content tokens (silent hang).
- `app/remediation/context.py` — `ContextWindowDetector`, which recognises the
  upstream's context-window-fill error and reclassifies it non-retryable.
- `app/remediation/think.py` — `ThinkContentGuard`, a *transform* (not a
  `Detector`) that relocates leaked thinking tags out of visible content and
  guarantees a non-empty reply.
- `app/remediation/overflow.py` — `MessageOverflowGuard`, a request-side
  *transform* that warns/truncates oversized tool results (distinct from
  `context.py`'s response-side `ContextWindowDetector`).
- `app/remediation/nudge.py` — `NudgePolicy`, re-prompt a turn that finished
  empty (reasoning only, no content, no tool calls).
- `app/remediation/coast.py` — `CoastPolicy`, re-prompt a turn that announced
  a tool call but did not make one.
- `app/remediation/runaway.py` — `RunawayReasoningDetector` /
  `RunawayReasoningPolicy`, flag and re-prompt a turn that reasons endlessly
  without ever producing content.
- `app/remediation/tool_call.py` — `ToolCallGuard`, a *transform* that
  validates/repairs/flags malformed tool-call JSON.
- `app/remediation/loop_retry.py` — `LoopRetryPolicy`, re-submit a looped
  request with varied sampling (behaviour described under Loop detection).
- `app/remediation/verdicts.py` — `VerdictRoute` / `VERDICT_ROUTES` /
  `route_for`, the single registry that maps each `VerdictKind` to its abort
  response (message, HTTP status) and telemetry (outcome label, abort counter).

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
and endpoint's domain, so llm-gauze never invents values — a sampling parameter
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

Unlike `ThinkingLoopDetector`, `StallDetector` observes time, not text. All
three content watchdogs (`ThinkingLoopDetector`, `StallDetector`,
`RunawayReasoningDetector`) implement one uniform `ContentWatchdog` contract
(`note`/`check`/`remaining`/`reset`), so the streaming pipeline drives them
identically. `StallDetector` takes an injectable monotonic clock so it is
unit-testable without sleeping.


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
- A match produces
  `Diagnosis(retryable=False, code=DiagnosisCode.CONTEXT_WINDOW_EXCEEDED)`,
  which short-circuits the retry loop. The upstream's own error is then
  forwarded **verbatim** (its status and body — token count, `n_ctx`, fix
  hints — are passed through unchanged) rather than translated into an llm-gauze
  message. Only when no upstream body is available (e.g. the overflow arrived
  as an exception message) does llm-gauze synthesise a minimal named
  `context_window_exceeded` error with `context_window_abort_status` (default
  413).
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
- `NudgePolicy.apply(turn, body)` appends `{"role": "user", "content": <nudge>}`
  to a copy of the request (the nudge text is `THINK_NUDGE_TEXT`) without
  mutating the input. Its ladder trigger is `applies(turn, body)`, which
  forwards the assembled turn to `should_nudge`.
- The re-submitted request reuses the streaming path unchanged; the budget is
  `THINK_NUDGE_MAX_ATTEMPTS`, and each nudge pass is recorded with an
  `outcome` (`triggered`, `succeeded`, or `exhausted`) so the intervention is
  visible in `records.jsonl`.
- On exhaustion the placeholder floor (`guard_empty`) is applied as before, so
  the client still receives a non-empty turn.

Extracting a *usable answer* from the relocated chain-of-thought is deliberately
out of scope here (see #13): it is not deterministically fixable, and surfacing
raw reasoning as content would be a quality regression.

## Coast detection

A sibling of the nudge rung above, but for the *opposite* shape of turn: the
model announced work and then did none. Mid-way through a multi-step tool loop,
some models end a turn with `finish_reason: stop`, non-empty visible `content`,
and no `tool_calls` — even though the request's `tools` list was non-empty and
the conversation already contains a prior assistant tool-call turn. The model
"coasted": it regurgitated its memorised status line ("Chunk 5 done … Pulling
next chunk.") without actually generating the call. Copilot's agent loop only
continues while the assistant emits tool calls, so such a turn ends the
workflow silently — no error, no crash, no user indication (see #16).

Where nudge fires on *empty* turns, coast fires on *non-empty* turns whose
chain-of-thought collapsed to be byte-identical with the visible content — the
deterministic "coasting" fingerprint observed in the incident.

- `CoastPolicy.should_nudge(finish_reason, content, tool_calls, reasoning,
  tools, messages)` is a pure predicate over the assembled turn: stop +
  non-empty content + no tool calls + non-empty `tools` list + a prior
  assistant tool-call turn + reasoning byte-identical to content. Deterministic
  — no heuristics.
- `CoastPolicy.apply(turn, body)` replays the coasted assistant message (from
  `turn.content`) into the conversation and appends the re-prompt
  (`COAST_NUDGE_TEXT`). Unlike nudge (whose empty turn left nothing in
  context), the coasted message is replayed so the re-prompt refers to
  something the model actually said. Its ladder trigger is
  `applies(turn, body)`, forwarding to `should_nudge`.
- The re-submitted request reuses the streaming path unchanged; the budget is
  `COAST_MAX_ATTEMPTS`, and each pass is recorded with an `outcome`
  (`triggered`, `succeeded`, or `exhausted`) so the intervention is visible in
  `records.jsonl`.
- On exhaustion the coasted turn is returned as-is (visible content, no tool
  call, logged outcome) rather than synthesising anything.

The master switch is `COAST_DETECTION_ENABLED`.

## Runaway-reasoning detection

Some models, instead of producing a visible answer (or a tool call), emit a
continuous stream of *reasoning* tokens that never resolves into content. This
is invisible to both other watchdogs: the reasoning is non-repeating (high
entropy), so the loop detector — which looks for low entropy / verbatim
repetition — cannot see it; and tokens are flowing, so the stall watchdog —
which looks for *silence* — cannot see it either. The turn simply runs until it
hits the output-window limit and terminates with `finish_reason: "length"` and
an empty reply (see #17).

The fingerprint is the invariant **reasoning tokens keep flowing while content
tokens stay at zero**. Two windows observe it:

1. **Proactive (streaming)** — `RunawayReasoningDetector`, a token-count
   watchdog over the reasoning stream. Once the reasoning token budget
   (`RUNAWAY_REASONING_TOKEN_THRESHOLD`, ~4 chars/token) is exceeded with no
   content or tool call yet produced, the stream is aborted early so the
   remaining output budget can be spent on a retry that actually answers. Like
   `StallDetector` it is a `ContentWatchdog` driven by the streaming pipeline
   through the uniform `note`/`check` contract.
2. **Terminal** — the output window was exhausted (`finish_reason: "length"`)
   with reasoning present but no content and no tool calls: the model burned
   its entire budget thinking.

Remediation mirrors the nudge rung: `RunawayReasoningPolicy.apply(turn, body)`
appends a stop-thinking instruction (`RUNAWAY_REASONING_NUDGE_TEXT`) and
re-submits, capped by `RUNAWAY_REASONING_MAX_ATTEMPTS`. Its ladder trigger
`applies(turn, body)` fires on a `runaway_reasoning` verdict. On exhaustion the
gateway returns `RUNAWAY_REASONING_ABORT_STATUS` (default 502).

The master switch is `RUNAWAY_REASONING_ENABLED`.

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


## Message overflow

A tool result can be arbitrarily large, and a local model has no way to know it
should summarise or skip it — it just receives a huge `role: "tool"` message
that silently eats its context window (see #15). This guard is the
*request-side* mirror of think-cleanup: before the request is forwarded
upstream, any `role: "tool"` message whose content exceeds a threshold is (a)
warned and (b) optionally truncated to a bounded prefix.

This is distinct from **context-window overflow** above, which recognises the
upstream's *response-side* "context filled" error. Message overflow runs
*request-side*, before the model ever sees the oversized tool result.

- `MessageOverflowGuard.process(body)` returns `(body, changes)`: the request
  with oversized tool results rewritten, plus a list of `tool_overflow` change
  records for the recorder. The input is never mutated; when nothing trips the
  threshold the original dict is returned unchanged.
- The trigger is a pure size comparison — character count vs
  `MESSAGE_OVERFLOW_THRESHOLD` — no fuzzy heuristics. Characters (not bytes) so
  the trigger and the truncation prefix share one unit.
- Only `role: "tool"` messages are touched: they are the one input the model
  *requested and can re-request more cheaply*, so the warning is actionable.
  `user`/`system`/`assistant` content is never modified.
- When `MESSAGE_OVERFLOW_TRUNCATE` is true, the content is truncated to a
  bounded prefix; the warning (`MESSAGE_OVERFLOW_WARNING`) is prepended either
  way.

The master switch is `MESSAGE_OVERFLOW_ENABLED`.


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
3. **Detection** — analyse recorded exchanges, classify further failures. ✅
4. **Remediation** — loop/context/message-overflow/coast/runaway detection and
   the nudge/think-cleanup/tool-repair transforms. ✅
5. **Streaming** — SSE pass-through, per-chunk handling, and rebuilt streams
   after a transform. ✅
