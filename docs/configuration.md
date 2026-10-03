# Configuration

llm-gauze is configured entirely through environment variables (read from
`.env`). Copy `.env.example` to `.env` and adjust. Every setting is listed
below, grouped by feature, with its default.

> Defaults below are the code defaults in `source/app/config.py`. `.env.example`
> ships recommended working values and may differ in places (for example
> `RETRY_MAX_ATTEMPTS=8` there vs `3` here).

## Upstream & gateway

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_BASE_URL` | `http://host.docker.internal:14434` | Upstream local LLM (OpenAI-compatible API). |
| `PORT` | `9317` | Host port the gateway is published on. This is a compose `ports:` mapping only — the gateway always listens on port `8000` *inside* the container. |
| `REQUEST_TIMEOUT` | `300` | Upstream read/write timeout in seconds once connected — generation can be slow. |
| `CONNECT_TIMEOUT` | `10` | Upstream connection timeout in seconds; kept short so a dead/blackholed endpoint fails fast. |
| `DATA_DIR` | `/data` | Directory where exchanges are recorded. |
| `RECORD_FILE` | `records.jsonl` | Recording filename (appended, rotated). |
| `UID` / `GID` | `1000` | Host user the container runs as, so the bind-mounted `data/` stays writable and owned by you. |

## Retries / backoff

Transient upstream failures are retried with exponential backoff.

| Variable | Default | Description |
| --- | --- | --- |
| `RETRY_MAX_ATTEMPTS` | `3` | Max attempts per request (`1` = no retry). |
| `RETRY_BACKOFF_INITIAL` | `0.5` | Initial backoff delay, in seconds. |
| `RETRY_BACKOFF_BASE` | `2.0` | Exponential base: `delay = initial * base^(attempt-1)`. |
| `RETRY_BACKOFF_MAX` | `30.0` | Upper bound on any single backoff delay, in seconds. |
| `RETRY_BACKOFF_JITTER` | `true` | Apply full jitter to the computed delay. |
| `RETRYABLE_STATUS_CODES` | `408,429,500,502,503,504` | Comma-separated HTTP statuses that warrant a retry. |

## Loop detection

Chat completions are streamed from upstream and observed for repetitive output.

| Variable | Default | Description |
| --- | --- | --- |
| `LOOP_DETECTION_ENABLED` | `true` | Master switch for loop detection. |
| `LOOP_WINDOW_BYTES` | `32768` | Sliding output window (bytes) over which repetition is measured. |
| `LOOP_MIN_OUTPUT_FRACTION` | `0.25` | Fraction of the window that must be produced before detection arms. |
| `LOOP_COMPRESSION_RATIO` | `0.15` | Compression ratio below which the window is considered low-entropy (repetitive). |
| `LOOP_ABORT_STATUS` | `502` | HTTP status returned when a loop is detected and not remediated. |

## Loop remediation

A detected loop is first re-submitted with varied sampling before aborting.

| Variable | Default | Description |
| --- | --- | --- |
| `LOOP_RETRY_ENABLED` | `true` | Re-submit a looped request with varied sampling instead of aborting. |
| `LOOP_RETRY_MAX_ATTEMPTS` | `5` | Maximum loop re-submissions before giving up. |
| `LOOP_RETRY_INCREMENT` | `0.1` | Amount added to a sampling parameter the client already submitted. |
| `LOOP_RETRY_TEMPERATURE` | `1.2` | Fallback temperature when the client submitted none. |
| `LOOP_RETRY_REPEAT_PENALTY` | `1.2` | Fallback llama.cpp repeat penalty when the client submitted none. |
| `LOOP_RETRY_PRESENCE_PENALTY` | `0.3` | Fallback OpenAI presence penalty when the client submitted none. |
| `LOOP_RETRY_FREQUENCY_PENALTY` | `0.3` | Fallback OpenAI frequency penalty when the client submitted none. |

## Stall detection

A stream that returns headers but stops producing content tokens is aborted;
timers reset only on real tokens, never on SSE keepalives.

| Variable | Default | Description |
| --- | --- | --- |
| `STALL_DETECTION_ENABLED` | `true` | Master switch for stall detection. |
| `STALL_TTFT_SECONDS` | `120` | Max seconds to wait for the first content-bearing token. |
| `STALL_GAP_SECONDS` | `60` | Max seconds between content-bearing tokens once started. |
| `STALL_ABORT_STATUS` | `502` | HTTP status returned when a stall is detected. |

## Context-window overflow

The upstream's context-window-fill error is matched by exact substring and
failed fast instead of being retried; the upstream's own error is forwarded
verbatim.

| Variable | Default | Description |
| --- | --- | --- |
| `CONTEXT_WINDOW_DETECTION_ENABLED` | `true` | Master switch for context-window detection. |
| `CONTEXT_WINDOW_MARKERS` | *(see `.env.example`)* | Comma-separated substrings (case-insensitive) signalling a full context window. |
| `CONTEXT_WINDOW_ABORT_STATUS` | `413` | HTTP status returned when an overflow is recognised but no upstream body is available to pass through. |

## Think-tag cleanup

Leaked `<think>`/`<reasoning>` tags are relocated out of the visible content
into `reasoning_content` — never discarded — and a non-empty placeholder is
emitted when the model produced only a thinking tag.

| Variable | Default | Description |
| --- | --- | --- |
| `THINK_CLEANUP_ENABLED` | `true` | Master switch for think-tag cleanup. |
| `THINK_TAGS` | `think,thinking,reasoning` | Comma-separated tag names (case-insensitive) whose inner text is relocated into `reasoning_content`. |
| `THINK_EMPTY_RESPONSE_PLACEHOLDER` | `The model replied inside a thinking tag; see reasoning_content.` | Placeholder emitted as visible content when the model produced only a thinking tag. |

## Nudge (re-prompt empty turns)

A turn that produced only reasoning (`finish_reason: stop`, no content, no tool
calls) is re-submitted with a short re-prompt before falling back to the
placeholder.

| Variable | Default | Description |
| --- | --- | --- |
| `THINK_NUDGE_ENABLED` | `true` | Master switch for the nudge. |
| `THINK_NUDGE_TEXT` | *(see `.env.example`)* | Nudge text appended as a `user` message. |
| `THINK_NUDGE_MAX_ATTEMPTS` | `2` | Max re-submissions before the placeholder floor. |

## Coast detection

A turn that produced non-empty visible content but no tool call — even though a
tool call was possible and the model's reasoning collapsed to be identical to
its content — is re-submitted with a re-prompt.

| Variable | Default | Description |
| --- | --- | --- |
| `COAST_DETECTION_ENABLED` | `true` | Master switch for coast detection. |
| `COAST_NUDGE_TEXT` | *(see `.env.example`)* | Re-prompt text appended as a `user` message. |
| `COAST_MAX_ATTEMPTS` | `2` | Max coast re-submissions before returning the coasted turn as-is. |

## Runaway-reasoning detection

A turn that keeps emitting reasoning tokens while never producing content is
flagged (proactive token budget or exhausted output window) and re-prompted to
stop thinking.

| Variable | Default | Description |
| --- | --- | --- |
| `RUNAWAY_REASONING_ENABLED` | `true` | Master switch for runaway-reasoning detection. |
| `RUNAWAY_REASONING_TOKEN_THRESHOLD` | `2000` | Reasoning-token budget (~4 chars/token) above which a turn is flagged, provided no content has appeared. |
| `RUNAWAY_REASONING_NUDGE_TEXT` | *(see `.env.example`)* | Re-prompt text appended as a `user` message. |
| `RUNAWAY_REASONING_MAX_ATTEMPTS` | `2` | Max runaway re-submissions before aborting. |
| `RUNAWAY_REASONING_ABORT_STATUS` | `502` | HTTP status returned when a runaway turn is detected and every re-submission also ran away. |

## Message overflow

Oversized `role: "tool"` results are warned and optionally truncated before the
request is forwarded upstream.

| Variable | Default | Description |
| --- | --- | --- |
| `MESSAGE_OVERFLOW_ENABLED` | `true` | Master switch for message-overflow protection. |
| `MESSAGE_OVERFLOW_THRESHOLD` | `4096` | Size threshold, in characters, above which a tool-result message is treated as oversized. |
| `MESSAGE_OVERFLOW_TRUNCATE` | `true` | Truncate oversized tool results to a bounded prefix. |
| `MESSAGE_OVERFLOW_WARNING` | *(see `.env.example`)* | Warning prepended to an oversized tool result. |

## Tool-call syntax enforcement

Assembled tool calls are validated; truncated `function.arguments` are repaired
where deterministic, the rest are flagged without crashing the request.

| Variable | Default | Description |
| --- | --- | --- |
| `TOOL_CALL_GUARD_ENABLED` | `true` | Master switch for tool-call validation/repair. |
