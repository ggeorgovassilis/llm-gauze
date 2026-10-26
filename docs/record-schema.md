# Recorder record schema

This document describes the stabilised shape of the records written to
`records.jsonl`. The recorder itself (`source/app/recorder.py`) is
schema-agnostic: `Recorder.record()` appends whatever dict it is given, adding
only `id` and `timestamp`. The schema is imposed entirely by the proxy
(`source/app/proxy.py`), the sole production writer.

Every record shares a `base_entry` prefix, then carries a stable `outcome`
discriminator and a per-outcome tail. A consumer should route a record by its
`outcome` field rather than by key-sniffing which optional keys happen to be
present.

## Base fields

Every record carries:

| Field | Meaning |
| --- | --- |
| `id` | Stable unique id, assigned by the recorder. |
| `timestamp` | Unix time the record was written, assigned by the recorder. |
| `request_id` | The request's id (present on the buffered path and on records written by callers that pass a `base_entry`). |
| `method` / `path` / `query` | The inbound request method, path and query string. |
| `request_headers` | The inbound request headers (buffered path); `Authorization` and `Proxy-Authorization` values are replaced with `[redacted]`. |
| `request_body` | The request body actually forwarded upstream (decoded to a string). Request-side prompt trimming and message-overflow rewriting run before recording, so this is the post-trim body; `prompt_trim`'s `chars_before` values are the record-level evidence of what was removed. |
| `attempt` / `max_attempts` | Which upstream attempt this record describes, and the configured retry budget. |
| `duration` | Seconds spent on this attempt. |
| `outcome` | The stable discriminator (see below). |

Fields that do not apply to an outcome are omitted, or set to `null`.

## The `outcome` discriminator

Every record carries a top-level `outcome` drawn from the same vocabulary the
telemetry layer already emits as the `outcome` label on `requests_total`:

* `success` — a body was delivered to the client.
* `http_error` — the upstream answered with a 4xx/5xx that was passed through.
* `upstream_failure` — a transport error, deadline, or a failure with no usable
  status (see [Attempt-failure fields](#attempt-failure-fields)).
* `loop_aborted` — a loop verdict aborted the request, or a loop-retry
  re-submission was triggered.
* `stalled` — a stall verdict aborted the request.
* `runaway_reasoning_aborted` — a runaway-reasoning verdict aborted the request,
  or a runaway re-submission was triggered.
* `context_window_exceeded` — the request exceeded the model's context window.

This lets a consumer route a record by `outcome` without knowing which module
wrote it or which optional keys it carries.

## `response_body` has one meaning

`response_body` is always **the body actually delivered to the client**, or
`null` when nothing was delivered. Concretely:

* Final deliveries record the decoded body.
* Retried attempts, transport errors, aborts and remediation passes record
  `null`, because that attempt's body was never delivered (it was discarded,
  failed, or re-submitted).

The pre-remediation turn that an extraction, coast or empty-stop
re-submission discards is **not** written to `response_body`; it lives in
`pre_remediation_body` (a string, decoded from the reconstructed turn) so both
bodies stay distinguishable.

## Change records

A *change record* is a small dict that a guard or the proxy attaches to a
record to report a mutation it made to the exchange. Change records are carried
in four lists — `prompt_trim`, `message_overflow`, `think_cleanup` and
`tool_repair` — and each is `null` when that stage ran but nothing changed.
Some success records omit a list key entirely when the stage never ran (for
example a remediation-pass record that never reaches tool repair). Every
change record has the same shape:

* `kind` — **mandatory** on every change record; the discriminator listed in
  the table below.
* `index` — optional; the zero-based position of the affected item (a message
  index for request-side rewrites, a tool-call index for response-side
  repairs).
* `role` — optional; the role of the affected message, for emitters that
  record one.
* Every other field is kind-specific and must be listed for that kind in the
  table below. `index` and `role` are permitted on any kind and are therefore
  not repeated in the table. Consumers may rely on a kind carrying no
  undocumented fields.

Emit a change record as a dict literal (`{"kind": "..."}`) or a
`dict(kind="...")` call with a literal `kind` value. The schema guard
(`tests/test_change_records.py`) discovers emitters by scanning the guards and
the proxy source for those two forms, so a record assembled through a helper, a
`TypedDict` constructor, or any other call form is invisible to that scan and
must be added to the guard's emitter allowlist explicitly.

All size fields are **character** counts — never bytes or tokens. `size` is a
measurement compared against a threshold; a record that rewrites content
instead reports `chars_before` / `chars_after`, the length of the affected
content before and after the rewrite. (Request-side prompt trimming uses the
same pair.)

| `kind` | Emitted by | Fields besides `kind` | Meaning |
| --- | --- | --- | --- |
| `prompt_trim` | `PromptTrimTransform` (`remediation/prompt_trim.py`) | `rule`, `action`, `chars_before`, `chars_after` | A prompt-trim rule matched and rewrote (or, in dry-run, would rewrite) a message. `rule` is the rule id, `action` one of `drop`/`replace`/`sub`/`truncate`; the character counts describe the message content before and after that application. |
| `tool_overflow` | `MessageOverflowGuard` (`remediation/overflow.py`) | `size`, `truncated` | A `role: "tool"` message exceeded `MESSAGE_OVERFLOW_THRESHOLD`. `size` is its original character count; `truncated` says whether it was cut to a bounded prefix. |
| `relocated_think` | `ThinkContentGuard.relocate` (`remediation/think.py`) | `chars_before`, `chars_after`, `blocks` | Leaked thinking text was moved out of visible `content` into `reasoning`. The character counts describe `content`; `blocks` is the number of relocated blocks. |
| `empty_content_placeholder` | `ThinkContentGuard.guard_empty` (`remediation/think.py`) and the finalising proxy (`proxy.py`) | — | Visible content was empty, so the configured placeholder was substituted. |
| `repaired_arguments` | `ToolCallGuard.validate` (`remediation/tool_call.py`) | — | A tool call's `function.arguments` was repaired into well-formed JSON. |
| `flagged_malformed` | `ToolCallGuard.validate` (`remediation/tool_call.py`) | `reason` | A tool call was left unchanged because its arguments could not be repaired. `reason` is one of `not_an_object`, `missing_function`, `missing_name`, `arguments_not_string`, `unfixable_arguments`. |

## Attempt-failure fields

An attempt that ends in `upstream_failure` carries two parallel views of *why*:
a raw `error` and a classified `diagnosis`. Both are optional (a successful
attempt has neither; `http_error` records have a diagnosis for the status).

`error` is the raw failure, populated only when an exception escaped the
exchange:

| Field | Meaning |
| --- | --- |
| `error.type` | Stable label for the failure. Wall-clock deadline errors use their reason (`header_timeout`, `total_deadline`); everything else uses the exception's class name (`ReadTimeout`, `ConnectError`, `RemoteProtocolError`, …). |
| `error.message` | The exception's message. |
| `error.traceback` | Formatted traceback, for post-hoc forensics. |

`diagnosis` is the retry classifier's verdict on that failure (or on a
non-2xx status):

| Field | Meaning |
| --- | --- |
| `diagnosis.retryable` | Whether the failure is worth another attempt. |
| `diagnosis.reason` | Human-readable reason. Deadline failures use the distinct reasons `header_timeout` and `total_deadline`; transient exceptions read `transient upstream error <Class>: <message>`; statuses read `retryable HTTP status <code>` / `non-retryable HTTP status <code>`. |
| `diagnosis.code` | Present only for special classifications (e.g. `context_window_exceeded`). |

Because `error.type` and `diagnosis.reason` are distinct per failure mode, an
operator can tell a hung header phase (`header_timeout`) from a stream that
outlived its total budget (`total_deadline`) without reading tracebacks. The
same labels are emitted as the `type` label on `upstream_errors_total`.

## Remediation-pass records

The five re-submission rungs each write one record with their own nested block.
There are **two `outcome` keys**, which must not be confused:

* The **top-level `outcome`** is the stable discriminator described above.
* The **nested `*.outcome`** describes the nested step's own result and is
  always `"triggered"` on a pass record (e.g. `extract.outcome`,
  `coast.outcome`, `empty_stop.outcome`, `runaway.outcome`,
  `loop_retry.outcome`).

The top-level `outcome` differs across the five rungs because the vocabulary
has no extract/coast/empty-stop term:

* `extract` / `coast` / `empty_stop` passes stamp `outcome: "success"` — the
  turn was empty and the re-submission is the remediation, not an abort.
* `runaway` passes stamp `outcome: "runaway_reasoning_aborted"`.
* `loop_retry` passes stamp `outcome: "loop_aborted"`.

The nested block on a remediation pass also carries the exact re-submitted
request body for auditability (`runaway.resubmitted_body`,
`loop_retry.resubmitted_body`) or the pass counters
(`extract`/`coast`/`empty_stop`).

## Remediation summary on the final success record

When the ladder finalises a turn (`outcome: "success"`), the record carries a
whole-ladder summary in addition to the per-rung nested blocks:

| Field | Meaning |
| --- | --- |
| `rounds` | Total re-submissions fired across all rungs for this request. |
| `remediation_seconds` | Wall-clock seconds spent in the remediation phase, rounded to milliseconds. |
| `remediation_exhausted` | Which whole-ladder cap tripped — `"rounds"` or `"time"` — or `null` when the ladder stopped because no rung fired. |

These mirror the two per-request caps (`max_remediation_rounds` /
`max_remediation_seconds`) and make cap trips observable on the record,
alongside the `remediation_rounds_exhausted_total` /
`remediation_time_exhausted_total` telemetry counters.

## Abort records

A verdict that terminates the request (loop, stall, runaway) records the
`outcome` from the verdict routing registry (`source/app/remediation/verdicts.py`)
together with `abort_kind` (`loop`, `stalled`, `runaway_reasoning`),
`abort_reason`, and `abort_details`. Whatever the model produced before the
abort is preserved — separately from the (null) `response_body` — in the
`partial_*` fields:

* `partial_content` — the accumulated visible content.
* `partial_reasoning` — the accumulated reasoning stream.
* `partial_tool_calls` — any accumulated tool-call fragments.

These are diagnosis aids, not delivered output.
