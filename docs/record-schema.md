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
| `request_body` | The client's original request body (decoded to a string). |
| `attempt` / `max_attempts` | Which upstream attempt this record describes, and the configured retry budget. |
| `duration` | Seconds spent on this attempt. |
| `outcome` | The stable discriminator (see below). |

Fields that do not apply to an outcome are omitted, or set to `null`.

## The `outcome` discriminator

Every record carries a top-level `outcome` drawn from the same vocabulary the
telemetry layer already emits as the `outcome` label on `requests_total`:

* `success` — a body was delivered to the client.
* `http_error` — the upstream answered with a 4xx/5xx that was passed through.
* `upstream_failure` — a transport error, or a failure with no usable status.
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

The pre-remediation turn that a nudge or coast re-submission discards is **not**
written to `response_body`; it lives in `pre_remediation_body` (a string,
decoded from the reconstructed turn) so both bodies stay distinguishable.

## Remediation-pass records

The four re-submission rungs each write one record with their own nested block.
There are **two `outcome` keys**, which must not be confused:

* The **top-level `outcome`** is the stable discriminator described above.
* The **nested `*.outcome`** describes the nested step's own result and is
  always `"triggered"` on a pass record (e.g. `nudge.outcome`,
  `coast.outcome`, `runaway.outcome`, `loop_retry.outcome`).

The top-level `outcome` differs across the four rungs because the vocabulary
has no nudge/coast term:

* `nudge` / `coast` passes stamp `outcome: "success"` — the turn was empty and
  the re-submission is the remediation, not an abort.
* `runaway` passes stamp `outcome: "runaway_reasoning_aborted"`.
* `loop_retry` passes stamp `outcome: "loop_aborted"`.

The nested block on a remediation pass also carries the exact re-submitted
request body for auditability (`runaway.resubmitted_body`,
`loop_retry.resubmitted_body`) or the pass counters (`nudge`/`coast`).

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
