# Analysis toolkit

The recorder writes a JSONL history of every exchange (`records.jsonl` plus
timestamp-suffixed rotated files). The analysis toolkit reads that history back
and answers the recurring questions — why a run is stuck, what the endpoint is
doing, which remediation fired, and what actually happened in the conversation
— without ad-hoc `grep`/`jq`/one-liner scripting.

The implementation lives in `source/app/analysis/`; the CLI wrapper is
`scripts/analyse-records`.

## Stages

A pipeline is a quoted string of `|`-separated stages:

```
read | filter … | project … | digest …
```

- **`read`** — stream records from the rotated files *and* the live
  `records.jsonl`, merged in `timestamp` order. Corrupt, blank and zero-filled
  lines (a hard kill mid-write) are skipped; a truncated final line does not
  abort the read. `read <path>` selects a different recording;
  `read include_rotated=false` reads only the live file.
- **`filter`** — keep records matching every criterion. Repeated keys OR their
  values (`outcome=upstream_failure,outcome=http_error`); different keys are
  ANDed. Supported keys:
  `outcome`, `status`, `attempt`, `error` (transport error type/message,
  substring), `reason` (`diagnosis.reason`, substring), `abort_kind`, `path`,
  `query`, `since`/`until` (unix timestamp), `rung`, `has_tool_calls`,
  `request_id`, `kind`, `retryable`. Paths and queries are glob-matched when
  the pattern contains `*`, `?` or `[`, and compared exactly otherwise.
- **`project`** — reduce each record to the listed fields. Fields are dotted
  paths (`diagnosis.reason`); `alias=path` renames. Missing paths are emitted
  as `null`.
- **`digest`** — one of the four ready-made summaries below. A digest must be
  the last stage. Options are `key=value` pairs, e.g.
  `digest narrative max_text=1000`.

Output is JSON (`--format json`, pretty by default, `--compact` for one line)
or JSONL (`--format jsonl`, one record per line). Nothing is prose.

## The four digests

| Digest | Answers | Summary |
| --- | --- | --- |
| `stuck` | Why is this run stuck? | Abort timeline (`abort_kind`/`abort_stream`/reason/details/partial sizes), per-`request_id` abort, resubmission and retry counts, `stall_false_positives` (a stall abort that still produced partial output), and `retry_storms` (requests with retries). |
| `endpoint` | What problems talking to the LLM endpoint? | Transport errors by `error.type` and `diagnosis.reason`, HTTP errors by status and reason, context-window counts, retryable/non-retryable split, and the per-request retry distribution. |
| `remediation` | Which errors occurred and how were they remedied? | Per request: errors (with diagnosis), the pass records that fired (rung, attempt, discarded/re-submitted bodies) and the final ladder summary (`rounds`, `remediation_seconds`, cap trip, rung outcomes). Top level: error-reason counts, rung counts, `cap_trips` and `exhausted_rungs`. |
| `narrative` | What happened in domain terms? | Per request: the last user question, the ordered tool/MCP calls **for that turn** joined to their results, the reconstructed turn, and a prose-ish `summary`. Record-level events (errors/passes/aborts/final) are in `timeline`. `max_text` truncates long strings (default 500); `include_history=true` reconstructs the full message history from `request_body` instead of just the current turn. |

`stuck`, `endpoint` and `remediation` list only requests with relevant
activity (an abort, retry, error, pass or cap trip) by default — their
`totals` still cover every request. Pass `include_idle=true` in the digest
stage to list every request, e.g. `digest endpoint include_idle=true`.

## Disambiguating `outcome`

`outcome` is overloaded (see [record-schema.md](record-schema.md)). The toolkit
classifies records into a `kind`:

| Kind | Meaning |
| --- | --- |
| `attempt` | One raw upstream attempt (`success`/`http_error`/`upstream_failure`/`context_window_exceeded`). |
| `abort` | A content watchdog abort (has `abort_kind`). |
| `pass` | A remediation-pass record: `response_body` is `null` and a nested rung block says `triggered`. |
| `final` | The terminal `success` that delivers a body and carries the ladder summary (`rounds`/`remediation_seconds` or nested rung summaries). |
| `unknown` | Anything else. |

Use `filter kind=abort` (not `outcome=loop_aborted`) to select aborts, and
`kind=pass` to select remediation passes. `kind=error` is a shortcut for the
three error attempt outcomes; `kind=success` selects any success.

## Examples

```bash
# Why is the latest run stuck?
scripts/analyse-records 'read | digest stuck'

# Endpoint health, only the failures
scripts/analyse-records 'read | filter kind=error | digest endpoint'

# What remediation was tried for one request, and did it finish?
scripts/analyse-records 'read | filter request_id=2c4dd7926c4c4e4fa9f1a90032185bdb | digest remediation'

# The ordered tool/MCP calls of the last narrative
scripts/analyse-records 'read | digest narrative' --format jsonl

# A small projection instead of whole bodies
scripts/analyse-records 'read | filter abort_kind=stalled | project id,request_id,abort_reason,partial_content_chars'
```

Choose a recording with `--records <path/to/records.jsonl>` (rotated siblings
are discovered automatically), `--data-dir <dir>`, or the `DATA_DIR` /
`RECORD_FILE` environment variables.

## Python API

The same stages compose in Python:

```python
from app.analysis import Digest, Filter, Project, Read, read_records

records = read_records("data/records.jsonl")
errors = list(Filter(outcome="http_error").apply(records))
summary = (Read("data/records.jsonl") | Filter(kind="abort") | Digest("stuck")).run()
```

`run_digest(name, records, **options)` runs a digest directly. `record_kind`,
`is_abort`, `is_pass`, `is_final`, `is_attempt`, `pass_rung` and `tool_calls`
expose the classification helpers.
