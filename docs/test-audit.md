# Test suite audit

Scope: [#39](https://github.com/ggeorgovassilis/llm-gauze/issues/39). A holistic
review of the test suite across coverage, duplication, efficiency, unit-vs-
integration balance, and missing edge cases. The deliverable is this document:
findings, their risks, and prioritised recommendations, where each actionable
item has been filed as its own ticket.

## Baseline

Coverage tooling was added as the ticket's prerequisite (`pytest-cov` + a
`fail_under` floor in `pyproject.toml`). The suite is green with **89% line
coverage** (1157 statements, 126 missed, 117 tests, ~20s).

| Module | Line cov. | Notes |
| --- | --- | --- |
| `main.py` | **0%** | No tests for the gateway endpoints |
| `retry.py` | **72%** | `ExponentialBackoff.delay` and `diagnose_exception` untested |
| `proxy.py` | 85% | Streaming path covered; buffered path and helpers not |
| `base.py` | 97% | `Diagnosis.__str__` only |
| `telemetry.py` | 99% | one Prometheus render branch |
| `context.py` | 93% | two lines |
| `overflow.py` | 93% | warn-only / `truncate=False` path |
| `loop.py` | 95% | two lines |
| `tool_call.py` | 91% | six lines |
| `config.py`, `recorder.py`, `coast.py`, `loop_retry.py`, `nudge.py`, `runaway.py`, `stall.py`, `think.py`, `__init__.py` | 100% | |

## Findings

### F1 — Gateway endpoints untested (high risk)

`main.py` (the FastAPI app, `/health` and `/metrics`) has **0% coverage**. The
metrics exposition (both the JSON snapshot and the Prometheus text renderer) is
the gateway's observability surface; a regression here breaks monitoring
silently and nothing in the suite would notice. No live upstream is needed to
test it — the FastAPI test client is sufficient.

### F2 — Retry pacing and failure classification untested (high risk)

`retry.py` sits at 72%. `ExponentialBackoff.delay()` is **never called** by any
test: the exponential growth, the `maximum` cap, and the jitter range are all
unasserted. `RetryableDetector.diagnose_exception()` (the retryable-vs-
non-retryable classification that decides whether *any* failure is retried) is
untested. This is the core of the "retry transient failures" behaviour — a bug
here causes retry storms or silently swallows transient failures. The gap
exists because the buffered `forward()` path (which invokes `backoff.delay`)
has no test (see F3).

### F3 — Buffered (non-streaming) proxy path untested (high risk)

Only `_forward_streaming` is exercised. The buffered `forward()` path — the
retry loop, transport-error handling, the upstream-failure 502, and HTTP-error
passthrough for every *non*-chat-completion route — has no direct test. The same
gap leaves these pure helpers uncovered: `_decode`, `_diagnosis_entry`,
`_is_chat_completion`, `_ensure_stream`, `_filtered_headers`,
`_abort_error_response`, and the synthesised branch of
`_context_window_error_response`. `_abort_error_response` maps a detected
failure to a client-facing status; an untested mapping there is exactly the
"silently mangle the response" class of risk this audit was asked to find.

### F4 — Integration scaffolding duplicated across nine files (medium risk)

Nine files (`test_coast.py`, `test_context_window.py`, `test_loop_integration.py`,
`test_nudge.py`, `test_overflow.py`, `test_runaway.py`, `test_stall_integration.py`,
`test_think_cleanup.py`, `test_tool_call.py`) each re-implement the same
scaffolding: a `_chunk` helper, a `_MockHandler`/`_MockServer` pair over
`http.server`, and a `_run_forward` driver. Roughly 200 lines of near-identical
mock-upstream code drift apart, and there is **no `conftest.py`** and no shared
fixture anywhere in `tests/`.

### F5 — Global `settings` mutated with no teardown (medium risk)

Every integration test assigns fields on the process-wide `settings` singleton
(`settings.llm_base_url = …`, `settings.loop_window_bytes = …`, …) and never
restores them. The suite is **order-dependent**: a value changed by one test
leaks into the next, so a single-file run can pass while the full run fails (or
vice-versa), and the suite cannot be parallelised. There is no autouse fixture
or `monkeypatch` anywhere to restore state.

### F6 — Small uncovered lines (low risk)

- `base.py:30` — `Diagnosis.__str__` (trivial, used only in logging).
- `telemetry.py:147` — one Prometheus rendering branch.
- `overflow.py:57,60,66` — the warn-only / `truncate=False` path.
- `context.py:36,77`, `loop.py:65,103`, `tool_call.py:39,64,66,81-83,88,132`.

### F7 — No branch coverage (info)

Only line coverage is measured. Branch coverage (`branch = true` in
`[tool.coverage.run]`) would catch partially-tested conditionals (e.g. the
jitter on/off, truncate on/off) that line coverage treats as covered.

### F8 — Unit/integration split applied inconsistently (info)

Only `loop` and `stall` use the `test_X.py` (unit) / `test_X_integration.py`
convention. Seven other files (`nudge`, `coast`, `runaway`, `overflow`,
`context_window`, `think_cleanup`, `tool_call`) bundle their end-to-end tests
into the same file as the unit tests, so the fast/slow split is not
discoverable from filenames.

## Risks

| Finding | Risk |
| --- | --- |
| F1 | Observability regressions ship unnoticed |
| F2 | Retry storms / transient failures not retried |
| F3 | Non-chat routes return wrong status/body; abort-status mapping unverified |
| F4 | Mock drift ⇒ tests assert against different upstreams |
| F5 | Flaky, order-dependent suite; blocks `pytest-xdist` |
| F6/F7 | Trivial, but deferring them entrenches low coverage |
| F8 | Harder to run/read the fast unit tests vs slow integration tests |

## Recommendations (prioritised)

Prioritisation follows the ticket's rule: remediation/transport code that can
silently drop or mangle upstream traffic ranks above pure helper/formatting
concerns.

| # | Recommendation | Milestone | Ticket |
| --- | --- | --- | --- |
| R1 | Test `main.py` health + metrics (JSON and Prometheus) | current | [#53](https://github.com/ggeorgovassilis/llm-gauze/issues/53) |
| R2 | Unit-test `retry.py` (`ExponentialBackoff.delay`, `RetryableDetector`) | current | [#54](https://github.com/ggeorgovassilis/llm-gauze/issues/54) |
| R3 | Extract shared mock-upstream fixture + isolate `settings` (F4/F5) | current | [#55](https://github.com/ggeorgovassilis/llm-gauze/issues/55) |
| R4 | Cover the buffered `forward()` path and its helpers (F3) | backlog | [#56](https://github.com/ggeorgovassilis/llm-gauze/issues/56) |
| R5 | Close remaining gaps + enable branch coverage (F6/F7/F8) | backlog | [#57](https://github.com/ggeorgovassilis/llm-gauze/issues/57) |

**Efficiency note:** the suite's ~20s runtime is acceptable; the dominant cost
is scaffolding duplication (F4), not test logic. No test is meaningfully slow on
its own. The settings mutation (F5) is what blocks parallelisation, not runtime.
