# Architecture audit

Scope: [#37](https://github.com/ggeorgovassilis/llm-gauze/issues/37). A holistic
review of the llm-gauze architecture across design coherence, the proxy
request/response flow and recorder integration, the remediation module
interface and its composition, extension points and coupling, and the config
surface and runtime wiring. The deliverable is this document: findings, their
risks, and prioritised recommendations, where each actionable item is listed at
the end under *Recommended follow-up tickets* for the Product Owner to triage.

The audit was written against `main` at the point the ticket was picked up. The
author read `docs/architecture.md`, `docs/configuration.md`, `docs/test-audit.md`,
`CONTRIBUTING.md`, `DEVELOPING.md`, `.github/copilot-instructions.md`, every
module under `source/app/` (including `source/app/remediation/`), and skimmed
`tests/` to confirm how modules are wired.

---

## 1. Overall design coherence and separation of concerns

### Findings

**What is good**

- The three-layer separation stated in `docs/architecture.md` — *data
  collection* (`recorder.py`), *understanding* (`Detector`), and *action*
  (`Backoff`/`RetryPolicy`) — is genuinely realised in the buffered retry path.
  It is not just documentation: `_build_retry_policy()` in `proxy.py` composes a
  `RetryableDetector` (optionally wrapped by `ContextWindowDetector`) with an
  `ExponentialBackoff` into a `RetryPolicy`, and the two layers meet only
  through the `Diagnosis` value object.
- Detection and remediation are kept apart in *concept*, even where the code
  has not yet given them distinct interfaces (see §3). Detectors classify;
  policies and guards act. The content *transforms* (`ThinkContentGuard`,
  `ToolCallGuard`, `MessageOverflowGuard`) are explicitly documented as *not*
  detectors, which prevents a category error that would otherwise be easy to
  make.
- The "no assumptions, record first" principle is respected: the recorder is
  the shared substrate, and every remediation intervention is logged so a
  mutation is never silent.
- The remediation modules are deliberately dependency-free beyond the
  project's existing `httpx` (standard library otherwise) and mostly pure, so
  the *understanding* and *action* layers are
  unit-testable without a running gateway. This is a strong, deliberate
  property.

**What is not good**

- **The "understanding" layer has no single interface.** There are effectively
  four shapes of classifier, and only two of them share a base class:
  `Detector` (`diagnose_exception`/`diagnose_status`), `StreamDetector`
  (`feed`/`flush`/`reset`, implemented by exactly one class —
  `ThinkingLoopDetector`), plus `StallDetector` and `RunawayReasoningDetector`,
  which implement *no* interface and are "driven manually by the streaming
  loop" (`note_token`/`remaining`, `note_reasoning`/`note_content`/`triggered`).
  The streaming loop in `proxy.py` therefore hard-codes each watchdog's bespoke
  API.
- **The orchestration is a single object.** `proxy.py` is ~1,300 lines and
  `_stream_once()` alone mixes SSE line reading, metadata merging, tool-call
  accumulation, feeding four detectors, stall/runaway polling, the terminal
  `finish_reason == "length"` classification, transport/status retry, and the
  context-window pass-through. This is the one place in the codebase where the
  design stops being "pluggable".
- The aspirational principle "detectors reason over history" is not yet true:
  see §2.

### Risks

- The absence of a uniform understanding interface means a new content
  watchdog must be wired into `_stream_once()` by hand, in several places, and
  will silently diverge from the existing detectors.
- The monolithic streaming loop is the highest-risk code to change; it is where
  a regression would silently drop or mangle upstream traffic.

### Recommendations

- Introduce one uniform "content watchdog" contract that `ThinkingLoopDetector`,
  `StallDetector`, and `RunawayReasoningDetector` all satisfy, and move the
  streaming loop out of `proxy.py` into a dedicated pipeline runner (§4, R1).
- Preserve the three-layer separation but make the *action* layer a first-class
  interface (§3, R2).

---

## 2. Proxy request/response flow and recorder integration

### Findings

**What is good**

- One `request_id` is minted per client request and reused across every attempt
  and every remediation re-submission (`base_entry` in `forward()`), so the
  JSONL correlates a whole logical request — original body, retries, and any
  nudge/coast/runaway/loop-retry passes — under one key.
- Both paths record: the buffered `forward()` records every attempt including
  failures, and the streaming path records aborts (with `partial_content` /
  `partial_reasoning` / `partial_tool_calls` preserved for post-hoc diagnosis)
  and the final success in `_finalize_success()`.
- Transforms report exactly what they changed (`think_cleanup`, `tool_repair`,
  `tool_overflow` change lists), so the recorder can tell a mutated response
  from an untouched one.

**What is not good**

- **`LOOP_DETECTION_ENABLED` is the real master switch for the entire streaming
  content pipeline, not just loop detection.** `forward()` enters
  `_forward_streaming()` only when `settings.loop_detection_enabled` is true
  (`proxy.py:452`). Everything that lives only in the streaming path — stall
  detection, runaway-reasoning detection, nudge, coast, think-tag cleanup, and
  tool-call guard — is therefore silently disabled when loop detection is off,
  regardless of each feature's own switch. The individual switches
  (`STALL_DETECTION_ENABLED`, `RUNAWAY_REASONING_ENABLED`, `THINK_CLEANUP_ENABLED`,
  `THINK_NUDGE_ENABLED`, `COAST_DETECTION_ENABLED`, `TOOL_CALL_GUARD_ENABLED`)
  give a misleading impression of independence. Only `MessageOverflowGuard`
  runs outside this branch and is truly independent.
- **The retry loop is duplicated.** The buffered `forward()` and the streaming
  `_stream_once()` each re-implement the attempt loop, recording, telemetry
  increments, `should_retry`, backoff, sleep, and the context-window special
  case, with subtle differences (e.g. `_stream_once` has an extra terminal
  `finish_reason == "length"` branch and an abort path). This is a maintenance
  hazard: a fix to retry behaviour in one path is easy to miss in the other.
- **The record schema is ad-hoc per outcome.** An abort is recorded with
  `abort_kind`/`abort_stream`/`abort_reason`/`abort_details`; a context-window
  overflow with `diagnosis` + `status`; a success with `think_cleanup` +
  `tool_repair` + `nudge` + `coast`; a re-submission pass with a nested
  `nudge`/`coast`/`runaway`/`loop_retry` object. The field `response_body`
  means "the final response" in `_finalize_success` but "the pre-remediation
  turn" in `_record_nudge_pass`. Any future detector that reads this history
  must learn a bespoke field map per module.
- **The recorder is write-only.** Nothing reads the JSONL back; all
  understanding happens live during streaming. This matches the stated "later
  phases reason over history" plan, but it means the recorder's value as a
  detection substrate is still unproven, and its schema (above) is not yet
  constrained by a consumer.
- **Recording is blocking file I/O on the async path.** `Recorder.record()`
  opens, writes, and closes synchronously under a `threading.Lock`, called
  directly from the `async` proxy code. Fine at local-LLM throughput, but it
  can stall the event loop under concurrency or a slow/contended volume.
- **SSE reconstruction is lossy after a transform.** When think-cleanup or
  tool-call repair changes a streamed response, `_reconstruct_sse_lines()`
  rebuilds the SSE from the cleaned values in a fixed shape, discarding the
  upstream's chunk boundaries, per-chunk ids, and any usage frames. Untouched
  streams pass through verbatim, so this is a deliberate, bounded trade-off —
  but it is not stated as an API contract, and a streaming client can observe a
  structurally different response than a non-streaming one for the same turn.

### Risks

- Disabling loop detection to silence one false positive quietly disables six
  other safety features — a misconfiguration with an outsized blast radius.
- The duplicated retry loop invites drift between the two request shapes.
- The ad-hoc record schema raises the cost of the project's own next phase
  (reasoning over history) and of any operator tooling over `records.jsonl`.

### Recommendations

- Make the streaming pipeline gate independent of loop detection (enter the
  streaming path for chat completions whenever *any* streaming feature is
  enabled) — §6, R3.
- Unify the retry loop and the record schema — §6, R4 and R6.
- Offload recorder writes off the event loop and document the blocking-write
  trade-off — §6, R7.

---

## 3. Remediation module interface, ordering, and composition

### Findings

**What is good**

- Each module is small, focused, and — where a deterministic predicate is
  possible — deliberately heuristic-free (exact tag names, exact marker
  substrings, byte-identical reasoning/content for coast, a pure size
  comparison for overflow). This is a consistent, well-documented discipline.
- The `from_settings()` classmethods make every module config-driven and
  constructible from the environment in one place.
- The empty-response ladder (`nudge → coast → runaway → loop-retry`, then the
  placeholder floor) is explicitly ordered and documented, and the
  re-submissions all reuse the same streaming path rather than branching into
  bespoke code.

**What is not good**

- **There is no remediation/transform interface.** `docs/architecture.md`
  points future capabilities at the interfaces in
  `source/app/remediation/base.py`, and the coding convention in
  `.github/agents/coder.agent.md` says new "detectors/remediation subclass the
  interfaces" there, but `base.py` only defines the *detector* side
  (`Detector`, `StreamDetector`) and the retry *action* (`Backoff`,
  `RetryPolicy`). (`DEVELOPING.md` is more precise: its "Detectors" bullet says
  detectors subclass the interfaces, and it describes remediation policies
  separately.) The re-submission policies (`NudgePolicy`, `CoastPolicy`,
  `RunawayReasoningPolicy`, `LoopRetryPolicy`) and the transforms
  (`ThinkContentGuard`, `ToolCallGuard`, `MessageOverflowGuard`) subclass
  nothing. They are duck-typed, and their `apply()`/`process()`/`validate()`
  signatures differ: nudge/runaway/loop-retry take `(body)`, coast takes
  `(body, content)`, overflow returns `(body, changes)`, tool-call returns a
  bare change list, and think returns a `(content, reasoning, changes)` tuple.
  Composition is therefore hand-written per call site.
- **Ordering is a hard-coded `if`-chain.** The remediation ladder in
  `_forward_streaming()`'s `while True` loop is a sequence of conditional
  blocks with bespoke bookkeeping (`nudge_attempt`, `coast_attempt`,
  `runaway_attempt`, `loop_retry_attempt`). The fact that nudge (empty turn)
  and coast (non-empty turn) are mutually exclusive is an accident of their
  predicates, not a property the structure enforces; nothing stops a future
  step from overlapping an existing one.
- **Verdict routing is stringly-typed.** The proxy compares
  `diagnosis.code == CONTEXT_WINDOW_CODE` and `loop_verdict.kind` against the
  literals `"loop"`, `"stalled"`, and `"runaway_reasoning"` in several places:
  the re-submission branches, `_abort_error_response()`'s message map, the
  abort-status selection, and the telemetry increments. Adding a new verdict
  kind means touching four or five sites, and a typo fails at runtime, not at
  import.
- The terminal `finish_reason == "length"` signal is shared by *loop* and
  *runaway-reasoning*, and the split between them depends on a subtle
  conjunction (`reasoning` present, `content` empty, no tool calls) evaluated
  inside `_stream_once`. This is the kind of implicit, order-sensitive
  classification that is easy to get wrong when the ladder grows.

### Risks

- Without a common interface, each new remediation is wired differently and the
  project's "add capabilities without rearchitecting" goal erodes module by
  module.
- Stringly-typed verdicts and the hard-coded ladder are the most likely source
  of a misrouted abort (wrong status/body/telemetry) as detectors multiply.

### Recommendations

- Add a common `Remediation`/`Transform` protocol to `base.py` and express the
  ladder as an ordered list of steps, not a nested `if`-chain — §6, R2.
- Replace verdict string literals with typed constants (an `enum`) plus a
  single registry mapping verdict → response/telemetry — §6, R5.

---

## 4. Extension points and coupling between modules

### Findings

**What is good**

- The buffered `Detector` extension point is clean: `ContextWindowDetector`
  *wraps* the `RetryableDetector` and only overrides the verdict when its
  signature matches — a textbook decorator that slots into the existing
  `RetryPolicy` unchanged. A new buffered detector can be added the same way.
- Remediation modules import only `base` (or nothing) plus `config`; none
  import the proxy, so there is no downward dependency from the modules to the
  orchestrator.

**What is not good**

- **`proxy.py` knows every module by name.** It imports from all eleven
  remediation modules and instantiates their classes inline. The architecture
  promise that "future
  capabilities implement these interfaces rather than editing the proxy" is
  only true for buffered detectors; every transform, policy, and watchdog
  requires an edit to `proxy.py` (import, instantiate, and a new branch in the
  streaming loop). This is the central extension-point weakness.
- **The "pure, unit-testable" modules are coupled to the global `settings`
  singleton.** Several classes fall back to `settings` in their `__init__`
  defaults (`NudgePolicy(text=None) → settings.think_nudge_text`, and the same
  pattern in `CoastPolicy`, `RunawayReasoningPolicy`, `LoopRetryPolicy`,
  `MessageOverflowGuard`, `ThinkContentGuard`), so a bare `Foo()` reads process
  global state. `from_settings()` mitigates this but does not remove it. The
  test suite has had to add an autouse `_restore_settings` fixture plus a
  `settings_override` context manager precisely because this coupling exists.
- `telemetry` is likewise a process-wide singleton (acceptable for a metrics
  registry, but the same pattern, and it too must be handled carefully in
  tests).
- **The `httpx.AsyncClient` is never closed.** `Proxy.__init__` creates it and
  there is no `aclose()` and no FastAPI lifespan/shutdown hook; the client (and
  its connection pool) is abandoned at process exit.

### Risks

- The proxy as god-object means the most-loaded file is also the one every new
  capability must touch, concentrating change risk.
- Global-singleton coupling makes modules harder to isolate and the suite
  order-sensitive (already observed and mitigated, but the mitigation is
  evidence of the problem).

### Recommendations

- Introduce a registry (e.g. a list of configured remediations/guards assembled
  once) so adding a capability is "register, don't edit" — §6, R1/R2.
- Inject `settings` (and the clock, which `StallDetector` already does) into
  modules explicitly, removing the `settings` fallback in `__init__` defaults —
  §6, R8.
- Add a lifespan to close the `httpx` client — fold into §6, R7.

---

## 5. Config surface and runtime wiring

### Findings

**What is good**

- Configuration is entirely environment-driven via `pydantic-settings`, with
  `extra="ignore"`, grouped comments, and per-feature master switches. It is
  documented in three places (`config.py` defaults, `.env.example`,
  `docs/configuration.md`) and the docs explicitly flag where `.env.example`
  differs from the code defaults.
- The dev/test vs. consume split is clean: `docker-compose.dev.yml` builds and
  live-mounts source, `docker-compose.yml` consumes the published image, and the
  `test` service is gated behind the `dev` profile so `docker compose up` never
  runs the suite.

**What is not good**

- **The config surface is duplicated in three files and has already drifted.**
  `RETRY_MAX_ATTEMPTS` is `3` in `config.py` but `8` in `.env.example`;
  `RETRY_BACKOFF_INITIAL` is `0.5` vs `1`. This is documented, but it means
  "the default" depends on whether the operator copied `.env.example` or ran
  without it — a quiet behavioural difference (notably in worst-case retry
  latency).
- **There is no validation.** No `pydantic` field validators or constraints: a
  negative timeout, a `loop_min_output_fraction` outside `(0, 1]`, or an
  invalid abort status would surface at runtime (or not at all) rather than at
  startup.
- **`PORT`, `UID`, and `GID` are compose-only.** They live in `.env` but are
  read by Docker Compose, never by the application — a reasonable split, but
  the three files blur the line between "app setting" and "compose mapping".
- **Wiring happens at import time.** `settings`, `recorder`, and `proxy` are
  built at module import in `main.py`, and the record path is fixed there. A
  config change requires a restart (acceptable for a gateway), but it means
  configuration cannot be validated or varied without a process boundary.
- The setting count is growing (~50 knobs and counting); each feature adds a
  master switch plus several tuning values, which compounds the documentation
  duplication above.

### Risks

- Config drift between the three files is the most likely source of a
  "works in dev, misbehaves in prod" discrepancy.
- Absent validation, a bad value can disable a safety feature silently (e.g. a
  typo'd `LOOP_DETECTION_ENABLED=False` per §2) or cause non-obvious runtime
  failures.

### Recommendations

- Add `pydantic` validators (ranges, enums) and reconcile the three config
  copies — ideally generate `docs/configuration.md` and `.env.example` from
  `config.py` — §6, R9.

---

## Recommended follow-up tickets

Each item below is concrete enough to be filed as its own ticket; the Product
Owner should triage and file them (none are created by this audit). Priorities
follow the project's rule: code that can silently disable safety features or
drop/mangle upstream traffic ranks above pure structural or documentation
concerns.

| # | Priority | Recommendation |
| --- | --- | --- |
| R1 | **High** | Extract the streaming orchestration from `proxy.py` into a dedicated pipeline runner, and give the three content watchdogs (`ThinkingLoopDetector`, `StallDetector`, `RunawayReasoningDetector`) one uniform interface. |
| R2 | **High** | Add a common remediation/transform interface to `base.py` and express the remediation ladder as an ordered, composable list of steps instead of a hard-coded `if`-chain in `_forward_streaming`. |
| R3 | **High** | Decouple the streaming pipeline from `LOOP_DETECTION_ENABLED`: enter the streaming path for chat completions whenever *any* streaming feature is enabled, so each feature's own switch is honoured independently. |
| R4 | **High** | De-duplicate the retry loop shared by `forward()` and `_stream_once()` (attempt loop, recording, telemetry, backoff, context-window special case) into one implementation. |
| R5 | **High** | Replace stringly-typed verdict routing (`loop`/`stalled`/`runaway_reasoning`, `context_window_exceeded`) with typed constants and a single verdict → response/telemetry registry. |
| R6 | **Medium** | Unify the recorder record schema (one attempt/outcome schema, one meaning for `response_body`) so downstream consumers can query history without a per-module field map. |
| R7 | **Medium** | Add a FastAPI lifespan that closes the `httpx.AsyncClient`, and offload recorder writes off the event loop (or document the blocking-write trade-off). |
| R8 | **Medium** | Inject `settings` (and the clock) into remediation modules explicitly, removing the `settings` fallback in `__init__` defaults so modules are genuinely pure and isolated. |
| R9 | **Medium** | Add `pydantic` validation to `Settings` and reconcile the three config copies (`config.py`, `.env.example`, `docs/configuration.md`), ideally by generating the latter two from the first. |
| R10 | **Low** | Register `runaway_reasoning_aborts_total` in `telemetry._METRICS` (it is incremented in `proxy.py` but currently absent from the Prometheus text exposition, so it only appears in the JSON snapshot). |
| R11 | **Low** | Document the SSE-reconstruction fidelity contract (fixed chunk shape, dropped usage frames) so streaming and non-streaming clients know exactly how their responses may differ after a transform. |

**Note on scope.** This audit is documentation-only; it changes no behaviour and
therefore adds no tests and alters none. Findings that concern test coverage are
left to the existing test-suite audit
([#39](https://github.com/ggeorgovassilis/llm-gauze/issues/39) /
`docs/test-audit.md`) and its tickets.
