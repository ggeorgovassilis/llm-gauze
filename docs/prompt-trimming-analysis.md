# Prompt-trimming analysis: Copilot client preambles

Subtask 2 of epic [#150](https://github.com/ggeorgovassilis/llm-gauze/issues/150),
implementing [#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152).
The companion starter set is [`config/prompt_rules.example.json`](../config/prompt_rules.example.json);
the synthetic request fixture is
[`tests/fixtures/copilot_request_body.json`](../tests/fixtures/copilot_request_body.json).

## Summary

The dated corpus contains **2,084 forwarded requests** across **5 files** and
**13 conversations** (2,198 recorded exchanges; retries and passes of the same
request are counted once). Every request carries the same large system preamble:
a 6,654-character system message (median), a 4,605-character tool-schema array
(median), and small environment/workspace/date blocks. The system message alone
is **85.1%** of message content and **47.7%** of the request body at the median.

Three conservative, system-only rules are proposed. They save **553 characters
per request** (median; p95 553), which is **4.11%** of the request body and
**7.07%** of message content. The measured stable prefix falls from **7,006** to
**6,453** characters (median, consecutive turns in a conversation). The rules
are idempotent on all 2,084 requests and on the synthetic fixture.

The result is deliberately modest: the corpus contains no trim/no-trim pairs, so
the effect on answer quality and tool calling is **not measured**. Every
tool-linked block is therefore kept, and the rule set is opt-in, default-off,
and intended to be run in dry-run mode first (#153).

## Reproducibility

The analysis is a bounded, read-only script over the local corpus. The exact
command is:

```bash
./scripts/analyse-prompt-preambles --data-dir data
```

It prints one JSON document. The tables in this document quote that output; all
sizes are **characters**, never bytes or tokens. The script excludes the live,
still-growing `data/records.jsonl`; only the five dated files are analysed.

The existing `scripts/analyse-records` / `source/app/analysis/` toolkit answers
record-level questions (filters, projections and its four digests) but does not
expose per-message block sizes, deduplicate retries by `request_id`, or measure
rule matches and savings. The analysis script is a bounded, stdlib-only,
read-only extension for that gap; it imports no gateway code and changes no
production module.

Definitions used by the script and below:

| Term | Definition |
| --- | --- |
| request | One unique `request_id`. Retry/pass records of the same request count once. |
| turn | A request whose message list contains at least one `user` message. |
| verbatim recurrence | Share of occurrences whose text is identical after normalising session-scoped ids. |
| preamble | System content plus the environment, workspace, date and reminder blocks. It excludes the user's `userRequest` and conversation history. |
| preamble share before | Pre-trim preamble characters divided by the original serialised request body. |
| preamble share after | Post-trim preamble characters divided by the *trimmed* serialised request body. |
| stable prefix | Longest common prefix of concatenated message contents for consecutive requests in one conversation (`x-interaction-id`). |
| global prefix | Longest common prefix of concatenated message contents across every request in the corpus. |

## Corpus manifest

The stakeholder's floor is at least 100 forwarded requests over at least three
files, each containing a user turn. The corpus clears it: 2,084 requests over
five files, each with at least one turn. The script reports this as
`corpus.sample_floor` with `minimum_requests`, `minimum_files` and
`minimum_turns_per_file`; `met` is true only when all three hold.

| File | Bytes | SHA-256 | Records | Requests | Turns | Conversations |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `records-20261021-175741.jsonl` | 27,120,149 | `c74639e82990481d4792b8cbf10f71022cf1d99d20bf01f74d91c2c7e3997cd4` | 1,292 | 1,198 | 1,198 | 5 |
| `records-20261022-040049.jsonl` | 3,377,922 | `dbe5be0675e388328934bbd6302a9387f68ccf59ff134eea8c303ccec6812cd7` | 175 | 165 | 165 | 4 |
| `records-20261023-095318.jsonl` | 13,242,367 | `f8e6731ff13cf69b8a776b3fd8e438c279115d55d23b371a5b6ff6b3f5ca1969` | 715 | 709 | 709 | 2 |
| `records-20261026-042400.jsonl` | 463,737 | `bf4c4006d8ecfa4e3a0d8e08d5505d4568f961e868ef86ff8d7a07f5e4bbaace` | 5 | 1 | 1 | 1 |
| `records-20261026-051313.jsonl` | 1,067,201 | `18fa07ad247da0e1d4711e9a06f82989fb3bd8a0ade0ee3a8effabd2c70c866d` | 11 | 11 | 11 | 1 |
| **Total** | **45,271,376** | | **2,198** | **2,084** | **2,084** | **13** |

Requests and turns coincide in this corpus: every unique request contains at
least one `user` message, so `requests` and `turns` are equal per file.
`data/records.jsonl` was excluded as the live file specified in the ticket.
`corrupt_lines` is zero for every file.

## Request composition

| Measure | Median | p95 |
| --- | ---: | ---: |
| Request body characters | 13,944 | 26,563 |
| Message content characters | 7,822.5 | 10,665 |
| System message characters | 6,654 | 7,137 |
| Tool-schema characters | 4,605 | 9,517 |
| Preamble characters | 6,952 | 7,435 |
| `userRequest` characters | 65 | 1,895 |
| Conversation-history characters | 494 | 2,518 |
| Preamble share of request body | 49.82% | 58.08% |
| System share of message content | 85.08% | 94.79% |
| System share of request body | 47.69% | 55.59% |

The preamble is not a small tail: at the median it is half of the forwarded
body. The tool schemas are the other large fixed cost, but #153 only rewrites
`messages[*].content`, so this analysis does not propose tool-schema rules.

## Taxonomy

Each row is a recurring block. Frequency is the number of unique requests that
contain the block; sizes are per occurrence. "Variants" counts distinct texts
after normalising session-scoped ids; "dominant" is the share of the most
common variant. A high dominant share means the block recurs verbatim across
turns; a low one means the block carries the turn's own content. The
`conversation.history` row is the exception: it has no fixed boundary, so
"variants" and "dominant" count distinct history *sizes* in characters rather
than distinct texts.

| Block | Role / position | Requests | Median chars | p95 chars | Variants / dominant | Purpose |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `system.identity` | system, before `<instructions>` | 2,084 (100%) | 559 | 559 | 1 / 100% | Agent identity, self-description, content policy and tone. |
| `system.instructions` | system, base `<instructions>` | 2,084 (100%) | 1,806 | 1,806 | 3 / 97.6% | Base coding-agent planning and tool-use behaviour. |
| `system.tool_use` | system, `<toolUseInstructions>` | 2,084 (100%) | 1,344 | 1,344 | 3 / 97.6% | Tool-call schema, parallelism and path rules. |
| `system.output_formatting` | system, `<outputFormatting>` | 2,084 (100%) | 575 | 575 | 1 / 100% | Markdown/KaTeX/Mermaid presentation rules. |
| `system.output_formatting.example` | system, nested `<example>` | 2,084 (100%) | 194 | 194 | 1 / 100% | Illustrative output snippets. |
| `system.agents_catalog` | system, `<instructions><agents>` | 313 (15.0%) | 550 | 726 | 2 / 54.6% | Available subagents and their descriptions. |
| `system.skills_catalog` | system, `<instructions><skills>` | 50 (2.4%) | 4,016 | 4,016 | 2 / 76.0% | Available skills and their file locations. |
| `system.mode_instructions` | system, `<modeInstructions>` | 2,072 (99.4%) | 1,948 | 1,948 | 6 / 82.8% | Active mode's workflow and tool sequence. |
| `system.completion_instruction` | system, untagged tail | 61 (2.9%) | 307 | 307 | 1 / 100% | Instruction to call the terminal task-completion tool. |
| `system.template_variables` | system, untagged tail | 2,084 (100%) | 418 | 418 | 1 / 100% | Per-session template variables for placeholder substitution. |
| `user.environment` | user message 1 | 2,084 (100%) | 70 | 70 | 1 / 100% | Operating-system context. |
| `user.workspace` | user message 1 | 2,084 (100%) | 129 | 129 | 3 / 97.7% | Workspace folder context. |
| `user.context` | user message 2 | 2,084 (100%) | 52 | 52 | 3 / 65.4% | Current-date context. |
| `user.reminder` | user message 2 | 2,084 (100%) | 47 | 47 | 2 / 97.7% | User reminder instructions (empty in the dominant variant). |
| `user.request` | user message 2 | 2,084 (100%) | 65 | 1,895 | 410 / 9.1% | The user's task and attached context. |
| `user.editor_context` | user message 2/3 | 22 (1.1%) | 132 | 132 | 2 / 59.1% | Editor selection/context. |
| `request.tools` | request key `tools` | 2,084 (100%) | 4,605 | 9,517 | 7 / 82.3% | Tool JSON schemas sent alongside the messages. |
| `conversation.history` | assistant/tool/user messages after the preamble | 2,084 (100%) | 494 | 2,518 | 674 / 19.8% | Prior turn history; not a preamble block. |

The six `modeInstructions` variants correspond to different configured modes.
Their tool-call rates are 50.0%, 66.7%, 68.2%, 80.6%, 97.2% and 97.8%; the 11
requests without a mode block are 100%. The aggregate is 81.1%. That is evidence
that mode text drives tool use, not decoration.

## Decisions

| Block | Decision | Rationale | Risk if trimmed |
| --- | --- | --- | --- |
| `system.identity` | **Replace** | Fixed identity/policy/tone text. It is not referenced by a tool schema and does not name a tool. The compact rewrite keeps identity and every policy clause. | Identity/style drift and a changed refusal wording. No observed tool-call dependency; opt-in and dry-run first. |
| `system.instructions` | **Keep** | Directs planning and repeated tool use. There is no ablation evidence that a shorter version preserves behaviour. | Lower tool-call rate or worse task decomposition. |
| `system.tool_use` | **Keep** | Directly governs argument shape, parallel calls and paths. | Malformed or missing tool calls. |
| `system.output_formatting` | **Replace** | Presentation only; the verbose tail can be expressed compactly. | Formatting/math/diagram style drift. |
| `system.output_formatting.example` | **Discard** | Illustrative snippets only; no task or tool instruction. | Minor formatting style change. |
| `system.agents_catalog` | **Keep** | Subagent names/descriptions are needed to select and invoke them; 313 requests carry the block. | Subagent routing loses names or descriptions. |
| `system.skills_catalog` | **Keep** | Skills and their file locations are needed for discovery; 50 requests carry the block. | Skills are not found or used. |
| `system.mode_instructions` | **Keep** | Defines the active workflow and tool sequence; present in 99.4% of requests and linked to the mode-specific tool-call rates above. | The workflow/tool sequence breaks. |
| `system.completion_instruction` | **Keep** | Explicit terminal tool instruction; removing it can stop the agent signalling completion. | The agent never calls the completion tool. |
| `system.template_variables` | **Keep** | Supplies values for `{{...}}` placeholders. The per-session log value is the only session-varying token in the system message; dropping it can break a skill that references it. | A skill/instruction loses a placeholder value. |
| `user.environment` | **Keep** | 70 characters; informs OS-specific commands. | Minor command portability. |
| `user.workspace` | **Keep** | 129 characters; names the workspace root. | Path confusion. |
| `user.context` | **Keep** | 52 characters; supplies the current date. | Date-dependent answers. |
| `user.reminder` | **Keep** | 47 characters and empty in 97.7% of requests; the 2.3% long variant is user-authored content. | Losing a user reminder. |
| `user.request` | **Keep** | The actual task and attachments. This is never a trimming target. | The task is lost. |
| `user.editor_context` | **Keep** | Rare (1.1%) but carries the user's editor selection. | Loss of selection context. |
| `request.tools` | **Keep** | Tool JSON schemas; #153 only rewrites message content, so they are out of scope. | Tools cannot be called correctly. |
| `conversation.history` | **Keep** | Out of scope; the existing `MessageOverflowGuard` bounds oversized tool results. | Loss of conversation state. |

Conservative default applied: `system.instructions`, `system.tool_use`, the agent
and skill catalogues, mode instructions, the completion instruction, the
template values and all tool schemas are kept because their purpose is
established and no trim experiment shows the change is safe.

## Starter rule set

The shipped example has three rules. All are system-only, as required by #153's
safety invariants. The patterns are structural (tag/line anchors, not copied
prompt prose); they intentionally stop matching if the client changes the
structure, which is what #154 detects.

| Rule id | Decision | Evidence class | Action | Match rate | Median saved / match | Total saved (corpus) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `replace-identity-policy` | Replace | `system.identity` | `sub` | 2,084 / 2,084 (100%) | 243 chars | 506,412 chars |
| `compact-output-formatting-tail` | Replace | `system.output_formatting` | `sub` | 2,084 / 2,084 (100%) | 115 chars | 239,660 chars |
| `discard-output-formatting-example` | Discard | `system.output_formatting.example` | `sub` | 2,084 / 2,084 (100%) | 195 chars | 407,622 chars |

Patterns:

| Rule id | Structural pattern | What it does |
| --- | --- | --- |
| `replace-identity-policy` | `\A(?:[^\n]*\n){7}(?=<instructions>)` | Replaces exactly the seven-line identity/policy prefix before the first instructions block. |
| `compact-output-formatting-tail` | `(?<=</example>)\n[\s\S]{150,}?(?=</outputFormatting>)` | Replaces the verbose formatting tail after the example; leaves the Markdown instruction before the example intact. |
| `discard-output-formatting-example` | `<example>[\s\S]*?</example>\n?` | Removes the illustrative example block. |

Rule order matters: the formatting-tail rule runs before the example is removed.
After that removal its look-behind no longer matches. The tail rule also has a
150-character floor, so its 94-character replacement cannot re-match it. The
identity rule's seven-line anchor no longer matches after it has been rewritten,
and the example rule has nothing left to match. Each rule is therefore a no-op
on its own output, as well as on the full set. The script reports
`idempotent: true` and `non_idempotent_requests: 0`; the test suite checks the
same property against the synthetic fixture.

The engine field names follow #153's sketch: `when.role`,
`when.content_regex`, optional `when.min_chars`, and
`then.action` with `then.replacement` for `sub` (and `then.content` /
`then.chars` for the other actions). The extra `evidence_class`, `decision` and
`match_rate` keys are analysis metadata for this ticket and for the #154
baseline; an engine that ignores unknown metadata keys is sufficient.

## Measured saving

Applying the starter set to every unique request:

| Measure | Median | p95 |
| --- | ---: | ---: |
| Content characters saved | 553 | 553 |
| Request-body characters saved | 572 | 572 |
| Share of request body saved | 4.11% | 4.78% |
| Share of message content saved | 7.07% | 7.88% |
| Preamble characters before | 6,952 | 7,435 |
| Preamble characters after | 6,399 | 6,882 |
| Preamble share of body before | 49.82% | 58.08% |
| Preamble share of trimmed body after | 47.79% | 56.15% |
| Stable prefix before | 7,006 | 9,891 |
| Stable prefix after | 6,453 | 9,299 |

* Requests changed: 2,084 / 2,084.
* Total content saved: 1,153,694 characters (sum of the three per-rule totals).
* Stable-prefix sample: 2,071 consecutive turn pairs across 13 conversations.
* Global common prefix across all requests: 1,100 characters before, 857 after.
* Minimum/maximum per-request content saving: 553 / 760 characters. The spread
  comes from six requests that carry a second `<example>` block; the identity
  and single-example savings are constant.

The saving is intentionally concentrated in blocks whose removal cannot affect
tool selection. The larger tool-linked blocks remain, so the current set is a
safe starter baseline rather than an aggressive context reduction.

## Open questions

**JSON or YAML?** JSON, using the standard library. YAML would require a newly
pinned dependency and parser for no measured benefit, and #153 already specifies
stdlib JSON. If an operator later needs YAML, that should be a separate ticket
with a pinned dependency and its own tests.

**`sub` backreferences?** Yes, assumed available: the action should be
implemented with `re.sub`, so backreferences work. The starter set does not need
one; this is an engine requirement for #153, not a corpus finding, and should be
covered by a #153 test.

**Default enabled?** No. The example set remains opt-in and
`PROMPT_TRIM_ENABLED` defaults to false. The saving is real, but the quality
effect of the identity rewrite is not measured; dry-run is the safe first step.

**Which blocks measurably break tool calling when trimmed, with evidence?**
None were measured. The corpus has no trim/no-trim pairs, so no block can be
shown causally to break tool calling. The evidence of dependency is structural
and quantitative: 100% of requests carry tool schemas, 81.1% of delivered
responses are tool calls, 79.6% of requests carry tool calls in their history,
and the mode/agent/completion blocks name the tools used. Every tool-linked
block is therefore kept. The three shipped rules touch only identity/policy and
output formatting; neither is referenced by a tool schema or the tool-use
policy. Any future rule that touches a tool-linked block must be validated with
the #153 dry-run and an A/B run before shipping.

## Privacy

* The corpus lives under the gitignored `data/` directory; the live
  `data/records.jsonl` is excluded from the analysis and from the manifest.
* The analysis command emits aggregates, sizes and hashes only. It never writes
  prompt text, headers or bodies to stdout.
* This document contains no verbatim client prompt text. Rule patterns are
  structural anchors (`<instructions>`, `<example>`, tag boundaries, line
  counts), and replacement text is original wording.
* The synthetic fixture uses invented content and example paths
  (`/home/example/...`); it is not a copy of recorded traffic.
* The recorder already redacts `Authorization` and `Proxy-Authorization`.
  Credentials are neither needed nor present in the repository, this document,
  the fixture or the tests.
* The corpus itself remains local. No ticket comment includes prompt text; only
  aggregates are reported.

## Verification plan

1. **Reproduce the aggregates.** Run
   `./scripts/analyse-prompt-preambles --data-dir data` and compare the manifest
   hashes and counts with the table above. The script is read-only.
2. **Validate the rule set and aggregate maths.** Run the full
   `./scripts/test.sh`. To run only the new files without the coverage floor,
   add `--no-cov`:
   `./scripts/test.sh tests/test_prompt_rules_example.py tests/test_analyse_prompt_preambles.py --no-cov`.
   The tests load the analysis script, so they exercise its rule engine,
   validator and corpus reader: schema, unique ids, system-only roles (including
   text that would match in a user message), a match for every rule on the
   synthetic fixture, per-rule and full-set idempotence, preservation of the
   tool-linked blocks, a bounded saving, live-file exclusion, a clean
   empty-corpus report and that no prompt text reaches stdout. The aggregate
   tests pin the percentile, shared-prefix, taxonomy, sample-floor and saving
   calculations behind the published figures.
3. **Load it through #153.** `PROMPT_TRIM_ENABLED=true` with the example file;
   confirm validation succeeds. An invalid or unreadable file must log `ERROR`
   once and disable trimming while requests are still served.
4. **Dry-run first.** `PROMPT_TRIM_DRY_RUN=true`, then inspect the
   `prompt_trim` record entries and the `prompt_trim_rules_total{rule,action,mode}`
   counters for the expected 100% match rates. Recorded `request_body` remains
   the forwarded (post-trim) body; `chars_before` is the evidence of removal.
5. **Re-validate on drift (#154).** Follow [Re-validation](#re-validation):
   capture a fresh corpus into a new, empty directory with the
   [#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152) protocol,
   analyse that directory only, and compare the per-rule match rate and measured
   saving with the baseline. File a follow-up ticket to update or retire a stale
   rule; do not edit it silently.
6. **Quality check before enabling.** #153 should compare a small representative
   sample with trimming off/on (and with the identity rule alone) on the local
   model. The corpus proves the size and recurrence, not the answer quality.

## Re-validation

The rules match text produced by another product. A VS Code/Copilot update can
change the preamble wording and every rule silently stops matching, so context
is consumed as before and nothing alerts. This is the standing procedure for
[#154](https://github.com/ggeorgovassilis/llm-gauze/issues/154): it reuses the
[#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152) corpus protocol
and the committed analysis command. It adds no code; the
script already emits every metric the thresholds compare.

### Baseline

Measured in [#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152) on
the corpus above (2,084 requests, five dated files, 13 conversations) with
`config/prompt_rules.example.json`. The corpus was captured with prompt
trimming off (`PROMPT_TRIM_ENABLED=false`, the default;
[#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152) predates #153).
That matters because the record holds the forwarded `request_body`, which is
post-trim whenever trimming is applied and not in dry-run, so these baselines
describe the client's text before any rule ran. A re-validation corpus must be
captured under the same condition (step 1) or it is not comparable.

| Rule id | Baseline match rate | Baseline median saved / match |
| --- | ---: | ---: |
| `replace-identity-policy` | 100% (2,084 / 2,084) | 243 chars |
| `compact-output-formatting-tail` | 100% (2,084 / 2,084) | 115 chars |
| `discard-output-formatting-example` | 100% (2,084 / 2,084) | 195 chars |

The aggregate baseline is **553 content characters per request** (p95 553),
**4.11%** of the request body and **7.07%** of message content. The preamble
falls from 6,952 to 6,399 characters (median) and the stable prefix from 7,006
to 6,453. Per-rule totals and the full saving table are in
[Starter rule set](#starter-rule-set) and [Measured saving](#measured-saving).

### Regression thresholds

A result is a regression when a row falls below its threshold:

| Signal | Baseline | Regression when below |
| --- | ---: | ---: |
| `replace-identity-policy` match rate | 100% | 50% |
| `compact-output-formatting-tail` match rate | 100% | 50% |
| `discard-output-formatting-example` match rate | 100% | 50% |
| `replace-identity-policy` median saved / match | 243 chars | 121.5 chars |
| `compact-output-formatting-tail` median saved / match | 115 chars | 57.5 chars |
| `discard-output-formatting-example` median saved / match | 195 chars | 97.5 chars |
| Aggregate median content saved / request | 553 chars | 276.5 chars |
| Median share of request body saved | 4.11% | 2.055% |
| Median share of message content saved | 7.07% | 3.535% |

The thresholds are proportional (half the baseline): compare rates and medians,
not raw match counts, because a re-validation corpus can be smaller. Any single
row crossing its threshold is a regression for that rule. The common case is a
rule that stops matching entirely: its match rate falls to 0% and the aggregate
saving loses that rule's contribution.

A run is only comparable when `saving.idempotent` is `true` and
`rule_validation.errors` is empty. If either check fails, file a bug ticket
instead of treating the result as drift.

### Procedure

1. **Capture a fresh corpus with the
   [#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152) protocol
   into an isolated directory.**
   Point `DATA_DIR` at a new, empty directory for this run (in the container
   `/data/revalidation-2026-10`, which maps to the host's
   `data/revalidation-2026-10`; keep `RECORD_FILE=records.jsonl`) and run the
   gateway with `RECORDING_ENABLED=true`. Keep prompt trimming off for the
   capture: leave `PROMPT_TRIM_ENABLED=false` (the default), or set
   `PROMPT_TRIM_DRY_RUN=true`, which records the would-be changes but forwards
   the original body. Do not capture with `PROMPT_TRIM_ENABLED=true` and
   `PROMPT_TRIM_DRY_RUN=false`: every record then holds an already-trimmed
   body, the analysis applies the rules to their own output, and the idempotent
   rules match nothing, so nearly every rule reports 0% — a false regression
   that is not drift. Do not reuse `data/`: it
   still holds the five baseline `records-*.jsonl` files, and the script reads
   every matching file in the supplied directory, so fresh traffic would be
   aggregated with the baseline and a rule that stopped matching could never
   cross the thresholds. Confirm the directory contains no `records-*.jsonl`
   before starting. Use VS Code/Copilot normally against the gateway. Rotation
   happens only when the recorder starts, so restart the gateway once after the
   last capture to rotate the final live `records.jsonl` into a dated
   `records-<stamp>.jsonl`; the live file itself is excluded and only the dated
   files are read. Keep them local; `data/` is gitignored. The sample floor is
   at least 100 forwarded requests over at least three dated files, each
   containing a user turn.
2. **Confirm the sample before comparing.** Run the command with `--data-dir
   <isolated dir>` and check that `corpus.sample_floor.met` is `true`. Confirm
   `corpus.files` lists only this run's dated snapshots (no baseline file
   names) and that `corpus.totals.requests` matches the number of requests
   captured — use the `requests_total` increase over the capture window, or the
   gateway log, as the source for that count. `corpus.data_dir` echoes the
   directory that was read. Note the file hashes so the run is reproducible. A
   corpus below the floor, or one that includes baseline files, is not
   comparable.
3. **Confirm the capture was untrimmed.** Before trusting any match rate, check
   a chat-completion record and verify that its system `request_body` still
   carries the `<example>` block that `discard-output-formatting-example`
   targets. The committed set applies all three rules, so its presence also
   shows that the identity and formatting-tail rules had not run. Counting
   marker occurrences keeps prompt text off the terminal and out of the
   report:

   ```bash
   jq -r '.request_body | fromjson? | .messages[]? | select(.role == "system") | .content' \
     data/revalidation-2026-10/records-*.jsonl | grep -c '<example>'
   ```

   A count near zero can mean either that the corpus was captured with trimming
   applied or that the client no longer emits the block. First confirm step 1's
   capture settings. If the capture ran with `PROMPT_TRIM_ENABLED=false` or
   `PROMPT_TRIM_DRY_RUN=true` and the count is still near zero, treat the
   missing block as drift and continue to step 4; discard and recapture only
   when step 1's settings were not met. Do not infer the capture condition from
   `prompt_trim` entries — they record rule applications, not whether the
   forwarded body was trimmed, and dry-run also writes them.
4. **Run the committed analysis command** against the isolated directory (the
   same script and committed example rules as
   [#152](https://github.com/ggeorgovassilis/llm-gauze/issues/152)):

   ```bash
   ./scripts/analyse-prompt-preambles --data-dir data/revalidation-2026-10
   ```

   Replace the directory with the one from step 1. It is read-only, reads only
   the dated `records-*.jsonl` files in that directory (never the live
   `records.jsonl`), prints aggregates only and never emits prompt text. Use
   `--rules` only to check a different deployed rule file, and say so in the
   result, because the baselines above are for the committed example set.
5. **Compare per-rule results.** For each entry in `rules`, read `match_rate`
   and `median_chars_saved_per_matched_request` and check them against the
   threshold table. Check `matched_requests` too when the corpus size changed.
6. **Compare the aggregate saving.** In `saving`, read
   `content_chars_saved.median`, `share_of_request_body.median` and
   `share_of_message_content.median`. The p95 values are informative but are
   not thresholds.
7. **Act on a crossed threshold.** File a new follow-up ticket to update or
   retire the stale rule. Include the rule id, baseline versus observed match
   rate and saving, the command, the gateway commit and the corpus manifest
   (file hashes and counts); include no prompt text. Do not edit the rule or
   this baseline silently: the follow-up ticket owns the fix and the refreshed
   baseline. Retire a rule outright when the client no longer emits the block.
8. **Record a clean run.** If no threshold is crossed, note the date, gateway
   commit, corpus manifest and outcome (for example as a comment on the
   re-validation ticket). No rule or baseline change is needed.

If the committed script cannot answer a comparison this procedure needs, that is
a separate tooling ticket; do not add ad-hoc code as part of a re-validation.

### When to run

- After any VS Code/Copilot update that changes prompt composition: release
  notes about the system prompt, instructions, tools, agents or modes, or the
  first recorded requests showing new, renamed or reworded preamble blocks.
- Otherwise periodically: at least once a quarter, and whenever the runtime
  signals below show that a rule has stopped matching.

The runtime signals are an early warning, not a substitute: they show a drop on
live traffic, while this procedure measures it against the corpus baseline.

### Runtime signals

The runtime already exposes these signals; none needs new code.

- `prompt_trim_skipped_total{reason}` counts skipped-trim events. `reason` is
  `disabled` (trimming off by configuration) or `invalid_rules` (the rule file
  failed to load or validate), incremented once per affected request; or
  `scan_bound`, incremented once per message skipped because it exceeded the
  per-message scan limit, so a partial run contributes skipped messages rather
  than requests. Check it first: a nonzero `disabled` or `invalid_rules` rate
  means trimming is off, while `scan_bound` means it was only partly applied,
  so flat rule counters do not indicate drift.
- `prompt_trim_rules_total{rule,action,mode}` increments once per rule
  application; `mode` is `apply` or `dry_run`. If a rule stops matching, its
  series stays flat while requests continue. The counter has no request
  denominator, so divide its increase over an interval by the increase in
  `requests_total` over the same interval to estimate the observed match rate.
  Treat that ratio as an estimate: `requests_total` also counts
  non-chat-completion routes. Use the corpus match rate for the threshold
  comparison.
- Each recorded exchange carries `prompt_trim`: a list with one entry per
  application, or `null` when no rule applied (including when trimming is
  disabled). Each entry has `rule`, `action`, `index`, `role`, `chars_before`
  and `chars_after`. A rule id disappearing from records that used to carry it
  is the same drop; `chars_before - chars_after` is that application's saving.
  Dry-run records the entries without changing the forwarded body, which makes
  it the safest way to watch for drift.

Stale rules are retired or updated through a new ticket, never edited silently.

## Limitations

* The corpus is one developer's VS Code/Copilot sessions on a local model over
  five dates. It clears the sample-size floor but is not a controlled
  experiment, and the task mix is specific to those sessions.
* There is no trim/no-trim quality measurement. The decisions rely on purpose,
  structural coupling and conservative defaults, not on model output quality.
* Match rates are corpus facts, not guarantees. The rules are anchored to
  another product's structure and will drift; #154 is the mitigation.
* The analysis measures characters, not tokens or cache bytes. Character
  savings are comparable across requests but not a token budget.
* The live `data/records.jsonl` continues to grow and was excluded. Snapshotting
  it would add a sixth file and a different manifest; this analysis intentionally
  uses only the dated files.
