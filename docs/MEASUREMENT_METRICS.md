# Reporting metrics, version 4

These are reporting counters, not budget limits. Official BFCL responses,
Tau2/VitaBench simulation steps, AutomationBench response limits, and Terminal
agent/verifier seconds retain their existing budget policies.

## Agent turns

`agent_turns_definition = inspect-completed-generation-v4`.
One completed model generation in the metered main-algorithm scope counts once.
This follows the generic log counter used by the requested
[Inspect Evals repository](https://github.com/UKGovernmentBEIS/inspect_evals/tree/ac481c7a7b4fb05d6befdfea59b47fc61b839a4f),
whose [lockfile](https://github.com/UKGovernmentBEIS/inspect_evals/blob/ac481c7a7b4fb05d6befdfea59b47fc61b839a4f/uv.lock)
pins Inspect AI `ce5617d35a19a2f4ed0e30f00110126fe0be8f3e`.
That version calls `record_turn()` after a completed generation, following
retries/fallbacks; cache hits count too. See the pinned
[generation path](https://github.com/UKGovernmentBEIS/inspect_ai/blob/ce5617d35a19a2f4ed0e30f00110126fe0be8f3e/src/inspect_ai/model/_model.py#L1499)
and [turn meter](https://github.com/UKGovernmentBEIS/inspect_ai/blob/ce5617d35a19a2f4ed0e30f00110126fe0be8f3e/src/inspect_ai/util/_limit.py#L778).
Inspect Evals GAIA uses framework `react()`, GDPval uses `generate()`, and BFCL
single-turn uses `generate(..., tool_calls="none")`. Its BFCL multi-turn and
Tau2 dialogue loops are task-specific counters, not this generic log metric.

Our explicit `agent_turns_scope` includes Actor, planner, router, critic, workers
and protocol repair. Speculation and benchmark-owned user simulation/scoring are
outside this main-algorithm scope, with separate usage. Inspect provides scoped
meters and `suspend_turn_limit()`; excluding these roles is our explicit scope
choice, not an assumption that Inspect automatically excludes all subagents.

All completed generations count, including text-only, thinking-only, empty or
output-length-limited responses. A failed/aborted transport attempt that produces
no completed generation counts zero. Tool failures do not undo the generation.
Multiple tools from one generation count as one turn. Returning a final answer
does not add another turn. Do not sum both Pi message_end and turn_end events.

Example: 3 planning generations + 7 tool-generating responses + 1 final-answer
generation = 11 turns. A text-only method with 17 generations has 17 turns, not
one. No special suppression of turns is needed for no-external-tools profiles.

Turns are neither a cost measure nor an efficiency ratio. Report model-call
diagnostics and Actor/Speculator token buckets for spend; monetary comparisons
also need model pricing. Never claim a method is cheaper solely from fewer turns.
Only compute turn means within one benchmark. BFCL single-turn, GAIA workspaces
and Tau2/VitaBench conversations are different task populations even when their
generation counter has the same unit. Unknown benchmark provenance has no mean.

`tool_calls` retains the benchmark bridge contract: main-algorithm benchmark tool
invocations, excluding native user-message pseudo-tools. Declaration-only BFCL
can have tool calls with zero environment executions. Speculative execution is
not silently added to main-algorithm tool calls.

## Tokens

`token_definition = uncached-input-plus-output-v2`.
Actor and Speculator each have disjoint `input`, `output`, `cache_read`,
`cache_write` buckets. Actor includes all main-algorithm planning/worker roles.
Speculator includes prediction branches, model-generated briefs and capability
judgments when recorded. Benchmark-owned hidden user/evaluator usage stays in
`harness_usage`; it is not part of either agent channel.

- `total = input + output`, the established workspace reporting convention.
- `all_tokens = input + output + cache_read + cache_write`.
- Neither total is money. Cost needs model/provider pricing and a pricing version.
- OpenAI prompt/input includes cached tokens; subtract cache read/write once.
- Anthropic input and Pi input already exclude cache; do not subtract again.
- Output includes provider-reported reasoning tokens; do not add reasoning twice.
- Optional cache fields omitted from an otherwise reported provider usage object
  default to zero. This reflects available provider telemetry, not an independent
  audit of the provider's cache or billing system.

`usage_coverage.actor/speculator` contain `usage_complete` and
`usage_missing_requests`. Missing usage, unreturned requests, and legacy Pi
all-zero initialized usage are not evidence of zero spend. Numeric aggregates in
an incomplete row contain only reported usage; the table labels them incomplete.
Known failed transport retries without usage are counted as missing too. Retries
hidden inside a provider SDK cannot be inferred from these artifacts.

## Historical results

Version is attached per record. Changing parser definitions does not change
measurement identity or scores. A report may reconstruct a copy from the saved
immutable attempt paths and stamp `metrics_recomputed_from_version`; original
summaries and scorer artifacts remain untouched. Missing artifacts yield unknown
fields, not legacy numbers relabeled as version 4. Version 2 counted completed
Actor generations; version 3 counted outward task decisions; version 4 follows
the pinned Inspect generation counter with explicit scope and report provenance.
Never mix v3 values into a v4 mean. Reconstruct turns from llm_response/Actor
message events or authoritative completed-generation counters, never from v3
decision events. Reports display the target version and each row's source version;
`metrics_recomputed_from_turn_definition` retains the original definition too.
Preserve all recorded attempt
paths when reconstructing the cost of an arm that retried.

Cache statistics can be recovered where raw usage survived. The old Anthropic
adapter discarded cache fields: those old traces cannot recover them. Cancelled
streams without usage also cannot be precisely reconstructed. Neither issue can
be fixed by rescoring; existing scores are unaffected by this accounting repair.

Terminal and Markdown reports show agent turns, tool calls and the four token
buckets for each channel. Model call diagnostics remain available in artifacts,
but are not columns in the final report.
