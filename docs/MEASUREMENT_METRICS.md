# Reporting metrics, version 2

These are reporting counters, not budget limits. Official BFCL responses,
Tau2/VitaBench simulation steps, AutomationBench response limits, and Terminal
agent/verifier seconds retain their existing budget policies.

## Agent turns

`agent_turns_definition = completed-actor-model-response-v2`.
One completed main-algorithm model response, together with its tool batch, is one
agent turn. Planner, router, critic and protocol-repair responses count; Speculator
responses do not. Parallel workers each contribute their own responses: the total
is not serial depth. Multiple tool calls from one response count as one turn.
Final answers count through their model response, without an extra return/finish
increment. Provider error/aborted placeholders and HTTP retries are excluded.
A response ending at the output-token limit still counts as a response.

There is no framework-independent definition of a turn:

- [OpenAI Agents SDK Runner](https://openai.github.io/openai-agents-python/ref/run/)
  describes a turn as one AI invocation and its tool calls.
- [Pi agent core](https://github.com/earendil-works/pi/blob/main/packages/agent/README.md)
  exposes turn/message/tool events. The local Pi loop also ends failed turns, so
  our completed-response reporting counter deliberately excludes error/aborted
  events. Do not count both message_end and turn_end for the same response.
- [AutoGen team turns](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html)
  count participant responses. An [AssistantAgent](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html)
  can perform multiple model/tool iterations during one participant response.

Version 2 chooses a common model-response boundary across this project's methods,
not each framework's distinct team/conversation turn boundary. The former
"distinct tool response IDs + final return" metric represented committed decisions
and double-counted BFCL declaration-and-finish responses. It is not reused.

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
fields, not legacy numbers relabeled as version 2. Preserve all recorded attempt
paths when reconstructing the cost of an arm that retried.

Cache statistics can be recovered where raw usage survived. The old Anthropic
adapter discarded cache fields: those old traces cannot recover them. Cancelled
streams without usage also cannot be precisely reconstructed. Neither issue can
be fixed by rescoring; existing scores are unaffected by this accounting repair.

Terminal and Markdown reports show agent turns, tool calls and the four token
buckets for each channel. Model call diagnostics remain available in artifacts,
but are not columns in the final report.
