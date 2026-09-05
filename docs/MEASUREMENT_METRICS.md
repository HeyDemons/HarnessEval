# Reporting metrics, version 3

These are reporting counters, not budget limits. Official BFCL responses,
Tau2/VitaBench simulation steps, AutomationBench response limits, and Terminal
agent/verifier seconds retain their existing budget policies.

## Agent turns

`agent_turns_definition = committed-task-decision-v3`.
One decision submitted to the task environment or user is one agent turn:
one tool batch, one user message, or a final answer. Planner, router, critic,
protocol-repair and speculative work do not count until an outward decision is
submitted. Parallel workers each count their submitted batches; the sum is not
serial depth. Multiple tool calls from one response count as one batch.
BFCL's one declaration response, including its end-of-task boundary, counts once.
Failed API attempts, empty output and thinking-only messages do not count.
An invalid tool request or a submitted call that times out still counts as a
decision: success of execution is a different metric.

Example: 3 internal planning responses + 7 tool batches + 1 final answer =
8 agent turns. A batch of 3 tools contributes 1 agent turn and 3 tool calls.
User messages received from a simulator do not add agent turns.

Baseline tool dispatch emits `decision_committed` with a unique decision ID
before waiting for the result. Calls sharing an assistant response ID reuse the
same decision ID. Publishing a selected speculative/search result records its
Actor decision once, without counting the unpublished branch. Final submission
records a separate decision, except for BFCL, where it reuses the declaration's ID.
Perseus counts outward tool batches or text messages in the authoritative Actor
message stream once, excluding thinking-only and error/aborted messages.

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

Version 3 deliberately chooses task decisions, as requested for this project,
rather than OpenAI/Pi model-response turns or AutoGen participant turns.
Version 2's completed-model-response counter remains available as a model-call
diagnostic and must not be relabeled as task-decision turns. The older formula
"distinct tool response IDs + final return" double-counted BFCL; version 3 uses
the same decision ID for declaration and finish.

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
fields, not legacy numbers relabeled as version 3. Legacy baseline tool-request
response IDs plus the recorded final answer can reconstruct decision turns;
the new explicit decision events are preferred. Preserve all recorded attempt
paths when reconstructing the cost of an arm that retried.

Cache statistics can be recovered where raw usage survived. The old Anthropic
adapter discarded cache fields: those old traces cannot recover them. Cancelled
streams without usage also cannot be precisely reconstructed. Neither issue can
be fixed by rescoring; existing scores are unaffected by this accounting repair.

Terminal and Markdown reports show agent turns, tool calls and the four token
buckets for each channel. Model call diagnostics remain available in artifacts,
but are not columns in the final report.
