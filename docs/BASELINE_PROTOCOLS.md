# Baseline protocol corrections

## BFCL single-turn declarations

The frozen single-turn BFCL suite scores one assistant response. Supported
native profiles are `actor-only`, `react`, and `sa`: they generate once and
preserve the complete native call batch. SA makes no Speculator call here.
`multi-persona` retains its one-call text-only SPP protocol without function tools.

The other profiles require multi-response algorithms: plan-execute, cmas, dmas,
memgpt, lats, aflow, dylan, magentic-one, llmcompiler and rewoo. The compatibility
matrix rejects these combinations instead of truncating their algorithms or
merging calls from different responses. This does not claim they cannot solve
function-selection tasks under a separately defined multi-step evaluation.

The response boundary is frozen even for an empty batch. A second generation
cannot search for a later call. The tool environment returns local declaration
acknowledgements without invoking handlers/subprocesses. Malformed batches are
parsed before publication. Unexpected exceptions after declaration remain
failed measurements; only an expected lifecycle stop counts as completion.

Results include committed_response_id, declaration_protocol and environment_calls=0.
Isolated records cannot be merged or relabeled into one declaration response.
Outside BFCL, LATS retains each proposal's source ID; explicit SA adoption keeps
both the speculative source ID and authoritative Actor ID.

The workspace runner stamps the new protocol on native baseline measurement
identities. Old BFCL results cannot silently resume under it. PERSEUS's product
adapter is unchanged. Historical summaries remain included in merged reports
even if a method is no longer runnable. These rules are specific to the current
single-turn suite, not future stateful BFCL categories or native conversations.

Reference: [Inspect Evals single-turn solver](https://github.com/UKGovernmentBEIS/inspect_evals/blob/ac481c7a7b4fb05d6befdfea59b47fc61b839a4f/src/inspect_evals/bfcl/solve/single_turn_solver.py).
The reference informs generation/execution boundaries, not a replacement scorer.

## ReAct protocols

The workspace batch runner explicitly uses the native-tool adapter by default
(`HARNESS_REACT_PROTOCOL=native`). The model selects one native action, receives
the actual tool-result message, and submits its answer through a local
react_finish control action. Multiple calls in one response are rejected before
side effects to preserve serial interaction. The control action is not an
environment tool and does not expose scores or gold. This follows Inspect's
structured-message approach, not the original text parser.

Use `HARNESS_REACT_PROTOCOL=text` to select the corrected textual profile. Direct
library use retains text unless policy.react_protocol explicitly selects native.
Both choices have distinct measurement identities in the batch runner.

The text profile stops consuming output at the first line-level Observation:
or numbered Observation n: marker, before parsing and before appending the
assistant message to the next prompt. A fabricated observation or trailing
answer cannot override the selected action. Escaped newlines in JSON remain valid.

This portable local stop supports reasoning/Responses providers without a
compatible server-side stop parameter. Raw output and all provider usage remain
in the trace. It corrects consumed-output semantics, not generation latency or
tokens spent after the marker. The react_observation_stop event records both
lengths. BFCL uses the separate native declaration adapter described above.

Reference: [ReAct notebook](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/hotpotqa.ipynb).

## Magentic-One workspace adapter

The four participants retain distinct responsibilities:

| Participant | Available capabilities |
| --- | --- |
| FileSurfer | read_file, list_files, search_files |
| WebSurfer | web_search, web_browser, browse_web, fetch_url |
| Coder | Text/code generation, no native environment tools |
| Executor | Fenced scripts through run_command, no LLM call |

Only capabilities present in the benchmark are exposed. Executor consumes new
messages since its previous dispatch and resets its cursor when the ledger
resets the thread. It supports python/py/sh/shell/bash blocks. Scripts persist
in the task workspace; explicit filenames cannot escape it, including via
symlinks. Blocks run sequentially and stop on nonzero exit. Error codes and
empty output remain visible.

Execution uses the benchmark's existing run_command in its task workspace or
container, preserving proxy/environment handling and the official verifier's
state. The wrapper is not a sandbox; the existing benchmark runtime supplies
isolation. No host-scorer subprocess or separate task container is introduced.

Execution budgets differ from the pinned AutoGen executor. Its
`LocalCommandLineCodeExecutor` defaults to **60 seconds per code block**. Here
all blocks in one Executor dispatch share **one run_command budget**. Workspace
bridges default to 180 seconds, configurable by `HARNESS_COMMAND_TIMEOUT_S`;
Terminal uses the task's agent budget, with the outer agent phase enforcing the
remaining time. This is not a fresh task budget per block or per dispatch.
The verifier has its independent official phase budget. These differences can
change timeouts and end-to-end latency; neither timing nor failure boundaries
should be described as identical to the upstream per-block executor.
This adapter deliberately retains benchmark-owned budgets.

Reference: [pinned LocalCommandLineCodeExecutor](https://github.com/microsoft/autogen/blob/bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0/python/packages/autogen-ext/src/autogen_ext/code_executors/local/__init__.py).

References: [AutoGen team](https://github.com/microsoft/autogen/blob/bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0/python/packages/autogen-ext/src/autogen_ext/teams/magentic_one.py),
[CodeExecutorAgent](https://github.com/microsoft/autogen/blob/bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0/python/packages/autogen-agentchat/src/autogen_agentchat/agents/_code_executor_agent.py).
This restores specialist responsibilities and the non-LLM Executor. It remains
a workspace adapter: benchmark file/web tools are not AutoGen's complete
Markdown preview and Chromium backends. Native conversation and remote-API-only
benchmarks are gated rather than substituting arbitrary API tools for code execution.

## Other boundaries

LLMCompiler defaults to `llmcompiler_reference_mode="upstream"`. Before scheduling,
it scans serialized argument text with the pinned numeric-reference regex and
infers positive predecessor IDs smaller than the current task ID. Both `$1` and
`${1}` infer dependency 1; `$10` does not infer dependency 1. Explicit JSON-planner
dependencies remain additional ordering constraints (a declared adapter extension).
Their union is computed once and used by both readiness and substitution. Missing
or empty dependencies therefore cannot launch a referenced task prematurely.
`llmcompiler_dependencies` records declared, inferred and effective edges; the
raw plan remains unchanged in the trace. The replacement is `str(observation)`.
A `.txt` suffix is literal, not a field selection. Dict
values are traversed for the dynamic JSON tool adapter. The old typed field
syntax is retained only as explicit `legacy-json-fields` policy and recorded
in `llmcompiler_config`; it is a different dialect, not a compatibility superset.
The legacy dialect retains explicit-only dependencies. The batch runner selects
upstream semantics and records `inferred-text-references-v3`; v2 measurements
cannot silently resume under this change. References outside the pinned positive
predecessor range are not inferred, rather than inventing forward-reference support.

Reference: [pinned dependency parser](https://github.com/SqueezeAILab/LLMCompiler/blob/a00c9d35507507da70e8c637eee64efc8c1857ae/src/llm_compiler/output_parser.py).

AFlow XML operators preserve upstream optional fields. A missing `thought` is
allowed; consuming a missing `answer` still fails in the graph. Search defaults
to upstream convergence checking and regenerates repeated modifications in the
same round, preserving rejected replies and their usage. See [search and frozen
DyLAN teams](AFLOW_DYLAN.md) for configuration and split validation.

DMAS retains its declared cold-start inference contract; no AgentNet training
is added. Its LLM capability mapper and weighted entry selection differ from
upstream's task-type capability map. ReWOO requires worker evidence assignments,
uses JSON tool inputs and typed field references, and omits the brackets that
upstream adds around substituted evidence strings. These are adapter boundaries.
Plan-and-Execute's original-objective option, SPP's generic examples,
LLMCompiler's non-streaming planner and SA's source-like waiting are declared
configurations/adaptations, not modifications made in this correction.

LATS still lacks an applicable batch environment with the required branching
isolation and legitimate online reward. After rejecting BFCL's single-response
lifecycle, no current batch benchmark is runnable for it. Its standalone
model-value fallback is an adaptation; sequential proposal requests do not have
the source's batched-sampling latency. Do not expose hidden gold to make it
runnable or report these configurations as equivalent latency measurements.

**Decision: LATS is temporarily not participating in the current batch.** Keep
its profile and compatibility gates registered, but display its participation
as N/A, not a zero score. Do not include it in the current effective comparison
or denominator. Historical records remain intact and explicitly historical.
A new read-only/snapshot-capable benchmark is separate future integration work,
not part of this batch or a reason to relax the existing gate.
