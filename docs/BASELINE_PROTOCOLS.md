# Baseline protocol corrections

## BFCL runtime-harness declarations

The frozen 65-case comparison treats every method as the runtime harness under
test. Each receives the pinned BFCL `question[0]` messages with roles preserved
and the official native function schemas. Controller-only annotations such as
`parallel` and `read_only` are never shown to the model.

Planner, worker, router, memory, Actor and Speculator calls remain internal to
their respective harnesses and are fully included in time, turn and token
accounting. An internal tool selection receives a synthetic observation: the
BFCL function is not executed and no hidden state is read, but the method is
allowed to finish its normal action, planning, heartbeat or speculation loop.
The declaration contract is appended once to the initial benchmark task message.
Later tool results contain only the policy-neutral
`{declaration_only, proposal_only, execution, observation}` sentinel; they never
repeat instructions, function arguments, or decision guidance.

After the method finishes, the harness exposes exactly one outward assistant
response containing its complete selected native tool-call batch. The
deterministic `bfcl-native-declaration-boundary-v2` publisher makes no LLM call,
repair or deduplication. Multi-model final nodes use
`multi-model-declaration-aggregation-v1`; DyLAN selects one complete batch by its
T-FFN/reformation vote; actor-only, ReAct, SA, MemGPT and tool-adapted
Multi-Persona publish their selected action chain. Repeated calls therefore reach
the official scorer unchanged.

The boundary is committed even for an empty batch. Results include
`committed_response_id`, `source_response_ids`, `proposal_response_ids`,
`internal_llm_calls`, `publisher_llm_calls=0`,
`external_assistant_responses=1`, and `environment_calls=0` for real external
execution. Synthetic proposals are recorded separately. The workspace runner
stamps the method-specific output protocol into measurement identity so older
results cannot resume or merge under this contract.

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
lengths. On BFCL, ReAct runs its complete native serial loop; selected actions
receive an explicit synthetic observation without executing a BFCL function.
When ReAct naturally finishes, the runtime publishes the ordered selected-action
chain as the one outward native batch.

Reference: [ReAct notebook](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/hotpotqa.ipynb).

## Magentic-One workspace adapter

The four participants retain distinct responsibilities:

| Participant | Available capabilities |
| --- | --- |
| FileSurfer | read_file, list_files, search_files |
| WebSurfer | web_search/web-browser tools and benchmark-native remote-service APIs |
| Coder | Text/code generation, no native environment tools |
| Executor | Fenced scripts through run_command, no LLM call |

Only capabilities present in the benchmark are exposed. Unrecognized domain
API names are assigned to WebSurfer at this adapter boundary; the ledger and
four-participant topology remain unchanged. Executor consumes new
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
upstream semantics and records `validated-planner-ids-v5`; earlier measurements
cannot silently resume under this change. References outside the pinned positive
predecessor range are not inferred, rather than inventing forward-reference support.
The JSON planner validates positive numeric IDs, duplicate IDs and task/dependency
list structure inside its existing bounded protocol repair, before any scheduling.
This does not add planning passes or change v4 raw observation substitution.

Reference: [pinned dependency parser](https://github.com/SqueezeAILab/LLMCompiler/blob/a00c9d35507507da70e8c637eee64efc8c1857ae/src/llm_compiler/output_parser.py).

The v4 observation is the successful tool's own return value, unwrapped once
from the controller's `ok/result` envelope. Previously v3 substituted the entire
envelope, corrupting scalar arguments such as Base64 text. Transport envelopes
stay in the trace and failed-tool observations retain their error details. The
explicit legacy field dialect keeps its documented envelope. This fixes data
transport, not planning: the published default remains one planning pass.

AFlow XML operators preserve upstream optional fields. A missing `thought` is
allowed; consuming a missing `answer` still fails in the graph. Search defaults
to upstream convergence checking and regenerates repeated modifications in the
same round, preserving rejected replies and their usage. See [search and frozen
DyLAN teams](AFLOW_DYLAN.md) for configuration and split validation.

DMAS retains its declared cold-start inference contract; no AgentNet training
is added. Its LLM capability mapper and weighted entry selection differ from
upstream's task-type capability map. Nonempty ReWOO plans require worker evidence assignments,
uses JSON tool inputs and typed field references, and omits the brackets that
upstream adds around substituted evidence strings. These are adapter boundaries.
Zero-step plans pass an empty worker log to Solver as in the pinned source.
Unclosed worker brackets retain bounded protocol repair rather than copying
the source parser's blind last-character truncation of potentially structured API input.

ReWOO's evidence references are resolved before the worker input is required to
be an object, so a single `#E` variable may be the whole input when the evidence
it names is that object -- upstream substitutes into the input text and only then
calls the worker, and demanding the object shape first rejected such a plan while
it was still literally the string `#E3`. A field path may index an array as
`#E1.results.0.url` or `#E1.results[0].url`, and a path is walked into a JSON
*string* observation, which is the shape a benchmark tool actually returns; the
identical path over a Python dict already resolved, so the difference was an
execution boundary rather than a wrong plan. Measured on the 2026-09-06 tau2 sweep
before this change: ReWOO scored 0.400 with 9 failed arms, the lowest of nine
methods. This is a new measurement identity -- ReWOO results from before it are
not comparable and must be re-run, not rescored.
Plan-and-Execute's original-objective option, SPP's generic examples,
LLMCompiler's non-streaming planner and SA's source-like waiting are declared
configurations/adaptations, not modifications made in this correction.

LATS still lacks an applicable batch environment with the required branching
isolation and legitimate online reward. It remains excluded from the BFCL-65
method comparison and every other current batch benchmark. Its standalone
model-value fallback is an adaptation; sequential proposal requests do not have
the source's batched-sampling latency. Do not expose hidden gold to make it
runnable or report these configurations as equivalent latency measurements.

Two independent blockers, and the second is a principle rather than unfinished
wiring. First, `run_lats` refuses any environment exposing a non-read-only tool,
because MCTS backtracking needs branch-isolated snapshots that no benchmark here
provides. Second, the published reward is the gold answer. In the pinned source
the HotpotQA environment computes `score = normalize_answer(self.data[idx][1]) ==
normalize_answer(info['answer'])`, where `self.data[idx][1]` is the labelled
answer; the search stores that as `node.reward` and returns the moment a node
reaches `reward == 1`. Wiring `RunContext.evaluate_terminal` to a scorer would
therefore hand the agent the grader's verdict mid-search, so it returns `None`
by deliberate refusal, not by omission, and the value-model fallback changes the
algorithm's search signal rather than merely approximating it.

This is not a defect to repair in code. A faithful LATS needs a benchmark that
natively publishes an online reward to the agent -- as WebShop does with its
item/instruction relevance score -- and whose tools are read-only or
snapshottable. Adding one is separate integration work.

Reference: [pinned HotpotQA reward](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/hotpot/wrappers.py)
and [its search loop](https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/853d81614607dd27433faf17c7b0a7d660f95d22/hotpot/lats.py).

**Decision: LATS is temporarily not participating in the current batch.** Keep
its profile and compatibility gates registered, but display its participation
as N/A, not a zero score. Do not include it in the current effective comparison
or denominator. Historical records remain intact and explicitly historical.
A new read-only/snapshot-capable benchmark is separate future integration work,
not part of this batch or a reason to relax the existing gate.
