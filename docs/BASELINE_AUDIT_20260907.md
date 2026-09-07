# Baseline fidelity and benchmark contract audit (2026-09-07)

Scope: all 15 registered profiles, their shared provider/tool boundaries and the
seven in-scope batch benchmark lifecycles (VitaBench excluded by user request).
A compatibility fix preserves the algorithm's
decisions, roles and budgets while implementing the benchmark/provider contract.
It is not permission to add replanning, worker quotas or task-specific strategies.

| Profile | Source/contract reviewed | Outcome and limitation |
| --- | --- | --- |
| actor-only | Local control; JSON loop and AutomationBench native batches | Native schema and episode argument-boundary fixes apply. Native and JSON controls are distinct configurations. |
| react | Pinned Thought/Action/Observation loop; native/text protocols | Remove controller-only flags from provider-native function definitions. Preserve one action and existing finish/stop semantics. |
| plan-execute | Pinned LangChain planner, sequential steps, previous results, final step | No new algorithm defect found in reviewed paths; shared native episode boundary corrected. |
| cmas | Declared local-control independent worker wave and synthesis | Keep design; cancellation/draining already implemented. No invented dependency DAG or synthesis quota. |
| dmas | AgentNet forward/split/execute and result handoff | Reject NaN capability weights before selection. Cold-start, LLM capability mapping and generic tools remain declared adaptations, not full cross-case AgentNet training. |
| lats | Pinned MCTS/value/reflection; branch and online-reward requirements | Retain N/A for current batch. No hidden reward access or gate relaxation. |
| memgpt | Pinned processor/functions, heartbeat, memory tiers and warning | Use last response input+output tokens for memory pressure, as upstream, instead of prompt-only/accumulated repair usage. |
| aflow | Pinned HotpotQA operators and frozen workflow search | QA profile retained; requires disjoint optimization artifact; no tool capability added under this name. |
| aflow-tools | Explicit tool-workflow adaptation | Keep separate identity. Repair missing native/task matrix test fixtures; do not label it an original AFlow reproduction. |
| dylan | Paper Algorithm 1/B.1 and published teams; released MMLU code | Keep paper-team transfer identity. Released MMLU code also calls nodes serially; do not invent parallel deployment or replace the declared paper protocol with another task's variant. |
| magentic-one | Pinned ledgers, specialist turns, stall hysteresis, round guard | Restore upstream pre-increment `n_rounds > max_turns` boundary: N permits N+1 ledger rounds. Stall count is intentionally carried across re-entry, as upstream. |
| multi-persona | Pinned one-response SPP collaboration prompt | Keep text-only topology and declared generic examples; no external tool loop grafted on. |
| llmcompiler | Pinned DAG/reference/scheduling/joiner semantics | Prior 9b15a1a restores raw tool observations; one planning pass remains unchanged. |
| rewoo | Pinned fixed Plan/Evidence/Solver protocol | Keep existing data-flow/JSON adapter. Current invalid inputs examined were null/non-request evidence, not fenced JSON parsing defects. No new schema-constrained LLM worker or recovery planner. |
| sa | Independent predictor, safe pre-actions, exact-match adoption | Isolated reads must respect the same validation policy as normal calls. Native handler defaults cannot turn into cached schema failures. Compare with a matching JSON Actor control for losslessness claims. |

## Fixes and measurement identity

- `ToolSpec.native_schema()` emits only name/description/parameters. Read-only
  and parallel metadata stay in the controller and text schemas. ReAct and BFCL
  declaration paths use it; AutomationBench native Actor and Magentic specialists
  already filtered these fields. No tool descriptions or argument schemas change.
- Isolated reads honor `ToolEnvironment.validate_schema`, matching normal reads.
  With native validation delegated, both paths invoke the same handler using the
  original arguments. Strict validation remains enabled where the bridge owns it.
- Tau2 opts into delegating known tool arguments to its native controller,
  including invalid attempts. Native defaults/error handling and native step/error
  budgets therefore see the same requests. Empty virtual user messages retain their
  explicit local protocol check. Trace/config identity: `benchmark-controller-v2`.
  Its public domain policy is preserved as a system message for every role,
  including assignment-only CMAS workers (`system-domain-policy-v2`). Other
  EpisodeBroker callers retain their existing defaults. VitaBench's dedicated
  adapter, lifecycle and participation gate are outside this change.
- MemGPT uses the latest Actor completion's `prompt_tokens + completion_tokens`,
  including cached prompt context. This is context pressure, not billable-token
  accounting. JSON-repair and summarizer calls are not accidentally summed into
  the processor's current response. Identity: `source-token-pressure-v3`.
- Magentic-One's published round guard checks `>` before incrementing. Preserve
  this boundary without enlarging any official benchmark deadline or response
  allowance. Identity: `source-round-guard-v2`.
- DMAS finite-score validation changes only invalid inputs; capability selection
  for valid scores is unchanged. Existing protocol-repair limits remain unchanged.

Root run_bench records the changed identities. Existing results retain their
recorded identity, rather than being relabeled as runs of the new implementation.
Changed decisions/observations/budget accounting require new measurements for a
new-version comparison; rescoring cannot reconstruct an unexecuted action. Native
schema filtering has no external-payload effect where a provider adapter already
removed the extra metadata, but no old artifact is silently upgraded.

## Sources

- ReAct: `ysymyth/ReAct@6bdb3a1`, hotpotqa.ipynb.
- Plan-and-Execute: `langchain-ai/langchain@0207dc1`, experimental plan_and_execute/agent_executor.py.
- AgentNet: `zoe-yyx/AgentNet@325d39f`, AgentNet_Code/src/agentgraph.py and agent.py.
- MemGPT: `cpacker/MemGPT@134df8f`, memgpt/agent.py (Agent.step current_total_tokens).
- AFlow: `FoundationAgents/AFlow@3f45721`, HotpotQA workflow operators and optimizer.
- DyLAN: 2310.02170v2 Algorithm 1/B.1; `SALT-NLP/DyLAN@006e440`, code/MMLU/LLMLP.py. The release does not supply the WebShop implementation.
- Magentic-One: `microsoft/autogen@bd5a24b`, _magentic_one_orchestrator.py (_orchestrate_step, _reenter_outer_loop).
- SPP: `MikeWangWZHL/Solo-Performance-Prompting@619c8a0`, prompts/logic_grid_puzzle.py.
- LLMCompiler: `SqueezeAILab/LLMCompiler@a00c9d3`, TaskFetchingUnit and published configs.
- ReWOO: `billxbf/ReWOO@9cd0283`, algos/PWS.py.
- SA: `naimengye/speculative-action@dc938b9`, speculative_workflow/Speculative_Chess.py.
- Tau2: `sierra-research/tau2-bench@79975ac`, environment/tool.py and native environment controller.

Full test and real-smoke evidence is recorded separately in the workspace audit
report. A successful matrix component test is not a claim that every method/task
combination or every external environment has completed a real scored run.

## Independent review follow-up

The additional source review identified defects beyond the initial table above:

- SPP now applies the pinned `tasks/trivia_creative_writing.py::prompt_unwrap`,
  preserving multiline final answers and the raw collaboration in the trace.
  No marker means return the original response; no extra generation is added.
  Identity: `source-final-answer-unwrapping-v2`.
- LLMCompiler validates numeric positive IDs, uniqueness, task/dependency list
  shape before scheduling, inside the existing JSON protocol repair loop. Numeric
  strings are normalized to the same integer ID used by source substitution.
  Missing IDs retain the existing positional default. No task is executed from
  a partially invalid plan; no new planning passes are allowed. Raw observations
  from v4 remain intact. Identity: `validated-planner-ids-v5`.
- ReWOO accepts a zero-step plan and calls Solver with an empty worker log, as
  pinned PWS does. Malformed evidence, unmatched Plan lines and unclosed brackets
  still use the existing bounded repair. **Declared source-parser deviation:**
  pinned PWS blindly removes the last character of an unclosed worker input;
  the dynamic JSON adapter does not reproduce that truncation, which would alter
  the requested API arguments. This is not a claim of byte-for-byte parser parity.
  Identity: `empty-plan-solver-v2`.
- Magentic-One uses source truthiness for satisfaction, speaker validation and
  stall decisions. In particular, both nonempty strings `"true"` and `"false"`
  are truthy in the pinned source. Structural checks, retry context, carried
  stalls and N+1 round guard remain unchanged. Identity: `source-ledger-truthiness-v3`.
- Shared action JSON accepts annotation fields, as the pinned LangChain parser
  ignores fields other than its action/input. Required action fields/types, known
  tools, mutually exclusive final/tool shapes and final-slot restrictions remain.
  Schema errors include branch-specific reasons. DyLAN canonicalizes only action
  fields for voting (`annotated-action-consensus-v3`), preserving ratings/raw text.
  Common contract identity: `role-schema-v2`.
- JSON Actor and SA pass finalizing to the parser; an invalid final response
  ends with budget failure, with no extra repair request or environment action.
- ReAct registry provenance is now `protocol-dependent-adaptation`; the batch
  runner's native/text identity already distinguishes the selected protocol.

Budget exhaustion remains a failed algorithm measurement, not a successful answer.
No synthesis reserve, worker quota, extra Solver call or grader modification is
introduced. Existing native reward evidence and budget termination are retained.

Two boundaries remain explicit: the retired DyLAN text/DM helper's ranking guard
differs from its release reference but is not reachable through registered `dylan`;
GAIA's shared baseline/product last-line normalization still truncates multiline
predictions. SPP extraction fixes its method output and native user transcript,
but does not by itself change that cross-arm GAIA scoring policy. Neither limitation
is silently presented as resolved by this patch.

Source links: [SPP unwrapping](https://github.com/MikeWangWZHL/Solo-Performance-Prompting/blob/619c8a0ff4205bfd39e33f0867647b40e1703b94/tasks/trivia_creative_writing.py),
[ReWOO PWS](https://github.com/billxbf/ReWOO/blob/9cd0283043ff4be0c9d614fda2789d143ca6ffd1/algos/PWS.py),
[LangChain output parser](https://github.com/langchain-ai/langchain/blob/0207dc1431c29379b724f51c09fa49e6b0333639/libs/langchain/langchain/agents/structured_chat/output_parser.py),
[Magentic orchestrator](https://github.com/microsoft/autogen/blob/bd5a24ba72ba01c4ec7509f027caaa7454b5f6d0/python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py).
