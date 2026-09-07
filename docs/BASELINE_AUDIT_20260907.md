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
