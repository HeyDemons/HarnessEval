# Baseline And Tool Compatibility

HarnessEval separates source fidelity, tool transport, benchmark lifecycle, and
scoring. Run `harnesseval matrix --json` for the current machine-readable table.

## What a provenance label claims

Every profile carries a `provenance` string. It describes **how the implemented
algorithm relates to its source**, and nothing else. In particular, no label
claims that a measurement here is comparable to a number printed in the source
paper: every profile runs on benchmarks its authors did not use, under this
project's common budget, transport and scoring.

| Label | Claims | Does not claim |
| --- | --- | --- |
| `protocol-reproduction` | The algorithm matches the pinned `source`+`revision`, and any deliberate divergence is recorded in the profile notes | The paper's configuration, hyper-parameters, benchmark or reported results |
| `paper-specification` | The algorithm follows the published specification | That a reference implementation existed to diff against |
| `paper-configuration-transfer` | Reuses published optimized teams and declares the new benchmark adapter | Optimization on the target benchmark or reproduction of the original benchmark's results |
| `local-adaptation` | An explicit, named departure from the published algorithm | To be the published method |
| `local-control` | A control condition built for this project | Any upstream at all; `source` and `revision` are null |

Two consequences worth stating plainly. Reproducing an algorithm is not
reproducing an experiment: `dylan` transfers the published optimized decision
teams to a new tool environment; its visible-state router is an explicit adapter.
And a label is not a
promise that the profile runs: LATS is a `protocol-reproduction` whose published
reward channel this harness deliberately refuses to wire, and it participates
nowhere. See [baseline protocol corrections](BASELINE_PROTOCOLS.md).

## Baselines

| Profile | Tool contract | Fidelity boundary |
| --- | --- | --- |
| Actor-only | Dynamic | Native API-tool loop on AutomationBench; single native declaration on BFCL; JSON control on other bridges |
| ReAct | Dynamic | Batch default: native serial tool loop with explicit finish; optional text protocol with local Observation stop; separate single-response adapter on BFCL |
| Plan-and-Execute | Dynamic | Minimal planner; sequential executors receive the original objective, previous steps and current objective (the source's optional include_task_in_prompt mode); last step response is returned |
| CMAS | Dynamic | Local centralized control with a manager, assignment-isolated parallel workers, and manager synthesis |
| DMAS | Dynamic decentralized DAG | AgentNet-aligned capability entry, per-agent Router/Executor, forward/split/execute, result-only handoff, and acyclic unchanged-task forwarding; cold-start evaluation has no cross-case RAG memory |
| LATS | Dynamic branch-isolated | **N/A — temporarily not participating** in the current batch; profile retained, no valid batch environment; historical records excluded from current comparisons |
| MemGPT | Dynamic virtual memory | Core/recall/archival memory functions, function executor, and heartbeat queue |
| AFlow | Official core + declared dynamic adapter | Pinned official MCTS and QA/math/code operator profiles; capability-limited ToolSession/ToolDecision only in the separate benchmark adapter artifact; see [artifact workflow](AFLOW_DYLAN.md) |
| DyLAN | Dynamic | Single published-team tool policy; exact action consensus and one controller commit per decision; generic state routing is an explicit [adapter](DYLAN_POLICY.md) |
| Magentic-One | Workspace specialists | Ledger topology, separate file/web tools, tool-free Coder and non-LLM code Executor |
| Multi-Persona | No external tools | SPP profile protocol with two complete demonstrations, dynamic participant profiles, iterative criticism/revision, and one model call |
| LLMCompiler | Dynamic | Dependencies inferred from predecessor `$1`/`${1}` references plus explicit ordering edges; scheduling and text substitution share the effective graph; literal suffixes; legacy dialect requires explicit policy; non-streaming planner |
| ReWOO | Dynamic | Source Plan/#E protocol; plan all calls first, execute explicit sequential Evidence Workers (dynamic tools or LLM worker), then solve from the complete evidence log |
| SA | Dynamic read-only speculation | Independent `HARNESS_SA_MODEL` predicts top-k safe actions concurrently on every Actor turn; only an exact Actor match commits a pre-executed read |

Git-backed revisions use full 40-character commits in the profile registry;
paper-backed configurations pin the paper version. Protocol reproductions are
not described as vendored upstream applications.

## Benchmark Lifecycles

| Benchmark | Tool loading | Built-in baseline bridge | Native score status |
| --- | --- | --- | --- |
| GAIA | Isolated workspace, argv command, DDGS web search | Implemented | Public answer normalization can finalize a run |
| GDPval | Isolated writable Office workspace, argv command, and DDGS web search | Implemented | Automated rubric remains a proxy to expert pairwise grading |
| TRAJECT-Bench | Per-case native API schemas | Implemented; external ToolBench service credentials required | Parallel set exact / sequential ordered exact, inclusion, parameter-use, and answer diagnostics are finalized after each arm |
| BFCL V4 | Per-case declared functions | The light suite contains independently scoreable single-turn categories and uses the official AST checker | Official single-turn category score; stateful categories remain full-suite only |
| VitaBench | Native stateful environment and hidden user simulator | Implemented through the official episode lifecycle | Native trajectory evaluator is enabled by the light runner |
| tau2/tau3 | Native stateful environment and hidden user simulator | Implemented through the official episode lifecycle | Official native reward is enabled by default |
| Terminal-Bench 2 | Task container filesystem | Implemented with separate agent and verifier containers | Official task reward |
| SWE-bench Verified | Nested official task containers | Implemented through the official controller and fresh evaluator container | Official repository tests; macOS ARM64 currently supports the configured digest-pinned case |

The matrix tests exercise lifecycle routes with scripted protocol responses.
This proves bridge and tool-contract compatibility, not model task success.
Multi-Persona executes without external tools. AFlow uses the benchmark-tool adaptation documented below.
The single DyLAN profile now executes its selected benchmark actions.
A tool-dependent task may end in a normal capability
failure. Exposing hidden user scenarios as prompts, replacing task containers
with text questions, or silently giving either method a ReAct loop would produce
an easier but invalid comparison.

BFCL single-turn accepts actor-only, ReAct, SA and text-only SPP. Multi-response
methods are gated instead of truncated or merged into one response. LATS also
lacks a branch-safe environment on the other batch benchmarks, so there is
currently no runnable batch benchmark for it. See [protocol corrections](BASELINE_PROTOCOLS.md).

The generic harness does not expose hidden benchmark answers to a baseline
during execution. LATS therefore uses its language-model value evaluations for
trajectory progress and terminal success; official benchmark scoring still
runs only after delivery. This preserves evaluator isolation but is a disclosed
boundary from task-specific LATS environments that return an online exact
reward for a terminal action.

DMAS reproduces the evaluation-time control flow of AgentNet revision
`325d39f2a940be5fa903d28c411bd3426b8007f5`: ten agents by default, a complete
directed communication graph, capability-matched entry, a three-hop unchanged-
task forwarding path, and up to thirty local executions. Router reasoning is
private to the current node; only completed subtask results enter the task
context passed to a peer. AgentNet's cross-task edge evolution, capability
updates, and RAG memories require a disjoint training phase and frozen state.
HarnessEval does not learn them from evaluation cases, so the built-in default
is explicitly a cold-start inference baseline.

On Apple Silicon, the SWE bridge accepts only the catalog's configured,
digest-pinned ARM64 case. Other SWE case IDs fail before execution instead of
falling back to an architecture or image with different task semantics.

## Tool Isolation

GAIA and GDPval cases are copied from sanitized input into one attempt-local
writable workspace. `run_command` accepts an argv array, not a shell string,
and runs as uid/gid 65534 with a minimal environment. Model API credentials are
absent from the tool process. Tool stdout and stderr are retained completely;
the bridge has no character or byte slicing threshold.

GAIA and GDPval expose `web_search` through DDGS with structured JSON input and
complete result records. Mutating commands are never marked safe for SA
pre-execution.
