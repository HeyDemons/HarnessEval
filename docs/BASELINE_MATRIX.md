# Baseline And Tool Compatibility

HarnessEval separates source fidelity, tool transport, benchmark lifecycle, and
scoring. Run `harnesseval matrix --json` for the machine-readable 14 x 8 table.

## Baselines

| Profile | Tool contract | Fidelity boundary |
| --- | --- | --- |
| Actor-only | Dynamic | Shared JSON tool loop control |
| ReAct | Dynamic | Batch default: native serial tool loop with explicit finish; optional text protocol with local Observation stop; separate single-response adapter on BFCL |
| Plan-and-Execute | Dynamic | Minimal planner; sequential executors receive the original objective, previous steps and current objective (the source's optional include_task_in_prompt mode); last step response is returned |
| CMAS | Dynamic | Local centralized control with a manager, assignment-isolated parallel workers, and manager synthesis |
| DMAS | Dynamic decentralized DAG | AgentNet-aligned capability entry, per-agent Router/Executor, forward/split/execute, result-only handoff, and acyclic unchanged-task forwarding; cold-start evaluation has no cross-case RAG memory |
| LATS | Dynamic branch-isolated | Published MCTS proposal, value, rollout, reflection, and backpropagation; requires read-only tools or environment snapshots |
| MemGPT | Dynamic virtual memory | Core/recall/archival memory functions, function executor, and heartbeat queue |
| AFlow | No external tools (QA operators) | Frozen Python graph from a disjoint search; distinct Custom/AnswerGenerate and candidate-preserving ScEnsemble; see [artifact workflow](AFLOW_DYLAN.md) |
| DyLAN | No external tools (text profile) | Frozen team from cross-query mean importance on a disjoint optimization split; evaluation performs only inference; see [configuration](AFLOW_DYLAN.md) |
| DyLAN query-local | No external tools (local adaptation) | Explicit `dylan-query-local` variant: per-query trial, importance, selection and fresh solve; not offline team optimization |
| Magentic-One | Workspace specialists | Ledger topology, separate file/web tools, tool-free Coder and non-LLM code Executor |
| Multi-Persona | No external tools | SPP profile protocol with two complete demonstrations, dynamic participant profiles, iterative criticism/revision, and one model call |
| LLMCompiler | Dynamic | Immediate dependency-ready scheduling; declared `$1`/`${1}` references insert str(observation) with literal suffixes; legacy JSON-field dialect requires explicit policy; non-streaming planner; `max_replans` counts total planning passes |
| ReWOO | Dynamic | Source Plan/#E protocol; plan all calls first, execute explicit sequential Evidence Workers (dynamic tools or LLM worker), then solve from the complete evidence log |
| SA | Dynamic read-only speculation | Independent `HARNESS_SA_MODEL` predicts top-k safe actions concurrently on every Actor turn; only an exact Actor match commits a pre-executed read |

All source-backed revisions are stored as full 40-character commits in the
profile registry. Protocol reproductions are not described as vendored upstream
applications.

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

All 112 baseline x benchmark cells have an explicit lifecycle route, and each
route is exercised by a scripted protocol subtest (56 single-turn, 28 native
conversation, and 28 task-container). This proves bridge and tool-contract
compatibility, not model task success. The selected AFlow QA, DyLAN text and
Multi-Persona profiles execute without external tools. This does not imply that
all experiments in the DyLAN paper prohibit tools. A tool-dependent task may end in a normal capability
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
