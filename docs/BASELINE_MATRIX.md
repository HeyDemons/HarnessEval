# Baseline And Tool Compatibility

HarnessEval separates source fidelity, tool transport, benchmark lifecycle, and
scoring. Run `harnesseval matrix --json` for the machine-readable 13 x 8 table.

## Baselines

| Profile | Tool contract | Fidelity boundary |
| --- | --- | --- |
| Actor-only | Dynamic | Shared JSON tool loop control |
| ReAct | Dynamic | Published Thought/Action/Observation protocol |
| Plan-and-Execute | Dynamic | Source-aligned minimal text planner; sequential executors receive only previous steps and the current objective; the last step response is returned |
| CMWS | Dynamic | Local control with a central manager, assignment-isolated parallel workers, and manager synthesis |
| LATS | Dynamic branch-isolated | Published MCTS proposal, value, rollout, reflection, and backpropagation; requires read-only tools or environment snapshots |
| MemGPT | Dynamic virtual memory | Core/recall/archival memory functions, function executor, and heartbeat queue |
| AFlow | Dynamic frozen workflow | Evaluation requires a workflow optimized on a disjoint split |
| DyLAN | No external tools | Published text-agent network is not silently converted to ReAct |
| Magentic-One | Dynamic | Ledger, speaker selection, stall and replan topology |
| Multi-Persona | No external tools | Published single-model persona collaboration |
| LLMCompiler | Dynamic | Dependency DAG and parallel ready-task scheduler |
| ReWOO | Dynamic | Plan evidence before sequential execution |
| SA | Dynamic read-only speculation | Only declared parallel read-only calls may be pre-executed |

All source-backed revisions are stored as full 40-character commits in the
profile registry. Protocol reproductions are not described as vendored upstream
applications.

## Benchmark Lifecycles

| Benchmark | Tool loading | Built-in baseline bridge | Native score status |
| --- | --- | --- | --- |
| GAIA | Isolated workspace, argv command, DDGS web search | Implemented | Public answer normalization can finalize a run |
| GDPval | Isolated writable Office workspace, argv command, and DDGS web search | Implemented | Automated rubric remains a proxy to expert pairwise grading |
| TRAJECT-Bench | Per-case native API schemas | Implemented; external ToolBench service credentials required | Native trajectory finalizer still required |
| BFCL V4 | Per-case declared functions | Implemented for trajectory capture | Official category scorer still required |
| VitaBench | Native stateful environment and hidden user simulator | Implemented through the official episode lifecycle | Native evaluator is optional and explicitly recorded |
| tau2/tau3 | Native stateful environment and hidden user simulator | Implemented through the official episode lifecycle | Official native reward is enabled by default |
| Terminal-Bench 2 | Task container filesystem | Implemented with separate agent and verifier containers | Official task reward |
| SWE-bench Verified | Nested official task containers | Implemented through the official controller and fresh evaluator container | Official repository tests; macOS ARM64 currently supports the configured digest-pinned case |

All 104 baseline x benchmark cells have an explicit lifecycle route, and each
route is exercised by a scripted protocol subtest (52 single-turn, 26 native
conversation, and 26 task-container). This proves bridge and tool-contract
compatibility, not model task success. DyLAN and Multi-Persona are
executed without external tools because their published methods do not define a
tool loop; a tool-dependent task may therefore end in a normal capability
failure. Exposing hidden user scenarios as prompts, replacing task containers
with text questions, or silently giving either method a ReAct loop would produce
an easier but invalid comparison.

LATS is marked non-runnable for benchmark lifecycles that do not expose a
branch snapshot/restore contract. It can run directly on all-read-only declared
toolsets, including the current TRAJECT and BFCL schema bridges. Executing
several mutating branches in one shared environment would not be LATS and is
therefore rejected.

The generic harness does not expose hidden benchmark answers to a baseline
during execution. LATS therefore uses its language-model value evaluations for
trajectory progress and terminal success; official benchmark scoring still
runs only after delivery. This preserves evaluator isolation but is a disclosed
boundary from task-specific LATS environments that return an online exact
reward for a terminal action.

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
