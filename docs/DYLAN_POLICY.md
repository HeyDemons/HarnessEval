# Single DyLAN benchmark profile

`dylan` is the only runnable DyLAN method. Its identity is
`annotated-action-consensus-v3`. The former `dylan-inference`,
`dylan-query-local` and `dylan-dm` names are retired, without renaming their
historical result files or merging them into the new method.

## Paper configuration and adaptation boundary

The decision-making experiment in [2310.02170v2, Appendix B.1](https://arxiv.org/html/2310.02170v2#Sx2.SS1)
publishes three optimized four-agent teams. This runtime transfers those selection
results; no training, new team artifact, evaluation gold or online team optimizer
is needed. It does not claim that these teams were optimized on AutomationBench,
Tau2 or Terminal-Bench.

| State | Published team, with normalized role spelling |
| --- | --- |
| searching | SearchOptimizer, BudgetAnalyst, InstructionAnalyst, DecisionReflector |
| exploring | DecisionMaker, BudgetAnalyst, ProductExplorer, InstructionAnalyst |
| item | BudgetAnalyst, DescriptionReader, DecisionMaker, ResultEstimator |

The network preserves T=4, temperature 0, shuffled predecessor messages, exact
parsed-action consensus above two thirds, and a listwise top-2 ranker at layer 3.
The reformation layer copies selected messages as Algorithm 1 specifies, rather
than adding an extra generation by every selected node. A full layer is evaluated
before consensus. The ranker's own model response is metered normally.

Peer ratings and propagated Agent Importance Scores are recorded using equations
10–12. These diagnostics do not silently replace the published teams with a team
selected on an evaluation case. The model and benchmark response budget remain
the experiment's configured values, rather than the paper's GPT-3.5 setup.

Generic benchmarks have no WebShop page types. The explicit adapter
`visible-tool-observation-v1` uses only committed public observations: an initial
state/new user message/error routes to searching; search/list results or public
collections route to exploring; other observed details route to item. Role wording
uses generic resources and records instead of literal WebShop products. This
router and the JSON action encoding are local adaptations, not released WebShop
code. The pinned public repository does not contain the WebShop implementation.

## Execution and accounting

Nodes only propose actions. After deliberation the controller executes exactly
one selected action and includes its real observation in the next decision.
There are no speculative writes by nodes and no environment access during ranking.
Schemas come from the benchmark; native user communication remains available as
`send_message_to_user` where that bridge defines it. A final answer can finish a
task without a tool call. Invalid proposals are excluded from votes.
Action identity includes only `tool`/`arguments` or `final`; explanation and
confidence annotations do not split identical decisions or invalidate votes.
Raw replies and peer ratings remain in the trace. Mixed final/tool proposals,
wrong field types and unknown tools are still invalid.

One node response or ranker response counts as one main-algorithm generation.
The existing benchmark budget is shared by all of them. Budget exhaustion is
recorded and does not enable extra calls or a tool action after the final-response
boundary. A one-case smoke can therefore establish tool integration while still
failing the task under the official model-response allowance.

Legacy text-team artifacts are not accepted by this runtime. Component helpers
remain available for inspecting old artifacts, but are not additional method
entries. Old results require a fresh agent run to represent this method; rescoring
cannot change the agents, prompts, decisions or token usage of an old trajectory.

Algorithm provenance: [paper](https://arxiv.org/html/2310.02170v2),
[released code snapshot](https://github.com/SALT-NLP/DyLAN/tree/006e440a519f7cf21e2826f3b8033d84ae9bf07c/code).
