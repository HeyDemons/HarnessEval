# AFlow tool workflow adaptation

`aflow-tools` is an opt-in adaptation distinct from `aflow`, whose pinned
HotpotQA operators and historical results remain unchanged. It transfers
AFlow's offline Python graph search, parent selection, experience, validation
and freezing to the benchmark's real tool lifecycle. It does not claim the
original AFlow release evaluated these tool benchmarks.

The workflow has the usual `Workflow(name, llm_config, dataset)` constructor
and async `__call__(problem)`. The QA operators remain available for planning,
criticism and selection. Two extra operators connect them to the environment:

- `ToolSession(llm, problem)` owns one invocation's conversation and observations.
- `ToolDecision(llm)(session, instruction="")` proposes a tool or final answer.
  `await session.commit(proposal)` performs the selected action once and appends
  its real observation. Unselected proposals never execute tools. A stale or
  already consumed proposal is rejected. Return the completed `session.answer`.

Every generation, including auxiliary QA operators, uses RunContext's shared
model-response budget and token accounting. The final response slot prohibits
new actions; proposal/commit cannot bypass it by inserting another generation.
Benchmark system/developer instructions remain authoritative. Native user
messages and the existing episode scorer keep their original lifecycle.
Terminal actions use the same task container as its verifier.

Artifacts use `aflow-tools-python-v1`, incompatible with QA artifacts. Frozen
artifacts must include a disjoint optimization/evaluation manifest and complete
search provenance. Batch evaluation reads `HARNESS_AFLOW_TOOLS_ARTIFACT`, or
`HARNESS_AFLOW_TOOLS_ARTIFACT_<BENCHMARK>` with hyphens replaced by underscores.
The entire evaluation suite must exclude optimization cases, even when running
a single evaluation case. The artifact and algorithm identity enter resume
checks. BFCL's single-response suite remains incompatible.

Search example from a configured workspace:

```sh
PYTHONPATH=HarnessEval python -m benchmark_platform.harnesses.aflow_search \
  --adapter tools --problem-type 'interactive benchmark tool tasks' \
  --split-manifest /absolute/path/split.json \
  --evaluate-command '["/absolute/path/python", "/absolute/path/scripts/evaluate_aflow_candidate.py"]' \
  --output /absolute/path/runs/aflow-search/new-search
```

The evaluator receives only optimization IDs, runs candidate graphs inside the
ordinary agent container, and reads the official verdict after agent exit. It
never imports generated Python in the host scorer. Missing measurements abort
search, while algorithm failures remain scored failures. Initialization is
allowed only via that explicit evaluator path and component tests; it is not
accepted as an optimized batch baseline. Each repeated validation gets a fresh
environment and immutable output directory.

Defaults retain 20 expansions and five repeated validations. Any smaller search
used to validate integration must be reported with its actual budget; it is not
evidence of converged or paper-budget optimization. A completed search can
legitimately select the unchanged initialization. No smoke result is imported
into a formal evaluation.

Generated Python is executable code: the import shim is not a security sandbox.
Keep gold, future user messages, other cases and the scorer outside the agent's
accessible workspace. This is the same isolation requirement as the QA adapter.
