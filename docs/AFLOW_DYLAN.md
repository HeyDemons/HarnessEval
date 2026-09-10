# AFlow and DyLAN fidelity profiles

These implementations preserve the selected algorithms while declaring their
benchmark adapters. They are not the complete upstream applications or every
task-specific experiment in the papers.

## AFlow

`aflow` first reproduces the pinned FoundationAgents/AFlow optimizer and
dataset-profiled operator library. Benchmark integration is a second layer.
It executes a frozen **Python** `Workflow`, preserving data dependencies,
custom prompts, loops and conditions; an operator-name list is not an AFlow
graph.

The supported official operator profiles at FoundationAgents/AFlow
`3f457218fc716093fe53f6df8a5d5e6379d66346` are:

| Profile | Official datasets | Operators |
| --- | --- | --- |
| `qa` | HotpotQA, DROP | Custom, AnswerGenerate, ScEnsemble |
| `math` | MATH, GSM8K | Custom, Programmer, ScEnsemble |
| `code` | MBPP, HumanEval | Custom, CustomCodeGenerate, ScEnsemble, Test |

Their union is six classes, but the upstream optimizer never exposes all six
to every dataset. `Programmer` execution and `Test` public cases are disabled
unless an isolated public-test sandbox explicitly enables them. Hidden
benchmark verifiers are never operator inputs.

For Tau2, AutomationBench and Terminal-Bench, the public method consumes an
artifact with operator profile `benchmark-tools`. That secondary adapter adds
ToolSession/ToolDecision as documented in [AFLOW_TOOLS.md](AFLOW_TOOLS.md); it
does not relabel those operators as official AFlow. Monetary cost
is unknown (`None`); provider token usage is recorded by RunContext. The default
operator budget is 100 calls, configurable as `aflow_max_operator_calls`; the
outer benchmark deadline still applies. XML parsing preserves upstream optional
fields and extra parsed tags. Missing `thought` is allowed; graphs still fail if
they access an absent `answer`, and invalid ensemble letters are errors. Provider
transport retries use HarnessEval's configured retry policy.

### Offline search and freezing

`python -m benchmark_platform.harnesses.aflow_search --help` exposes the search
driver. Official profiles are selected by `--operator-profile qa|math|code`;
the separate integration profile is `benchmark-tools`. It implements the
pinned optimizer's top-score parent pool, mixed
uniform/softmax selection (`lambda=.3`, `alpha=.2` on scores multiplied by 100),
LLM graph/prompt edits, repeated validation, parent-indexed success/failure
experience, and final selection by mean validation score. Defaults are 20
expansions, five validation repetitions, and four parent candidates. The search
prompt adapts the official prompt to this API. Automatic convergence stopping
uses the pinned defaults: top three means, z=0, five unchanged transitions.
`--no-check-convergence` explicitly disables it. Empty or repeated modifications
are regenerated within the same round, including parent reselection. Each reply,
parent, rejection and provider usage is saved; retries do not consume rounds.
`--max-generation-attempts` optionally bounds this loop; the default is unbounded
as upstream, with cancellation controlled by the caller. Exhausting an explicit
cap raises instead of freezing an incomplete search. Other syntax/graph-invalid
expansions remain unscored rounds. Provider/evaluator failures abort the search.

Example split manifest (IDs must come from actual separate splits):

```json
{
  "benchmark": "gaia",
  "optimization_case_ids": ["train-case-id"],
  "evaluation_case_ids": ["evaluation-case-id"]
}
```

```bash
python -m benchmark_platform.harnesses.aflow_search \
  --split-manifest split.json \
  --evaluate-command '["python", "/absolute/path/to/evaluate_optimization.py"]' \
  --output /path/to/new/search-directory
```

The evaluator receives two additional arguments: candidate JSON path, and an
optimization-only case manifest path. It must execute the candidate in an
agent sandbox and invoke the scorer **after agent completion**. It prints
`{"score": 0.5, "feedback": ...}` to stdout, with a finite score in [0,1].
Optional feedback may contain only optimization-split diagnostics. Hidden
evaluation labels must never be available to generated code or the optimizer.
The CLI itself never imports generated graphs and never loads answer keys.
Its evaluator subprocess has a default 900-second timeout; an evaluator that
starts containers owns their cleanup. Use unique output directories.

Outputs include the search history, per-attempt raw optimizer expansions,
`generations.json`, and `frozen.json`
with code checksum, source revision, split membership, selected round and score,
and search-history checksum. An unchanged initialization can legitimately win
a completed search; it is not silently labeled as an improvement. Split IDs
and checksums make provenance reviewable, but are not cryptographic proof of
how an externally supplied artifact was produced.

Both offline optimizers also record the model, reasoning, stream and request
configuration, with only a hash of the endpoint and no API credentials. DyLAN
trial records include token usage; AFlow generation records include rejected
attempt usage. An external importance table explicitly has unreported provider
provenance unless its caller supplies configuration.

Official AFlow uses separate optimizer and executor LLM configurations. The
executor uses `HARNESS_MODEL`; the optimizer requires the explicit
`HARNESS_AFLOW_OPT_MODEL`. Optimizer endpoint/key/transport inherit the actor
unless corresponding `HARNESS_AFLOW_OPT_*` variables override them.

The workspace batch runner reads `HARNESS_AFLOW_ARTIFACT` or a benchmark-specific
override such as `HARNESS_AFLOW_ARTIFACT_GAIA`. It validates the benchmark and
case membership, embeds the artifact in the agent request, and records its
identity for resume checks. Missing artifacts cause a default sweep to skip
AFlow; explicitly selecting it fails before model calls. Existing results are
still included when merged summaries are rebuilt.

For isolated operator tests only, `make_artifact()` plus the explicit policy
`aflow_allow_initialization=True` runs round-one Custom. This is not the default
evaluation path and must not be reported as an optimized AFlow result. Historical
`aflow_workflow: ["Custom"]` requests are rejected, not reinterpreted.

Artifact formats are versioned. `aflow-official-python-v2` records an official
operator profile; `aflow-benchmark-tools-python-v2` records the secondary tool
adapter. Earlier `aflow-python-v1` and `aflow-tools-python-v1` artifacts are
historical and cannot silently resume or run under this implementation.

### Code execution boundary

Graph loading is a compatibility shim, **not a security sandbox**. Artifacts
contain executable Python. Run them inside the benchmark's existing isolated
agent environment. Do not import them into a host evaluator with gold data.

## DyLAN

Only `dylan` is runnable. It transfers the paper's published optimized
four-agent decision-making teams through the benchmark's actual tool lifecycle.
It does not need a frozen text-team artifact or a separate training run.
See [the current protocol and its adaptation boundaries](DYLAN_POLICY.md).

The old `dylan-inference`, `dylan-query-local`, and `dylan-dm` method entries
are retired. Their records remain historical and must not be relabeled or
merged as the current method. The legacy text optimizer and network component
helpers remain solely for inspecting old artifacts and regression tests.
