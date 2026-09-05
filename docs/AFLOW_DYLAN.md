# AFlow and DyLAN fidelity profiles

These implementations preserve the selected algorithms while declaring their
benchmark adapters. They are not the complete upstream applications or every
task-specific experiment in the papers.

## AFlow

`aflow` executes a frozen **Python** `Workflow`, not a list of operator names.
It preserves data dependencies, custom instructions, loops, and conditions.
The adapter redirects the infrastructure imports of the pinned HotpotQA graphs
to the HarnessEval provider, while leaving the graph body intact.

The supported operator library is FoundationAgents/AFlow
`3f457218fc716093fe53f6df8a5d5e6379d66346`, HotpotQA:

- `Custom(input, instruction)` makes one plain text generation with the literal
  concatenation `instruction + input` and returns `{"response": text}`.
- `AnswerGenerate(input)` uses its own prompt and XML `thought`/`answer` fields.
- `ScEnsemble(solutions)` selects a candidate letter and returns that original
  solution. An invalid letter is an error, never a newly invented answer.

These QA operators have no benchmark tool loop. Code-generation-specific
operators and arbitrary dynamic tool workflows are not implemented by this
adapter. The shared no-external-tools compatibility rules apply. Monetary cost
is unknown (`None`); provider token usage is recorded by RunContext. The default
operator budget is 100 calls, configurable as `aflow_max_operator_calls`; the
outer benchmark deadline still applies. XML parsing preserves upstream optional
fields and extra parsed tags. Missing `thought` is allowed; graphs still fail if
they access an absent `answer`, and invalid ensemble letters are errors. Provider
transport retries use HarnessEval's configured retry policy.

### Offline search and freezing

`python -m benchmark_platform.harnesses.aflow_search --help` exposes the search
driver. It implements the pinned optimizer's top-score parent pool, mixed
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

### Code execution boundary

Graph loading is a compatibility shim, **not a security sandbox**. Artifacts
contain executable Python. Run them inside the benchmark's existing isolated
agent environment. Do not import them into a host evaluator with gold data.

## DyLAN

Default `dylan` requires a frozen team from a disjoint optimization split. Each
public optimization question runs one demo text network and backward pass.
Importance sums across layers, then averages across optimization questions;
stable top-k candidate IDs define a single team for the held-out split.
Evaluation never runs a trial or selects agents using the current answer.

This is an open-ended text adapter of cross-query importance aggregation. It
does not claim to implement the pinned MMLU subject/subset analysis scripts:
those use fixed CLI roles, single-choice answers and offline subject analysis.
No gold scorer is used for agent selection in this adapter.

Use the split manifest above and a separate optimization file containing only
public questions, for example `[{"id":"train-case-id","prompt":"Public question"}]`.
Only id/prompt fields are accepted, and case IDs must exactly match the
optimization split. Export the Harness provider configuration before running:

```bash
python -m benchmark_platform.harnesses.dylan_team \
  --split-manifest split.json --optimization-cases public-questions.json \
  --roles Assistant Programmer Mathematician Historian --team-size 2 \
  --rounds 3 --seed 0 --output /path/to/new/team-directory
```

The directory contains per-question JSONL traces, `trials.json` and `team.json`.
The team pins candidate roles, selected IDs, mean importance, rounds, seed,
source/importance revisions, split membership and checksums. Public prompts
must be sanitized by the caller; checksums establish identity, not proof of
an externally supplied artifact's origin.

The batch runner loads `HARNESS_DYLAN_TEAM` or a benchmark override such as
`HARNESS_DYLAN_TEAM_GAIA` once per process. It rejects optimization overlap with
the actual evaluation suite and checks coverage of selected cases before any
model calls. Missing teams skip default sweeps; explicit `--methods dylan`
fails preflight. Team identity is included in resume checks. Direct library
requests must provide `dylan_team_artifact`, `dylan_benchmark`, `dylan_case_id`;
candidate/round/seed overrides are rejected because the artifact is authoritative.

The old per-query trial+solve remains the opt-in `dylan-query-local` profile,
classified as a local adaptation. It defaults to four Assistant candidates,
selects two, runs three rounds per phase, and uses temperature 1.0 and seed 0.
Its policy accepts `dylan_agents`, `dylan_roles`, `dylan_team_size`,
`dylan_rounds`, and `dylan_seed` (or `seed`). At defaults with no consensus early
stop it uses 17 calls, not 22: 11 trial and 6 solve. Trial replies do not enter solve.

The first two layers exchange all active responses. From the third layer,
the listwise ranker selects two agents when more than two remain. Predecessor
messages are still retained as defined by the paper's Eq.7. Once two remain,
no invalid two-input call to the demo ranker is attempted. Agent identities
remain stable after shuffling; the public demo's index/mask confusion is not
copied. The final ranking pair must contain distinct in-range IDs; malformed
rankings use a seeded random pair, with fallback recorded in the trace.

Open-ended consensus retains complete responses and uses pinned sacrebleu
2.3.1 lowercase sentence BLEU >= 90, with strictly more than two-thirds of the
active population required. Invalid rating lists use uniform incoming weights.
Stable score ties are broken by original agent ID. The trace records effective
configuration, nodes/edge weights, rank selections, importance and chosen team.

Within `dylan-query-local`, `dylan_team_optimization=False` is an explicit
inference-only ablation, used by low-level consensus tests. Both profiles
use the public text protocol. The full paper also discusses tool nodes and
reports separate code-interpreter/WebShop experiments; it is inaccurate to
describe the entire DyLAN paper as prohibiting tools. Adding such tools to the
existing arm would require a separately specified adapter and new identity.

Both revised methods change prompts, trajectories and call costs. Their old
measurements cannot be converted by rescoring. Preserve them as historical
adaptations and run new measurements under the new implementation identity.

Sources: [AFlow paper](https://arxiv.org/html/2410.10762v3),
[AFlow operators](https://github.com/FoundationAgents/AFlow/blob/3f457218fc716093fe53f6df8a5d5e6379d66346/workspace/HotpotQA/workflows/template/operator.py),
[AFlow selection](https://github.com/FoundationAgents/AFlow/blob/3f457218fc716093fe53f6df8a5d5e6379d66346/scripts/optimizer_utils/data_utils.py),
[DyLAN paper](https://arxiv.org/html/2310.02170v2),
[DyLAN network](https://github.com/SALT-NLP/DyLAN/blob/006e440a519f7cf21e2826f3b8033d84ae9bf07c/code/demo/LLMLP.py).
