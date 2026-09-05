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
outer benchmark deadline still applies. XML validation rejects missing fields
rather than relying on upstream's permissive formatter defaults. Provider
transport retries use HarnessEval's configured retry policy.

### Offline search and freezing

`python -m benchmark_platform.harnesses.aflow_search --help` exposes the search
driver. It implements the pinned optimizer's top-score parent pool, mixed
uniform/softmax selection (`lambda=.3`, `alpha=.2` on scores multiplied by 100),
LLM graph/prompt edits, repeated validation, parent-indexed success/failure
experience, and final selection by mean validation score. Defaults are 20
expansions, five validation repetitions, and four parent candidates. The search
prompt adapts the official prompt to this API; automatic convergence stopping
is not enabled. Syntax/format-invalid expansions consume a search round and
are unscored. Provider or evaluator transport failures abort the search rather
than being assigned a zero score.

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

Outputs include the search history, raw optimizer expansions, and `frozen.json`
with code checksum, source revision, split membership, selected round and score,
and search-history checksum. An unchanged initialization can legitimately win
a completed search; it is not silently labeled as an improvement. Split IDs
and checksums make provenance reviewable, but are not cryptographic proof of
how an externally supplied artifact was produced.

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

The default `dylan` text profile now runs both stages from the COLM 2024 paper:

1. A preliminary temporal network exchanges responses and 1–5 peer ratings.
2. Incoming weights are normalized, terminal supporters receive equal mass,
   and importance propagates backward; per-agent scores sum across layers.
3. The top agents form a fresh solving network. Trial responses are not passed
   into the new network.

Defaults: four `Assistant` candidates (the public demo's role configuration),
select two, three rounds per phase, temperature 1.0, and a run-local RNG seeded
from `dylan_seed` or `seed` (default 0). `dylan_roles` accepts one pinned role
name per candidate: Assistant, Mathematician, Programmer, Lawyer, Historian,
Economist, Psychologist, Doctor. Set `dylan_agents`, `dylan_team_size`, and
`dylan_rounds` explicitly when reproducing another configuration. This default
is query-local team optimization, not a claim to use the paper's optimized
MMLU subject teams.

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

`dylan_team_optimization=False` is an explicit inference-only ablation. It is
used by low-level consensus tests; it is not the default baseline. This profile
uses the public text protocol. The full paper also discusses tool nodes and
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
