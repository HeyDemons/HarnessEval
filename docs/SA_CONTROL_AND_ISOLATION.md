# SA control and native isolation

SA/Tau2 uses a separate native speculative execution and commit channel. A
declared benchmark `READ` tool with `mutates_state=False` is executed against a
deep copy of the current Tau2 environment. A discarded prediction never reaches
the native transcript, primary world or native step counter. On an exact Actor
match, EpisodeBroker publishes the Actor's canonical ToolCall; Tau2 records that
one action and serves the shadow ToolMessage instead of executing the primary
read again. A miss executes the Actor call normally. Mutating and undeclared
tools are never speculated.

The shadow channel receives neither evaluator state nor gold. If a shadow result
cannot be adopted, the canonical Tau2 tool path remains the authority. Previous
Tau2/SA trajectories produced before this adapter are historical contaminated
measurements and require rerunning; rescoring cannot remove their extra native
calls. Deterministic broker tests cover exact-match publication, while a real
Tau2 smoke is required before a campaign can rely on environment deepcopy and
native ToolMessage adoption in the pinned image.

VitaBench's dedicated bridge is outside this audit and no Vita benchmark is run.
EpisodeBroker still rejects SA when a native bridge does not provide both shadow
execution and adoption hooks.
Other ToolEnvironment adapters retain their existing isolated-read capability;
ordinary authoritative calls continue to reach the same handler.

## AutomationBench matched Actor control

The batch runner now accepts an explicit pair:

```bash
HARNESS_API_STREAM=1 .venv/bin/python scripts/run_bench.py \
  --benchmark automationbench --mode light --methods actor-only,sa \
  --matched-sa-control --concurrency 1 --retry-failed infra \
  --out-root runs/automationbench-sa-json-control
```

Configure `HARNESS_SA_MODEL` as usual. This flag selects the existing JSON
actor-only protocol after environment loading; it requires exactly this pair
and benchmark. Native actor-only remains an independent default configuration.
Use a new output root for the pair: native-control summaries must not be resumed
as JSON control. The existing per-method measurement identities enforce this.

SA and JSON actor-only use the same action prompt, response schema, observation
encoding and final-slot instruction. Deterministic tests compare their actual
model request messages through a cache hit and finalization. Their predictor
usage remains separate. Protocol matching is necessary for a lossless comparison,
but does not itself prove equality across stochastic model runs or different
model/budget/suite configurations.

Merged summaries and rendered reports inspect the saved **per-method** configs
to report `sa_actor_protocol_comparison`. An old native control remains unmatched
even when the latest invocation used JSON for another method.

## Other second-review fixes

- ReWOO searches the original text from the evidence cursor with a line-anchored
  Plan-header regex. Same-line trailing notes do not become orphan plans; actual
  empty Plan lines are still rejected. Identity: `original-text-plan-anchors-v3`.
- Magentic-One compares arbitrary JSON speaker values against participant names
  without hashing dict/list values. They enter the existing bounded local ledger
  retry. This is compatibility handling; pinned upstream may terminate an invalid
  speaker with ValueError. Identity: `typed-speaker-validation-v4`.
- DMAS at its own execution-count limit returns the last completed executor
  result, matching pinned AgentNet TaskChain.final_result. It emits
  `dmas_execution_limit`; it adds no synthesis or model call. Official response
  and time budgets still fail at their own boundaries. Identity:
  `retained-executor-result-v2`.
- SA's aligned Actor finalization uses `isolated-channel-matched-actor-v3`;
  Tau2 additionally records the native shadow/adoption adapter identity.

SPP's source-compatible empty answer behavior and DyLAN's declared visible-state
router are unchanged. Scorers and official budgets are unchanged.
