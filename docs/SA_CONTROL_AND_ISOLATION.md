# SA control and native isolation

SA/Tau2 is incompatible until a separate native speculative execution **and**
commit channel exists. A business `READ` tool with `mutates_state=False` can
still change the native conversation and consume native steps. EpisodeBroker
routes ordinary tool handlers through that conversation and therefore sets
`ToolEnvironment.isolated_calls_supported=False`. Both speculative execution
and adoption reject that environment; SA rejects safe-preaction use before a
model request. The Tau2 entrypoint and compatibility matrix reject SA outright,
including attempts to run it with an empty speculation allowlist. This does not
silently turn the method into an Actor-only run under the SA label.

This is a compatibility gate, not an implementation of shadow episodes.
Re-enabling requires showing that discarded predictions have no native transcript,
state or budget effect, while adopting a prediction publishes exactly the Actor's
canonical native action and preserves native evaluator evidence. No evaluator or
gold may be used by the speculative channel. The previous Tau2/SA trajectories
are retained as historical records, but cannot establish losslessness and require
new measurements after a correct adapter is available. Rescoring cannot repair them.

VitaBench's dedicated bridge is outside this audit and no Vita benchmark is run.
The shared EpisodeBroker protection applies wherever that broker is used.
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
- SA's new isolation check and aligned Actor finalization use
  `isolated-channel-matched-actor-v3`.

SPP's source-compatible empty answer behavior and DyLAN's declared visible-state
router are unchanged. Scorers and official budgets are unchanged.
