# Benchmark budgets

`benchmark_platform/budgets.py` is the authority for budget units, public defaults,
validation and local wall envelopes. Resolve it after environment loading. A
`BudgetPlan` is configuration; a `Deadline` is the one monotonic clock for an
actual measurement. A retry receives its remaining time, never a new deadline.
The parent workspace batch runner enforces the local wall envelope. Low-level
`run_profile()` calls do not create a second independent wall clock; standalone
callers must supply their own enclosing timeout/deadline. An iteration guard
alone is not a wall-time guarantee.

## Official limits versus local policy

| Benchmark | Public/native constraint | Local measurement wall envelope |
| --- | --- | --- |
| BFCL single-turn suite | One assistant response, including its complete declaration batch | 900 s |
| Tau2 | 200 simulation steps, 10 consecutive errors by the pinned native defaults | 900 s |
| VitaBench | 300 simulation steps, 10 consecutive errors by the pinned CLI defaults | 1200 s |
| AutomationBench public | 50 model-response steps by the pinned CLI default | 900 s |
| Terminal-Bench-2 | Independent agent/verifier timeout_sec from each task.toml | Agent + verifier + local 120 s overhead |
| GAIA | No universal wall deadline is supplied by this public task/scorer adapter | 900 s |
| GDPVal | No universal wall deadline is supplied by this public dataset/scorer adapter | 1200 s |
| Trajectory-Bench | Remote-tool task/scorer contract; no common wall value inherited here | 900 s |

These are different units. Native simulation steps are not LLM calls, environment
calls or ReAct loop iterations. The 900/1200-second values are operator choices,
not official leaderboard requirements. This does not claim that every paper or
agent configuration uses the same resources as these local defaults. The optional
SWE-bench-verified adapter also has a local 900-second envelope in this resolver;
it is not added to the batch runner's runnable suite by this change.

Sources pinned by the current catalog:

- [Tau2 config](https://github.com/sierra-research/tau2-bench/blob/79975ac5741e23fbb1d2ac44262d62398a6d87bd/src/tau2/config.py): 200 steps/10 errors. [CLI](https://github.com/sierra-research/tau2-bench/blob/79975ac5741e23fbb1d2ac44262d62398a6d87bd/src/tau2/cli.py) has no wall timeout by default.
- [Vita config](https://github.com/meituan-longcat/vitabench/blob/742e240855bf8686a0842360749d5ea970ea3987/src/vita/config.py) and [CLI](https://github.com/meituan-longcat/vitabench/blob/742e240855bf8686a0842360749d5ea970ea3987/src/vita/cli.py): 300 steps, rather than Orchestrator's separate class default.
- [AutomationBench public CLI](https://github.com/zapier/AutomationBench/blob/4a8e1061254004d9dac807054eed33fad7d1ff14/README.md): max-steps=50 model responses.
- [Terminal-Bench task source](https://github.com/harbor-framework/terminal-bench-2/tree/2fd12b88aafdd04a52c298e3940bcb189f9766d6) and [Harbor task structure](https://www.harborframework.com/docs/tasks): task-specific agent/verifier budgets.
- [BFCL declaration adapter](BASELINE_PROTOCOLS.md) follows the frozen single-turn suite, not stateful BFCL categories.
- [GDPval methodology](https://arxiv.org/html/2510.04374v1) does not make this project's 1200-second local envelope an official limit.

## Wall time and cleanup

Cancellation grace is **inside** the local envelope. At the defaults:

```text
general: 890 s process execution + up to 10 s cancellation = 900 s
GDP/Vita: 1190 s process execution + up to 10 s cancellation = 1200 s
Terminal: agent + verifier + 110 s overhead, then up to 10 s cancellation
```

For Terminal, cancellation grace is capped by the local overhead. If overhead=0,
neither official phase is reduced: the outer timeout is their sum. The inner
agent and verifier still receive their complete independent task values. The
outer envelope is a local infrastructure guard; excessive setup or teardown can
still exhaust it, which is recorded as an arm-envelope timeout, not falsely
identified as the agent's official phase expiring.

The host starts its monotonic deadline before launching the arm, so preparation,
model calls, tools and in-process native evaluation consume the same envelope.
It sends SIGINT when process time ends, waits the recorded cancellation grace,
then kills the owned process group and removes owned containers if necessary.
Final housekeeping can take additional infrastructure time after the model has
stopped; it does not grant the agent more work time. Direct scorer containers
receive unique names so cancelling a Docker client does not orphan them.

Do not rewrite HARNESS_ARM_TIMEOUT_S to a smaller value inside a child and then
subtract cancellation grace again. The host passes remaining seconds directly
to the process watchdog. Call-level API/socket/tool timeouts remain separate
I/O safeguards and cannot reset or extend the case deadline.

## Loop guards and final answers

The batch runner no longer injects 20/40/50 as a generic loop cap. Baseline
loops default to a **1000-iteration safety guard**. This is not an official
step budget and not a portable unit across methods: a planner/worker loop and a
single ReAct loop remain different scopes. The primary cross-arm budget is wall
time; the guard is explicitly named baseline_loop_safety_limit in the record.
PERSEUS's internal loop implementation is not silently assigned this baseline
guard. Any guard hit must remain visible rather than being assumed negligible.

For AutomationBench the shared RunContext additionally enforces 50 Actor-channel
generation requests, across internal workers/planners. Speculator usage stays
separate; it is not silently counted as an authoritative Actor response. Provider
transport retries inside one generation are distinct from this response counter.

When finalization is enabled, native/text ReAct, the JSON agent loop and SA
reserve their last permitted loop turn for an answer, with no further tool
actions. MemGPT uses send_message for this closing response. Native ReAct exposes
only its finish tool in that slot. Finalization is counted within the loop or
response limit and the same wall deadline; failure to submit still fails normally.
There is no new model request after a hard wall deadline or after BFCL's response.
Methods with JSON repair or memory summarization can make more than one model
request per loop iteration; the shared model-response cap still counts those
requests separately and never allows a repair to bypass its limit.

Method-defined search/coordination parameters (DyLAN layers, LATS iterations,
Magentic-One rounds, compiler planning passes, etc.) remain intact. This change
does not reinterpret all of them as a common step count.

## Retries and scoring

Whole-arm automatic PERSEUS retries remain available only for static tasks
without a native step/response/phase budget, and share one remaining deadline.
BFCL, native episodes and Terminal do not restart their official allowances
inside one measurement. An explicitly requested infra retry is a new attempt.
Reported token use and execution time across automatic retries are accumulated;
attempt details retain their own paths and outcomes. Cancelled streams may still
lack provider usage, which this refactor does not invent or estimate.

External host scorer subprocesses have independent local envelopes: BFCL and
Trajectory default to 900 s; GDPVal to 600 s. These also include cancellation
grace. Native evaluators remain in their existing native lifecycle; Terminal's
verifier uses its own official phase. No gold or scorer feedback enters the agent.

Outer expiry is recorded as arm_timeout plus a measurement_envelope budget event.
An underlying provider failure is preserved, including eligibility for infra
retry. Existing native verifier failure classifications remain authoritative.
An agent's official timeout and a verifier's numeric score remain separate facts.

## Configuration and migration

| Variable | Meaning |
| --- | --- |
| HARNESS_ARM_TIMEOUT_S | General local total envelope, default 900 |
| GDPVAL_ARM_TIMEOUT_S / VITABENCH_ARM_TIMEOUT_S | Local envelopes, defaults 1200 |
| HARNESS_ARM_CANCEL_GRACE_S | Cancellation allowance inside the envelope, default 10 |
| TERMINAL_BENCH_OUTER_GRACE_S | Local overhead beyond official phases, default 120 |
| HARNESS_LOOP_SAFETY_LIMIT | Baseline loop guard, default 1000 |
| GDPVAL_MAX_TURNS / TERMINAL_BENCH_MAX_TURNS | Explicit legacy overrides of that baseline guard |
| HARNESS_SCORER_TIMEOUT_S | External BFCL/Trajectory scorer envelope, default 900; also a GDP fallback |
| GDPVAL_SCORER_TIMEOUT_S | GDP scorer-specific envelope, default 600 |

Values must be finite and valid for their unit. Resolve after dotenv loading so
environment changes cannot be ignored by import-time constants. `budget_policy`
is part of measurement identity; records also carry the resolved budget and any
budget termination. Old results remain immutable. Use a new measurement identity
for comparisons under the new limits; do not silently merge old truncated runs
or merely rescore them as if they had received the new budget.
