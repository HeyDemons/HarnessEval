# HarnessEval

HarnessEval is an independent, Docker-native control plane for evaluating
agent harnesses without an Inspect runtime dependency. It standardizes
isolation, source provenance, singleton persistence, resume, logs, and result
envelopes while leaving task semantics, tools, and scoring with each benchmark.

## What It Provides

- Eight benchmark adapters with pinned sources and explicit scoring claims.
- A generic `run` path for any harness command already available in a task image.
- `product-run pi`, `product-run codex`, and `product-run claude` paths that keep product CLIs on the
  host while routing complete tool calls through benchmark-owned Docker
  environments.
- A portable harness request contract for user-supplied tools and
  OpenAI-compatible APIs.
- Thirteen theory profiles, including LATS, MemGPT, AFlow, DyLAN,
  Magentic-One, Multi-Persona, LLMCompiler, ReWOO, and Speculative Actions.
- Atomic per-case results, append-only attempts, process locks, resume, and
  complete terminal/JSONL logs.
- No prompt, tool-result, log, or artifact slicing by character count.

HarnessEval is a control plane, not a universal replacement for benchmark
scorers. A benchmark result is official only when its catalog entry says so.

## Evaluation Scale

Experiments use two explicit suite modes. `light` resolves a frozen,
outcome-independent subset for broad baseline comparisons; `full` delegates to
the complete case registry owned by the pinned benchmark release.

```bash
harnesseval suite all --mode light
harnesseval suite gaia --mode light --ids-only
harnesseval suite all --mode full --json
```

The current light design includes 56 locally scoreable GAIA cases (10/20/26 by
level), 27 sector-balanced GDPval tasks, 100 endpoint-audited TRAJECT cases,
the unbiased VitaBench-60, and representative tau2, BFCL, and Terminal-Bench 2
subsets. Exact denominators, scoring limits, and selection integrity rules are
in [the evaluation suite specification](docs/EVALUATION_SUITES.md).

The repository-level matrix runner accepts every ready non-SWE light suite. Run a
server preflight before spending model tokens:

```bash
python3 scripts/run_bench.py --benchmark vitabench --preflight
python3 scripts/run_bench.py --benchmark trajectory-bench --preflight
python3 scripts/run_bench.py --benchmark bfcl --preflight

python3 scripts/run_bench.py --benchmark vitabench --limit 3 --methods react,perseus
python3 scripts/run_bench.py --benchmark bfcl --limit 3 --methods react,perseus
```

TRAJECT requires a ToolBench-compatible execution endpoint in `API_URL`.
`TOOLBENCH_KEY` is optional: it is sent in both the JSON body and request header when
set, while StableToolBench MirrorAPI supports the documented empty key. For a
StableToolBench service running on the same server as Docker, use
`API_URL=http://localhost:<port>/virtual`; the runner translates host loopback to
`host.docker.internal` inside benchmark containers and adds Docker's `host-gateway`
mapping on Linux. The explicit `host.docker.internal` form is also accepted. For a
remote or protected endpoint, set its full URL and a non-empty `TOOLBENCH_KEY`.

```bash
export API_URL=http://localhost:8080/virtual
export TOOLBENCH_KEY=
python3 scripts/run_bench.py --benchmark trajectory-bench --preflight
```

GDPval scores are independent model-rubric proxy scores, consistent with its catalog
comparability claim; they are not represented as expert pairwise grades. The BFCL light
suite is limited to the 13 independently scoreable single-turn categories; stateful
multi-turn, memory, and web-search categories remain full-suite work.

## Install

Requirements are Python 3.11+, Docker, and Git.

```bash
git clone https://github.com/HuiCir/HarnessEval.git
cd HarnessEval
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
harnesseval list
harnesseval harnesses
harnesseval matrix
```

The host package has no runtime dependency outside the standard library.
Host, architecture, BuildKit, proxy, and image-download details are documented
in [the macOS installation guide](docs/INSTALLATION.md).

## CLI Execution Modes

HarnessEval exposes one CLI but preserves four distinct execution boundaries:

| Command | Agent location | Benchmark tools, state, and scorer | Intended use |
| --- | --- | --- | --- |
| `harnesseval run ... -- COMMAND` | Inside the declared task image | Inside the benchmark lifecycle | Native benchmark or externally packaged harness commands |
| `harnesseval harness-run METHOD ...` | Inside a portable harness image | Request-defined tools and finalizer | Contract tests and user-supplied tool packages |
| `harnesseval bridge-run METHOD BENCHMARK ...` | Inside HarnessEval's isolated baseline runtime | Benchmark-owned Docker bridge and native scorer | Built-in paper and theory baselines |
| `harnesseval product-run PRODUCT BENCHMARK ...` | Product CLI on the macOS host | Benchmark-owned Docker bridge and native scorer | Pi, Codex CLI, and Claude Code comparisons |

For `product-run`, the product executable itself is not copied into Docker.
Only the benchmark-declared tool schemas cross the loopback bridge; complete
calls execute in the benchmark environment and return to the host CLI. Product
credentials remain host-only, while benchmark service credentials cross the
container boundary only through explicit `--pass-env` allowlists. This is an
end-to-end benchmark run, not a host-only approximation.

## Benchmark Smoke

Set `--orch-root` to a directory containing any local datasets named in the
catalog, or provide a customized catalog with `--catalog`.

```bash
harnesseval --orch-root /path/to/benchmark-data doctor gaia gdpval trajectory-bench
harnesseval --orch-root /path/to/benchmark-data smoke gaia gdpval trajectory-bench \
  --run-dir runs/infrastructure-smoke
```

Infrastructure/oracle smokes prove source, data, container, and scorer wiring.
They are marked `oracle_smoke: true` and are not model scores.

To execute a benchmark-native harness command:

```bash
harnesseval run vitabench \
  --case delivery-H0717001 \
  --run-dir runs/vitabench \
  --pass-env OPENAI_API_KEY \
  -- vita run --domain delivery --task-ids H0717001 --num-trials 1
```

Everything after `--` executes inside the benchmark image. HarnessEval does not
rewrite it into a generic loop or infer scores from terminal text.

## Built-In Harnesses

Configure any OpenAI-compatible provider with names that are safe to record:

```bash
export HARNESS_API_BASE=https://provider.example/v1
export HARNESS_API_KEY=...
export HARNESS_MODEL=model-id
```

Anthropic Messages providers use `HARNESS_API_TYPE=anthropic-messages` and
require `HARNESS_MAX_OUTPUT_TOKENS`. Official Anthropic authentication defaults
to `HARNESS_API_AUTH=x-api-key`; compatible gateways that issue bearer tokens
can select `HARNESS_API_AUTH=bearer` explicitly. `HARNESS_API_USER_AGENT` may
override the recorded `HarnessEval/0.1` transport identifier when a provider
requires a specific client identifier.

For example, the same built-in method can use an Anthropic-compatible gateway
without changing its harness logic:

```bash
export HARNESS_API_BASE=https://provider.example
export HARNESS_API_KEY=...
export HARNESS_MODEL=claude-model-id
export HARNESS_API_TYPE=anthropic-messages
export HARNESS_API_AUTH=x-api-key
export HARNESS_MAX_OUTPUT_TOKENS=16384
```

Run the complete API -> harness -> tool -> answer loop inside Docker:

```bash
harnesseval harness-run react \
  --request examples/harness-request.json \
  --case arithmetic-smoke \
  --run-dir runs/harness-smoke \
  --mount "$PWD/examples/tools:/tools:ro" \
  --pass-env HARNESS_API_BASE \
  --pass-env HARNESS_API_KEY \
  --pass-env HARNESS_MODEL
```

The same command accepts a benchmark task image through `--image`, read-only or
writable task mounts through `--mount`, and a request-defined official scorer
through `finalizer.command`. See [the harness contract](docs/HARNESS_CONTRACT.md).

For benchmark-owned tools, use the isolated bridge instead of copying tool
implementations into a baseline:

```bash
harnesseval bridge-run sa gaia \
  --case CASE_ID \
  --run-dir runs/sa-gaia \
  --pass-env HARNESS_API_BASE \
  --pass-env HARNESS_API_KEY \
  --pass-env HARNESS_MODEL
```

The same command preserves stateful and task-container lifecycles:

```bash
# VitaBench case IDs are the native task IDs.
harnesseval bridge-run magentic-one vitabench --case H0717001 \
  --run-dir runs/magentic-vita --pass-env HARNESS_API_BASE \
  --pass-env HARNESS_API_KEY --pass-env HARNESS_MODEL

# tau task IDs are namespaced because IDs repeat across official task sets.
harnesseval bridge-run react tau2 --case mock:create_task_1 \
  --run-dir runs/react-tau --pass-env HARNESS_API_BASE \
  --pass-env HARNESS_API_KEY --pass-env HARNESS_MODEL
```

GAIA and GDPval inject their benchmark workspace tools plus structured DDGS
`web_search`. TRAJECT-Bench and BFCL inject each case's declared schemas. Vita
and tau keep the hidden user and mutable official environment outside the
baseline. Terminal-Bench and SWE-bench give the baseline only the task
workspace; verifier tests and reference solutions never enter its container.

AFlow additionally requires a frozen operator list produced on a disjoint
optimization split, for example `--policy '{"aflow_workflow":["Custom"]}'`.
DyLAN and Multi-Persona intentionally receive no external tools because their
published protocols do not define a tool loop.

LATS preserves tree expansion over independent environment states. It runs only
when every declared tool is read-only; benchmark environments with mutating
tools require a snapshot/restore bridge and are rejected instead of being
silently reduced to a serial loop. MemGPT exposes its core, recall, and archival
memory functions alongside the benchmark's dynamic tools and chains work through
function-result heartbeats until `send_message`.

`harnesseval matrix` reports every baseline x benchmark cell. `runnable` means
the lifecycle bridge exists; it does not mean the case succeeded or that a
publishable native score is available. See
[the baseline matrix](docs/BASELINE_MATRIX.md).

## Local Product CLIs

HarnessEval can drive an installed Pi CLI without copying the user's Pi home or
credentials into a benchmark container:

```bash
harnesseval product-run pi gaia \
  --case CASE_ID \
  --run-dir runs/pi-gaia \
  --provider deepseek \
  --model deepseek-v4-flash \
  --thinking high
```

The benchmark prompt and schemas come from the isolated Docker adapter. Pi gets
only a generated extension for those declared tools; built-in tools, ambient
extensions, skills, context files, prompt templates, and session reuse are
disabled. Complete Pi events, tool arguments/results, native scores, attempts,
and resume identities are persisted. Static, native multi-turn, and live task
container adapters use the same command. Benchmark-side API variables cross
only when explicitly named with `--pass-env`. Custom provider variables for Pi
use `--pi-env` and never enter benchmark Docker.

An installed official Codex CLI uses the same task, Docker bridge, scorer,
attempt, and resume closure. For an OpenAI Responses-compatible custom
provider:

```bash
export PACKY_API_KEY=...
harnesseval product-run codex gaia \
  --case CASE_ID \
  --run-dir runs/codex-gaia \
  --provider packy \
  --base-url https://cf.api.fan/v1 \
  --api-key-env PACKY_API_KEY \
  --model gpt-5.6-terra \
  --thinking high
```

The key value is not written to config, commands, identities, or artifacts.
Codex runs under an attempt-local home with personal plugins, shell, built-in
search, multi-agent tools, hooks, and memories disabled. The benchmark's exact
tool schemas are injected over a required local MCP server; complete calls and
results remain in the benchmark trace without character-count slicing.

Claude Code can use the same benchmark and scoring closure through its native
Anthropic Messages protocol:

```bash
export PACKY_API_KEY=...
harnesseval product-run claude gaia \
  --case CASE_ID \
  --run-dir runs/claude-gaia \
  --provider packy \
  --base-url https://cf.api.fan \
  --api-key-env PACKY_API_KEY \
  --model claude-sonnet-5 \
  --thinking high
```

The adapter runs Claude Code in bare, non-persistent mode under attempt-local
home and configuration directories. Built-in tools and ambient resources are
disabled; an explicit MCP allowlist exposes only the complete benchmark-owned
tool schemas. Provider keys remain host-only and are mapped to Claude Code's
native authentication environment without being serialized.

## Registered Benchmarks

| Benchmark | Runtime | Scoring claim |
| --- | --- | --- |
| GAIA | Read-only dataset/tool image | Public leaderboard answer normalization |
| GDPval | Artifact workspace image | Official rubrics; automated judging remains a proxy |
| TRAJECT-Bench | Pinned API-only source image | Native trajectory metrics |
| VitaBench | Pinned official package/data image | Native assertions and rubric evaluator |
| tau2/tau3 | Pinned source and `uv.lock` | Native state/action reward |
| BFCL V4 | Pinned source/data and full dependency image | Native scorer required for publication |
| Terminal-Bench 2 | Official task image and verifier | Native task reward |
| SWE-bench Verified | Official controller over task images | Native repository tests |

OSWorld is intentionally not advertised or stubbed: faithful execution needs a
KVM-capable Linux computer-use environment and gated assets. ClawBench,
SkillsBench, and PinchBench have been assessed but are not silently reduced to
metadata-only scores; see [extension status](docs/EXTENSIONS.md).

## Integrity Rules

- Secrets are passed only by allow-listed environment variable name. Values are
  never serialized into commands, manifests, or results.
- Tool subprocesses do not inherit model API credentials unless the tool itself
  explicitly declares a `pass_env` name.
- One case owns one isolated directory and one advisory lock. Different cases
  remain independently parallelizable.
- Resume skips terminal results. `--retry-failed` appends a new attempt and
  never overwrites old evidence.
- Docker images with pinned source labels are rebuilt when labels are stale.
- `/var/run/docker.sock` and root execution are exceptions for official nested
  Docker controllers, never generic harness defaults.

Detailed fidelity and local smoke evidence are in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md). Pi and Codex application-level
integration, together with the external PERSEUS speculative swarm path, is
documented in [docs/PRODUCT_AGENTS.md](docs/PRODUCT_AGENTS.md).
