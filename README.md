# HarnessEval

HarnessEval is an independent, Docker-native control plane for evaluating
agent harnesses without an Inspect runtime dependency. It standardizes
isolation, source provenance, singleton persistence, resume, logs, and result
envelopes while leaving task semantics, tools, and scoring with each benchmark.

## What It Provides

- Eight benchmark adapters with pinned sources and explicit scoring claims.
- A generic `run` path for any harness command already available in a task image.
- A portable harness request contract for user-supplied tools and
  OpenAI-compatible APIs.
- Eleven source-pinned theory profiles, including AFlow, DyLAN,
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

The current light design includes GAIA 10/20/30 by level, 27 sector-balanced
GDPval tasks, the unbiased VitaBench-60, and representative tau2, BFCL, and
Terminal-Bench 2 subsets. TRAJECT-Bench is held until replay endpoints are
verified. Exact denominators, scoring limits, and selection integrity rules are
in [the evaluation suite specification](docs/EVALUATION_SUITES.md).

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

`harnesseval matrix` reports every baseline x benchmark cell. `runnable` means
the lifecycle bridge exists; it does not mean the case succeeded or that a
publishable native score is available. See
[the baseline matrix](docs/BASELINE_MATRIX.md).

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
