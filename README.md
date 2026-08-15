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
- Four bundled theory baselines: Actor-only, ReAct, Plan-and-Execute, and CMWS.
- Atomic per-case results, append-only attempts, process locks, resume, and
  complete terminal/JSONL logs.
- No prompt, tool-result, log, or artifact slicing by character count.

HarnessEval is a control plane, not a universal replacement for benchmark
scorers. A benchmark result is official only when its catalog entry says so.

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
```

The host package has no runtime dependency outside the standard library.

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

## Registered Benchmarks

| Benchmark | Runtime | Scoring claim |
| --- | --- | --- |
| GAIA | Read-only dataset/tool image | Public leaderboard answer normalization |
| GDPval | Artifact workspace image | Official rubrics; automated judging remains a proxy |
| TRAJECT-Bench | Pinned API-only source image | Native trajectory metrics |
| VitaBench | Pinned official package/data image | Native assertions and rubric evaluator |
| tau2/tau3 | Pinned source and `uv.lock` | Native state/action reward |
| BFCL V4 | Pinned source/data image | Official agentic subset only |
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
integration is documented in [docs/PRODUCT_AGENTS.md](docs/PRODUCT_AGENTS.md).
