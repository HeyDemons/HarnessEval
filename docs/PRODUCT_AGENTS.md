# Product Agent Integration

Product agents are kept separate from bundled paper baselines. Their native
distributions, prompts, tools, authentication, and version strings are part of
the evaluated system and must be recorded.

## Common Pattern

HarnessEval supports two faithful product layouts. A packaged product may run
inside a benchmark-specific image through `harnesseval run`. Pi, Codex CLI, and
Claude Code use `harnesseval product-run`: the executable stays on the macOS
host, while a versioned loopback bridge exposes only benchmark-declared tools
and keeps environment state and native scoring in Docker.

In either layout, the adapter must:

1. Read the benchmark-owned prompt and workspace.
2. Run the product in non-interactive mode.
3. Preserve its complete event stream under `/job`.
4. Invoke the benchmark's native scorer.
5. Write native metrics to `/job/payload.json`.

Then launch it with `harnesseval run ... -- adapter-command`. Pass credentials
by allow-listed variable name and never put a key in the command or catalog.

For a supported host product, use this general CLI form instead:

```bash
harnesseval product-run PRODUCT BENCHMARK \
  --case CASE_ID \
  --run-dir runs/PRODUCT-BENCHMARK \
  [product-specific model and provider options]
```

The host process receives the benchmark prompt and exact tool schemas, but not
the hidden scorer state. Calls are executed by the benchmark Docker lifecycle;
the resulting native score, complete event stream, execution time, and resume
identity are written to the same per-attempt result envelope as container-side
harnesses. Thus `product-run` is a CLI transport mode, not a different task or
scoring protocol.

Do not compare a host-native product with a Docker baseline unless filesystem,
network, tool access, model settings, and scorer closure are equivalent and
reported. Container isolation is part of the evaluated harness.

## Pi

Pi provides a non-interactive JSON mode and explicit tool allow-lists.
HarnessEval includes a host-native adapter, so an installed Pi can be evaluated
end to end without installing it in every benchmark image:

```bash
harnesseval product-run pi gaia \
  --case CASE_ID \
  --run-dir runs/pi-gaia \
  --provider deepseek \
  --model deepseek-v4-flash \
  --thinking high
```

The adapter records the resolved Pi executable, version and digest, selected
provider/model/thinking level, generated extension digest, and prepared-case
digest in its resume identity. It starts the existing benchmark Docker bridge,
registers the complete case-specific schemas as Pi tools, streams complete
arguments and results over loopback HTTP, and invokes the existing native
scorer or records its explicit external/not-run status. VitaBench and tau retain
their hidden-user multi-turn lifecycle; Terminal-Bench and SWE-bench retain
their live task containers and official verifier finalizers.

Pi runs with `--no-session`, `--offline`, `--no-context-files`, `--no-skills`,
`--no-prompt-templates`, `--no-extensions`, and `--no-builtin-tools`. Only the
HarnessEval extension and benchmark-declared tool names are enabled. Local
projects and automatic Pi resources therefore cannot change benchmark
authority. Authentication stays in the local Pi installation; credentials are
never copied into Docker or serialized. Benchmark-side API variables can cross
only by allow-listed name:

```bash
harnesseval product-run pi tau2 \
  --case mock:create_task_1 \
  --run-dir runs/pi-tau \
  --provider deepseek \
  --model deepseek-v4-flash \
  --thinking high \
  --pass-env HARNESS_API_BASE \
  --pass-env HARNESS_API_KEY \
  --pass-env HARNESS_MODEL
```

For a Pi custom provider whose `models.json` refers to an environment variable,
pass that name only to Pi:

```bash
harnesseval product-run pi gaia \
  --case CASE_ID \
  --run-dir runs/pi-custom-gaia \
  --provider custom-provider \
  --model custom-model \
  --pi-env CUSTOM_PROVIDER_API_KEY
```

`--pi-env` and `--pass-env` are intentionally separate. The former is visible
only to the host Pi process; the latter is reserved for a benchmark service or
hidden-user model that explicitly needs it.

Each case is atomic and resumable. `--retry-failed` appends an attempt without
overwriting evidence. Changing the Pi binary, model, provider, thinking level,
benchmark policy, or prepared case requires a new run directory or
`--no-resume`. Pi execution time is reported separately from Docker bridge
startup and official scoring time.

## Codex CLI

HarnessEval includes a host-native adapter for the official Codex CLI. It uses
the same benchmark-owned Docker bridge and scorer closure as Pi, while an
attempt-local `CODEX_HOME` prevents personal configuration from entering the
experiment. For a custom Responses-compatible provider:

```bash
export PROVIDER_API_KEY=...
harnesseval product-run codex gaia \
  --case CASE_ID \
  --run-dir runs/codex-gaia \
  --provider provider-id \
  --base-url https://provider.example/v1 \
  --api-key-env PROVIDER_API_KEY \
  --model model-id \
  --thinking high
```

The adapter prefers the Codex binary bundled with ChatGPT on macOS and falls
back to an executable `codex` on `PATH`. It records the resolved binary,
version, digest, model/provider/reasoning settings, MCP bridge digest, and
prepared-case digest. Custom provider keys are read only from the environment
name supplied by `--api-key-env`; neither their values nor a personal Codex
home are serialized or mounted into Docker.

Personal plugins, shell, built-in web search, multi-agent tools, hooks, goals,
and memories are disabled. A required local MCP bridge exposes only the
benchmark-declared schemas. Tool arguments, results, event JSONL, stderr, and
terminal output are preserved completely. The benchmark's native multi-turn or
live-container lifecycle remains authoritative, and `agent_execution_seconds`
is separate from environment setup and scoring time.

`--codex-env NAME` passes an additional named variable only to Codex. It is
separate from `--pass-env`, which is restricted to the benchmark's declared
allowlist and crosses into benchmark Docker only when explicitly requested.

The Codex CLI reference is maintained in the
[official OpenAI documentation](https://developers.openai.com/codex/cli/reference/).

## Claude Code

HarnessEval can evaluate an installed Claude Code CLI through its native
Anthropic Messages protocol while retaining the same Docker tool bridge and
native scorer closure used by the other product agents:

```bash
export PROVIDER_API_KEY=...
harnesseval product-run claude gaia \
  --case CASE_ID \
  --run-dir runs/claude-gaia \
  --provider provider-id \
  --base-url https://provider.example \
  --api-key-env PROVIDER_API_KEY \
  --model claude-model-id \
  --thinking high
```

Claude Code runs with `--bare`, no session persistence, no slash commands, no
Chrome integration, and no built-in tools. An explicit strict MCP configuration
and tool allowlist expose only benchmark-declared schemas. Attempt-local home and
configuration directories exclude personal settings, projects, hooks, plugins,
skills, memory, and authentication state. `--claude-env NAME` passes an
additional variable only to the local Claude Code process. The selected provider
key is mapped to `ANTHROPIC_API_KEY` in process memory and is never written to a
command, config, identity, Docker mount, or result artifact.

## PERSEUS

PERSEUS is integrated as an external product harness rather than a built-in
HarnessEval theory profile. This keeps its Actor/Speculator control plane,
dependencies, version, and traces independently attributable.

The public repository includes a custom HarnessEval catalog, Docker product
image, adapter, exact-file oracle smoke, and matched-control protocol:

```bash
git clone https://github.com/HuiCir/Perseus.git
cd Perseus
bash integrations/harnesseval/run-smoke.sh
```

For benchmark experiments, let HarnessEval retain task-image, case, attempt,
resume, logging, and native-scorer ownership. Install or mount PERSEUS into the
task image and execute it through `harnesseval run ... --`. Workspace tools can
remain native; service simulators require a versioned Pi extension that maps
the benchmark's complete tool schemas and results without changing task
semantics. Only benchmark-declared read/query tools belong in the speculative
safe list. Mutations remain authoritative Actor calls.

Run every PERSEUS case against `PERSEUS_ENABLED=0` with the same Actor model,
prompt, image, tools, and scorer. Use distinct case keys or run directories so
resume never conflates the paired configurations.

## Scoring Closure

Pi, Codex, or Claude Code exiting successfully is not task success. The adapter must still run
the benchmark's official verifier or scorer. If a benchmark uses an external
human/pairwise stage, label local model judging as a proxy rather than promoting
it to an official score.
