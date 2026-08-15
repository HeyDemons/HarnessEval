# Product Agent Integration

Product agents are kept separate from bundled paper baselines. Their native
distributions, prompts, tools, authentication, and version strings are part of
the evaluated system and must be recorded.

## Common Pattern

Create a benchmark-specific image containing the product CLI and its runtime,
or mount a versioned adapter into an existing task image. The adapter must:

1. Read the benchmark-owned prompt and workspace.
2. Run the product in non-interactive mode.
3. Preserve its complete event stream under `/job`.
4. Invoke the benchmark's native scorer.
5. Write native metrics to `/job/payload.json`.

Then launch it with `harnesseval run ... -- adapter-command`. Pass credentials
by allow-listed variable name and never put a key in the command or catalog.

Do not compare a host-native product with a Docker baseline unless filesystem,
network, tool access, model settings, and scorer closure are equivalent and
reported. Container isolation is part of the evaluated harness.

## Pi

Pi provides a non-interactive text/JSON mode and explicit tool allow-lists. A
minimal adapter invocation inside a prepared task image is:

```bash
pi --mode json --print --no-session \
  --provider PROVIDER \
  --model MODEL \
  --tools read,bash,edit,write \
  "$(cat /task/prompt.txt)" \
  > /job/pi-events.jsonl
```

Pin the installed Pi version in the image. Use `--no-context-files` and disable
automatic skills/extensions when the benchmark does not grant them; otherwise
record every enabled resource as part of the harness configuration. Let the
benchmark adapter decide the tool allow-list instead of using one global list.

## Codex CLI

Install and pin the official Codex CLI in the task image, then use its
non-interactive `codex exec` path. A typical adapter shape is:

```bash
codex exec --json \
  --sandbox workspace-write \
  --skip-git-repo-check \
  -C /workspace \
  "$(cat /task/prompt.txt)" \
  > /job/codex-events.jsonl
```

Confirm flags against `codex exec --help` for the pinned release. Authentication
may use the product's documented login or API-key path, but do not mount an
entire personal home directory into a benchmark container. Mount only the
minimum credential/config material and never include it in artifacts.

The Codex CLI reference is maintained in the
[official OpenAI documentation](https://developers.openai.com/codex/cli/reference/).

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

Pi or Codex exiting successfully is not task success. The adapter must still run
the benchmark's official verifier or scorer. If a benchmark uses an external
human/pairwise stage, label local model judging as a proxy rather than promoting
it to an official score.
