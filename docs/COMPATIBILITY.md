# Compatibility And Evidence

The control plane is fully self-built and has no Inspect import, runner, task,
tool, or scorer dependency. "Self-built" does not mean benchmark semantics are
reimplemented casually: official sources, data, task images, and scorers remain
the authority whenever they can run faithfully in Docker.

## Fidelity Matrix

| Benchmark | Pinned authority | Local boundary | Score claim | Current smoke evidence |
| --- | --- | --- | --- | --- |
| GAIA | `gaia-benchmark/GAIA` snapshot and public leaderboard scorer | Read-only data mount in workspace-core | Official answer normalization | `evidence/smoke-summary.json` |
| GDPval | Local official task/rubric/gold snapshot and OpenAI grading contract | Read-only data plus office-capable workspace | Proxy only; expert pairwise remains standard | `evidence/smoke-summary.json` |
| TRAJECT-Bench | `2723fd890778dbfb6af9e3aa8ee1c22272979468` | Pinned source/data image | Official metric wiring; smoke is not a model score | `evidence/smoke-summary.json` |
| VitaBench | `742e240855bf8686a0842360749d5ea970ea3987` | Pinned native package/data image | Official native evaluator | `evidence/smoke-summary.json` |
| tau2/tau3 | `79975ac5741e23fbb1d2ac44262d62398a6d87bd` | Pinned `uv.lock` package image | Official native reward | `evidence/smoke-summary.json` |
| BFCL V4 | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | Pinned source/data image with official package dependencies | Native scorer required; a bridge trajectory alone is not a score | `evidence/smoke-summary.json` |
| Terminal-Bench 2 | `2fd12b88aafdd04a52c298e3940bcb189f9766d6`, `regex-log` task image | Agent and verifier run in separate containers; tests are verifier-only and solution is oracle-smoke-only | Official task reward | `evidence/smoke-summary.json` |
| SWE-bench Verified | Harness v4.1.0 `726c5461e2ef52d83cf1ea2107870a8bb3328d57`; dataset `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` | Official controller/scorer over host Docker socket; official task images on x86_64 and a digest-pinned Epoch task image for the ARM64 smoke | Native repository tests; ARM64 image provenance is disclosed | `evidence/smoke-summary.json` |

## Verified Smokes

These are infrastructure/oracle results, not agent leaderboard measurements.

| Benchmark | Status | Execution seconds | Native check |
| --- | ---: | ---: | --- |
| GAIA | completed | 0.34 | 21 workspace tools; DDGS structured stdin |
| GDPval | completed | 0.43 | 22 workspace/Office tools; DDGS structured stdin |
| TRAJECT-Bench | completed | 1.34 | 5,910 task records; 1,228 tool records; no invalid schemas |
| VitaBench | completed | 2.31 | 400 native tasks; 82 tools; no invalid schemas |
| tau2/tau3 | completed | 10.53 | 2,804 tasks; 77 tools across six domains; no invalid schemas |
| BFCL V4 | completed | 0.50 | 9,970 records; 9,314 function schemas; full evaluator dependencies |
| Terminal-Bench 2 | completed | 143.79 | isolated official `regex-log` verifier reward `1.0` |
| SWE-bench Verified | completed | 67.52 | official report `resolved=1`, `error=0` |

The measurements above are from one clean, current-image run on 2026-08-16.
The local `runs/` directory is ignored. A sanitized portable summary is
committed under `evidence/`; no workstation path or task content is published.

## Harness Contract Smoke

The current Plan-and-Execute, CMWS, LATS, and MemGPT profiles completed a real
OpenAI-compatible API, Docker harness, tool, and final-answer loop on
2026-08-19. The identical arithmetic request produced the following transport
evidence; it is not a benchmark score.

| Profile | Harness seconds | LLM calls | Tool calls | Answer |
| --- | ---: | ---: | ---: | --- |
| Plan-and-Execute, source-aligned run 1 | 17.58 | 9 | 4 | `42` |
| Plan-and-Execute, source-aligned run 2 | 56.59 | 8 | 4 | `252` |
| CMWS, assignment-isolated run 1 | 12.37 | 10 | 5 | `42` |
| CMWS, assignment-isolated run 2 | 16.02 | 10 | 5 | `42` |
| LATS | 16.64 | 8 | 3 | `42` |
| MemGPT | 7.70 | 4 | 3 | `42` |

Plan-and-Execute is aligned to `langchain-experimental==0.0.65` at revision
`0207dc1431c29379b724f51c09fa49e6b0333639`: each executor sees previous steps
and its current objective, the full task is not injected by default, and the
last step response is returned without another synthesis call. The repeated
run still exposed a source-method limitation rather than a transport failure:
the model multiplied during the beta-retrieval step, recorded `42` as beta, and
the next executor correctly computed `6 * 42 = 252`. The first aligned run was
correct, and an Actor-only control completed the same request as `42` in 5.36
seconds with four LLM calls and three tool calls.

CMWS is explicitly a local conventional control, not an attributed paper
reproduction. Workers now receive only their assignment; the manager retains
the original task for synthesis. Both repeated runs were correct. The manager
still placed a dependent multiplication assignment in the same nominally
independent wave, so that worker re-fetched alpha and beta. This is visible
baseline behavior, not hidden workflow repair. HarnessEval does not add
task-specific prompts or post-hoc answer correction.

The expanded theory-profile suite covers all thirteen registered profiles: 52
single-turn, 26 native-conversation, and 26 task-container protocol subtests.
All 104 baseline x benchmark lifecycle cells have an explicit bridge contract.
LATS cells without read-only tools or snapshot/restore support pass by refusing
the invalid shared-state execution before an LLM call. Native
conversation and task-container rows use the same profile implementations but
different benchmark-owned lifecycle brokers; they are not flattened into
single-turn text tasks. These scripted protocol tests establish routing and
tool-contract correctness, not model task success. The suite also exercises
nested structured observations, credential isolation, and a 200,000-character
tool value with no content slicing. A
Plan-and-Execute retry retained its failed first attempt, completed as attempt
two after a structured-result protocol correction, and an identical third
invocation resumed without creating another attempt.

A tau2 integration run traversed the official hidden-user tool call, assistant
tool call, environment mutation, stop condition, and native evaluator in one
episode, returning reward `1.0`. It used a deterministic local API fixture to
validate transport and lifecycle only; it is not reported as a model score.

## Result Contract

Each singleton case owns this append-only attempt structure:

```text
RUN_DIR/<benchmark>/<case>/
  result.json
  attempts/0001/
    request.json
    events.jsonl
    terminal.log
    payload.json
    result.json
```

`result.json` is an atomic pointer to the latest completed attempt. Resume skips
terminal states; `--retry-failed` creates a new numbered attempt. No prior log or
payload is overwritten, and no prompt, tool result, terminal stream, or payload
is character-sliced by the platform.

An advisory lock serializes only identical benchmark/case keys. It is released
by the kernel if the controller exits, so crash recovery does not depend on a
wall-clock timeout or a stale-lock guess.

The generic payload contract is deliberately small. A benchmark may write the
complete native metrics to `/job/payload.json`; the control plane embeds that
JSON without inventing a score from terminal text. Infrastructure/oracle smokes
set `oracle_smoke: true` and must not be entered into model leaderboards.

## Explicit Exceptions

- GDPval automated judges are diagnostic proxies, not substitutes for the
  published expert pairwise process.
- BFCL image dependency completeness and native score completeness are separate.
  A recorded function-call trajectory must pass the official category scorer
  before it is reported as a BFCL result.
- The tau2 `banking_knowledge` schema probe uses the official all-tools class
  without constructing its eager dense index. Real episodes still use the
  official environment and require an embedding provider if no embeddings
  cache is present.
- SWE-bench needs direct Docker socket access because its official controller
  creates task containers. Only this catalog entry may request root execution.
  Linux x86_64 follows the official image path unchanged. The ARM64 smoke path
  verifies the Epoch image digest/instance label, binds only the official
  `make_test_spec` `arch` argument to `arm64`, and gives the image a matching
  local name. The pinned official data row, generated eval script, task text,
  patches, tests, and scorer remain unchanged.

OSWorld and other computer-use suites are not cataloged. This repository does
not replace their VM/browser state with a text-only approximation.
