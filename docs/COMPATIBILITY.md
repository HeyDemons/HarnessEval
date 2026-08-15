# Compatibility And Evidence

The control plane is fully self-built and has no Inspect import, runner, task,
tool, or scorer dependency. "Self-built" does not mean benchmark semantics are
reimplemented casually: official sources, data, task images, and scorers remain
the authority whenever they can run faithfully in Docker.

## Fidelity Matrix

| Benchmark | Pinned authority | Local boundary | Score claim | Current smoke evidence |
| --- | --- | --- | --- | --- |
| GAIA | `gaia-benchmark/GAIA` snapshot and public leaderboard scorer | Read-only data mount in workspace-core | Official answer normalization | `runs/harnesseval_release_core_20260815` |
| GDPval | Local official task/rubric/gold snapshot and OpenAI grading contract | Read-only data plus office-capable workspace | Proxy only; expert pairwise remains standard | `runs/harnesseval_release_core_20260815` |
| TRAJECT-Bench | `2723fd890778dbfb6af9e3aa8ee1c22272979468` | Pinned source/data image | Official metric wiring; smoke is not a model score | `runs/harnesseval_release_core_20260815` |
| VitaBench | `742e240855bf8686a0842360749d5ea970ea3987` | Pinned native package/data image | Official native evaluator | `runs/harnesseval_release_core_20260815` |
| tau2/tau3 | `79975ac5741e23fbb1d2ac44262d62398a6d87bd` | Pinned `uv.lock` package image | Official native reward | `runs/harnesseval_release_core_20260815` |
| BFCL V4 | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | Pinned source/data image | Official agentic scorer subset only | `runs/harnesseval_release_core_20260815` |
| Terminal-Bench 2 | `2fd12b88aafdd04a52c298e3940bcb189f9766d6`, `regex-log` task image | Official task image with solution/verifier mounts | Official task reward | `runs/harnesseval_release_terminal_20260815` |
| SWE-bench Verified | Harness v4.1.0 `726c5461e2ef52d83cf1ea2107870a8bb3328d57`; dataset `c104f840cc67f8b6eec6f759ebc8b2693d585d4a` | Official controller/scorer over host Docker socket; official task images on x86_64 and digest-pinned Epoch task images on Apple ARM64 | Native repository tests; ARM64 image provenance is disclosed | `runs/harnesseval_release_swe_20260815` |

## Verified Smokes

These are infrastructure/oracle results, not agent leaderboard measurements.

| Benchmark | Status | Execution seconds | Native check |
| --- | ---: | ---: | --- |
| GAIA | completed | 1.71 | public scorer oracle `1.0` |
| GDPval | completed | 1.84 | dataset/rubric integrity `1.0` |
| TRAJECT-Bench | completed | 0.91 | 5,910 records, 1,228 tools |
| VitaBench | completed | 2.14 | 400 native tasks, package integrity `1.0` |
| tau2/tau3 | completed | 16.59 | CLI, data, six domains, registry integrity `1.0` |
| BFCL V4 subset | completed | 0.28 | 70 data JSON files, agentic scorer oracle `1.0` |
| Terminal-Bench 2 | completed | 133.92 | official `regex-log` verifier reward `1.0` |
| SWE-bench Verified | completed | 45.22 | official report `resolved=1`, `error=0` |

The ignored local `runs/` paths above are development evidence. A sanitized,
portable summary is committed under `evidence/`; no workstation path or task
content is published.

## Harness Contract Smoke

All four built-in profiles completed a real OpenAI-compatible API, harness,
tool, and final-answer loop in Docker on 2026-08-15. Actor-only used 20.25
seconds, ReAct 21.69, Plan-and-Execute 12.28, and CMWS 24.28; each returned
`42`. CMWS issued two independent tool requests within 0.001 seconds and
received both results within 0.002 seconds, exercising a real concurrent wave.

The theory-profile unit suite also exercises nested structured observations,
credential isolation, and a 200,000-character tool value with no content
slicing. A Plan-and-Execute retry retained its failed first attempt, completed
as attempt two after a structured-result protocol correction, and an identical
third invocation resumed without creating another attempt.

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
- BFCL currently excludes AST/executable and vector-memory profiles. The image
  must not be described as full BFCL V4 support.
- SWE-bench needs direct Docker socket access because its official controller
  creates task containers. Only this catalog entry may request root execution.
  Linux x86_64 follows the official image path unchanged. The Apple ARM64 path
  verifies the Epoch image digest/instance label, binds only the official
  `make_test_spec` `arch` argument to `arm64`, and gives the image a matching
  local name. The pinned official data row, generated eval script, task text,
  patches, tests, and scorer remain unchanged.

OSWorld and other computer-use suites are not cataloged. This repository does
not replace their VM/browser state with a text-only approximation.
