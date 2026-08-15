# Benchmark Extension Status

HarnessEval registers only adapters whose local execution and scoring boundary
have been exercised. A repository clone or import check is not reported as an
end-to-end benchmark.

## Assessed Candidates

| Candidate | Official source assessed | Decision |
| --- | --- | --- |
| Claw Bench | `claw-bench/claw-bench` at `1fc25add8fe77aa498d58fb564ea91a87307da76` | Deferred. Tasks have Docker workspaces and weighted pytest verifiers, but the native product-agent sandbox/controller must be integrated without replacing its lifecycle with a file-only oracle. |
| SkillsBench 1.1 | `benchflow-ai/skillsbench` at `9a1f4dd5f7659f75707435da3ce854b6e48321d1` | Deferred. It is compatible in principle through BenchFlow task containers, but paired with-Skills/without-Skills execution and per-task image orchestration must stay intact. |
| PinchBench | `pinchbench/skill` at `819384ae830492365b8363fc26bc2602e73f216d` | External product profile. Its runner evaluates OpenClaw-specific behavior and optional leaderboard upload, so a generic harness substitution would change the benchmark. |
| Browser ClawBench | `TIGER-AI-Lab/ClawBench` | Deferred with other computer-use suites because it depends on browser/live-site execution and request interception. |
| OSWorld | `xlang-ai/OSWorld-V2` | Not provided. It requires a KVM-capable Linux computer-use environment and gated assets. |

These decisions avoid two misleading shortcuts: reporting dataset integrity as
agent success, and replacing an official stateful/GUI/product lifecycle with a
one-shot prompt. Users can still add a private catalog entry or run an official
controller through the generic command adapter while preserving its scorer.

## Adding A Benchmark

1. Pin the official repository commit, release, dataset revision, and task-image
   digest where available.
2. Keep benchmark task generation, environment state, tools, and scorer native.
3. Build a minimal controller image; do not copy a workstation dependency freeze.
4. Mount datasets read-only and place generated artifacts under `/job`.
5. Add an offline infrastructure/oracle smoke and label it as such.
6. Add one real agent singleton before claiming end-to-end support.
7. Document official, subset, proxy, or external scoring precisely.

Use `--catalog path/to/catalog.json` to maintain organization-specific entries
without forking the control plane.
