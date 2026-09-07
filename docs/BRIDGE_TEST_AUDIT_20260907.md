# Bridge contract test review — 2026-09-07

Scope: BFCL, Tau2, AutomationBench, GAIA and Terminal-Bench-2; plus a
defensive rejection check for LATS/Trajectory-Bench. No Perseus execution,
VitaBench integration work, algorithm changes or scoring changes.

## Findings and fixes

1. **BFCL matrix: confirmed.** The previous matrix constructed a generic
   ToolEnvironment without declaration-only wiring. It therefore exercised a
   different lifecycle and accepted eleven multi-response profiles. The matrix
   now calls `bridges.runner.execute`, asserts their rejection before any model
   request, and checks one model response / zero environment calls for supported
   profiles. LATS/BFCL is rejected by the declaration gate, not the branch gate.
   Existing batch-declaration tests remain in `test_declaration_protocol.py`.
2. **Tau2 broker fixture: confirmed.** An unmarked lookup never enabled SA
   speculation. The fixture now exposes a parallel read-only lookup and asserts
   that SA fails before model calls or native queued actions. This is a broker
   protocol test, not a claim that every profile is eligible for Tau2; the
   production Tau2 entrypoint/compatibility rejection is covered separately in
   `test_native_isolation_and_limits.py`.
3. **AutomationBench matrix and injection: confirmed.** `run_episode` now accepts
   an optional `speculator_client`; omission retains the existing environment
   factory. The new matrix covers all eleven currently eligible profiles through
   `run_episode`, each executing an environment tool before post-agent scoring.
   It retains native Actor/ReAct and each other method's own protocol, checks
   public system instructions/private assertion separation, strict and partial
   score fields, and the 50-response policy. The AFlow-tools initial graph is an
   explicitly marked test fixture, not a searched artifact or an evaluation run.
4. **SA bridge adoption coverage: confirmed.** New hit/miss cases call the GAIA,
   TB2 and AutomationBench production entrypoints. They check actual handler
   invocation counts, one published Actor call, Actor response attribution,
   prediction events, hit events, Actor/Speculator response counts, and absence
   of discarded prediction arguments from Actor messages. GAIA handlers read real
   temporary files; TB2 uses real handler/path/command construction with only the
   Docker subprocess boundary mocked. AutomationBench uses a synthetic episode.
5. **LATS/Trajectory runtime bypass: confirmed.** The runner already supplies
   `branch_safe_tools=[]`, but LATS ignored it. LATS now requires all tools to
   satisfy the supplied allowlist as well as read-only metadata, and rejects an
   environment without isolated execution support. The production-runner matrix
   checks rejection before model calls. This does not enable LATS anywhere.
   The feedback's reference to `run_sa` at this point should read `run_lats`:
   SA already filters `speculation_safe_tools`.
6. **Stale helper and finalization coverage: confirmed.** Removed the unused
   legacy DyLAN artifact-policy helper. The workspace matrix uses
   `baseline_limits`; dedicated small-boundary GAIA tests check final-slot
   submission and rejection of new tool actions without execution or free
   Speculator responses.

## Validation and result reuse

- Full HarnessEval regression: **413 passed / 196 subtests passed**.
- After strengthening GAIA handler-count assertions: targeted bridge speculation
  tests **2 passed / 8 subtests passed**.
- Root runner self-check, root runner compilation, BFCL scorer self-check and
  `git diff --check`: passed.
- No live provider/Docker/scorer smoke or scored benchmark experiment was run.
  These tests establish deterministic bridge behavior, not model performance or
  end-to-end deployment readiness. No recommendation to restart experiments is
  implied.
- Local changes only; the server, active experiments and historical results
  were not modified. Existing eligible baseline results need neither rerunning
  nor rescoring because of these changes. Any manually bypassed historical
  LATS/Trajectory results are unsupported measurements, not candidates for
  rescoring. This does not rehabilitate earlier SA/Tau2 measurements rejected by
  the previous isolation audit.
