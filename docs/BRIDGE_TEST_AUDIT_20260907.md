# Bridge contract test review — 2026-09-07

> Historical audit note: this review predates the current
> `bfcl-native-declaration-boundary-v2` runtime-harness contract. Marker-based
> method-final publication was subsequently retired; current output nodes emit
> native declaration batches and publication makes no model call.

Scope: BFCL, Tau2, AutomationBench, GAIA and Terminal-Bench-2; plus a
defensive rejection check for LATS/Trajectory-Bench. No Perseus execution,
VitaBench integration work, algorithm changes or scoring changes.

## Findings and fixes

1. **BFCL matrix: confirmed.** The previous matrix constructed a generic
   ToolEnvironment without declaration-only wiring. It therefore exercised a
   different lifecycle and accepted eleven multi-response profiles. The matrix
   now calls `bridges.runner.execute`. A later all-baseline campaign added the
   explicit multi-model declaration aggregation protocol. That intermediate
   protocol was subsequently retired by the publisher contract noted above.
2. **Tau2 broker fixture: confirmed.** An unmarked lookup never enabled SA
   speculation. The fixture now exposes a parallel read-only lookup and asserts
   that a broker without a native isolation adapter fails before model calls or
   queued actions. Tau2 now supplies a shadow-execution/adoption adapter; its
   broker publication behavior is covered separately.
3. **AutomationBench matrix and injection: confirmed.** `run_episode` now accepts
   an optional `speculator_client`; omission retains the existing environment
   factory. The matrix now covers all fourteen participating profiles through
   `run_episode`; tool-capable profiles act before scoring while the published
   text-only AFlow/SPP profiles retain zero environment calls.
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

- The counts below describe the review commit at the time it was written; see
  the latest campaign report for validation after all-baseline integration.
- Full HarnessEval regression at that review commit: **413 passed / 196 subtests passed**.
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
