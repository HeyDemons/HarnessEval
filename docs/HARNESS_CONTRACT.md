# Harness Contract

The built-in runner consumes one JSON request and writes a full JSON result plus
an append-only JSONL trace. This contract is benchmark-neutral: it carries no
task answer, benchmark prompt patch, or hidden scoring rule.

## Request

```json
{
  "schema_version": 1,
  "task": {"id": "case-id", "prompt": "complete task prompt"},
  "tools": [
    {
      "name": "read_record",
      "description": "Read one record by id.",
      "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"]
      },
      "command": ["python", "/tools/read_record.py"],
      "parallel": true,
      "read_only": true,
      "pass_env": []
    }
  ],
  "policy": {"max_turns": 8, "protocol_repairs": 1, "max_parallel": 4},
  "finalizer": {
    "command": ["python", "/scorer/score.py"],
    "pass_env": []
  }
}
```

`max_turns`, `protocol_repairs`, and `max_parallel` are explicit algorithm
parameters, not hidden platform limits. HarnessEval does not set a character,
byte, log, prompt, or tool-result cap. `HARNESS_MAX_OUTPUT_TOKENS` is omitted by
default and is sent only when the user explicitly configures it.

Paper-profile policy parameters are also explicit. LATS uses
`lats_iterations`, `lats_generate_samples`, `lats_value_samples`,
`lats_rollout_width`, `lats_tree_depth`, `lats_rollout_depth`,
`lats_failure_memory`, `lats_reflection_limit`, and `lats_temperature`. Its
source search shape is preserved, while `lats_max_parallel` (default `1`) and
`lats_max_llm_calls` (default `64`) bound local transport fan-out and total HTTP
requests independently of case-level concurrency. Proposal and value-sampling
waves reserve their complete initial-call budget before dispatch. The official
implementation obtains multiple proposal samples with one provider request;
the generic compatible transport does not expose that batching primitive, so
those samples are sent through the disclosed LATS-local semaphore.
Each remembered trajectory remains complete. MemGPT uses `memgpt_core_memory_chars` and
`memgpt_memory_warning_tokens`, matching the original core-memory and active
context warning thresholds. Memory pressure is handled through semantic
summarization into recall memory, never by slicing tool results or messages.

Each tool is an argv command, never a shell string. Arguments arrive as one JSON
object on stdin. A successful tool must write one complete JSON value to stdout.
Nonzero exits and malformed output are returned to the agent as structured
failures with complete stdout/stderr; the control plane does not cut the output.
Successful JSON is exposed as `{"ok": true, "result": <complete-value>}` so
plan references use paths such as `$s1.result.field`.

Tool commands receive a minimal process environment. API credentials are not
visible unless a tool opts in by variable name through `pass_env`.

## Result

The runner writes:

```text
RUN_DIR/harness-<profile>/<case>/attempts/0001/
  request.json
  events.jsonl
  terminal.log
  harness_request.json
  harness_trace.jsonl
  harness_result.json
  payload.json             # only when a finalizer writes it
  result.json
```

`harness_result.json` contains the final answer, topology/provenance, wall time,
LLM/tool call counts, and token usage. `harness_trace.jsonl` retains complete
messages, provider responses, tool arguments, and tool results.

A finalizer receives only `HARNESS_RESULT_PATH` plus explicitly allow-listed
environment variables. It should call the benchmark's native scorer and write
the complete native metrics to `/job/payload.json`. HarnessEval embeds that JSON
without renaming metrics or guessing success from text.

## Custom Harnesses

There are two supported paths:

1. Put the harness executable in the benchmark image or mount it read-only, then
   use `harnesseval run BENCHMARK ... -- harness-command ...`.
2. Implement this request/result contract and use `harnesseval harness-run` with
   a custom `--image`.

The first path preserves a benchmark or product's native lifecycle. The second
path makes theory harnesses directly comparable on one tool transport.
