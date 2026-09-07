# Role-specific JSON replies

`role-schema-v2` separates each caller's reply structure from the benchmark's
tool argument schemas. Plans require a validated `steps`/`assignments` list;
DMAS routing has its own `decision` or `status` reply; executors accept one tool
action or a final answer; MemGPT requires its function-call envelope. SA's
predictor uses an `actions` list separately from the Actor's action contract.

Responses requests carry these schemas through `submit_benchmark_json.response`
when function transport is selected. Text transport uses a named JSON-schema
envelope and unwraps it before handing the reply to the algorithm. Compatible
clients without this capability retain their normal transport and use the same
local validation. Provider schema guidance is not treated as a guarantee: every
reply is validated before dispatch. Unknown action names, mixed tool/final
objects, null finals and progress objects cannot become environment calls.
Harmless annotations such as `explanation` or `confidence` are accepted; reserved
tool/final keys stay mutually exclusive. Branch errors include the missing or
invalid field rather than only reporting that no reply shape matched.

`complete_json` repairs syntax and caller-specific structure errors using the
existing `protocol_repairs` allowance. Feedback reports the actual validation
error and schema; it never globally forbids `status` or another field required
by a caller. Provider failures propagate without being reclassified as syntax
errors. Every completed repair response retains its generation and token usage.

Final-response reservations are recorded per asynchronous Actor task. MemGPT's
repair request switches to the final-message schema if it consumes the reserved
last slot. The processor and environment boundary both prevent a last-slot
reply from performing a new tool action. Earlier concurrent workers retain
their own response scope. Native user communication and BFCL declaration-only
acknowledgements retain their existing lifecycle. Explicitly disabled
finalization remains disabled.

The batch measurement identity records `json_reply_contract=role-schema-v2` for
affected methods, action controllers use `external-controller-v3`, and MemGPT
uses `source-token-pressure-v3`. Native actor-only on AutomationBench,
native BFCL declarations and other unaffected methods do not acquire an unused
JSON-contract identity. Old affected results require new measurements; they
cannot silently resume as this version. Scorers, task data and official budgets
are unchanged.

`final_response_action_contract=task-local-final-slot-v1` separately versions
the shared environment guard for baselines with a reserved final model response,
including non-JSON planners. BFCL declaration-only and product arms are excluded.

API format reference: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
