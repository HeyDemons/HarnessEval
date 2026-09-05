# Responses transport for baseline harnesses

Opt in per run with `HARNESS_API_TYPE=openai-responses`. The existing
`HARNESS_API_BASE`, key, model, stream, timeout, retry, reasoning and output-budget
settings still apply. This does not change the default transport or any method's
action/tool/scoring contract. SA can inherit this transport or explicitly select
it with `HARNESS_SA_API_TYPE`.

The client uses `/responses`, explicit top-level `instructions`, `store=false`,
typed terminal stream events, and `text.format` for JSON mode. Never concatenate
both deltas and done snapshots. Multiple text messages remain in generation
order so the existing method-specific parser decides which action to execute.
When a method has no system prompt, `instructions="."` is a neutral non-whitespace
sentinel: the relay was observed replacing whitespace-only instructions with
roughly 4,380 extra input tokens. Real system instructions remain unchanged.
Hosted tools are not enabled: native function calls/results stay within the
benchmark's bridge. Images remain typed content, not base64 text in prompts.

Native continuation preserves output Items and encrypted reasoning in an
in-memory, prefix-scoped bounded replay cache. No previous_response_id or stored
conversation is used. Text JSON loops do not replay later hypothetical actions;
their caller's canonical action history is authoritative. Normalized trace usage
retains input/cache/output fields, but never stores encrypted reasoning payloads.

Some relays require the word JSON in input rather than instructions. When JSON
mode is requested and input lacks it, a neutral `Return JSON.` format reminder is
added at serialization only. Provider `seed` is unsupported: the client warns and
records `responses_request.provider_seed_requested` with `provider_seed_applied=false`,
without sending an unsupported API field. Tau2's native episode seed remains
independently configured and applied; this is not provider-level seeded equivalence.

The protocol is part of measurement identity. Do not silently combine latency or
token measurements across Chat Completions and Responses, nor treat endpoint
switching as permission to rerun or overwrite prior measurements. Validate an
isolated native episode and task-container run before a full campaign.

References: [migration](https://developers.openai.com/api/docs/guides/migrate-to-responses),
[function calling](https://developers.openai.com/api/docs/guides/function-calling).
