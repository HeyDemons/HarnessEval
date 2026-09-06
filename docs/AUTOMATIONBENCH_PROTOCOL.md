# AutomationBench integration protocol

The source and native scorer remain pinned to AutomationBench 1.0.6,
`4a8e1061254004d9dac807054eed33fad7d1ff14`. This integration change affects
agent inputs and interaction, not the task world or scoring assertions.

## Public instructions

`automationbench_prompt_protocol=native-task-roles-v1` preserves public
dataset system/developer messages at their actual instruction role on every
Actor, planner, worker and Speculator request. Algorithm-specific instructions
are retained. Text-based algorithms receive the remaining public conversation
as their task string. Native actor-only receives the original role-preserving
conversation. No `info`, initial world or assertions enter these messages.

Responses transport moves system messages into `instructions`, and sends user
messages as user input. `bridge_manifest.json` records the public message list;
`harness_trace.jsonl` records the actual requests after instruction attachment.

## Native actor-only

`automationbench_actor_protocol=native` is the default for AutomationBench.
It uses only `api_search`, `api_fetch`, and `base64_encode`, with schemas generated
by the pinned verifiers package exactly as `StatefulToolEnv.add_tool` does, hiding
the controller-owned `world` argument. No JSON wrapper or synthetic finish tool
is supplied to the model. A response without tool calls terminates the episode.

Every call in a native response is processed in order, matching the official
stateful environment; multiple state-changing calls are allowed in one batch.
Each observation preserves its call ID and successful API string content. The
normal controller trace, token accounting, model-response budget and post-agent
native assertion scorer remain in use. One response with several calls is one
Actor generation. Tool failures still produce observations; no scorer feedback
is available during the agent loop.

`automationbench_argument_protocol=official-function-defaults-v1` uses the native
function's argument binding, like the official environment. The upstream model
schema marks defaulted parameters as required; runtime calls may nevertheless omit
them or provide the documented `{}` sentinel. AutomationBench therefore does not
apply the generic controller schema validator ahead of the official handler.
Other benchmarks retain their existing validation. Model-supplied `world` values
remain rejected.

The existing project budget remains 50 Actor-channel responses, with its final
reserved response disallowing new actions. This local finalization convention is
kept explicit; this control is not a claim that every official provider/default
setting is identical.

## Diagnostic JSON control and result provenance

The root runner accepts `HARNESS_AUTOMATIONBENCH_ACTOR_PROTOCOL=json` for a
diagnostic JSON single-action control. This still receives corrected task roles
and official schemas; it does not recreate the older flattened-role bug.
Use separate output roots for native and JSON comparisons.

Root summaries record prompt, tool-schema and actor protocols in `baseline_policy`.
Changing them prevents automatic resume/merge with older results. The native
Actor's identity does not include the unused Responses JSON-envelope setting.

Old trajectories and scores remain valid historical artifacts of the old protocol.
They can be re-rendered to display strict score and partial completion, but cannot
be relabeled as results of this repair. Measuring the repaired protocol requires
new agent runs; rescoring alone cannot reconstruct changed model decisions.

The official Calendly guest/assertion inconsistency documented in the workspace
audit is not changed by this patch.

Sources: [pinned AutomationBench environment](https://github.com/zapier/AutomationBench/blob/4a8e1061254004d9dac807054eed33fad7d1ff14/automationbench/runner.py),
[Responses function calling](https://developers.openai.com/api/docs/guides/function-calling).
