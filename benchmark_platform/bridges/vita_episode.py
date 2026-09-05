from __future__ import annotations
from benchmark_platform.budgets import native_steps, native_errors

import argparse
import asyncio
import json
import random
import traceback
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import CompletionClient, completion_client_from_env

from .episode import ActionRequest, EpisodeBroker, FinalResponse, NativeTool, UsageMeter, visible_text


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    pending.replace(path)


# VitaBench declares tool semantics upstream via @is_tool(tool_type=ToolType.READ /
# WRITE / GENERIC), stored by the decorator on the wrapped function as
# `__tool_type__`. Tool keeps that function in `_func`, but it is absent from
# `openai_schema` (name/description/parameters only), so reading only the schema drops
# the declaration. Every tool then fell back to NativeTool's conservative
# read_only=False, which empties the speculation set for `sa` (paper_methods.run_sa
# filters on `tool.read_only and tool.parallel`) and for PERSEUS. Both degenerated to
# actor-only plus one wasted predictor call, making speculation unmeasurable here.
#
# Only ToolType.READ is treated as safe for pre-execution. WRITE mutates state.
# GENERIC is a pure utility in practice (distance, weather, holiday lookup) but the
# name is not a read guarantee, so it is excluded — default-deny, since a wrong
# read-only mark would let a mutating call execute speculatively.
_TOOL_TYPE_ATTR = "__tool_type__"


def _declared_read_only(tool: Any) -> bool | None:
    """Upstream ToolType for a tool, or None when the declaration is unreachable."""
    declared = getattr(getattr(tool, "_func", None), _TOOL_TYPE_ATTR, None)
    if declared is None:
        return None
    return str(getattr(declared, "name", declared)).upper() == "READ"


def _native_tools(tools: list[Any]) -> list[NativeTool]:
    declared = []
    for tool in tools:
        schema = tool.openai_schema.get("function", tool.openai_schema)
        # An unreachable declaration must not silently become "safe".
        read_only = _declared_read_only(tool) is True
        declared.append(
            NativeTool(
                name=str(schema["name"]),
                description=str(schema.get("description", "")),
                parameters=schema.get("parameters") or {"type": "object", "properties": {}},
                read_only=read_only,
                parallel=read_only,
            )
        )
    return declared


def _tool_contract(tools: list[Any]) -> dict[str, Any]:
    """Provenance of the contract, recorded per run so the mapping is auditable."""
    by_type: dict[str, list[str]] = {}
    undeclared: list[str] = []
    for tool in tools:
        name = str(tool.openai_schema.get("function", tool.openai_schema)["name"])
        raw = getattr(getattr(tool, "_func", None), _TOOL_TYPE_ATTR, None)
        if raw is None:
            undeclared.append(name)
            continue
        by_type.setdefault(str(getattr(raw, "name", raw)).upper(), []).append(name)
    return {
        "provenance": "benchmark-declared",
        "source": "@is_tool(tool_type=ToolType.*) via Tool._func.__tool_type__",
        "safe_for_prelaunch": "ToolType.READ only",
        "total": len(tools),
        "by_tool_type": {key: sorted(value) for key, value in sorted(by_type.items())},
        "undeclared": sorted(undeclared),
    }


def _language(policy: dict[str, Any]) -> str | None:
    """Leave the default language to VitaBench itself; None means "whatever upstream says".

    The environment is keyed by language and answers a lookup made in any other one with an
    empty result rather than an error, so a default that disagrees with upstream stays
    invisible until every name-based tool call has already missed -- an English environment
    under a Chinese-speaking model returned "No trains found" for 12 of 13 calls. Holding a
    copy of the constant here would reintroduce exactly that risk on a revision bump, and
    every call site this bridge touches (task loader, environment constructor, get_weekday,
    get_prompts) already falls back to vita.config.DEFAULT_LANGUAGE on None. The resolved
    value is recorded in the result so a run is never ambiguous about which one it used.
    """
    language = policy.get("language")
    return str(language) if language is not None else None


# VitaBench's English database matches natural-language entities in English. A model that
# answers an otherwise English episode in Chinese therefore gets plausible-looking empty
# search results instead of an explicit language error. Keep this instruction at the task
# policy boundary: every baseline Actor receives it, and the product manifest carries the
# same prompt into both the authoritative PERSEUS Actor and its Speculator context.
_ACTOR_LANGUAGE_DIRECTIVES = {
    "english": (
        "Respond to the user in English. Use English for every natural-language string "
        "passed to tools, including place, product, store, hotel, attraction, station, and "
        "city names. Do not translate English entity names into Chinese. Preserve opaque "
        "identifiers, dates, numbers, and enum values verbatim."
    )
}


def _actor_language_directive(language: str | None) -> str | None:
    if language is None:
        from vita.config import DEFAULT_LANGUAGE

        language = DEFAULT_LANGUAGE
    return _ACTOR_LANGUAGE_DIRECTIVES.get(str(language))


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return json.dumps(
            [
                {
                    "id": item.id,
                    "name": item.name,
                    "arguments": item.arguments,
                    "requestor": item.requestor,
                }
                for item in tool_calls
            ],
            ensure_ascii=False,
        )
    tool_messages = getattr(message, "tool_messages", None)
    if tool_messages:
        return json.dumps(
            [
                {"id": item.id, "name": item.name, "content": item.content, "error": item.error}
                for item in tool_messages
            ],
            ensure_ascii=False,
        )
    return ""


def _visible_history(messages: list[Any]) -> str:
    rows = []
    for message in messages:
        role = getattr(message, "role", type(message).__name__)
        rows.append(f"{role}: {_message_text(message)}")
    return "\n".join(rows)


def _task_clock(environment: Any, language: str | None) -> tuple[str, str]:
    """Return the benchmark database time and the official policy rendering value."""
    from vita.utils.utils import get_weekday

    system_time = str(environment.tools.db.time)
    return system_time, f"{system_time} {get_weekday(system_time, language)}"


def _render_domain_policy(environment: Any, language: str | None) -> tuple[str, str]:
    system_time, policy_time = _task_clock(environment, language)
    policy = environment.get_policy().format(time=policy_time)
    if directive := _actor_language_directive(language):
        policy = f"{policy}\n\n# Language\n- {directive}"
    return policy, system_time


# Upstream's user-simulator prompt carries no language directive at all -- neither version
# does. It relies on the template's own language being the cue, which holds for the Chinese
# primary dataset and breaks on the English translation: the romanised Chinese entity names
# and the Meituan-style scenario outweigh it, and the simulator opened in Chinese 8 times out
# of 8 on an English prompt containing zero Chinese characters. The agent then mirrors the
# user, queries the English-keyed database with Chinese strings, and 12 of 13 lookups come
# back empty -- so an English run measures translation luck rather than agent skill.
#
# One appended line restores it (8/8 English, measured; a longer, firmer wording bought
# nothing). It is added as a separate message so the official template stays byte-identical,
# and only for the language that needs it -- under Chinese the cue already works and this
# would be a gratuitous deviation. The Actor has its own policy-level directive above because
# the measured product Actor still switched to Chinese despite an all-English visible user.
# Both deviations are recorded per run.
_LANGUAGE_DIRECTIVES = {"english": "Write every message in English."}


def _patch_vita_generation(client: CompletionClient, language: str | None = None) -> dict[str, UsageMeter]:
    """Route VitaBench's own LLM roles through the harness client, metered and de-leaked."""
    from vita.data_model.message import AssistantMessage
    from vita.utils.llm_utils import format_messages

    from vita.config import DEFAULT_LANGUAGE

    directive = _LANGUAGE_DIRECTIVES.get(language or DEFAULT_LANGUAGE)

    def build(meter: UsageMeter, extra_system: str | None = None):
        def generate(*, model: str, messages: list[Any], tools=None, tool_choice=None, enable_think=False, **kwargs):
            if tools:
                raise RuntimeError("HarnessEval's hidden-user bridge does not expose assistant tools to the user simulator")
            formatted = format_messages(messages)
            compatible = [
                {"role": str(item["role"]), "content": str(item.get("content") or "")}
                for item in formatted
                if item.get("role") in {"system", "user", "assistant"}
            ]
            if extra_system:
                # After the official system block, before the conversation: the shape the
                # 8/8 measurement used.
                index = 0
                while index < len(compatible) and compatible[index]["role"] == "system":
                    index += 1
                compatible.insert(index, {"role": "system", "content": extra_system})
            # Same reasoning as tau_episode: this hook is synchronous and already off the loop.
            completion = client.complete_sync(compatible, temperature=kwargs.get("temperature"))
            meter.add(completion)
            return AssistantMessage(
                role="assistant",
                content=visible_text(completion.content),
                cost=0.0,
                usage={
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                },
                raw_data=completion.raw,
            )

        return generate

    import vita.evaluator.evaluator_traj as evaluator_traj
    import vita.user.user_simulator as user_simulator

    meters = {"user_simulator": UsageMeter(), "evaluator": UsageMeter()}
    # Only the simulator: the evaluator emits structured verdicts, and its prompt is the one
    # the relays reject, so it gets nothing added.
    user_simulator.generate = build(meters["user_simulator"], directive)
    evaluator_traj.generate = build(meters["evaluator"])
    return meters


def _find_task(case_id: str, language: str):
    """Return the task and the task set it came from; the set is needed to repair _domain."""
    from vita.registry import registry

    for task_set in registry.get_task_sets():
        tasks = registry.get_tasks_loader(task_set)(language=language)
        for task in tasks:
            if str(task.id) == case_id:
                return task, task_set
    raise KeyError(f"VitaBench case not found: {case_id}")


def _domain(task: Any, task_set: str) -> str:
    """The registry key for this task's environment, repaired when the dataset translated it.

    tasks_en.json translated the domain field itself on part of the in-store set: 25 of its
    100 tasks say "in-store consumption" and 2 say "in-store", where the registry key is
    "instore". get_env_constructor raises KeyError on all 27, which is 4 of the 60 cases in
    the light suite. The Chinese dataset is clean on all 400, so this is a defect in the
    English translation, not a naming convention we should be following.

    The task set carries the same fact the field is meant to carry, and the three
    single-domain sets are named exactly like the domains -- so fall back to it, and only
    when the field itself does not resolve. Cross-domain tasks say "delivery,ota,instore",
    which resolves, so they never take this path.
    """
    from vita.registry import registry

    known = set(registry.get_domains())
    declared = str(task.domain)
    if all(part.strip() in known for part in declared.split(",")):
        return declared
    if task_set in known:
        return task_set
    raise KeyError(f"VitaBench task {task.id} declares an unresolvable domain: {declared!r}")


def _build_environment(task: Any, language: str, domain: str):
    from vita.environment.environment import get_cross_environment
    from vita.registry import registry

    if "," in domain:
        return get_cross_environment(domain, task.environment, language)
    return registry.get_env_constructor(domain)(task.environment, language)


def _tool_results(message: Any) -> dict[str, tuple[Any, bool]]:
    from vita.data_model.message import MultiToolMessage, ToolMessage

    if isinstance(message, ToolMessage):
        return {message.id: (message.content, bool(message.error))}
    if isinstance(message, MultiToolMessage):
        return {
            item.id: (item.content, bool(item.error))
            for item in message.tool_messages
        }
    return {}


def _user_message(message: Any) -> str | None:
    from vita.data_model.message import UserMessage

    if isinstance(message, UserMessage) and not message.is_tool_call():
        return str(message.content or "")
    return None


def run_episode(profile: str, case_id: str, policy: dict[str, Any], job: Path) -> dict[str, Any]:
    from vita.agent.base import BaseAgent
    from vita.data_model.message import AssistantMessage, ToolCall
    from vita.config import DEFAULT_LANGUAGE, DEFAULT_SEED
    from vita.evaluator.evaluator import evaluate_simulation
    from vita.orchestrator.orchestrator import Orchestrator
    from vita.user.user_simulator import UserSimulator

    language = _language(policy)
    # vita.config.DEFAULT_SEED, so the recorded value matches an official run -- but it is
    # inert on this path and must not be read as a reproducibility guarantee. Upstream uses a
    # seed for two things: deriving per-trial seeds in run.py (we construct the Orchestrator
    # directly, so that code never runs) and pushing llm_args["seed"] to the provider (our
    # generate hook forwards only temperature, so it is dropped before the request). Nothing
    # in vita's episode path uses Python's random either. Wiring it through would still leave
    # the actor unseeded on both arms, so it would buy a reproducibility claim we cannot make.
    seed = int(policy.get("seed", DEFAULT_SEED))
    random.seed(seed)
    client = completion_client_from_env()
    harness_usage = _patch_vita_generation(client, language)
    task, task_set = _find_task(case_id, language)
    domain = _domain(task, task_set)
    environment = _build_environment(task, language, domain)
    domain_policy, _system_time = _render_domain_policy(environment, language)

    class HarnessAgent(BaseAgent[dict[str, Any]]):
        STOP_TOKEN = "###STOP###"

        def __init__(self):
            self.broker: EpisodeBroker | None = None
            self.history: list[Any] = []

        def set_seed(self, value: int):
            random.seed(value)

        def get_init_state(self, message_history=None):
            self.history = list(message_history or [])
            return {"initialized": True}

        def _start(self, incoming: Any) -> None:
            history = [*self.history, incoming]
            prompt = (
                "Act as the benchmark assistant under the complete domain policy below. The user simulator's "
                "private scenario is not available; rely only on visible conversation messages. Use "
                "send_message_to_user whenever another user turn is required.\n\n"
                f"<domain_policy>\n{domain_policy}\n</domain_policy>\n\n"
                f"<visible_conversation>\n{_visible_history(history)}\n</visible_conversation>"
            )
            self.broker = EpisodeBroker(
                profile=profile,
                prompt=prompt,
                tools=_native_tools(environment.get_tools()),
                trace_path=job / "harness_trace.jsonl",
                policy=policy,
                client=client,
            )
            self.broker.start()

        def generate_next_message(self, message, state):
            if self.broker is None:
                self._start(message)
                wave = self.broker.next_wave()
            else:
                wave = self.broker.next_wave(
                    tool_results=_tool_results(message),
                    user_message=_user_message(message),
                )
            if isinstance(wave, FinalResponse):
                return AssistantMessage(role="assistant", content=f"{wave.answer}\n{self.STOP_TOKEN}"), state
            message_actions = [item for item in wave if item.is_message]
            if message_actions:
                if len(message_actions) != 1 or len(wave) != 1:
                    raise RuntimeError("VitaBench cannot mix user communication and tool calls in one turn")
                return AssistantMessage(role="assistant", content=str(message_actions[0].arguments["content"])), state
            calls = [
                ToolCall(id=item.id, name=item.name, arguments=item.arguments, requestor="assistant")
                for item in wave
            ]
            return AssistantMessage(role="assistant", tool_calls=calls), state

    agent = HarnessAgent()
    user = UserSimulator(
        persona=str(task.user_scenario.user_profile),
        instructions=str(task.instructions),
        llm="harnesseval-hidden-user",
        llm_args={},
        language=language,
    )

    def measured(**outcome: Any) -> dict[str, Any]:
        """What the episode measured, whether or not it reached its own end.

        An exhausted baseline turn budget raises out of the orchestrator, and that used to
        land in main()'s except branch, which writes a six-field failure result: the tokens
        the arm burned, the tool contract it ran under and the language it ran in were all
        discarded. bench_runtime recovers the actor's own tokens from the trace, but nothing
        records the hidden user simulator's -- so every truncated arm was invisible in the
        cost accounting, and those are exactly the long, expensive ones. The agent-side
        numbers exist either way; only the simulation and its score do not.
        """
        return {
            "schema_version": 1,
            "benchmark": "vitabench",
            "case_id": case_id,
            "profile": profile,
            "native_lifecycle": True,
            "language": language or DEFAULT_LANGUAGE,
            "harness_usage": {name: meter.as_dict() for name, meter in harness_usage.items()},
            "actor_language_directive": _actor_language_directive(language),
            "user_simulator_language_directive": _LANGUAGE_DIRECTIVES.get(language or "chinese"),
            "tool_contract": _tool_contract(environment.get_tools()),
            **(agent.broker.metrics() if agent.broker else {}),
            **outcome,
        }

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        # Orchestrator's own class default is 100, but no official entry point uses it:
        # vita/cli.py passes DEFAULT_MAX_STEPS = 300. The gap matters because reaching the
        # ceiling sets TerminationReason.MAX_STEPS, which evaluate_simulation turns into a
        # flat 0.0 with no rubric grading at all -- a harness budget silently overwriting the
        # benchmark's verdict. A 40-turn agent already produces ~90 orchestrator steps.
        max_steps=native_steps("vitabench", policy),
        max_errors=native_errors("vitabench", policy),
        seed=seed,
        language=language,
    )
    try:
        simulation = orchestrator.run()
    except Exception as exc:
        _write(job / "episode_error.json",
               {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return measured(status="failed", error=f"{type(exc).__name__}: {exc}")
    native_reward = None
    native_score_error = None
    if policy.get("native_evaluate") is True:
        # The scorer failing must not destroy the episode that already ran. Raising here sent
        # run_episode down main()'s except branch, which writes a six-field failure result --
        # tokens, tool contract, language and the whole simulation gone, over a provider fault
        # in grading. The product bridge already keeps the episode and reports the scorer
        # fault separately; this is the same contract on the baseline side.
        try:
            native_reward = evaluate_simulation(
                domain=domain,
                task=task,
                simulation=simulation,
                evaluation_type=str(policy.get("evaluation_type", "trajectory")),
                llm_evaluator="harnesseval-evaluator",
                llm_args_evaluator={},
                language=language,
            ).model_dump(mode="json")
        except Exception as exc:
            native_score_error = f"{type(exc).__name__}: {exc}"
            _write(job / "native_evaluator_error.json",
                   {"error": native_score_error, "traceback": traceback.format_exc()})
    return measured(
        status="completed",
        termination_reason=simulation.termination_reason,
        duration=simulation.duration,
        messages=len(simulation.messages),
        native_reward=native_reward,
        native_score=native_reward.get("reward") if native_reward is not None else None,
        native_score_status=(
            "completed"
            if native_reward is not None
            else "error"
            if native_score_error is not None
            else "not_requested"
        ),
        native_score_error=native_score_error,
        simulation=simulation.model_dump(mode="json"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--job", type=Path, default=Path("/job"))
    args = parser.parse_args()
    try:
        policy = json.loads(args.policy)
        result = run_episode(args.profile, args.case, policy, args.job)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "benchmark": "vitabench",
            "case_id": args.case,
            "profile": args.profile,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write(args.job / "harness_result.json", result)
    _write(args.job / "payload.json", result)
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
