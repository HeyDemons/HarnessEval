from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import ApiConfig, OpenAICompatibleClient

from .episode import ActionRequest, EpisodeBroker, FinalResponse, NativeTool


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


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    tool_messages = getattr(message, "tool_messages", None)
    if tool_messages:
        return json.dumps(
            [
                {"id": item.id, "name": item.name, "content": item.content, "error": item.error}
                for item in tool_messages
            ],
            ensure_ascii=False,
        )
    return str(message)


def _visible_history(messages: list[Any]) -> str:
    rows = []
    for message in messages:
        role = getattr(message, "role", type(message).__name__)
        rows.append(f"{role}: {_message_text(message)}")
    return "\n".join(rows)


def _patch_vita_generation(client: OpenAICompatibleClient) -> None:
    from vita.data_model.message import AssistantMessage
    from vita.utils.llm_utils import format_messages

    def generate(*, model: str, messages: list[Any], tools=None, tool_choice=None, enable_think=False, **kwargs):
        if tools:
            raise RuntimeError("HarnessEval's hidden-user bridge does not expose assistant tools to the user simulator")
        formatted = format_messages(messages)
        compatible = [
            {"role": str(item["role"]), "content": str(item.get("content") or "")}
            for item in formatted
            if item.get("role") in {"system", "user", "assistant"}
        ]
        completion = asyncio.run(client.complete(compatible, temperature=kwargs.get("temperature")))
        return AssistantMessage(
            role="assistant",
            content=completion.content,
            cost=0.0,
            usage={
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
            raw_data=completion.raw,
        )

    import vita.evaluator.evaluator_traj as evaluator_traj
    import vita.user.user_simulator as user_simulator

    user_simulator.generate = generate
    evaluator_traj.generate = generate


def _find_task(case_id: str, language: str):
    from vita.registry import registry

    for task_set in registry.get_task_sets():
        tasks = registry.get_tasks_loader(task_set)(language=language)
        for task in tasks:
            if str(task.id) == case_id:
                return task
    raise KeyError(f"VitaBench case not found: {case_id}")


def _build_environment(task: Any, language: str):
    from vita.environment.environment import get_cross_environment
    from vita.registry import registry

    domain = str(task.domain)
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
    from vita.evaluator.evaluator import evaluate_simulation
    from vita.orchestrator.orchestrator import Orchestrator
    from vita.user.user_simulator import UserSimulator

    language = str(policy.get("language", "english"))
    seed = int(policy.get("seed", 42))
    random.seed(seed)
    client = OpenAICompatibleClient(ApiConfig.from_env())
    _patch_vita_generation(client)
    task = _find_task(case_id, language)
    environment = _build_environment(task, language)

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
                f"<domain_policy>\n{environment.get_policy()}\n</domain_policy>\n\n"
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
    simulation = Orchestrator(
        domain=str(task.domain),
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=int(policy.get("native_max_steps", 100)),
        max_errors=int(policy.get("native_max_errors", 10)),
        seed=seed,
        language=language,
    ).run()
    native_reward = None
    if policy.get("native_evaluate") is True:
        native_reward = evaluate_simulation(
            domain=str(task.domain),
            task=task,
            simulation=simulation,
            evaluation_type=str(policy.get("evaluation_type", "trajectory")),
            llm_evaluator="harnesseval-evaluator",
            llm_args_evaluator={},
            language=language,
        ).model_dump(mode="json")
    broker_metrics = agent.broker.metrics() if agent.broker else {}
    return {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "vitabench",
        "tool_contract": _tool_contract(environment.get_tools()),
        "case_id": case_id,
        "profile": profile,
        "native_lifecycle": True,
        "termination_reason": simulation.termination_reason,
        "duration": simulation.duration,
        "messages": len(simulation.messages),
        "native_reward": native_reward,
        "native_score_status": "completed" if native_reward is not None else "not_requested",
        "simulation": simulation.model_dump(mode="json"),
        **broker_metrics,
    }


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
