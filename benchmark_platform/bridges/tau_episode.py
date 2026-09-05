from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import CompletionClient, completion_client_from_env
from benchmark_platform.budgets import TAU2_MAX_STEPS, native_steps, native_errors

from .episode import EpisodeBroker, FinalResponse, NativeTool, visible_text


TASK_SET_DOMAINS = {
    "telecom_full": "telecom",
    "telecom_small": "telecom",
}

# Pinned to the official tau2-bench defaults in the benchmark image used by this
# workspace (sierra-research/tau2-bench@79975ac).  Keep these explicit instead of
# relying on TextRunConfig defaults: the native product bridge constructs the config
# itself, and a silent upstream/default drift would otherwise change measurement.
TAU2_USER_TEMPERATURE = 0.0
TAU2_SEED = 300


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


# tau2 declares tool semantics upstream via @is_tool(tool_type=ToolType.*,
# mutates_state=bool), which its toolkit stores on the wrapped function as
# `__tool_type__` and `__mutates_state__`; Tool keeps that function in `_func`.
# `openai_schema` carries none of it, so reading only the schema drops the declaration
# and every tool falls back to NativeTool's conservative read_only=False. That empties
# the speculation set (`run_sa` filters on `tool.read_only and tool.parallel`), which
# silently degrades `sa`/PERSEUS to actor-only plus one wasted predictor call and makes
# speculation unmeasurable on this benchmark. Same defect as vita_episode.py had.
#
# Both signals must agree before a tool may be pre-executed: `mutates_state=False` AND
# ToolType.READ. tau2 lets a tool override mutates_state independently of its type, so
# neither alone is sufficient. THINK and GENERIC are excluded — default-deny, because a
# wrong read-only mark lets a state-changing call execute speculatively.
_TOOL_TYPE_ATTR = "__tool_type__"
_MUTATES_STATE_ATTR = "__mutates_state__"


def _declared_read_only(tool: Any) -> bool | None:
    """True when tau2 declares the tool safe to pre-execute; None if undeclared."""
    func = getattr(tool, "_func", None)
    tool_type = getattr(func, _TOOL_TYPE_ATTR, None)
    if tool_type is None:
        return None
    is_read = str(getattr(tool_type, "value", tool_type)).lower().endswith("read")
    mutates = getattr(func, _MUTATES_STATE_ATTR, None)
    return bool(is_read) and mutates is False


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
    rows: dict[str, list[str]] = {}
    undeclared: list[str] = []
    for tool in tools:
        name = str(tool.openai_schema.get("function", tool.openai_schema)["name"])
        func = getattr(tool, "_func", None)
        tool_type = getattr(func, _TOOL_TYPE_ATTR, None)
        if tool_type is None:
            undeclared.append(name)
            continue
        key = f"{str(getattr(tool_type, 'value', tool_type)).upper()}/mutates={getattr(func, _MUTATES_STATE_ATTR, None)}"
        rows.setdefault(key, []).append(name)
    return {
        "provenance": "benchmark-declared",
        "source": "@is_tool(tool_type=, mutates_state=) via Tool._func.__tool_type__/__mutates_state__",
        "safe_for_prelaunch": "ToolType.READ and mutates_state is False",
        "total": len(tools),
        "by_declaration": {key: sorted(value) for key, value in sorted(rows.items())},
        "undeclared": sorted(undeclared),
    }


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
                {"id": item.id, "content": item.content, "error": item.error}
                for item in tool_messages
            ],
            ensure_ascii=False,
        )
    return ""


def _visible_history(messages: list[Any]) -> str:
    return "\n".join(
        f"{getattr(message, 'role', type(message).__name__)}: {_message_text(message)}"
        for message in messages
    )


def _tool_results(message: Any) -> dict[str, tuple[Any, bool]]:
    from tau2.data_model.message import MultiToolMessage, ToolMessage

    if isinstance(message, ToolMessage):
        return {message.id: (message.content, bool(message.error))}
    if isinstance(message, MultiToolMessage):
        return {
            item.id: (item.content, bool(item.error))
            for item in message.tool_messages
        }
    return {}


def _user_message(message: Any) -> str | None:
    from tau2.data_model.message import UserMessage

    if isinstance(message, UserMessage) and not message.is_tool_call():
        return str(message.content or "")
    return None


def _parse_case(case_id: str) -> tuple[str, str]:
    if ":" not in case_id:
        raise ValueError(
            "tau2 case IDs must use TASK_SET:CASE_ID (for example airline:0); "
            "bare IDs are ambiguous across official task sets"
        )
    task_set, native_id = case_id.split(":", 1)
    if not task_set or not native_id:
        raise ValueError("tau2 case IDs must use TASK_SET:CASE_ID")
    return task_set, native_id


def _load_task(case_id: str):
    from tau2.registry import registry

    task_set, native_id = _parse_case(case_id)
    if task_set not in registry.get_task_sets():
        raise KeyError(f"Unknown tau2 task set: {task_set}")
    matches = [
        task
        for task in registry.get_tasks_loader(task_set)()
        if str(task.id) == native_id
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one tau2 task for {case_id}, found {len(matches)}"
        )
    return task_set, matches[0]


def _patch_tau_generation(client: CompletionClient) -> None:
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.utils.llm_utils import to_litellm_messages

    def generate(
        *,
        model: str,
        messages: list[Any],
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        compatible = to_litellm_messages(messages)
        schemas = [tool.openai_schema for tool in tools] if tools else None
        started = time.perf_counter()
        # Called synchronously from tau2's own worker thread: go straight to the blocking
        # client instead of spinning up an event loop per turn just to await a wrapper.
        completion = client.complete_sync(
            compatible,
            tools=schemas,
            tool_choice=tool_choice,
            temperature=kwargs.get("temperature"),
            seed=kwargs.get("seed"),
        )
        raw_message = completion.raw["choices"][0]["message"]
        parsed_calls = []
        for raw_call in raw_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("Native tool-call arguments must decode to an object")
            parsed_calls.append(
                ToolCall(
                    id=str(raw_call.get("id") or uuid.uuid4().hex),
                    name=str(function["name"]),
                    arguments=arguments,
                    requestor="assistant",
                )
            )
        return AssistantMessage(
            role="assistant",
            # The relay may inline the model's reasoning as <think>...</think>. Tau2's
            # native evaluator expects a bare JSON object, so hidden user/evaluator
            # reasoning must not be fed back into its parser or graded transcript.
            content=visible_text(raw_message.get("content")),
            tool_calls=parsed_calls or None,
            cost=0.0,
            usage={
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
            raw_data=completion.raw,
            generation_time_seconds=time.perf_counter() - started,
        )

    import importlib

    modules = (
        "tau2.agent.llm_agent",
        "tau2.environment.utils.interface_agent",
        "tau2.evaluator.auth_classifier",
        "tau2.evaluator.evaluator_nl_assertions",
        "tau2.evaluator.hallucination_reviewer",
        "tau2.evaluator.review_llm_judge",
        "tau2.evaluator.review_llm_judge_user_only",
        "tau2.user.user_simulator",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        if hasattr(module, "generate"):
            module.generate = generate


def run_episode(profile: str, case_id: str, policy: dict[str, Any], job: Path) -> dict[str, Any]:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.build import _build_env_kwargs, build_text_orchestrator
    from tau2.runner.simulation import run_simulation

    seed = int(policy.get("seed", TAU2_SEED))
    random.seed(seed)
    client = completion_client_from_env()
    _patch_tau_generation(client)
    task_set, task = _load_task(case_id)
    domain = TASK_SET_DOMAINS.get(task_set, task_set)
    agent_name = f"harnesseval_{uuid.uuid4().hex}"

    class HarnessAgent(HalfDuplexAgent[dict[str, Any]]):
        def __init__(self, tools, domain_policy, **kwargs):
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.broker: EpisodeBroker | None = None
            self.brokers: list[EpisodeBroker] = []
            self.history: list[Any] = []

        def set_seed(self, value: int):
            random.seed(value)

        def get_init_state(self, message_history=None):
            self.history = list(message_history or [])
            return {"initialized": True}

        def _start(self) -> None:
            prompt = (
                "Act as the benchmark assistant under the complete domain policy below. The hidden user "
                "scenario is unavailable; rely only on visible conversation. Use send_message_to_user "
                "whenever another user turn is required. Do not claim completion until the user's request "
                "has been handled under policy.\n\n"
                f"<domain_policy>\n{self.domain_policy}\n</domain_policy>\n\n"
                f"<visible_conversation>\n{_visible_history(self.history)}\n</visible_conversation>"
            )
            self.broker = EpisodeBroker(
                profile=profile,
                prompt=prompt,
                tools=_native_tools(self.tools),
                trace_path=job / "harness_trace.jsonl",
                policy=policy,
                client=client,
            )
            self.brokers.append(self.broker)
            self.broker.start()

        def generate_next_message(self, message, state):
            # tau2 owns the episode and invokes this method once per incoming user/tool
            # message. A profile's ordinary return value is therefore one assistant turn,
            # not an instruction to terminate the native conversation. Profiles that use
            # send_message_to_user remain inside one broker; profiles that return text are
            # restarted on the next user turn with the complete visible history.
            if message is not None:
                self.history.append(message)
            if self.broker is None:
                self._start()
                assert self.broker is not None
                wave = self.broker.next_wave()
            else:
                wave = self.broker.next_wave(
                    tool_results=_tool_results(message),
                    user_message=_user_message(message),
                )
            if isinstance(wave, FinalResponse):
                response = AssistantMessage(role="assistant", content=wave.answer, cost=0.0)
                self.broker = None
            else:
                message_actions = [item for item in wave if item.is_message]
                if message_actions:
                    if len(message_actions) != 1 or len(wave) != 1:
                        raise RuntimeError("tau2 cannot mix user communication and tool calls in one turn")
                    response = AssistantMessage(
                        role="assistant",
                        content=str(message_actions[0].arguments["content"]),
                        cost=0.0,
                    )
                else:
                    calls = [
                        ToolCall(
                            id=item.id,
                            name=item.name,
                            arguments=item.arguments,
                            requestor="assistant",
                        )
                        for item in wave
                    ]
                    response = AssistantMessage(role="assistant", tool_calls=calls, cost=0.0)
            self.history.append(response)
            return response, state

        def metrics(self) -> dict[str, int]:
            totals = {
                "agent_turns": 0,
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "actor_llm_calls": 0,
                "actor_prompt_tokens": 0,
                "actor_completion_tokens": 0,
                "speculator_llm_calls": 0,
                "speculator_prompt_tokens": 0,
                "speculator_completion_tokens": 0,
                "tool_calls": 0,
                "user_messages": 0,
            }
            for broker in self.brokers:
                for name, value in broker.metrics().items():
                    # RunContext may add a new counter without changing the
                    # native adapter. Do not lose an already-scored simulation
                    # during final metrics collection (e.g. agent_turns).
                    totals[name] = totals.get(name, 0) + int(value)
            return totals

    registry.register_agent_factory(HarnessAgent, agent_name)
    config = TextRunConfig(
        domain=domain,
        task_set_name=task_set,
        agent=agent_name,
        user="user_simulator",
        llm_agent="harnesseval-baseline",
        llm_user="harnesseval-hidden-user",
        llm_args_agent={},
        llm_args_user={
            "temperature": float(
                policy.get("native_user_temperature", TAU2_USER_TEMPERATURE)
            )
        },
        max_steps=native_steps("tau2", policy),
        max_errors=native_errors("tau2", policy),
        seed=seed,
        enforce_communication_protocol=True,
    )
    orchestrator = build_text_orchestrator(config, task, seed=seed)
    if policy.get("native_evaluate", True) is False:
        simulation = orchestrator.run()
        simulation.policy = orchestrator.environment.get_policy()
    else:
        evaluation_type = EvaluationType(str(policy.get("evaluation_type", "all")))
        simulation = run_simulation(
            orchestrator,
            evaluation_type=evaluation_type,
            env_kwargs=_build_env_kwargs(config, task),
        )
    reward = simulation.reward_info.reward if simulation.reward_info is not None else None
    return {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "tau2",
        "case_id": case_id,
        "task_set": task_set,
        "domain": domain,
        "profile": profile,
        "native_lifecycle": True,
        "termination_reason": simulation.termination_reason.value,
        "duration": simulation.duration,
        "messages": len(simulation.get_messages()),
        "native_reward": simulation.reward_info.model_dump(mode="json") if simulation.reward_info else None,
        "native_score": reward,
        "native_score_status": "completed" if reward is not None else "not_requested",
        "simulation": simulation.model_dump(mode="json"),
        "tool_contract": _tool_contract(getattr(orchestrator.agent, "tools", None) or []),
        **orchestrator.agent.metrics(),
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
            "benchmark": "tau2",
            "case_id": args.case,
            "profile": args.profile,
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write(args.job / "harness_result.json", result)
    _write(args.job / "payload.json", result)
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
