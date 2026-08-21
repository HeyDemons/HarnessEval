from __future__ import annotations

import argparse
import json
import random
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import completion_client_from_env

from .episode import SEND_MESSAGE_TOOL
from .product_episode import PendingProductAction, ProductEpisodeBridge, serve
from .vita_episode import (
    _build_environment,
    _find_task,
    _native_tools,
    _patch_vita_generation,
    _render_domain_policy,
    _tool_contract,
    _tool_results,
    _user_message,
    _visible_history,
)


def decode_result(content: Any, error: bool) -> dict[str, Any]:
    if error:
        return {"ok": False, "error": "native_tool_failed", "detail": content}
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass
    return {"ok": True, "result": content}


def run_native_episode(bridge: ProductEpisodeBridge, case_id: str, policy: dict[str, Any]) -> None:
    from vita.agent.base import BaseAgent
    from vita.data_model.message import AssistantMessage, ToolCall, ToolMessage
    from vita.evaluator.evaluator import evaluate_simulation
    from vita.orchestrator.orchestrator import Orchestrator
    from vita.user.user_simulator import UserSimulator

    language = str(policy.get("language", "english"))
    seed = int(policy.get("seed", 42))
    random.seed(seed)
    client = completion_client_from_env()
    _patch_vita_generation(client)
    task = _find_task(case_id, language)
    environment = _build_environment(task, language)
    domain_policy, system_time = _render_domain_policy(environment, language)
    native_tools = _native_tools(environment.get_tools())
    safe_tools = {tool.name for tool in native_tools if tool.read_only and tool.parallel}
    final_pending: PendingProductAction | None = None

    def native_message(response: dict[str, Any], call: Any) -> Any:
        error = response.get("ok") is not True
        content = response.get("result") if not error else response.get("detail", response.get("error"))
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        return ToolMessage(
            id=call.id,
            name=call.name,
            content=content,
            requestor=call.requestor,
            role="tool",
            error=error,
        )

    original_get_response = environment.get_response

    def get_response(call: Any) -> Any:
        replay = bridge.replay_response(call.id)
        return native_message(replay, call) if replay is not None else original_get_response(call)

    environment.get_response = get_response

    def speculate(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in safe_tools:
            return {"ok": False, "error": "tool_not_safe_for_speculation"}
        call = ToolCall(
            id=f"spec-{uuid.uuid4().hex}",
            name=name,
            arguments=arguments,
            requestor="assistant",
        )
        message = original_get_response(call)
        return decode_result(message.content, bool(message.error))

    class ProductAgent(BaseAgent[dict[str, Any]]):
        STOP_TOKEN = "###STOP###"

        def __init__(self):
            self.history: list[Any] = []
            self.pending: PendingProductAction | None = None
            self.started = False

        def set_seed(self, value: int):
            random.seed(value)

        def get_init_state(self, message_history=None):
            self.history = list(message_history or [])
            return {"initialized": True}

        def _resolve_incoming(self, message: Any) -> tuple[PendingProductAction, dict[str, Any]] | None:
            if self.pending is None:
                return None
            pending = self.pending
            if self.pending.name == SEND_MESSAGE_TOOL:
                user = _user_message(message)
                response = (
                    {"ok": True, "result": {"user_message": user}}
                    if user is not None
                    else {"ok": False, "error": "native_user_reply_missing"}
                )
            else:
                results = _tool_results(message)
                content, error = results.get(
                    self.pending.id,
                    ("Native tool result id did not match the submitted action", True),
                )
                response = decode_result(content, error)
            self.pending = None
            return pending, response

        def _publish_manifest(self) -> None:
            prompt = (
                "Act as the benchmark assistant under the complete domain policy below. The user simulator's "
                "private scenario is not available; rely only on visible conversation messages. Use "
                "send_message_to_user whenever another user turn is required.\n\n"
                f"<domain_policy>\n{domain_policy}\n</domain_policy>\n\n"
                f"<visible_conversation>\n{_visible_history(self.history)}\n</visible_conversation>"
            )
            bridge.publish_manifest(
                prompt=prompt,
                tools=native_tools,
                metadata={
                    "native_lifecycle": True,
                    "system_time": system_time,
                    "tool_contract": _tool_contract(environment.get_tools()),
                    "speculation_policy": "benchmark_declared_read_tools_with_native_commit_replay",
                    "requires_speculative_commit": True,
                },
                speculative_executor=speculate,
                allow_speculation=True,
            )

        def generate_next_message(self, message, state):
            nonlocal final_pending
            self.history.append(message)
            if not self.started:
                self._publish_manifest()
                self.started = True
            else:
                resolved = self._resolve_incoming(message)
                if resolved is not None and resolved[0].name == SEND_MESSAGE_TOOL:
                    self._publish_manifest()
                if resolved is not None:
                    bridge.resolve(*resolved)
            pending = bridge.next_action()
            if pending.kind == "final":
                final_pending = pending
                outgoing = AssistantMessage(
                    role="assistant",
                    content=f"{pending.arguments['answer']}\n{self.STOP_TOKEN}",
                )
                self.history.append(outgoing)
                return outgoing, state
            self.pending = pending
            if pending.name == SEND_MESSAGE_TOOL:
                outgoing = AssistantMessage(role="assistant", content=str(pending.arguments["content"]))
            else:
                outgoing = AssistantMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id=pending.id,
                            name=pending.name,
                            arguments=pending.arguments,
                            requestor="assistant",
                        )
                    ],
                )
            self.history.append(outgoing)
            return outgoing, state

    try:
        agent = ProductAgent()
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
        native_score_error = None
        if policy.get("native_evaluate") is True:
            try:
                native_reward = evaluate_simulation(
                    domain=str(task.domain),
                    task=task,
                    simulation=simulation,
                    evaluation_type=str(policy.get("evaluation_type", "trajectory")),
                    llm_evaluator="harnesseval-evaluator",
                    llm_args_evaluator={},
                    language=language,
                ).model_dump(mode="json")
            except Exception as exc:
                native_score_error = f"{type(exc).__name__}: {exc}"
                from benchmark_platform.util import atomic_json

                atomic_json(
                    bridge.job / "native_evaluator_error.json",
                    {
                        "error": native_score_error,
                        "traceback": traceback.format_exc(),
                    },
                )
        bridge.complete(
            {
                "schema_version": 1,
                "status": "completed",
                "benchmark": "vitabench",
                "case_id": case_id,
                "native_lifecycle": True,
                "termination_reason": simulation.termination_reason,
                "duration": simulation.duration,
                "messages": len(simulation.messages),
                "native_reward": native_reward,
                "native_score": native_reward.get("reward") if native_reward is not None else None,
                "native_score_status": (
                    "completed"
                    if native_reward is not None
                    else "error"
                    if native_score_error is not None
                    else "not_requested"
                ),
                "native_score_error": native_score_error,
                "simulation": simulation.model_dump(mode="json"),
                "tool_contract": _tool_contract(environment.get_tools()),
            },
            final_pending,
        )
    except Exception as exc:
        from benchmark_platform.util import atomic_json

        atomic_json(
            bridge.job / "native_error.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        bridge.fail(exc, final_pending)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--job", type=Path, default=Path("/job"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    policy = json.loads(args.policy)
    bridge = ProductEpisodeBridge("vitabench", args.case, args.job)
    thread = threading.Thread(
        target=run_native_episode,
        args=(bridge, args.case, policy),
        name="vitabench-native-episode",
        daemon=True,
    )
    thread.start()
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
