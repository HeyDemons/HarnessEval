from __future__ import annotations

import argparse
import json
import random
import threading
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import ApiConfig, OpenAICompatibleClient

from .episode import SEND_MESSAGE_TOOL
from .product_episode import PendingProductAction, ProductEpisodeBridge, serve
from .vita_episode import (
    _build_environment,
    _find_task,
    _native_tools,
    _patch_vita_generation,
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
    native_tools = _native_tools(environment.get_tools())
    final_pending: PendingProductAction | None = None

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

        def _resolve_incoming(self, message: Any) -> None:
            if self.pending is None:
                return
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
            bridge.resolve(self.pending, response)
            self.pending = None

        def generate_next_message(self, message, state):
            nonlocal final_pending
            if not self.started:
                history = [*self.history, message]
                prompt = (
                    "Act as the benchmark assistant under the complete domain policy below. The user simulator's "
                    "private scenario is not available; rely only on visible conversation messages. Use "
                    "send_message_to_user whenever another user turn is required.\n\n"
                    f"<domain_policy>\n{environment.get_policy()}\n</domain_policy>\n\n"
                    f"<visible_conversation>\n{_visible_history(history)}\n</visible_conversation>"
                )
                bridge.publish_manifest(
                    prompt=prompt,
                    tools=native_tools,
                    metadata={
                        "native_lifecycle": True,
                        "tool_contract": _tool_contract(environment.get_tools()),
                        "speculation_policy": "disabled_until_native_commit_hook_is_available",
                    },
                    allow_speculation=False,
                )
                self.started = True
            else:
                self._resolve_incoming(message)
            pending = bridge.next_action()
            if pending.kind == "final":
                final_pending = pending
                return AssistantMessage(
                    role="assistant",
                    content=f"{pending.arguments['answer']}\n{self.STOP_TOKEN}",
                ), state
            self.pending = pending
            if pending.name == SEND_MESSAGE_TOOL:
                return AssistantMessage(role="assistant", content=str(pending.arguments["content"])), state
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id=pending.id,
                        name=pending.name,
                        arguments=pending.arguments,
                        requestor="assistant",
                    )
                ],
            ), state

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
                "native_score_status": "completed" if native_reward is not None else "not_requested",
                "simulation": simulation.model_dump(mode="json"),
                "tool_contract": _tool_contract(environment.get_tools()),
            },
            final_pending,
        )
    except Exception as exc:
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
