from __future__ import annotations

import argparse
import json
import random
import threading
import uuid
from pathlib import Path
from typing import Any

from benchmark_platform.harnesses.api import ApiConfig, OpenAICompatibleClient

from .episode import SEND_MESSAGE_TOOL
from .product_episode import PendingProductAction, ProductEpisodeBridge, serve
from .tau_episode import (
    TASK_SET_DOMAINS,
    _load_task,
    _native_tools,
    _patch_tau_generation,
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
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.build import _build_env_kwargs, build_text_orchestrator
    from tau2.runner.simulation import run_simulation

    seed = int(policy.get("seed", 42))
    random.seed(seed)
    client = OpenAICompatibleClient(ApiConfig.from_env())
    _patch_tau_generation(client)
    task_set, task = _load_task(case_id)
    domain = TASK_SET_DOMAINS.get(task_set, task_set)
    agent_name = f"harnesseval_product_{uuid.uuid4().hex}"
    final_pending: PendingProductAction | None = None

    class ProductAgent(HalfDuplexAgent[dict[str, Any]]):
        def __init__(self, tools, domain_policy, **kwargs):
            super().__init__(tools=tools, domain_policy=domain_policy)
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
            if pending.name == SEND_MESSAGE_TOOL:
                user = _user_message(message)
                response = (
                    {"ok": True, "result": {"user_message": user}}
                    if user is not None
                    else {"ok": False, "error": "native_user_reply_missing"}
                )
            else:
                results = _tool_results(message)
                content, error = results.get(
                    pending.id,
                    ("Native tool result id did not match the submitted action", True),
                )
                response = decode_result(content, error)
            self.pending = None
            return pending, response

        def _publish_manifest(self) -> None:
            prompt = (
                "Act as the benchmark assistant under the complete domain policy below. The hidden user "
                "scenario is unavailable; rely only on visible conversation. Use send_message_to_user "
                "whenever another user turn is required. Do not claim completion until the user's request "
                "has been handled under policy.\n\n"
                f"<domain_policy>\n{self.domain_policy}\n</domain_policy>\n\n"
                f"<visible_conversation>\n{_visible_history(self.history)}\n</visible_conversation>"
            )
            native_tools = _native_tools(self.tools)
            bridge.publish_manifest(
                prompt=prompt,
                tools=native_tools,
                metadata={
                    "native_lifecycle": True,
                    "tool_contract": _tool_contract(self.tools),
                    "speculation_policy": "disabled_until_native_commit_hook_is_available",
                },
                allow_speculation=False,
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
                    content=str(pending.arguments["answer"]),
                    cost=0.0,
                )
                self.history.append(outgoing)
                return outgoing, state
            self.pending = pending
            if pending.name == SEND_MESSAGE_TOOL:
                outgoing = AssistantMessage(
                    role="assistant",
                    content=str(pending.arguments["content"]),
                    cost=0.0,
                )
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
                    cost=0.0,
                )
            self.history.append(outgoing)
            return outgoing, state

    try:
        registry.register_agent_factory(ProductAgent, agent_name)
        config = TextRunConfig(
            domain=domain,
            task_set_name=task_set,
            agent=agent_name,
            user="user_simulator",
            llm_agent="harnesseval-product",
            llm_user="harnesseval-hidden-user",
            llm_args_agent={},
            llm_args_user={},
            max_steps=int(policy.get("native_max_steps", 100)),
            max_errors=int(policy.get("native_max_errors", 10)),
            seed=seed,
            enforce_communication_protocol=True,
        )
        orchestrator = build_text_orchestrator(config, task, seed=seed)
        if policy.get("native_evaluate", True) is False:
            simulation = orchestrator.run()
            simulation.policy = orchestrator.environment.get_policy()
        else:
            simulation = run_simulation(
                orchestrator,
                evaluation_type=EvaluationType(str(policy.get("evaluation_type", "all"))),
                env_kwargs=_build_env_kwargs(config, task),
            )
        pending = getattr(orchestrator.agent, "pending", None)
        if pending is not None and pending.name == SEND_MESSAGE_TOOL:
            for message in reversed(simulation.get_messages()):
                user = _user_message(message)
                if user is not None:
                    bridge.resolve(pending, {"ok": True, "result": {"user_message": user}})
                    orchestrator.agent.pending = None
                    break
        reward = simulation.reward_info.reward if simulation.reward_info is not None else None
        bridge.complete(
            {
                "schema_version": 1,
                "status": "completed",
                "benchmark": "tau2",
                "case_id": case_id,
                "task_set": task_set,
                "domain": domain,
                "native_lifecycle": True,
                "termination_reason": simulation.termination_reason.value,
                "duration": simulation.duration,
                "messages": len(simulation.get_messages()),
                "native_reward": simulation.reward_info.model_dump(mode="json") if simulation.reward_info else None,
                "native_score": reward,
                "native_score_status": "completed" if reward is not None else "not_requested",
                "simulation": simulation.model_dump(mode="json"),
                "tool_contract": _tool_contract(getattr(orchestrator.agent, "tools", None) or []),
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
    bridge = ProductEpisodeBridge("tau2", args.case, args.job)
    thread = threading.Thread(
        target=run_native_episode,
        args=(bridge, args.case, json.loads(args.policy)),
        name="tau2-native-episode",
        daemon=True,
    )
    thread.start()
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()
