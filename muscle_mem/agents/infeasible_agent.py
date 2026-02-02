import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from muscle_mem.agents.tool_loop import (
    extract_response_content,
    extract_text_blocks,
    normalize_content_list,
)
from muscle_mem.agents.tools import ExecutionToolProvider, ToolRegistry, UIActions
from muscle_mem.agents.tools.registry import tool_action
from muscle_mem.core.mllm import LMMAgent
from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY
from muscle_mem.utils.common_utils import call_llm_safe

logger = logging.getLogger("desktopenv.agent")

DEFAULT_INFEASIBLE_TOOL_ALLOW = [
    "web_search",
    "web_fetch",
    "call_code_agent",
    "click",
    "click_image",
    "switch_applications",
    "open",
    "scroll",
    "wait",
    "hold_and_press",
    "report_infeasible",
    "report_feasible",
]


def _format_llm_message_for_log(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        lines.append(f"[{role}]")
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    lines.append(str(block))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    lines.append(block.get("text", ""))
                elif block_type == "image":
                    lines.append("[image omitted]")
                else:
                    lines.append(f"[{block_type} omitted]")
    return "\n".join(line for line in lines if line is not None)


def _format_llm_response_for_log(response: Any) -> str:
    if isinstance(response, dict):
        content = response.get("content", [])
    else:
        content = []
    normalized = normalize_content_list(content)
    text_blocks = extract_text_blocks(normalized)
    return "\n".join(text_blocks).strip() or "(no text content)"


@dataclass
class InfeasibleAgentResult:
    task_instruction: str
    summary: str
    completion_reason: str
    steps_executed: int
    budget: int
    feasible: Optional[bool] = None
    reason: Optional[str] = None
    evidence: Optional[str] = None
    execution_history: Optional[List[Dict[str, Any]]] = None


@dataclass
class InfeasibleAgentStepResult:
    info: Dict[str, Any]
    actions: List[str]
    status: str
    payload: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None


class InfeasibleResultToolProvider:
    def __init__(self, manager: "InfeasibleAgentManager") -> None:
        self.manager = manager

    @tool_action
    def report_infeasible(self, reason: str, evidence: str) -> Dict[str, str]:
        """
        Report the infeasibility conclusion.

        Args:
            reason: Why the task is infeasible.
            evidence: Evidence supporting the infeasibility.
        """
        payload = {
            "reason": (reason or "").strip(),
            "evidence": (evidence or "").strip(),
        }
        self.manager.last_reported_result = payload
        return payload

    @tool_action
    def report_feasible(self, reason: str, evidence: str) -> Dict[str, str]:
        """
        Report the feasible conclusion.

        Args:
            reason: Why the task is feasible within the constraints.
            evidence: Evidence supporting the feasibility.
        """
        payload = {
            "reason": (reason or "").strip(),
            "evidence": (evidence or "").strip(),
        }
        self.manager.last_reported_result = payload
        return payload


InfeasibleResultToolProvider.report_infeasible.tool_input_schema = {
    "type": "object",
    "description": "Report the infeasibility conclusion with reason and evidence.",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Why the task is infeasible within the constraints.",
        },
        "evidence": {
            "type": "string",
            "description": "Evidence supporting the infeasibility.",
        },
    },
    "required": ["reason", "evidence"],
    "additionalProperties": False,
}


InfeasibleResultToolProvider.report_feasible.tool_input_schema = {
    "type": "object",
    "description": "Report the feasibility conclusion with reason and evidence.",
    "properties": {
        "reason": {
            "type": "string",
            "description": "Why the task is feasible within the constraints.",
        },
        "evidence": {
            "type": "string",
            "description": "Evidence supporting the feasibility.",
        },
    },
    "required": ["reason", "evidence"],
    "additionalProperties": False,
}


class InfeasibleUIActions(UIActions):
    def report_infeasible(self, reason: str, evidence: str):
        return super().report_infeasible(reason, evidence)


class InfeasibleAgentManager:
    def __init__(
        self,
        engine_params: Dict[str, Any],
        grounding,
        budget: int = 20,
    ) -> None:
        self.engine_params = dict(engine_params or {})
        self.grounding = grounding
        self.budget = budget
        self.tool_allow = list(DEFAULT_INFEASIBLE_TOOL_ALLOW)
        self.tool_deny: List[str] = []
        self.last_reported_result: Optional[Dict[str, str]] = None
        self.ui_action_names = {
            "click",
            "switch_applications",
            "open",
            "drag_and_drop",
            "scroll",
            "hotkey",
            "wait",
            "hold_and_press",
            "highlight_text_span",
            "set_cell_values",
            "type",
        }
        self.ui_actions = InfeasibleUIActions(grounding)
        self._tool_registry: Optional[ToolRegistry] = None
        self._llm_agent: Optional[LMMAgent] = None
        self._messages: List[Dict[str, Any]] = []
        self._task_instruction: Optional[str] = None
        self._max_rounds = int(budget or self.budget)
        self.turn_count = 0
        self._execution_steps: List[Dict[str, Any]] = []
        self.reset()

    def reset(self) -> None:
        self._llm_agent = LMMAgent(
            engine_params=self.engine_params,
            system_prompt=PROCEDURAL_MEMORY.INFEASIBLE_AGENT_PROMPT,
        )
        self._tool_registry = self._build_tool_registry()
        self._messages = []
        self._task_instruction = None
        self._max_rounds = int(self.budget)
        self.turn_count = 0
        self._execution_steps = []
        self.last_reported_result = None

    def _build_tool_registry(self) -> ToolRegistry:
        tool_registry = ToolRegistry()
        tool_registry.register_action_provider(InfeasibleResultToolProvider(self))
        env_controller = None
        if getattr(self.grounding, "env", None) is not None:
            env_controller = getattr(self.grounding.env, "controller", None)
        exec_tools = ExecutionToolProvider(
            env_controller, engine_params=self.engine_params
        )
        tool_registry.register_action_provider(exec_tools)
        tool_registry.register_action_provider(self.ui_actions)
        if getattr(self.grounding, "code_agent_tools", None) is not None:
            tool_registry.register_action_provider(self.grounding.code_agent_tools)
        return tool_registry

    def _ensure_session(
        self, task_instruction: str, screenshot: Optional[Any], budget: Optional[int]
    ) -> None:
        if not task_instruction:
            return
        budget_value = int(budget or self.budget)
        if self._task_instruction != task_instruction:
            self.reset()
            self._task_instruction = task_instruction
            self._max_rounds = budget_value
            user_text = "用户任务:\n" f"{task_instruction}\n\n"
            blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            if screenshot:
                blocks.extend(
                    self._build_screenshot_blocks("Current screenshot:", screenshot)
                )
            self._messages = [{"role": "user", "content": blocks}]
            return
        if screenshot:
            self._messages.append(
                {
                    "role": "user",
                    "content": self._build_screenshot_blocks(
                        "Current screenshot:", screenshot
                    ),
                }
            )

    def _append_tool_result(
        self, tool_use_id: str, content: Any, is_error: bool = False
    ) -> None:
        result = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        self._messages.append({"role": "user", "content": [result]})

    def _is_exec_tool(self, tool_name: str) -> bool:
        if self._tool_registry is None:
            return False
        exec_tools = None
        for provider in (getattr(self.grounding, "exec_tools", None),):
            if provider is not None:
                exec_tools = provider
                break
        if exec_tools is None:
            return False
        handler = getattr(type(exec_tools), tool_name, None)
        return bool(handler and getattr(handler, "is_tool_action", False))

    def _safe_wait(self, seconds: float) -> str:
        try:
            return self.grounding.call_tool("wait", {"time": seconds})
        except Exception:
            return f"import time; time.sleep({seconds})"

    def _get_ui_action_thoughts(self, tool_name: Optional[str]) -> Optional[str]:
        if tool_name not in {"click_image"}:
            return None
        return getattr(self.grounding, "last_grounding_thoughts", None) or None

    def _record_execution_history(
        self,
        step_idx: int,
        plan: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if not hasattr(self.grounding, "record_execution_history"):
            return
        entry = {
            "step": step_idx,
            "plan": plan,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": error_message,
        }
        self.grounding.record_execution_history(entry)

    def _build_screenshot_blocks(
        self, label: str, image_bytes: Any
    ) -> List[Dict[str, Any]]:
        llm_agent = self._llm_agent
        if llm_agent is None:
            return [{"type": "text", "text": f"{label} (no llm agent)"}]
        try:
            base64_image = llm_agent.encode_image(image_bytes)
        except Exception as exc:
            logger.warning("Failed to attach screenshot to infeasible agent: %s", exc)
            return [{"type": "text", "text": f"{label} (failed to encode screenshot)"}]
        return [
            {"type": "text", "text": label},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                },
            },
        ]

    def _build_decision_payload(
        self,
        feasible: Optional[bool],
        reason: Optional[str],
        evidence: Optional[str],
        summary: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "feasible": feasible,
            "infeasible": bool(feasible is False),
            "reason": reason,
            "evidence": evidence,
            "summary": summary,
        }

    def generate_next_action(
        self,
        task_instruction: str,
        obs: Dict,
        budget: Optional[int] = None,
    ) -> InfeasibleAgentStepResult:
        if not task_instruction:
            info = {
                "plan": "NO_TASK",
                "plan_code": "",
                "exec_code": "DONE",
                "reflection": None,
                "reflection_thoughts": None,
                "code_agent_output": None,
            }
            return InfeasibleAgentStepResult(
                info=info,
                actions=[],
                status="NO_DECISION",
                payload=None,
                summary="(no task instruction)",
            )

        screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
        self._ensure_session(task_instruction, screenshot, budget)
        if hasattr(self.grounding, "assign_screenshot"):
            self.grounding.assign_screenshot(obs)
        if hasattr(self.grounding, "set_task_instruction"):
            self.grounding.set_task_instruction(task_instruction)

        if self.turn_count >= int(budget or self._max_rounds):
            summary_text = "Stopped after max rounds"
            payload = self._build_decision_payload(
                feasible=None, reason=None, evidence=None, summary=summary_text
            )
            info = {
                "plan": "NO_CLEAR_DECISION",
                "plan_code": "",
                "exec_code": "DONE",
                "reflection": None,
                "reflection_thoughts": None,
                "code_agent_output": None,
                "infeasible": payload,
            }
            return InfeasibleAgentStepResult(
                info=info,
                actions=[],
                status="NO_DECISION",
                payload=payload,
                summary=summary_text,
            )

        tool_allow = self.tool_allow
        tool_deny = self.tool_deny
        allow_set = None
        if tool_allow and tool_allow != ["*"] and tool_allow != "*":
            allow_set = set(tool_allow)
        deny_set = set(tool_deny or [])

        def _is_tool_allowed(name: Optional[str]) -> bool:
            if not name:
                return False
            if allow_set is not None and name not in allow_set:
                return False
            if name in deny_set:
                return False
            return True

        while True:
            tools = (
                self._tool_registry.build_tools(allow=tool_allow, deny=tool_deny)
                if self._tool_registry is not None
                else []
            )
            messages_with_system = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PROCEDURAL_MEMORY.INFEASIBLE_AGENT_PROMPT,
                        }
                    ],
                },
                *self._messages,
            ]
            response = call_llm_safe(
                self._llm_agent,
                messages=messages_with_system,
                temperature=self._llm_agent.engine.temperature or 0.0,
                max_new_tokens=8000,
                tools=tools,
                tool_choice={"type": "any"},
            )
            # logger.info(
            #     "Infeasible agent LLM request:\n%s",
            #     _format_llm_message_for_log(messages_with_system),
            # )
            # logger.info(
            #     "Infeasible agent LLM response:\n%s",
            #     _format_llm_response_for_log(response),
            # )

            content_blocks, stop_reason = extract_response_content(response)
            normalized_blocks = normalize_content_list(content_blocks)
            tool_use = next(
                (
                    block
                    for block in normalized_blocks
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ),
                None,
            )
            if tool_use is None:
                summary_text = "\n".join(extract_text_blocks(normalized_blocks)).strip()
                if not summary_text:
                    summary_text = "(no output)"
                payload = self._build_decision_payload(
                    feasible=None,
                    reason=None,
                    evidence=None,
                    summary=summary_text,
                )
                info = {
                    "plan": "NO_CLEAR_DECISION",
                    "plan_code": "",
                    "exec_code": "DONE",
                    "reflection": None,
                    "reflection_thoughts": None,
                    "code_agent_output": None,
                    "infeasible": payload,
                }
                self.turn_count += 1
                return InfeasibleAgentStepResult(
                    info=info,
                    actions=[],
                    status="NO_DECISION",
                    payload=payload,
                    summary=summary_text,
                )

            tool_name = tool_use.get("name")
            tool_input = tool_use.get("input", {}) or {}
            tool_use_id = tool_use.get("id", "tool_use")
            plan = (
                f"TOOL_USE name={tool_name} "
                f"input={json.dumps(tool_input, ensure_ascii=False)}"
            )
            self._messages.append({"role": "assistant", "content": normalized_blocks})
            self._record_execution_history(
                self.turn_count + 1, plan, tool_name, tool_input
            )

            if not _is_tool_allowed(tool_name):
                self._append_tool_result(
                    tool_use_id,
                    f"Tool '{tool_name}' is not allowed for infeasibility checks.",
                    is_error=True,
                )
                continue

            sanitized_input = tool_input if isinstance(tool_input, dict) else {}
            if tool_name == "call_code_agent":
                base_task = (sanitized_input.get("task") or task_instruction).strip()
                readonly_task = (
                    "只读核验任务可行性，禁止修改任何文件或系统状态。\n"
                    f"原始任务：{base_task}"
                )
                sanitized_input = dict(sanitized_input)
                sanitized_input["task"] = readonly_task

            error_message = None
            try:
                tool_output = (
                    self._tool_registry.dispatch(tool_name, sanitized_input)
                    if self._tool_registry is not None
                    else None
                )
            except Exception as exc:
                error_message = str(exc)
                tool_output = None

            if error_message:
                self._append_tool_result(
                    tool_use_id,
                    error_message,
                    is_error=True,
                )
                exec_code = self._safe_wait(1.333)
                info = {
                    "plan": plan,
                    "plan_code": "",
                    "exec_code": exec_code,
                    "reflection": None,
                    "reflection_thoughts": None,
                    "code_agent_output": None,
                }
                self.turn_count += 1
                return InfeasibleAgentStepResult(
                    info=info,
                    actions=[exec_code],
                    status="ACTION",
                    payload=None,
                )

            if tool_name in {"report_infeasible", "report_feasible"}:
                payload_dict = tool_output if isinstance(tool_output, dict) else {}
                feasible = tool_name == "report_feasible"
                summary_prefix = "FEASIBLE" if feasible else "INFEASIBLE"
                summary_text = (
                    f"{summary_prefix}: {payload_dict.get('reason') or ''}".strip()
                )
                payload = self._build_decision_payload(
                    feasible=feasible,
                    reason=payload_dict.get("reason"),
                    evidence=payload_dict.get("evidence"),
                    summary=summary_text or "(no output)",
                )
                self._append_tool_result(
                    tool_use_id,
                    (
                        payload
                        if isinstance(payload, str)
                        else json.dumps(payload, ensure_ascii=False)
                    ),
                )
                info = {
                    "plan": summary_prefix,
                    "plan_code": "",
                    "exec_code": "FAIL" if not feasible else "DONE",
                    "reflection": None,
                    "reflection_thoughts": None,
                    "code_agent_output": None,
                    "infeasible": payload,
                }
                self.turn_count += 1
                return InfeasibleAgentStepResult(
                    info=info,
                    actions=[],
                    status="INFEASIBLE" if not feasible else "FEASIBLE",
                    payload=payload,
                    summary=summary_text,
                )

            output_text = (
                tool_output
                if isinstance(tool_output, str)
                else json.dumps(tool_output, ensure_ascii=False)
            )

            if tool_name in {
                "call_pac_agent",
                "call_code_agent",
                "call_infeasible_agent",
            }:
                if not output_text:
                    output_text = "(no output)"
                self._append_tool_result(
                    tool_use_id, [{"type": "text", "text": output_text}]
                )
                exec_code = tool_output if isinstance(tool_output, str) else ""
                info = {
                    "plan": plan,
                    "plan_code": "",
                    "exec_code": exec_code,
                    "reflection": None,
                    "reflection_thoughts": None,
                    "code_agent_output": None,
                }
                self.turn_count += 1
                return InfeasibleAgentStepResult(
                    info=info,
                    actions=[exec_code] if exec_code else [],
                    status="ACTION",
                    payload=None,
                )
            if self._is_exec_tool(tool_name) or tool_name in {
                "TodoWrite",
                "save_scratchpad",
                "read_scratchpad",
            }:
                if not output_text:
                    output_text = "(no output)"
                self._append_tool_result(
                    tool_use_id, [{"type": "text", "text": output_text}]
                )
                continue

            if tool_name in {"done", "fail"}:
                text = "OK"
                self._append_tool_result(tool_use_id, [{"type": "text", "text": text}])
                exec_code = tool_output if isinstance(tool_output, str) else ""
                info = {
                    "plan": plan,
                    "plan_code": "",
                    "exec_code": exec_code,
                    "reflection": None,
                    "reflection_thoughts": None,
                    "code_agent_output": None,
                }
                self.turn_count += 1
                return InfeasibleAgentStepResult(
                    info=info,
                    actions=[exec_code] if exec_code else [],
                    status="ACTION",
                    payload=None,
                )

            if error_message:
                text = error_message
            else:
                thoughts = self._get_ui_action_thoughts(tool_name)
                if thoughts is not None:
                    text = (
                        thoughts + "\nDone."
                        if thoughts
                        else "(no grounding thoughts returned)"
                    )
                else:
                    text = "Done."
            self._append_tool_result(tool_use_id, [{"type": "text", "text": text}])
            exec_code = tool_output if isinstance(tool_output, str) else ""
            info = {
                "plan": plan,
                "plan_code": "",
                "exec_code": exec_code,
                "reflection": None,
                "reflection_thoughts": None,
                "code_agent_output": None,
            }
            self.turn_count += 1
            return InfeasibleAgentStepResult(
                info=info,
                actions=[exec_code] if exec_code else [],
                status="ACTION",
                payload=None,
            )

    def run_task(
        self,
        task_instruction: str,
        screenshot: str = "",
        budget: Optional[int] = None,
    ) -> InfeasibleAgentResult:
        max_rounds = int(budget or self.budget)
        llm_agent = LMMAgent(
            engine_params=self.engine_params,
            system_prompt=PROCEDURAL_MEMORY.INFEASIBLE_AGENT_PROMPT,
        )

        tool_registry = ToolRegistry()
        tool_registry.register_action_provider(InfeasibleResultToolProvider(self))
        env_controller = None
        if getattr(self.grounding, "env", None) is not None:
            env_controller = getattr(self.grounding.env, "controller", None)
        exec_tools = ExecutionToolProvider(
            env_controller, engine_params=self.engine_params
        )
        tool_registry.register_action_provider(exec_tools)
        tool_registry.register_action_provider(self.ui_actions)
        if getattr(self.grounding, "code_agent_tools", None) is not None:
            tool_registry.register_action_provider(self.grounding.code_agent_tools)

        tool_allow = self.tool_allow
        tool_deny = self.tool_deny
        allow_set = None
        if tool_allow and tool_allow != ["*"] and tool_allow != "*":
            allow_set = set(tool_allow)
        deny_set = set(tool_deny or [])

        def _is_tool_allowed(name: Optional[str]) -> bool:
            if not name:
                return False
            if allow_set is not None and name not in allow_set:
                return False
            if name in deny_set:
                return False
            return True

        user_text = "用户任务:\n" f"{task_instruction}\n\n"

        blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if screenshot:
            blocks.extend(
                self._build_screenshot_blocks("Current screenshot:", screenshot)
            )

        messages: List[Dict[str, Any]] = [{"role": "user", "content": blocks}]
        execution_steps: List[Dict[str, Any]] = []
        steps_executed = 0

        for round_index in range(1, max_rounds + 1):
            steps_executed = round_index
            tools = tool_registry.build_tools(allow=tool_allow, deny=tool_deny)
            messages_with_system = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PROCEDURAL_MEMORY.INFEASIBLE_AGENT_PROMPT,
                        }
                    ],
                },
                *messages,
            ]
            response = call_llm_safe(
                llm_agent,
                messages=messages_with_system,
                temperature=llm_agent.engine.temperature or 0.0,
                max_new_tokens=8000,
                tools=tools,
                tool_choice={"type": "any"},
            )
            logger.info(
                "Infeasible agent LLM request:\n%s",
                _format_llm_message_for_log(messages_with_system),
            )
            logger.info(
                "Infeasible agent LLM response:\n%s",
                _format_llm_response_for_log(response),
            )

            content_blocks, stop_reason = extract_response_content(response)
            normalized_blocks = normalize_content_list(content_blocks)
            if stop_reason == "tool_use":
                tool_uses = [
                    block
                    for block in normalized_blocks
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ]
                results = []
                reported_payload = None
                reported_tool_name = None
                for tool_use in tool_uses:
                    name = tool_use.get("name")
                    tool_input = tool_use.get("input", {}) or {}
                    tool_use_id = tool_use.get("id")
                    if not _is_tool_allowed(name):
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": f"Tool '{name}' is not allowed for infeasibility checks.",
                                "is_error": True,
                            }
                        )
                        continue
                    try:
                        sanitized_input = (
                            tool_input if isinstance(tool_input, dict) else {}
                        )
                        if name == "call_code_agent":
                            base_task = (
                                sanitized_input.get("task") or task_instruction
                            ).strip()
                            readonly_task = (
                                "只读核验任务可行性，禁止修改任何文件或系统状态。\n"
                                f"原始任务：{base_task}"
                            )
                            sanitized_input = dict(sanitized_input)
                            sanitized_input["task"] = readonly_task
                        output = tool_registry.dispatch(name, sanitized_input)
                        output_text = (
                            output
                            if isinstance(output, str)
                            else json.dumps(output, ensure_ascii=False)
                        )
                        if (
                            name in self.ui_action_names
                            and isinstance(output, str)
                            and output.strip()
                            and output not in {"DONE", "FAIL"}
                        ):
                            output_text = "Done."
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": output_text,
                            }
                        )
                        if name in {"report_infeasible", "report_feasible"}:
                            reported_payload = (
                                output if isinstance(output, dict) else None
                            )
                            reported_tool_name = name
                    except Exception as exc:
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": str(exc),
                                "is_error": True,
                            }
                        )
                messages.append({"role": "assistant", "content": normalized_blocks})
                messages.append({"role": "user", "content": results})
                execution_steps.append(
                    {
                        "step": round_index,
                        "tool_uses": [
                            str(tool.get("name") or "") for tool in tool_uses
                        ],
                        "tool_results": [
                            str(result.get("content") or "") for result in results
                        ],
                    }
                )
                if reported_payload:
                    feasible = reported_tool_name == "report_feasible"
                    summary_prefix = "FEASIBLE" if feasible else "INFEASIBLE"
                    summary_text = (
                        f"{summary_prefix}: {reported_payload.get('reason')}".strip()
                    )
                    return InfeasibleAgentResult(
                        task_instruction=task_instruction,
                        summary=summary_text or "(no output)",
                        completion_reason="REPORTED",
                        steps_executed=steps_executed,
                        budget=max_rounds,
                        feasible=feasible,
                        reason=reported_payload.get("reason"),
                        evidence=reported_payload.get("evidence"),
                        execution_history=execution_steps or None,
                    )
                continue

            messages.append({"role": "assistant", "content": normalized_blocks})
            summary_text = "\n".join(extract_text_blocks(normalized_blocks)).strip()
            if not summary_text:
                summary_text = "(no output)"
            return InfeasibleAgentResult(
                task_instruction=task_instruction,
                summary=summary_text,
                completion_reason="DONE",
                steps_executed=steps_executed,
                budget=max_rounds,
                feasible=None,
                execution_history=execution_steps or None,
            )

        return InfeasibleAgentResult(
            task_instruction=task_instruction,
            summary="Stopped after max rounds",
            completion_reason="DONE",
            steps_executed=steps_executed,
            budget=max_rounds,
            feasible=None,
            execution_history=execution_steps or None,
        )


class InfeasibleAgentToolProvider:
    def __init__(self, manager: InfeasibleAgentManager, grounding) -> None:
        self.manager = manager
        self.grounding = grounding

    @tool_action
    def call_infeasible_agent(self):
        """
        Invoke the infeasible agent to judge task feasibility.
        """
        task_value = getattr(self.grounding, "current_task_instruction", None)
        if not task_value:
            raise ValueError(
                "No task instruction available for call_infeasible_agent tool"
            )

        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling Infeasible Agent")
        logger.info("=" * 50)
        logger.info("Checking feasibility for task: %s", task_value)

        screenshot = ""
        if getattr(self.grounding, "obs", None):
            screenshot = self.grounding.obs.get("screenshot", "")

        result = self.manager.run_task(task_value, screenshot=screenshot)

        if hasattr(self.grounding, "last_infeasible_agent_result"):
            self.grounding.last_infeasible_agent_result = result

        logger.info("Infeasible agent execution completed")
        logger.info("Summary: %s", result.summary)
        logger.info("=" * 50)

        payload = {
            "feasible": result.feasible,
            "infeasible": bool(result.feasible is False),
            "reason": result.reason,
            "evidence": result.evidence,
            "summary": result.summary,
        }
        return json.dumps(payload, ensure_ascii=False)


InfeasibleAgentToolProvider.call_infeasible_agent.tool_input_schema = {
    "type": "object",
    "description": "Parameters for invoking the Infeasible Agent.",
    "additionalProperties": False,
}
