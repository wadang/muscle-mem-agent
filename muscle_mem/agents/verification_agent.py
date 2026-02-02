import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from muscle_mem.agents.tool_loop import (
    extract_response_content,
    extract_text_blocks,
    normalize_content_list,
)
from muscle_mem.agents.tools import ExecutionToolProvider, ToolRegistry
from muscle_mem.agents.tools.registry import tool_action
from muscle_mem.core.mllm import LMMAgent
from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY
from muscle_mem.utils.common_utils import call_llm_safe


logger = logging.getLogger("desktopenv.agent")

DEFAULT_VERIFICATION_TOOL_ALLOW = [
    "web_search",
    "web_fetch",
    "call_code_agent",
    "click",
    "switch_applications",
    "open",
    "scroll",
    "wait",
    "hold_and_press",
    "report_verification_plan",
    "report_verification_result",
]

VERIFICATION_CONCLUSIONS = ("IMPOSSIBLE", "ERROR", "SUCCESS")


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
class VerificationAgentResult:
    task_instruction: str
    summary: str
    completion_reason: str
    steps_executed: int
    budget: int
    conclusion: Optional[str] = None
    explanation: Optional[str] = None
    execution_history: Optional[List[Dict[str, Any]]] = None


class VerificationResultToolProvider:
    def __init__(self, manager: "VerificationAgentManager") -> None:
        self.manager = manager

    @tool_action
    def report_verification_plan(
        self,
        task_understanding: str,
        possible_failures: str,
        screenshot_observation: str,
        verification_plan: str,
    ) -> Dict[str, str]:
        """
        Record the verification plan.

        Args:
            task_understanding: How you interpret the task.
            possible_failures: What could cause the verification to fail.
            screenshot_observation: What you observed from the screenshots.
            verification_plan: Your verification plan.
        """
        payload = {
            "task_understanding": (task_understanding or "").strip(),
            "possible_failures": (possible_failures or "").strip(),
            "screenshot_observation": (screenshot_observation or "").strip(),
            "verification_plan": (verification_plan or "").strip(),
        }
        self.manager.last_reported_plan = payload
        return payload

    @tool_action
    def report_verification_result(
        self, conclusion: str, explanation: str
    ) -> Dict[str, str]:
        """
        Report the final verification result.

        Args:
            conclusion: One of IMPOSSIBLE, ERROR, SUCCESS.
            explanation: Brief evidence or reasoning.
        """
        normalized = (conclusion or "").strip().upper()
        if normalized not in VERIFICATION_CONCLUSIONS:
            raise ValueError(
                f"Invalid conclusion '{conclusion}'. Must be one of: {', '.join(VERIFICATION_CONCLUSIONS)}."
            )
        payload = {"conclusion": normalized, "explanation": (explanation or "").strip()}
        self.manager.last_reported_result = payload
        return payload


VerificationResultToolProvider.report_verification_result.tool_input_schema = {
    "type": "object",
    "description": "Report the verification conclusion and explanation.",
    "properties": {
        "conclusion": {
            "type": "string",
            "enum": list(VERIFICATION_CONCLUSIONS),
            "description": "Verification conclusion.",
        },
        "explanation": {
            "type": "string",
            "description": "Brief evidence or reasoning.",
        },
    },
    "required": ["conclusion", "explanation"],
    "additionalProperties": False,
}

VerificationResultToolProvider.report_verification_plan.tool_input_schema = {
    "type": "object",
    "description": "Report the verification plan, including observations, task understanding, and plan.",
    "properties": {
        "task_understanding": {
            "type": "string",
            "description": "How you interpret the task.",
        },
        "possible_failures": {
            "type": "string",
            "description": "What could cause the verification to fail.",
        },
        "screenshot_observation": {
            "type": "string",
            "description": "What you observed from the screenshots.",
        },
        "verification_plan": {
            "type": "string",
            "description": "Your verification plan.",
        },
    },
    "required": [
        "task_understanding",
        "possible_failures",
        "screenshot_observation",
        "verification_plan",
    ],
    "additionalProperties": False,
}


class VerificationAgentManager:
    def __init__(
        self,
        engine_params: Dict[str, Any],
        grounding,
        budget: int = 12,
    ) -> None:
        self.engine_params = dict(engine_params or {})
        self.grounding = grounding
        self.budget = budget
        self.tool_allow = list(DEFAULT_VERIFICATION_TOOL_ALLOW)
        self.tool_deny: List[str] = []
        self.last_reported_result: Optional[Dict[str, str]] = None
        self.last_reported_plan: Optional[Dict[str, str]] = None
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

    def _capture_local_obs(self) -> Optional[Dict[str, Any]]:
        try:
            import pyautogui
            from PIL import Image
        except Exception:
            return None

        target_size = None
        if getattr(self.grounding, "obs", None):
            try:
                current_bytes = self.grounding.obs.get("screenshot")
                if current_bytes:
                    image = Image.open(io.BytesIO(current_bytes))
                    target_size = image.size
            except Exception:
                target_size = None

        screenshot = pyautogui.screenshot()
        if target_size:
            screenshot = screenshot.resize(target_size, Image.LANCZOS)
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        return {"screenshot": buffered.getvalue()}

    def _execute_ui_action(
        self, action_code: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        env = getattr(self.grounding, "env", None)
        if env is not None and hasattr(env, "step"):
            sleep_after = getattr(env, "sleep_after_execution", None)
            if sleep_after is None:
                sleep_after = getattr(self.grounding, "sleep_after_execution", 1.0)
            try:
                try:
                    obs, _reward, _done, _info = env.step(action_code, sleep_after)
                except TypeError:
                    obs, _reward, _done, _info = env.step(action_code)
                return obs, None
            except Exception as exc:
                return None, str(exc)

        try:
            exec("import time\n" + action_code, {})
        except Exception as exc:
            return None, str(exc)
        return self._capture_local_obs(), None

    def run_task(
        self,
        task_instruction: str,
        second_screenshot: str = "",
        screenshot: str = "",
        is_code_agent_verification: bool = False,
        budget: Optional[int] = None,
    ) -> VerificationAgentResult:
        max_rounds = int(budget or self.budget)
        llm_agent = LMMAgent(
            engine_params=self.engine_params,
            system_prompt=PROCEDURAL_MEMORY.VERIFICATION_AGENT_PROMPT,
        )

        tool_registry = ToolRegistry()
        tool_registry.register_action_provider(VerificationResultToolProvider(self))
        env_controller = None
        if getattr(self.grounding, "env", None) is not None:
            env_controller = getattr(self.grounding.env, "controller", None)
        exec_tools = ExecutionToolProvider(
            env_controller, engine_params=self.engine_params
        )
        tool_registry.register_action_provider(exec_tools)
        if getattr(self.grounding, "ui_actions", None) is not None:
            tool_registry.register_action_provider(self.grounding.ui_actions)
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

        user_text = (
            "Task description:\n"
            f"{task_instruction}\n\n"
            "You will receive the second submitted and current screenshots.\n"
            "Use them to verify completion and respond in the required format."
        )
        if is_code_agent_verification:
            user_text = (
                f"{user_text}\n\n" "CRITICAL: You are verifying the Code Agent's work. "
            )

        def _build_screenshot_blocks(
            label: str, image_bytes: Any
        ) -> List[Dict[str, Any]]:
            try:
                base64_image = llm_agent.encode_image(image_bytes)
            except Exception as exc:
                logger.warning(
                    "Failed to attach screenshot to verification agent: %s", exc
                )
                return [
                    {"type": "text", "text": f"{label} (failed to encode screenshot)"}
                ]
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

        blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if second_screenshot:
            blocks.extend(
                _build_screenshot_blocks(
                    "Second submitted screenshot:", second_screenshot
                )
            )
        if screenshot:
            blocks.extend(_build_screenshot_blocks("Current screenshot:", screenshot))

        messages: List[Dict[str, Any]] = [{"role": "user", "content": blocks}]
        execution_steps: List[Dict[str, Any]] = []
        steps_executed = 0

        for round_index in range(1, max_rounds + 1):
            steps_executed = round_index
            tools = tool_registry.build_tools(allow=tool_allow, deny=tool_deny)
            tool_names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
            # logger.info(
            #     "Verification agent tools submitted to LLM: %s",
            #     ", ".join(name for name in tool_names if name),
            # )
            messages_with_system = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PROCEDURAL_MEMORY.VERIFICATION_AGENT_PROMPT,
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
                "Verification agent LLM request:\n%s",
                _format_llm_message_for_log(messages_with_system),
            )
            logger.info(
                "Verification agent LLM response:\n%s",
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
                latest_screenshot = None
                for tool_use in tool_uses:
                    name = tool_use.get("name")
                    tool_input = tool_use.get("input", {}) or {}
                    tool_use_id = tool_use.get("id")
                    if not _is_tool_allowed(name):
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": f"Tool '{name}' is not allowed for verification.",
                                "is_error": True,
                            }
                        )
                        continue
                    try:
                        output = tool_registry.dispatch(
                            name, tool_input if isinstance(tool_input, dict) else {}
                        )
                        output_text = (
                            output
                            if isinstance(output, str)
                            else json.dumps(output, ensure_ascii=False)
                        )
                        exec_error = None
                        if (
                            name in self.ui_action_names
                            and isinstance(output, str)
                            and output.strip()
                            and output not in {"DONE", "FAIL"}
                        ):
                            new_obs, exec_error = self._execute_ui_action(output)
                            if new_obs and isinstance(new_obs, dict):
                                latest_screenshot = (
                                    new_obs.get("screenshot") or latest_screenshot
                                )
                                self.grounding.assign_screenshot(new_obs)
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": (
                                    f"{output_text}\n\n[execution_error] {exec_error}"
                                    if exec_error
                                    else output_text
                                ),
                            }
                        )
                        if name == "report_verification_result":
                            reported_payload = (
                                output if isinstance(output, dict) else None
                            )
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
                if latest_screenshot:
                    messages.append(
                        {
                            "role": "user",
                            "content": _build_screenshot_blocks(
                                "Updated screenshot after tool execution:",
                                latest_screenshot,
                            ),
                        }
                    )
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
                    summary_text = (
                        f"{reported_payload.get('conclusion')}: "
                        f"{reported_payload.get('explanation')}"
                    ).strip()
                    return VerificationAgentResult(
                        task_instruction=task_instruction,
                        summary=summary_text or "(no output)",
                        completion_reason="REPORTED",
                        steps_executed=steps_executed,
                        budget=max_rounds,
                        conclusion=reported_payload.get("conclusion"),
                        explanation=reported_payload.get("explanation"),
                        execution_history=execution_steps or None,
                    )
                continue

            messages.append({"role": "assistant", "content": normalized_blocks})
            summary_text = "\n".join(extract_text_blocks(normalized_blocks)).strip()
            if not summary_text:
                summary_text = "(no output)"
            return VerificationAgentResult(
                task_instruction=task_instruction,
                summary=summary_text,
                completion_reason="DONE",
                steps_executed=steps_executed,
                budget=max_rounds,
                execution_history=execution_steps or None,
            )

        return VerificationAgentResult(
            task_instruction=task_instruction,
            summary="Stopped after max rounds",
            completion_reason="DONE",
            steps_executed=steps_executed,
            budget=max_rounds,
            execution_history=execution_steps or None,
            conclusion="SUCCESS",
            explanation="Verification completed.",
        )


class VerificationAgentToolProvider:
    def __init__(self, manager: VerificationAgentManager, grounding) -> None:
        self.manager = manager
        self.grounding = grounding

    # @tool_action
    def call_verification_agent(
        self,
        task: Optional[str] = None,
        is_code_agent_verification: Optional[bool] = None,
    ):
        """
        Invoke the verification agent to validate task completion.
        """
        if is_code_agent_verification is None:
            raise ValueError(
                "Missing required field 'is_code_agent_verification' for call_verification_agent tool"
            )
        task_value = (task or "").strip() or getattr(
            self.grounding, "current_task_instruction", None
        )
        if not task_value:
            raise ValueError(
                "No task instruction available for call_verification_agent tool"
            )

        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling Verification Agent")
        logger.info("=" * 50)
        logger.info("Verifying task: %s", task_value)

        screenshot = ""
        if getattr(self.grounding, "obs", None):
            screenshot = self.grounding.obs.get("screenshot", "")

        second_screenshot = getattr(self.grounding, "second_screenshot", "")
        if not second_screenshot:
            second_screenshot = getattr(self.grounding, "initial_screenshot", "")
        previous_task = getattr(self.grounding, "current_task_instruction", None)
        if task is not None:
            self.grounding.current_task_instruction = task_value
        try:
            result = self.manager.run_task(
                task_value,
                second_screenshot=second_screenshot,
                screenshot=screenshot,
                is_code_agent_verification=bool(is_code_agent_verification),
            )
        finally:
            if task is not None:
                self.grounding.current_task_instruction = previous_task

        if hasattr(self.grounding, "last_verification_agent_result"):
            self.grounding.last_verification_agent_result = result

        logger.info("Verification agent execution completed")
        logger.info("Summary: %s", result.summary)
        logger.info("=" * 50)

        payload = {
            "conclusion": result.conclusion,
            "explanation": result.explanation,
            "summary": result.summary,
        }
        return json.dumps(payload, ensure_ascii=False)


VerificationAgentToolProvider.call_verification_agent.tool_input_schema = {
    "type": "object",
    "description": "Parameters for invoking the Verification Agent.",
    "properties": {
        "task": {
            "type": "string",
            "description": "Optional. Leave blank by default; only fill in for a special task. If omitted, uses the full task.",
        },
        "is_code_agent_verification": {
            "type": "boolean",
            "description": "Required. True if this verification targets Code Agent work.",
        },
    },
    "required": ["is_code_agent_verification"],
    "additionalProperties": False,
}
