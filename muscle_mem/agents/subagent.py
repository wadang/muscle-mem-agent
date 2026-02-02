import io
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from muscle_mem.agents.tool_loop import (
    extract_response_content,
    extract_text_blocks,
    normalize_content_list,
    summarize_tool_use,
)
from muscle_mem.agents.tools.exec_tools import ExecutionToolProvider
from muscle_mem.agents.tools.registry import ToolRegistry, tool_action
from muscle_mem.core.mllm import LMMAgent
from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY
from muscle_mem.utils.common_utils import call_llm_safe

logger = logging.getLogger("desktopenv.agent")

DEFAULT_SUBAGENT_TOOL_ALLOW = [
    "web_search",
    "click",
    "click_image_area",
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
]

UI_SUBAGENT_TOOL_ALLOW = [
    "click",
    "click_image_area",
    "drag_and_drop",
    "scroll",
    "hotkey",
    "wait",
    "hold_and_press",
    "highlight_text_span",
    "type",
]

UI_ACTION_NAMES = {
    "click",
    "click_image_area",
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

SUB_AGENT_TYPES = {
    "Pac-Agent": {
        "description": "当需要对当前界面上的一系列元素执行相同的操作，但该界面缺乏批量处理功能时，激活此 Agent。",
        "tools": DEFAULT_SUBAGENT_TOOL_ALLOW,
        "prompt": (
            "你是 **Meticulous GUI Executor (极度严谨的 GUI 执行者)**。"
            "在处理图形界面上的重复任务（如逐张处理图片、逐个清除红点）时，你必须确保列表中的**每一个**目标都被成功处理。绝不允许在未确认“清零”的情况下草率结束任务。"
        ),
        "max_rounds": 20,
    },
}


@dataclass
class SubAgentResult:
    agent_type: str
    task_instruction: str
    summary: str
    completion_reason: str
    steps_executed: int
    budget: int
    execution_history: Optional[List[Dict[str, Any]]] = None


class SubAgentManager:
    def __init__(
        self,
        engine_params_for_generation: Dict[str, Any],
    ) -> None:
        self.engine_params_for_generation = dict(engine_params_for_generation or {})

    def get_agent_descriptions(self) -> str:
        return "\n".join(
            f"- {name}: {cfg['description']}" for name, cfg in SUB_AGENT_TYPES.items()
        )

    def _build_screenshot_blocks(
        self, llm_agent: LMMAgent, label: str, image_bytes: Any
    ) -> List[Dict[str, Any]]:
        if not image_bytes:
            return []
        try:
            base64_image = llm_agent.encode_image(image_bytes)
        except Exception as exc:
            logger.warning("Failed to attach screenshot to sub-agent: %s", exc)
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

    def _capture_local_obs(self, grounding) -> Optional[Dict[str, Any]]:
        try:
            import pyautogui
            from PIL import Image
        except Exception:
            return None

        target_size = None
        if grounding and getattr(grounding, "obs", None):
            try:
                current_bytes = grounding.obs.get("screenshot")
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
        self, grounding, action_code: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        env = getattr(grounding, "env", None) if grounding is not None else None
        if env is not None and hasattr(env, "step"):
            sleep_after = getattr(env, "sleep_after_execution", None)
            if sleep_after is None:
                sleep_after = getattr(grounding, "sleep_after_execution", 1.0)
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
        return self._capture_local_obs(grounding), None

    def _generate_summary(
        self,
        execution_history: List[Dict[str, Any]],
        task_instruction: str,
        final_response_text: str,
    ) -> str:
        if not execution_history and not final_response_text:
            return "No actions were executed."

        execution_context = f"Task: {task_instruction}\n"
        if execution_history:
            execution_context += "\nExecution Steps:\n"
            for step in execution_history:
                step_num = step.get("step")
                tool_uses = step.get("tool_uses", [])
                tool_results = step.get("tool_results", [])
                execution_context += f"\nStep {step_num}:\n"
                if tool_uses:
                    execution_context += "Tool Uses:\n" + "\n".join(tool_uses) + "\n"
                if tool_results:
                    execution_context += (
                        "Tool Results:\n" + "\n".join(tool_results) + "\n"
                    )

        if final_response_text:
            execution_context += f"\nFinal Response:\n{final_response_text}\n"

        summary_prompt = f"""
{execution_context}

Please provide a concise summary of the sub-agent session. Focus on:

1. The tools used and their outcomes
2. The sequence of actions taken
3. Any final response or conclusions

Keep the summary under 150 words and use clear, factual language.
"""

        try:
            summary_agent = LMMAgent(engine_params=self.engine_params_for_generation)
            summary_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PROCEDURAL_MEMORY.SUBAGENT_SUMMARY_AGENT_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": summary_prompt}],
                },
            ]
            response = call_llm_safe(
                summary_agent,
                messages=summary_messages,
                temperature=summary_agent.engine.temperature or 0.0,
                max_new_tokens=1200,
            )
            content = normalize_content_list(
                response.get("content", [])
                if isinstance(response, dict)
                else getattr(response, "content", [])
            )
            summary_text = "\n".join(extract_text_blocks(content)).strip()
            if not summary_text and not content:
                summary_text = str(response).strip()
            if not summary_text:
                return final_response_text or "(no output)"
            return summary_text
        except Exception as exc:
            logger.warning("Sub-agent summary generation failed: %s", exc)
            return final_response_text or f"Summary generation failed: {str(exc)}"

    def run_task(
        self,
        agent_type: str,
        task_instruction: str,
        *,
        env_controller=None,
        grounding=None,
        screenshot: str = "",
        budget: Optional[int] = None,
    ) -> SubAgentResult:
        if agent_type not in SUB_AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        config = SUB_AGENT_TYPES[agent_type]
        return self._run_light_agent(
            agent_type,
            task_instruction,
            env_controller=env_controller,
            grounding=grounding,
            screenshot=screenshot,
            budget=budget or config.get("max_rounds"),
        )

    def _run_light_agent(
        self,
        agent_type: str,
        task_instruction: str,
        *,
        env_controller=None,
        grounding=None,
        screenshot: str = "",
        budget: Optional[int] = None,
    ) -> SubAgentResult:
        config = SUB_AGENT_TYPES[agent_type]
        max_rounds = int(budget or config.get("max_rounds") or 4)
        system_prompt = config.get("prompt") or "You are a helpful assistant."

        llm_agent = LMMAgent(engine_params=self.engine_params_for_generation)
        tool_registry = ToolRegistry()
        if grounding is not None and getattr(grounding, "ui_actions", None) is not None:
            tool_registry.register_action_provider(grounding.ui_actions)
        exec_tools = ExecutionToolProvider(
            env_controller, engine_params=self.engine_params_for_generation
        )
        tool_registry.register_action_provider(exec_tools)

        blocks: List[Dict[str, Any]] = [{"type": "text", "text": task_instruction}]
        if screenshot:
            blocks.extend(
                self._build_screenshot_blocks(
                    llm_agent, "Current screenshot:", screenshot
                )
            )
        messages: List[Dict[str, Any]] = [{"role": "user", "content": blocks}]
        steps_executed = 0
        execution_history: List[Dict[str, Any]] = []
        allow_list = config.get("tools")
        allow_set = None
        if allow_list and allow_list != ["*"] and allow_list != "*":
            allow_set = set(allow_list)

        def _is_tool_allowed(name: Optional[str]) -> bool:
            if not name:
                return False
            if allow_set is not None and name not in allow_set:
                return False
            return True

        for round_index in range(1, max_rounds + 1):
            steps_executed = round_index
            tools = tool_registry.build_tools(allow=config.get("tools"))
            messages_with_system = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                *messages,
            ]
            response = call_llm_safe(
                llm_agent,
                messages=messages_with_system,
                temperature=llm_agent.engine.temperature or 0.0,
                max_new_tokens=16000,
                tools=tools,
                tool_choice={"type": "any"},
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
                                "content": f"Tool '{name}' is not allowed for sub-agent.",
                                "is_error": True,
                            }
                        )
                        continue
                    try:
                        output = tool_registry.dispatch(
                            name, tool_input if isinstance(tool_input, dict) else {}
                        )
                        if isinstance(output, str):
                            output_text = output
                        else:
                            try:
                                output_text = json.dumps(output, ensure_ascii=False)
                            except Exception:
                                output_text = str(output)
                        exec_error = None
                        if (
                            name in UI_ACTION_NAMES
                            and isinstance(output, str)
                            and output.strip()
                            and output not in {"DONE", "FAIL"}
                        ):
                            new_obs, exec_error = self._execute_ui_action(
                                grounding, output
                            )
                            if new_obs and isinstance(new_obs, dict):
                                latest_screenshot = (
                                    new_obs.get("screenshot") or latest_screenshot
                                )
                                if grounding is not None:
                                    if hasattr(grounding, "assign_screenshot"):
                                        grounding.assign_screenshot(new_obs)
                                    else:
                                        grounding.obs = new_obs
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
                            "content": self._build_screenshot_blocks(
                                llm_agent,
                                "Updated screenshot after tool execution:",
                                latest_screenshot,
                            ),
                        }
                    )
                execution_history.append(
                    {
                        "step": round_index,
                        "tool_uses": [
                            summarize_tool_use(tool_use) for tool_use in tool_uses
                        ],
                        "tool_results": [
                            str(result.get("content") or "") for result in results
                        ],
                    }
                )
                continue

            messages.append({"role": "assistant", "content": normalized_blocks})
            final_response_text = "\n".join(
                extract_text_blocks(normalized_blocks)
            ).strip()
            if not final_response_text:
                final_response_text = "(no output)"
            summary_text = self._generate_summary(
                execution_history, task_instruction, final_response_text
            )
            return SubAgentResult(
                agent_type=agent_type,
                task_instruction=task_instruction,
                summary=summary_text,
                completion_reason="DONE",
                steps_executed=steps_executed,
                budget=max_rounds,
                execution_history=execution_history or None,
            )

        summary_text = self._generate_summary(
            execution_history,
            task_instruction,
            "Stopped after max rounds without a final response.",
        )
        return SubAgentResult(
            agent_type=agent_type,
            task_instruction=task_instruction,
            summary=summary_text,
            completion_reason="BUDGET_EXHAUSTED",
            steps_executed=steps_executed,
            budget=max_rounds,
            execution_history=execution_history or None,
        )


class SubAgentToolProvider:
    def __init__(self, manager: SubAgentManager, grounding) -> None:
        self.manager = manager
        self.grounding = grounding

    # @tool_action 模型不肯使用暂时注释掉
    def gui_batch_automation(self, instruction: str):
        """当需要对GUI上的一系列元素执行相似的操作，可以调用此工具。

        例如：
            - 删除页面中所有的红色区域。
            - 找到页面中所有的兔子，挨个给他们戴上帽子

        Args：
            instruction：指令，需说明如何操作。

        """

        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling call_pac_agent (%s)", subagent)
        logger.info("=" * 50)
        logger.info("Executing task: %s", instruction)

        screenshot = ""
        if getattr(self.grounding, "obs", None):
            screenshot = self.grounding.obs.get("screenshot", "")

        result = self.manager.run_task(
            "Pac-Agent",
            instruction,
            env_controller=getattr(
                getattr(self.grounding, "env", None), "controller", None
            ),
            grounding=self.grounding,
            screenshot=screenshot,
            budget=20,
        )

        if hasattr(self.grounding, "last_subagent_result"):
            self.grounding.last_subagent_result = result
        logger.info("Sub-agent execution completed")
        logger.info("Summary: %s", result.summary)
        logger.info("=" * 50)

        return result.summary or "(no output)"

    # @tool_action
    def call_subagent(
        self,
        subagent: str,
        instruction: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ):
        """
        Run a sub-agent task with isolated context.

        Args:
            subagent: One of light.
            instruction: Task or subtask to execute. If omitted, uses the current full task.
            max_rounds: Optional max rounds override.

        **CRITICAL GUIDELINES:**
        - **ONLY pass an instruction parameter for SPECIFIC subtasks** (e.g., "Calculate sum of column B", "Filter data by date")
        - **NEVER pass an instruction parameter for full tasks** - let it default to the original task instruction
        - **NEVER rephrase or modify the original task** - this prevents hallucination corruption
        - **If unsure, omit the instruction parameter entirely** to use the original task instruction
        """
        task_to_execute = instruction or getattr(
            self.grounding, "current_task_instruction", None
        )
        if not task_to_execute:
            raise ValueError("No task instruction available for call_subagent tool")

        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling Sub-Agent (%s)", subagent)
        logger.info("=" * 50)
        logger.info("Executing task: %s", task_to_execute)

        screenshot = ""
        if getattr(self.grounding, "obs", None):
            screenshot = self.grounding.obs.get("screenshot", "")

        result = self.manager.run_task(
            subagent,
            task_to_execute,
            env_controller=getattr(
                getattr(self.grounding, "env", None), "controller", None
            ),
            grounding=self.grounding,
            screenshot=screenshot,
            budget=max_rounds,
        )

        if hasattr(self.grounding, "last_subagent_result"):
            self.grounding.last_subagent_result = result
        logger.info("Sub-agent execution completed")
        logger.info("Summary: %s", result.summary)
        logger.info("=" * 50)

        return result.summary or "(no output)"


SubAgentToolProvider.call_subagent.tool_input_schema = {
    "type": "object",
    "description": (
        "Invoke a sub-agent to execute a focused task (analysis or device actions).\n\n"
    ),
    "properties": {
        "subagent": {
            "type": "string",
            "enum": list(SUB_AGENT_TYPES.keys()),
            "description": ("Which sub-agent to run."),
        },
        "instruction": {
            "type": "string",
            "description": (
                "Optional. A specific, narrow, verifiable subtask instruction. If omitted, the sub-agent "
                "will receive the ORIGINAL full task instruction. ONLY provide this for focused subtasks; "
                "NEVER paraphrase or rewrite the full task."
            ),
        },
        "max_rounds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": (
                "Optional. Override the maximum execution rounds for the sub-agent. Use this only when "
                "necessary; keep it minimal."
            ),
        },
    },
    "required": ["subagent"],
    "additionalProperties": False,
}
