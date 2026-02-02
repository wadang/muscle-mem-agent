import json
import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from muscle_mem.agents.grounding import ACI
from muscle_mem.core.module import BaseModule
from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY
from muscle_mem.utils.common_utils import (
    call_llm_safe,
)

logger = logging.getLogger("desktopenv.agent")


class Worker(BaseModule):
    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = "ubuntu",
        max_trajectory_length: int = 8,
        enable_reflection: bool = True,
    ):
        """
        Worker receives the main task and generates actions, without the need of hierarchical planning
        Args:
            worker_engine_params: Dict
                Parameters for the worker agent
            grounding_agent: Agent
                The grounding agent to use
            platform: str
                OS platform the agent runs on (darwin, linux, windows)
            max_trajectory_length: int
                The amount of images turns to keep
            enable_reflection: bool
                Whether to enable reflection
        """
        super().__init__(worker_engine_params, platform)

        self.temperature = worker_engine_params.get("temperature", 0.0)
        model_name = worker_engine_params.get("model", "")
        if model_name.startswith("anthropic/"):
            worker_engine_params["engine_type"] = "anthropic"
            worker_engine_params["model"] = model_name.split("/", 1)[1]
        if not worker_engine_params.get("engine_type", "").startswith("anthropic"):
            raise ValueError("Worker only supports Anthropic tool use in this branch.")
        self.use_thinking = worker_engine_params.get("use_thinking", False)
        self.max_todo_retries = worker_engine_params.get("max_todo_retries")
        self.max_initial_done_retries = worker_engine_params.get(
            "max_initial_done_retries", 3
        )
        self.grounding_agent = grounding_agent
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = False

        self.reset()

    def reset(self):
        if self.platform != "linux":
            skipped_actions = ["set_cell_values"]
        else:
            skipped_actions = []

        # Hide sub-agent action entirely if no env/controller is available
        if not getattr(self.grounding_agent, "env", None) or not getattr(
            getattr(self.grounding_agent, "env", None), "controller", None
        ):
            skipped_actions.extend(["call_subagent", "call_code_agent"])

        self.skipped_actions = skipped_actions
        sudo_password = None
        if self.grounding_agent is not None:
            env = getattr(self.grounding_agent, "env", None)
            sudo_password = getattr(env, "client_password", None)
        sys_prompt = PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            sudo_password
        )
        sys_prompt = sys_prompt.replace("CURRENT_OS", self.platform)

        self.generator_agent = self._create_agent(sys_prompt)
        self.reflection_agent = None

        self.turn_count = 0
        self.worker_history = []
        self.reflections = []
        self.cost_this_turn = 0
        self.screenshot_inputs = []
        self.initial_done_retries = 0
        self.todo_request_seen = False
        if hasattr(self.grounding_agent, "reset_task_state"):
            self.grounding_agent.reset_task_state()
        elif hasattr(self.grounding_agent, "reset_execution_history"):
            self.grounding_agent.reset_execution_history()

    def _extract_tool_use(self, response):
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            content = []
        for block in content:
            if block.get("type") == "tool_use":
                return block
        return None

    def _append_tool_result(self, tool_use_id: str, content):
        tool_result_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        }
        self.generator_agent.messages.append(tool_result_message)

    def _extract_text_from_response(self, response: Any) -> str:
        content = []
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            content = getattr(response, "content", [])
        texts: List[str] = []
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts).strip()
        return str(response) if response is not None else ""

    def _get_todo_items(self, tool_use: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(tool_use, dict):
            return []
        tool_input = tool_use.get("input")
        if not isinstance(tool_input, dict):
            return []
        items = tool_input.get("items")
        if isinstance(items, list):
            return items
        return []

    def _has_todo_with_min_items(
        self, candidates: List[Dict[str, Any]], min_items: int
    ) -> bool:
        for candidate in candidates:
            if len(self._get_todo_items(candidate)) >= min_items:
                return True
        return False

    def _format_todo_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for idx, candidate in enumerate(candidates, start=1):
            items = self._get_todo_items(candidate)
            try:
                payload = json.dumps(items, ensure_ascii=False, indent=2)
            except Exception:
                payload = str(items)
            parts.append(f"候选 {idx}:\n{payload}")
        return "\n\n".join(parts)

    def _build_todo_retry_messages(
        self,
        base_messages: List[Dict[str, Any]],
        instruction: str,
        previous_candidate: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items = self._get_todo_items(previous_candidate or {})
        try:
            items_payload = json.dumps(items, ensure_ascii=False, indent=2)
        except Exception:
            items_payload = str(items)
        prompt = (
            "上一次 TodoWrite 候选如下，请给出更有利于 Agent 替用户完成操作的 TodoWrite 候选。\n"
            "请直接返回 TodoWrite 工具调用。\n\n"
            f"需要执行的任务：{instruction}\n\n"
            f"上一次 TodoWrite items：\n{items_payload}"
        )
        messages = deepcopy(base_messages)
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        return messages

    def _build_selection_messages(
        self,
        instruction: str,
        screenshot: Any,
        candidates: List[Dict[str, Any]],
        system_message: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates_text = self._format_todo_candidates(candidates)
        prompt = (
            "你将看到原任务、当前截图，以及若干个 TodoWrite 候选结果。\n"
            "请只选择一个最有可能正确执行任务的候选。\n"
            '只返回 JSON：{"choice": <数字>}，不要解释。\n\n'
            f"任务：{instruction}\n\n"
            f"候选结果（编号从 1 开始）：\n{candidates_text}"
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if screenshot is not None:
            try:
                base64_image = self.generator_agent.encode_image(screenshot)
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Failed to attach screenshot for todo selection: %s", exc
                )
        return [
            deepcopy(system_message),
            {
                "role": "user",
                "content": content,
            },
        ]

    def _parse_choice_index(self, text: str, max_index: int) -> int:
        if not text:
            return 1
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                choice = parsed.get("choice")
                if isinstance(choice, str) and choice.isdigit():
                    choice = int(choice)
                if isinstance(choice, int):
                    return choice if 1 <= choice <= max_index else 1
        except Exception:
            pass
        match = re.search(r"\b(\d+)\b", text)
        if match:
            choice = int(match.group(1))
            if 1 <= choice <= max_index:
                return choice
        return 1

    def _collect_todo_candidates(
        self,
        base_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Dict[str, Any],
        instruction: str,
        min_repeats: int,
        max_repeats: int,
        min_items: Optional[int] = None,
        existing_candidates: Optional[List[Dict[str, Any]]] = None,
        reference_candidate: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = list(existing_candidates or [])
        total_repeats = max_repeats
        anchor_candidate = reference_candidate or (
            candidates[0] if candidates else None
        )
        for idx in range(total_repeats):
            if idx >= min_repeats and min_items is not None:
                if self._has_todo_with_min_items(candidates, min_items):
                    break
            retry_messages = (
                self._build_todo_retry_messages(
                    base_messages, instruction, anchor_candidate
                )
                if anchor_candidate is not None
                else deepcopy(base_messages)
            )
            response = call_llm_safe(
                self.generator_agent,
                temperature=self.temperature,
                use_thinking=self.use_thinking,
                tools=tools,
                tool_choice=tool_choice,
                messages=retry_messages,
            )
            tool_use = self._extract_tool_use(response)
            if tool_use is None:
                logger.warning(
                    "TODO retry %d/%d did not return a tool use",
                    idx + 1,
                    total_repeats,
                )
                continue
            name = tool_use.get("name")
            name_normalized = name.lower() if isinstance(name, str) else ""
            if name_normalized != "todowrite":
                logger.warning(
                    "TODO retry %d/%d returned non-todo tool: %s",
                    idx + 1,
                    total_repeats,
                    name,
                )
                continue
            candidates.append(tool_use)
        return candidates

    def _select_best_todo_candidate(
        self,
        instruction: str,
        screenshot: Any,
        candidates: List[Dict[str, Any]],
        system_message: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not candidates:
            return {}
        if len(candidates) == 1:
            return candidates[0]
        selection_messages = self._build_selection_messages(
            instruction, screenshot, candidates, system_message
        )
        response = call_llm_safe(
            self.generator_agent,
            temperature=self.temperature,
            use_thinking=self.use_thinking,
            messages=selection_messages,
        )
        selection_text = self._extract_text_from_response(response)
        choice_index = self._parse_choice_index(selection_text, len(candidates))
        return candidates[choice_index - 1]

    def _is_exec_tool(self, tool_name: str) -> bool:
        exec_tools = getattr(self.grounding_agent, "exec_tools", None)
        if exec_tools is None:
            return False
        handler = getattr(type(exec_tools), tool_name, None)
        return bool(handler and getattr(handler, "is_tool_action", False))

    def _safe_wait(self, seconds: float) -> str:
        try:
            return self.grounding_agent.call_tool("wait", {"time": seconds})
        except Exception:
            return f"import time; time.sleep({seconds})"

    def _get_ui_action_thoughts(self, tool_name: Optional[str]) -> Optional[str]:
        if tool_name not in {"click_image"}:
            return None
        return getattr(self.grounding_agent, "last_grounding_thoughts", None) or None

    def _has_required_action_history(self) -> bool:
        required_tools = {"call_code_agent", "click", "click_image", "hotkey"}
        for entry in self.worker_history:
            if not isinstance(entry, str):
                continue
            match = re.search(r"name=([^\s]+)", entry)
            if not match:
                continue
            if match.group(1).lower() in required_tools:
                return True
        return False

    def _record_execution_history(
        self,
        step_idx: int,
        plan: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict] = None,
        error_message: Optional[str] = None,
    ) -> None:
        if not hasattr(self.grounding_agent, "record_execution_history"):
            return
        entry = {
            "step": step_idx,
            "plan": plan,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error": error_message,
        }
        self.grounding_agent.record_execution_history(entry)

    def flush_messages(self):
        """Flush messages based on the model's context limits.

        This method ensures that the agent's message history does not exceed the maximum trajectory length.

        Side Effects:
            - Modifies the messages of generator, reflection, and bon_judge agents to fit within the context limits.
        """
        # Flush strategy for long-context models: keep all text, only keep latest images
        max_images = self.max_trajectory_length
        for agent in [self.generator_agent, self.reflection_agent]:
            if agent is None:
                continue
            # keep latest k images
            img_count = 0
            for i in range(len(agent.messages) - 1, -1, -1):
                for j in range(len(agent.messages[i]["content"])):
                    if "image" in agent.messages[i]["content"][j].get("type", ""):
                        img_count += 1
                        if img_count > max_images:
                            del agent.messages[i]["content"][j]

    def generate_next_action(self, instruction: str, obs: Dict) -> Tuple[Dict, List]:
        """
        Predict the next action(s) based on the current observation.
        """

        step_idx = self.turn_count + 1
        logger.info("=" * 80)
        logger.info("WORKER STEP %d START", step_idx)
        logger.info("Instruction: %s", instruction)
        logger.info("=" * 80)

        self.grounding_agent.assign_screenshot(obs)
        self.grounding_agent.set_task_instruction(instruction)

        feasible_report_message = None
        # Feasible report injection is disabled per request.

        generator_message = (
            "current observation received."
            # if self.turn_count > 0
            # else "当前桌面截图已提供。"
        )

        # Load the task into the system prompt
        if self.turn_count == 0:
            prompt_with_instructions = self.generator_agent.system_prompt.replace(
                "TASK_DESCRIPTION", instruction
            )
            self.generator_agent.add_system_prompt(prompt_with_instructions)

        # Get the grounding agent's scratchpad buffer
        # generator_message += (
        #     f"\nCurrent Text Buffer = [{','.join(self.grounding_agent.scratchpad)}]\n"
        # )

        # Finalize the generator message
        if feasible_report_message:
            self.generator_agent.add_message(feasible_report_message, role="user")
            logger.info(
                "WORKER STEP %d FEASIBLE REPORT MESSAGE:\n%s",
                step_idx,
                feasible_report_message,
            )
        self.generator_agent.add_message(
            generator_message, image_content=obs["screenshot"], role="user"
        )
        logger.info("WORKER STEP %d CONTEXT MESSAGE:\n%s", step_idx, generator_message)

        # Generate the next action via Anthropic tool use
        tools = self.grounding_agent.get_anthropic_tools()
        if self.skipped_actions:
            tools = [
                tool for tool in tools if tool.get("name") not in self.skipped_actions
            ]
        while True:
            base_messages = deepcopy(self.generator_agent.messages)
            response = call_llm_safe(
                self.generator_agent,
                temperature=self.temperature,
                use_thinking=self.use_thinking,
                tools=tools,
                tool_choice={"type": "any"},
            )
            tool_use = self._extract_tool_use(response)
            if tool_use is None:
                logger.error("WORKER STEP %d TOOL USE EXTRACTION FAILED", step_idx)
                plan = str(response)
                logger.info("WORKER STEP %d PLAN RAW RESPONSE:\n%s", step_idx, plan)
                exec_code = self._safe_wait(1.333)
                plan_code = ""
                self.worker_history.append(plan)
                self._record_execution_history(step_idx, plan)
                self.generator_agent.add_message(plan, role="assistant")
                break

            tool_name = tool_use.get("name")
            tool_name_normalized = (
                tool_name.lower() if isinstance(tool_name, str) else ""
            )
            if tool_name_normalized == "todowrite" and not self.todo_request_seen:
                self.todo_request_seen = True
                todo_items_count = len(self._get_todo_items(tool_use))
                if todo_items_count < 4:
                    logger.info(
                        "WORKER STEP %d initial todo has %d items; requesting more todo candidates (min 3, max 9 repeats if still < 4)",
                        step_idx,
                        todo_items_count,
                    )
                    candidates = [tool_use]
                    candidates = self._collect_todo_candidates(
                        base_messages,
                        tools,
                        {"type": "any"},
                        instruction,
                        min_repeats=3,
                        max_repeats=9,
                        min_items=4,
                        existing_candidates=candidates,
                        reference_candidate=tool_use,
                    )
                    selected = self._select_best_todo_candidate(
                        instruction,
                        obs.get("screenshot"),
                        candidates,
                        base_messages[0] if base_messages else {},
                    )
                    if selected:
                        tool_use = selected
                        response = {"content": [tool_use], "stop_reason": "tool_use"}
                        tool_name = tool_use.get("name")
                        tool_name_normalized = (
                            tool_name.lower() if isinstance(tool_name, str) else ""
                        )
                    else:
                        logger.warning(
                            "TODO candidate selection failed; falling back to initial todo"
                        )
            tool_input = tool_use.get("input", {})
            if (
                tool_name_normalized == "done"
                and not self._has_required_action_history()
                and self.initial_done_retries < self.max_initial_done_retries
            ):
                self.initial_done_retries += 1
                logger.warning(
                    "WORKER STEP %d received done without required actions; returning guidance and retrying (%d/%d)",
                    step_idx,
                    self.initial_done_retries,
                    self.max_initial_done_retries,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": (
                        response.get("content", [tool_use])
                        if isinstance(response, dict)
                        else [tool_use]
                    ),
                }
                self.generator_agent.messages.append(assistant_message)
                tool_use_id = tool_use.get("id", "tool_use")
                self._append_tool_result(
                    tool_use_id,
                    [
                        {
                            "type": "text",
                            "text": "你的任务不是提供建议，而是替用户执行他想做的事情。请将任务直接交给 Code Agent 。",
                        }
                    ],
                )
                continue
            plan = f"TOOL_USE name={tool_name} input={json.dumps(tool_input, ensure_ascii=False)}"
            self.worker_history.append(plan)
            self._record_execution_history(step_idx, plan, tool_name, tool_input)
            assistant_message = {
                "role": "assistant",
                "content": (
                    response.get("content", [tool_use])
                    if isinstance(response, dict)
                    else [tool_use]
                ),
            }
            self.generator_agent.messages.append(assistant_message)
            tool_use_id = tool_use.get("id", "tool_use")
            logger.info("WORKER STEP %d PLAN RAW RESPONSE:\n%s", step_idx, plan)
            error_message = None
            try:
                tool_output = self.grounding_agent.call_tool(tool_name, tool_input)
                exec_code = tool_output
            except Exception as e:
                logger.error(
                    "Could not execute tool %s with input %s: %s",
                    tool_name,
                    tool_input,
                    e,
                )
                error_message = str(e)
                exec_code = self._safe_wait(1.333)
            plan_code = ""
            if tool_name in {
                "call_pac_agent",
                "call_code_agent",
                "call_infeasible_agent",
            }:
                text = error_message if error_message else str(tool_output or "")
                if not text:
                    text = "(no output)"
                content = [{"type": "text", "text": text}]
                self._append_tool_result(tool_use_id, content)
                if hasattr(self.grounding_agent, "last_subagent_result"):
                    self.grounding_agent.last_subagent_result = None
                if hasattr(self.grounding_agent, "last_code_agent_result"):
                    self.grounding_agent.last_code_agent_result = None
                if hasattr(self.grounding_agent, "last_infeasible_agent_result"):
                    self.grounding_agent.last_infeasible_agent_result = None
                # 验证 Agent 功能已停用
                break
            elif self._is_exec_tool(tool_name) or tool_name in {
                "TodoWrite",
                "save_scratchpad",
                "read_scratchpad",
            }:
                text = error_message if error_message else str(tool_output or "")
                if not text:
                    text = "(no output)"
                self._append_tool_result(tool_use_id, [{"type": "text", "text": text}])
                exec_code = ""
                continue
            elif tool_name in {"done", "fail", "report_infeasible"}:
                text = error_message if error_message else "OK"
                self._append_tool_result(tool_use_id, [{"type": "text", "text": text}])
                break
            else:
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
                break

        executor_info = {
            "plan": plan,
            "plan_code": plan_code,
            "exec_code": exec_code,
            "reflection": None,
            "reflection_thoughts": None,
            "code_agent_output": None,
        }
        self.turn_count += 1
        self.screenshot_inputs.append(obs["screenshot"])
        self.flush_messages()
        logger.info("WORKER STEP %d EXECUTABLE CODE:\n%s", step_idx, exec_code)
        logger.info(
            "WORKER STEP %d END | reflection=%s | actions_generated=%d",
            step_idx,
            "no",
            1 if exec_code else 0,
        )
        logger.info("=" * 80)
        return executor_info, [exec_code]
