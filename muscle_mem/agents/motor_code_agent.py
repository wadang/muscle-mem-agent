#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from muscle_mem.agents.tool_loop import (
    extract_response_content,
    extract_text_blocks,
    normalize_content_list,
    summarize_tool_use,
)
from muscle_mem.agents.tools import (
    ExecutionToolProvider,
    ScratchpadToolProvider,
    TodoManager,
    TodoRenderConfig,
    TodoToolProvider,
    ToolRegistry,
)
from muscle_mem.agents.tools.registry import tool_action
from muscle_mem.core.mllm import LMMAgent
from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY
from muscle_mem.utils.local_env import LocalEnv


logger = logging.getLogger("desktopenv.agent")


def _redact_base64(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        value_type = value.get("type") if isinstance(value.get("type"), str) else ""
        media_type = (
            value.get("media_type") if isinstance(value.get("media_type"), str) else ""
        )
        for key, item in value.items():
            if key == "data" and isinstance(item, str):
                if value_type == "base64" or media_type.startswith("image/"):
                    redacted[key] = "<base64 omitted>"
                else:
                    redacted[key] = _redact_base64(item)
                continue
            if key == "url" and isinstance(item, str):
                if item.startswith("data:image") and "base64," in item:
                    prefix = item.split("base64,", 1)[0] + "base64,"
                    redacted[key] = prefix + "<base64 omitted>"
                    continue
            redacted[key] = _redact_base64(item)
        return redacted
    if isinstance(value, list):
        return [_redact_base64(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_base64(item) for item in value)
    if isinstance(value, set):
        return [_redact_base64(item) for item in value]
    if hasattr(value, "__dict__"):
        return _redact_base64(getattr(value, "__dict__", {}))
    return value


def log_error_debug(tag: str, info: Any) -> None:
    try:
        payload = json.dumps(_redact_base64(info), ensure_ascii=False, indent=2)
    except Exception:
        payload = "(unserializable info)"
    if len(payload) > 4000:
        payload = payload[:4000] + "\n...<truncated>"
    print(f"⚠️  {tag}:")
    print(payload)


def _serialize_for_logging(value: Any) -> str:
    try:
        return json.dumps(
            _redact_base64(value),
            ensure_ascii=False,
            indent=2,
            default=lambda o: getattr(o, "__dict__", str(o)),
        )
    except Exception:
        return str(value)


def split_thinking_response(full_response: str) -> Tuple[str, str]:
    try:
        thoughts = full_response.split("<thoughts>")[-1].split("</thoughts>")[0].strip()
        answer = full_response.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer, thoughts
    except Exception:
        return full_response, ""


def _normalize_code_agent_engine_params(
    engine_params: Dict[str, Any],
) -> Dict[str, Any]:
    normalized = dict(engine_params or {})
    engine_type = normalized.get("engine_type") or "anthropic"
    model_name = normalized.get("model")
    if model_name.startswith("anthropic/"):
        engine_type = "anthropic"
        model_name = model_name.split("/", 1)[1]
    normalized["engine_type"] = engine_type
    normalized["model"] = model_name
    return normalized


DEFAULT_SUDO_PASSWORD = "password"


def _resolve_sudo_password(sudo_password: Optional[str]) -> str:
    return sudo_password or DEFAULT_SUDO_PASSWORD


def build_system_prompt(sudo_password: Optional[str] = None) -> str:
    resolved_password = _resolve_sudo_password(sudo_password)

    return (
        "你是一位资深的软件工程师，你擅长用代码解决用户的问题。你细致、耐心，考虑周全，决不会马虎大意，也决不会做到一半就放弃。\n"
        "规则：\n"
        "- 使用工具执行操作，任务完成后必须调用 Done 工具结束。\n"
        "- 目前的环境是linux，行动前先观察环境。\n"
        "- 复杂任务可以使用Todo工具来维护多步骤计划。\n"
        f"- 使用sudo的方式：\"echo '{resolved_password}' | sudo -S [命令]\"\n"
        '- 用户名："user"\n'
        f'- 密码："{resolved_password}"\n'
        "\n"
        "# CRITICAL: Data Format Guidelines\n"
        "- 将日期存储为正确的日期对象，而非文本字符串\n"
        "- 将数字存储为数值类型，而非带符号的格式化文本\n"
        "- 保留数据类型以便进行计算和求值\n"
        "- 迁移数据时，保持字段的完整性（例如：完整的雇主名称），记账请严格遵循‘会计主体假设’，尊重原表中的类型，而非自己编造\n"
        "- 对电子表格列应用数据验证时，仅限制包含实际数据的行范围，而非整列\n"
        "- 创建跨工作表引用时，使用单元格引用（例如：=Sheet1!A1），而非手动输入值\n"
        '- 当要求创建新工作表但未指定名称时，默认使用标准工作表名称（例如："Sheet1"、"Sheet2"等，也就是 ws.title="Sheet1"）\n'
        "- 导出邮件时，文件名尽可能保留原始标题中的特殊符号"
        "\n"
        "# CRITICAL: File Modification Strategy\n"
        "- 始终优先就地修改已打开的现有文件，而非创建新文件\n"
        "- 截图上下文显示当前打开的文件，该文件即为需要修改的目标\n"
        "- 对于已打开的文档（LibreOffice的.docx/.xlsx、文本编辑器等），直接修改现有文件\n"
        "- 使用适当的库（python-docx、openpyxl等）就地修改文件\n"
        "- OCR时，先安装适当的库（tesseract-ocr等）\n"
        "- 关键：修改文件时，执行完全覆盖，而非追加\n"
        "- 对于文档：用新内容替换所有段落/工作表\n"
        "- 对于文本文件：写入完整的新内容，覆盖旧内容\n"
        "- 仅在任务明确要求时才创建新文件\n"
        "- 验证你的推理是否符合用户对打开文件的意图\n"
        "\n"
        "# CRITICAL: Thorough File Inspection Guidelines\n"
        "- 修改前后务必检查文件内容和数据类型\n"
        "- 检查单元格值、格式、数据类型、数字格式、小数分隔符和格式属性\n"
        "- 对于电子表格：检查单元格值、数字格式、日期格式、货币格式和单元格属性\n"
        "- 对于文档：检查文本内容、格式、样式和结构元素\n"
        "- 验证修改确实改变了预期的属性（而不仅仅是值）\n"
        "- 对比修改前后的状态，确保更改正确应用\n"
        "\n"
        "# CRITICAL: Preserve Document Structure and Formatting\n"
        "- 修改文档/电子表格时，保留原始结构、标题行和格式\n"
        "- 除非明确要求，切勿修改列标题、行标题、文档标题或工作表名称\n"
        "- 保持字体、颜色、边框、单元格格式、段落样式等\n"
        "- 只更改内容/数据，不更改结构或视觉呈现\n"
        "- 使用支持格式保留的库（python-docx、openpyxl等）\n"
        "- 目标是保持文档外观完全一致，只是内容不同\n"
        "- 对于列重排：保持表格位置——在表格内重排列，不要移动表格本身\n"
        "- 优先判定文档显示效果是否由结构属性生成（如自动编号/列表、样式、字段）而非纯文本\n"
        "    - 若检测到结构属性（如段落存在 pPr/numPr / w:numPr）：新增/插入必须继承相邻同类对象的结构设置（numPr 的 numId/ilvl + 必要的缩进/制表位等），只替换内容文本；严禁仅复制 style 或 add_paragraph 后改 style 来“伪造”结构（会丢 numPr 导致编号断裂）\n"
        "    - 插入/修改后必须复检：新段落仍含 numPr，且 numId/ilvl 与相邻同类段落一致\n"
        "    - 若未检测到结构属性：才按纯文本方式插入/替换\n"
        "    - 例：不要先相信 para.text；先检测段落 pPr/numPr。若存在 numPr，新增条目必须复制上一条段落结构（含 numPr 的 numId/ilvl），仅替换文本内容\n"
        "- 当限制目录是 /home/<name> 形式时，优先理解为“用户主目录”；例如 useradd -d 创建。除非明确要求 SSH chroot 等更强隔离。\n"
        "\n"
        "# 关键：最终步骤要求\n"
        "- 在完成任务前的最后一步（即调用Done工具之前的一步），你**必须**打印出所有已修改文件的内容。\n"
        "- 使用适当的命令显示修改后的最终状态：\n"
        "    * 文本文件：`cat 文件名`，大文件使用 `head -n 50 文件名`。\n"
        "    * Python 文件：`cat filename.py`。\n"
        "    * 配置文件：`cat filename.conf`。\n"
        "    * 其他文件类型：使用相应的查看命令。\n"
        "- 这确保了用户可以清楚地看到文件具体发生了哪些更改。\n"


    )


class DoneToolProvider:
    @tool_action
    def Done(self, reason: Optional[str] = None) -> str:
        """Signal the Code Agent to terminate after this tool call."""
        return reason or "DONE"


def query(
    messages: List[Dict[str, Any]],
    tool_registry: ToolRegistry,
    opts: Dict[str, Any] | None = None,
    model: Optional[str] = None,
    budget: int = 30,
    system_prompt: Optional[str] = None,
    llm_agent: Optional[LMMAgent] = None,
    tool_allow: Optional[List[str]] = None,
    tool_deny: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    opts = opts or {}
    if llm_agent is None:
        raise ValueError("LLM agent is required for code agent queries.")
    max_rounds = budget
    resolved_system_prompt = system_prompt or build_system_prompt()
    continue_prompt = "请继续执行任务。完成后请务必调用 Done 工具结束。"
    for round_index in range(1, max_rounds + 1):
        print(f"Round {round_index}/{max_rounds}")
        try:
            messages_with_system = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": resolved_system_prompt}],
                },
                *messages,
            ]
            print("---- 提交给模型的消息 ----")
            print(_serialize_for_logging(messages_with_system))
            response = llm_agent.get_response(
                messages=messages_with_system,
                temperature=llm_agent.engine.temperature or 0.0,
                max_new_tokens=16000,
                tools=tool_registry.build_tools(allow=tool_allow, deny=tool_deny),
                **(
                    {"tool_choice": opts["tool_choice"]}
                    if "tool_choice" in opts
                    else {}
                ),
            )
            print("---- 模型返回的原始响应 ----")
            print(_serialize_for_logging(response))
        except Exception as e:
            print(f"API Error: {e}")
            raise e

        tool_uses: List[Any] = []
        content_blocks: List[Dict[str, Any]] = []
        stop_reason = None
        try:
            content_blocks, stop_reason = extract_response_content(response)
            for block in content_blocks or []:
                block_type = (
                    getattr(block, "type", None)
                    if not isinstance(block, dict)
                    else block.get("type")
                )
                if block_type == "text":
                    text_value = (
                        getattr(block, "text", None)
                        if not isinstance(block, dict)
                        else block.get("text")
                    )
                    print(text_value or "")
                if block_type == "tool_use":
                    tool_uses.append(block)
        except Exception as err:
            log_error_debug(
                "Iterating response content failed",
                {
                    "error": str(err),
                    "stop_reason": stop_reason,
                    "content_type": type(content_blocks).__name__,
                },
            )
            raise

        normalized_blocks = normalize_content_list(content_blocks)
        if stop_reason == "tool_use":
            results = []
            done_called = False
            for tool_use in tool_uses:
                name = tool_use.get("name")
                input_obj = tool_use.get("input", {}) or {}
                tool_use_id = tool_use.get("id")
                if isinstance(name, str) and name.lower() == "done":
                    done_called = True
                try:
                    output = tool_registry.dispatch(
                        name, input_obj if isinstance(input_obj, dict) else {}
                    )
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": output,
                        }
                    )
                except Exception as error:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": str(error),
                            "is_error": True,
                        }
                    )
            messages.append({"role": "assistant", "content": normalized_blocks})
            messages.append({"role": "user", "content": results})
            if done_called:
                return messages
            continue

        messages.append({"role": "assistant", "content": normalized_blocks})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": continue_prompt}],
            }
        )
        continue

    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"Stopped after {max_rounds} rounds without final response.",
                }
            ],
        }
    )
    return messages


class CodeAgent:
    """A drop-in compatible agent for code execution with tool use."""

    def __init__(
        self,
        engine_params: Optional[Dict[str, Any]] = None,
        budget: int = 30,
        env_controller=None,
    ):
        self.engine_params = _normalize_code_agent_engine_params(engine_params or {})
        self.budget = budget
        self.env_controller = env_controller
        self.system_prompt = build_system_prompt(
            self.engine_params.get("client_password")
        )
        self.llm_agent = LMMAgent(
            engine_params=self.engine_params, system_prompt=self.system_prompt
        )
        self.model = self.engine_params.get("model")
        self.reset()

    def reset(self) -> None:
        render_config = TodoRenderConfig(
            checkbox_pending="☐",
            checkbox_completed="☒",
            empty_text="☐ No todos yet",
            include_status=True,
        )
        if not hasattr(self, "scratchpad") or self.scratchpad is None:
            self.scratchpad: List[str] = []
        self.current_task_instruction = None
        self.todo_board = TodoManager(
            allow_alias_fields=False,
            default_active_form=False,
            render_config=render_config,
        )
        self.todo_tools = TodoToolProvider(self.todo_board)
        self.scratchpad_tools = ScratchpadToolProvider(self.scratchpad, owner=self)
        if not hasattr(self, "exec_tools") or self.exec_tools is None:
            self.exec_tools = ExecutionToolProvider(
                self.env_controller, engine_params=self.engine_params
            )
        else:
            self.exec_tools.set_env_controller(self.env_controller)
            self.exec_tools.set_engine_params(self.engine_params)
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_action_provider(self.exec_tools)
        self.tool_registry.register_action_provider(self.todo_tools)
        self.tool_registry.register_action_provider(self.scratchpad_tools)
        self.tool_registry.register_action_provider(DoneToolProvider())
        self.pending_context_blocks: List[Dict[str, str]] = []

    def _build_screenshot_block(self, screenshot: Any) -> Optional[Dict[str, Any]]:
        if not screenshot:
            return None
        try:
            base64_image = self.llm_agent.encode_image(screenshot)
        except Exception as exc:
            logger.warning("Failed to attach screenshot to code agent: %s", exc)
            return None
        engine_type = (self.engine_params.get("engine_type") or "").lower()
        if engine_type in ("anthropic", "anthropiclr"):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image,
                },
            }
        if engine_type == "vllm":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image;base64,{base64_image}"},
            }
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high",
            },
        }

    def _collect_step_info(
        self, new_messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], List[str]]:
        assistant_texts: List[str] = []
        tool_uses: List[str] = []
        tool_results: List[str] = []
        for message in new_messages:
            role = message.get("role")
            content = message.get("content", [])
            if role == "assistant":
                assistant_texts.extend(extract_text_blocks(content))
                for block in content or []:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_uses.append(summarize_tool_use(block))
            elif role == "user":
                for block in content or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_results.append(str(block.get("content") or ""))
        action_text = "\n".join(text for text in assistant_texts if text).strip()
        if tool_uses:
            action_text = (
                action_text + "\n\nTool Use:\n" + "\n".join(tool_uses)
            ).strip()
        if tool_results:
            action_text = (
                action_text + "\n\nTool Results:\n" + "\n".join(tool_results)
            ).strip()
        if not action_text:
            action_text = "(no assistant text)"
        return action_text, tool_uses, tool_results

    def execute(
        self,
        task_instruction: str,
        screenshot: str,
        env_controller=None,
        pending_context: Optional[str] = None,
    ) -> Dict:
        env_controller = env_controller or self.env_controller
        if env_controller is None:
            raise ValueError("env_controller is required for code execution")

        self.reset()
        self.current_task_instruction = task_instruction
        messages: List[Dict[str, Any]] = []
        blocks: List[Dict[str, Any]] = []

        if self.pending_context_blocks:
            blocks.extend(self.pending_context_blocks)
            self.pending_context_blocks.clear()
        task_text = task_instruction
        if screenshot:
            task_text = f"Task: {task_instruction}\n\nCurrent screenshot is provided for context."
        blocks.append({"type": "text", "text": task_text})
        screenshot_block = self._build_screenshot_block(screenshot)
        if screenshot_block:
            blocks.append(screenshot_block)
        
        if pending_context:
            blocks.append({"type": "text", "text": pending_context})    
        
        messages.append({"role": "user", "content": blocks})

        step_count = 0
        execution_history: List[Dict[str, Any]] = []
        completion_reason = None

        prev_len = len(messages)
        self.exec_tools.set_env_controller(env_controller)
        messages = query(
            messages,
            tool_registry=self.tool_registry,
            model=self.model,
            budget=self.budget,
            system_prompt=self.system_prompt,
            llm_agent=self.llm_agent,
        )
        new_messages = messages[prev_len:]
        action_text, tool_uses, tool_results = self._collect_step_info(new_messages)
        action, thoughts = split_thinking_response(action_text)
        done_called = any(
            str(tool_use).strip().lower().startswith("done") for tool_use in tool_uses
        )
        execution_history.append(
            {
                "step": step_count + 1,
                "action": action,
                "thoughts": thoughts,
                "tool_uses": tool_uses,
                "tool_results": tool_results,
                "done_called": done_called,
            }
        )

        action_upper = action.upper().strip()
        if done_called or action_upper == "DONE":
            completion_reason = "DONE"
        elif action_upper == "FAIL":
            completion_reason = "FAIL"
        else:
            completion_reason = "UNKNOWN"

        step_count += 1

        summary = self._generate_summary(execution_history, task_instruction)

        print("--- Execution Summary ---")
        print(summary)

        return {
            "task_instruction": task_instruction,
            "completion_reason": completion_reason,
            "summary": summary,
            "execution_history": execution_history,
            "steps_executed": step_count,
            "budget": self.budget,
        }

    # 调试专用
    def predict(self, instruction: str, obs: Dict) -> Tuple[Dict, List[str]]:
        result = self.execute(
            instruction,
            screenshot=str(obs.get("screenshot", "")) if isinstance(obs, dict) else "",
            env_controller=self.env_controller,
        )
        return result, ["DONE"]

    def _generate_summary(
        self, execution_history: List[Dict[str, Any]], task_instruction: str
    ) -> str:
        if not execution_history:
            return "No actions were executed."

        execution_context = f"Task: {task_instruction}\n\nExecution Steps:\n"
        for step in execution_history:
            step_num = step.get("step")
            thoughts = step.get("thoughts", "")
            action = step.get("action", "")
            tool_results = step.get("tool_results", [])
            execution_context += f"\nStep {step_num}:\n"
            if thoughts:
                execution_context += f"Thoughts: {thoughts}\n"
            execution_context += f"Action: {action}\n"
            if tool_results:
                execution_context += "Tool Results:\n" + "\n".join(tool_results) + "\n"

        summary_prompt = f"""
{execution_context}

Please provide a concise summary of the code execution session. Focus on:

1. The logic used at each step
2. The outputs and results produced by each tool execution
3. The progression of the solution approach

Do not make judgments about success or failure. Simply describe what was attempted and what resulted.

        Keep the summary under 150 words and use clear, factual language.
"""

        try:
            summary_messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": PROCEDURAL_MEMORY.CODE_SUMMARY_AGENT_PROMPT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": summary_prompt}],
                },
            ]
            response = self.llm_agent.get_response(
                messages=summary_messages, max_new_tokens=1200
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
                return "Summary generation failed - no response from LLM"
            return summary_text
        except Exception as exc:
            return f"Summary generation failed: {str(exc)}"


@dataclass
class CodeAgentResult:
    task_instruction: str
    summary: str
    completion_reason: str
    steps_executed: int
    budget: int
    execution_history: Optional[List[Dict[str, Any]]] = None


class CodeAgentManager:
    def __init__(
        self,
        engine_params: Dict[str, Any],
        budget: int = 40,
    ) -> None:
        self.engine_params = dict(engine_params or {})
        self.budget = budget
        self._code_agent: Optional[CodeAgent] = None

    def run_task(
        self,
        task_instruction: str,
        *,
        env_controller=None,
        screenshot: str = "",
        budget: Optional[int] = None,
        pending_context: Optional[str] = None,
    ) -> CodeAgentResult:
        if budget is None:
            budget = self.budget
        if self._code_agent is None or self._code_agent.budget != budget:
            self._code_agent = CodeAgent(
                self.engine_params,
                budget=budget,
                env_controller=env_controller,
            )
        result = self._code_agent.execute(
            task_instruction,
            screenshot,
            env_controller=env_controller,
            pending_context=pending_context,
        )
        return CodeAgentResult(
            task_instruction=result.get("task_instruction", task_instruction),
            summary=result.get("summary") or "(no summary)",
            completion_reason=result.get("completion_reason", "UNKNOWN"),
            steps_executed=int(result.get("steps_executed", 0)),
            budget=int(result.get("budget", budget)),
            execution_history=result.get("execution_history"),
        )


class CodeAgentToolProvider:
    def __init__(self, manager: CodeAgentManager, grounding) -> None:
        self.manager = manager
        self.grounding = grounding

    @tool_action
    def call_code_agent(
        self,
        task: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ):
        """
        Invoke the Code Agent to execute work that can be completed solely via code.

        Recommended use cases for the Code Agent:
        - Spreadsheets (LibreOffice Calc, Excel): data processing, filtering, sorting, calculations,
          formulas, bulk editing, cleanup.
        - Documents (LibreOffice Writer, Word): text processing, content editing, bulk formatting,
          reference cleanup.
        - Code editors (VS Code, text editors): code editing, file processing, configuration updates.
        - Data analysis: transformation, statistics, reporting.
        - File management: batch operations, extraction, merging, conversions.
        - Web information retrieval.
        - System utilities: automation, configuration, setup (within allowed constraints).

        **CRITICAL GUIDELINES (MUST FOLLOW):**
        1) Full-task invocation (DEFAULT):
           - Omit `instruction` entirely.
           - The Code Agent will use the ORIGINAL full task instruction.
           - DO NOT rephrase, summarize, or modify the original task instruction.
        2) Subtask invocation (ONLY WHEN NECESSARY):
           - Provide `instruction` ONLY for a specific, narrow, verifiable subtask.
           - Good examples: "Calculate the sum of column B and write it to cell D2",
             "Filter rows by date range and export".
           - Bad examples: paraphrasing the whole task, adding extra goals, or vague instructions like
             "clean the sheet".
        3) If unsure whether to pass `instruction`, OMIT it.

        """
        task_to_execute = task or getattr(
            self.grounding, "current_task_instruction", None
        )
        if not task_to_execute:
            raise ValueError("No task instruction available for call_code_agent tool")

        logger.info("=" * 50)
        logger.info("GROUNDING AGENT: Calling Code Agent")
        logger.info("=" * 50)
        logger.info("Executing task: %s", task_to_execute)

        screenshot = ""
        if getattr(self.grounding, "obs", None):
            screenshot = self.grounding.obs.get("screenshot", "")

        pending_context = None
        if hasattr(self.grounding, "pop_pending_feasible_report"):
            pending_context = self.grounding.pop_pending_feasible_report(
                for_code_agent=True
            )

        result = self.manager.run_task(
            task_to_execute,
            env_controller=getattr(
                getattr(self.grounding, "env", None), "controller", None
            ),
            screenshot=screenshot,
            budget=max_rounds,
            pending_context=pending_context,
        )

        if hasattr(self.grounding, "last_code_agent_result"):
            self.grounding.last_code_agent_result = result
        logger.info("Code agent execution completed")
        logger.info("Summary: %s", result.summary)
        logger.info("=" * 50)

        return result.summary or "(no output)"


CodeAgentToolProvider.call_code_agent.tool_input_schema = {
    "type": "object",
    "description": "Parameters for invoking the Code Agent.",
    "properties": {
        "task": {
            "type": "string",
            "description": (
                "If omitted, the Code Agent "
                "receives the ORIGINAL full task instruction."
                "If absolutely necessary, provide a specific, narrow, verifiable subtask instruction."
            ),
        },
        "max_rounds": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": "Override the maximum execution rounds for the Code Agent.",
        },
    },
    "additionalProperties": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Kode Agent (CLI)")
    parser.add_argument("instruction", type=str, help="Instruction for the agent")
    args = parser.parse_args()

    try:
        env = LocalEnv()
        agent = CodeAgent({}, budget=20, env_controller=env.controller)
        result = agent.execute(
            args.instruction, screenshot="", env_controller=env.controller
        )
        print(result.get("summary") or "")
    except Exception as error:
        print(f"Error: {str(error)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
