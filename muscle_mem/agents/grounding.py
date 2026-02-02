import logging
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from pytesseract import Output

from muscle_mem.agents.motor_code_agent import (
    CodeAgentManager,
    CodeAgentToolProvider,
)
from muscle_mem.agents.subagent import SubAgentManager, SubAgentToolProvider
from muscle_mem.agents.infeasible_agent import (
    InfeasibleAgentManager,
    InfeasibleAgentToolProvider,
)
# NOTE: Verification agent disabled per request (commented out integration).
# from motor_mem_agent.agents.verification_agent import (
#     VerificationAgentManager,
#     VerificationAgentToolProvider,
# )
from muscle_mem.agents.tools import (
    ExecutionToolProvider,
    ScratchpadToolProvider,
    TodoManager,
    TodoRenderConfig,
    TodoToolProvider,
    ToolRegistry,
    UIActions,
    tool_action,
)
from muscle_mem.core.mllm import LMMAgent
from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY
from muscle_mem.utils.common_utils import (
    call_llm_safe,
    call_llm_safe_with_thinking,
)

logger = logging.getLogger("desktopenv.agent")


class ACI:
    def __init__(self):
        self.scratchpad: List[str] = []

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        return []

    def call_tool(self, name: str, tool_input: Optional[Dict[str, Any]] = None):
        raise NotImplementedError("Tool use is not implemented for this agent")


# ACI primitives are parameterized by description, and coordinate generation uses a pretrained grounding model
class OSWorldACI(ACI):
    def __init__(
        self,
        env,
        platform: str,
        engine_params_for_generation: Dict,
        engine_params_for_grounding: Dict,
        engine_params_for_image_grounding: Optional[Dict] = None,
        width: int = 1920,
        height: int = 1080,
        code_agent_budget: int = 30,
        code_agent_engine_params: Dict = None,
    ):
        super().__init__()

        self.env = env
        self.platform = (
            platform  # Dictates how the switch_applications agent action works.
        )

        # Configure scaling
        self.width = width
        self.height = height

        # Maintain state for save_scratchpad
        self.scratchpad = []
        self.execution_history: List[Dict[str, Any]] = []

        # Screenshot used during ACI execution
        self.obs = None
        self.initial_screenshot = None
        self.second_screenshot = None
        self.last_grounding_thoughts: Optional[str] = None

        # Configure the visual grounding model responsible for coordinate generation
        grounding_engine_params = dict(engine_params_for_grounding)
        if grounding_engine_params.get("engine_type") == "anthropic":
            grounding_engine_params["prompt_caching"] = False

        self.grounding_model = LMMAgent(grounding_engine_params)
        self.engine_params_for_grounding = grounding_engine_params

        if engine_params_for_image_grounding:
            image_grounding_engine_params = dict(engine_params_for_image_grounding)
            if image_grounding_engine_params.get("engine_type") == "anthropic":
                image_grounding_engine_params["prompt_caching"] = False
            self.image_grounding_model = LMMAgent(image_grounding_engine_params)
            self.engine_params_for_image_grounding = image_grounding_engine_params
        else:
            self.image_grounding_model = self.grounding_model
            self.engine_params_for_image_grounding = self.engine_params_for_grounding

        # Configure text grounding agent
        text_span_engine_params = dict(engine_params_for_generation)
        if text_span_engine_params.get("engine_type") == "anthropic":
            text_span_engine_params["prompt_caching"] = False
        self.text_span_agent = LMMAgent(
            engine_params=text_span_engine_params,
            system_prompt=PROCEDURAL_MEMORY.PHRASE_TO_WORD_COORDS_PROMPT,
        )

        # Configure sub-agent manager (lightweight only)
        code_agent_engine_params = (
            code_agent_engine_params or engine_params_for_generation
        )
        if self.env is not None and hasattr(self.env, "client_password"):
            code_agent_engine_params = dict(code_agent_engine_params)
            code_agent_engine_params["client_password"] = self.env.client_password
        self.subagent_manager = SubAgentManager(
            engine_params_for_generation=engine_params_for_generation,
        )
        self.code_agent_manager = CodeAgentManager(
            engine_params=code_agent_engine_params,
            budget=code_agent_budget,
        )
        self.infeasible_agent_manager = InfeasibleAgentManager(
            engine_params=engine_params_for_generation,
            grounding=self,
        )
        # 验证 Agent 功能暂时停用（不注册到主 Agent / Code Agent 流程）
        # self.verification_agent_manager = VerificationAgentManager(
        #     engine_params=engine_params_for_generation,
        #     grounding=self,
        # )

        # Store task instruction for sub-agent routing
        self.current_task_instruction = None
        self.last_subagent_result: Optional[Any] = None
        self.last_code_agent_result: Optional[Any] = None
        self.last_infeasible_agent_result: Optional[Any] = None
        self.last_infeasible_report: Optional[Dict[str, str]] = None
        # self.last_verification_agent_result: Optional[Any] = None
        self.pending_feasible_report: Optional[Dict[str, str]] = None
        self.feasible_report_used_by_worker = False
        self.feasible_report_used_by_code_agent = False
        render_config = TodoRenderConfig(
            checkbox_pending="[ ]",
            checkbox_completed="[x]",
            empty_text="[ ] No todos yet",
            include_status=False,
        )
        self.todo_board = TodoManager(
            allow_alias_fields=True,
            default_active_form=True,
            render_config=render_config,
        )
        self.last_todo_board_view: Optional[str] = None
        self.last_todo_summary: Optional[str] = None
        self.todo_tools = TodoToolProvider(self.todo_board, owner=self)
        self.scratchpad_tools = ScratchpadToolProvider(self.scratchpad, owner=self)
        self.task_tools = SubAgentToolProvider(self.subagent_manager, self)
        self.code_agent_tools = CodeAgentToolProvider(self.code_agent_manager, self)
        self.infeasible_agent_tools = InfeasibleAgentToolProvider(
            self.infeasible_agent_manager, self
        )
        # self.verification_agent_tools = VerificationAgentToolProvider(
        #     self.verification_agent_manager, self
        # )
        self.exec_tools = ExecutionToolProvider(
            getattr(self.env, "controller", None),
            engine_params=engine_params_for_generation,
        )
        self.ui_actions = UIActions(self)
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_action_provider(self.ui_actions)
        self.tool_registry.register_action_provider(self.exec_tools)
        self.tool_registry.register_action_provider(self.todo_tools)
        self.tool_registry.register_action_provider(self.scratchpad_tools)
        self.tool_registry.register_action_provider(self.task_tools)
        self.tool_registry.register_action_provider(self.code_agent_tools)
        self.tool_registry.register_action_provider(self.infeasible_agent_tools)
        # self.tool_registry.register_action_provider(self.verification_agent_tools)
        self.tool_registry.register_action_provider(self)

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        return self.tool_registry.build_tools(
            deny=[
                "python",
                "web_fetch",
                "bash",
                "call_infeasible_agent",
                "scholarly_publication",
                "scholarly_author",
            ]
        )

    def call_tool(self, name: str, tool_input: Optional[Dict[str, Any]] = None):
        return self.tool_registry.dispatch(name, tool_input)

    def _get_client_password(self) -> str:
        if self.env is None:
            return "password"
        return getattr(self.env, "client_password", None) or "password"

    # Given the state and worker's referring expression, use the grounding model to generate (x,y)
    def generate_coords(
        self, ref_expr: str, obs: Dict, use_image_model: bool = False
    ) -> List[int]:

        # Reset the grounding model state
        grounding_model = (
            self.image_grounding_model if use_image_model else self.grounding_model
        )
        grounding_model.reset()

        # Configure the context, UI-TARS demo does not use system prompt
        prompt = f"Query:{ref_expr}\nOutput only the coordinate of one point in your response.\n"
        grounding_model.add_message(
            text_content=prompt, image_content=obs["screenshot"], put_text_last=True
        )

        # Generate and parse coordinates

        response_text, thinking = call_llm_safe_with_thinking(
            grounding_model,
            extra_body={"enable_thinking": True, "thinking_budget": 4096},
        )

        response_text = (
            response_text if isinstance(response_text, str) else str(response_text)
        )
        self.last_grounding_thoughts = thinking
        print("RAW GROUNDING MODEL RESPONSE:", response_text)
        point_match = re.search(r"<point>\s*(\d+)\s+(\d+)\s*</point>", response_text)
        if point_match:
            return [int(point_match.group(1)), int(point_match.group(2))]
        numericals = re.findall(r"\d+", response_text)
        assert len(numericals) >= 2
        return [int(numericals[0]), int(numericals[1])]

    # Calls pytesseract to generate word level bounding boxes for text grounding
    def get_ocr_elements(self, b64_image_data: str) -> Tuple[str, List]:
        image = Image.open(BytesIO(b64_image_data))
        image_data = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Clean text by removing leading and trailing spaces and non-alphabetical characters, but keeping punctuation
        for i, word in enumerate(image_data["text"]):
            image_data["text"][i] = re.sub(
                r"^[^a-zA-Z\s.,!?;:\-\+]+|[^a-zA-Z\s.,!?;:\-\+]+$", "", word
            )

        ocr_elements = []
        ocr_table = "Text Table:\nWord id\tText\n"
        # Obtain the <id, text, group number, word number> for each valid element
        grouping_map = defaultdict(list)
        ocr_id = 0
        for i in range(len(image_data["text"])):
            block_num = image_data["block_num"][i]
            if image_data["text"][i]:
                grouping_map[block_num].append(image_data["text"][i])
                ocr_table += f"{ocr_id}\t{image_data['text'][i]}\n"
                ocr_elements.append(
                    {
                        "id": ocr_id,
                        "text": image_data["text"][i],
                        "group_num": block_num,
                        "word_num": len(grouping_map[block_num]),
                        "left": image_data["left"][i],
                        "top": image_data["top"][i],
                        "width": image_data["width"][i],
                        "height": image_data["height"][i],
                    }
                )
                ocr_id += 1

        return ocr_table, ocr_elements

    # Given the state and worker's text phrase, generate the coords of the first/last word in the phrase
    def generate_text_coords(
        self, phrase: str, obs: Dict, alignment: str = ""
    ) -> List[int]:

        ocr_table, ocr_elements = self.get_ocr_elements(obs["screenshot"])

        alignment_prompt = ""
        if alignment == "start":
            alignment_prompt = "**Important**: Output the word id of the FIRST word in the provided phrase.\n"
        elif alignment == "end":
            alignment_prompt = "**Important**: Output the word id of the LAST word in the provided phrase.\n"

        # Load LLM prompt
        self.text_span_agent.reset()
        self.text_span_agent.add_message(
            alignment_prompt + "Phrase: " + phrase + "\n" + ocr_table, role="user"
        )
        self.text_span_agent.add_message(
            "Screenshot:\n", image_content=obs["screenshot"], role="user"
        )

        # Obtain the target element
        response = call_llm_safe(self.text_span_agent)
        print("TEXT SPAN AGENT RESPONSE:", response)
        numericals = re.findall(r"\d+", response)
        if len(numericals) > 0:
            text_id = int(numericals[-1])
        else:
            text_id = 0
        elem = ocr_elements[text_id]

        # Compute the element coordinates
        if alignment == "start":
            coords = [elem["left"], elem["top"] + (elem["height"] // 2)]
        elif alignment == "end":
            coords = [elem["left"] + elem["width"], elem["top"] + (elem["height"] // 2)]
        else:
            coords = [
                elem["left"] + (elem["width"] // 2),
                elem["top"] + (elem["height"] // 2),
            ]
        return coords

    def assign_screenshot(self, obs: Dict):
        self.obs = obs
        if self.initial_screenshot is None and isinstance(obs, dict):
            self.initial_screenshot = obs.get("screenshot")
        elif self.second_screenshot is None and isinstance(obs, dict):
            self.second_screenshot = obs.get("screenshot")

    def set_task_instruction(self, task_instruction: str):
        """Set the current task instruction for sub-agent routing."""
        self.current_task_instruction = task_instruction

    def set_pending_feasible_report(self, report: Optional[Dict[str, str]]) -> None:
        self.pending_feasible_report = report
        self.feasible_report_used_by_worker = False
        self.feasible_report_used_by_code_agent = False

    def set_infeasible_report(self, report: Optional[Dict[str, str]]) -> None:
        self.last_infeasible_report = report

    def _format_feasible_report_text(self, report: Dict[str, str]) -> str:
        reason = (report.get("reason") or "").strip() or "(empty)"
        evidence = (report.get("evidence") or "").strip() or "(empty)"
        return f"已验证此任务可执行\\nreason: {reason}\\nevidence: {evidence}\\n **注意：** 选择Code Agent和GUI中效率最高的方式。处理数据、文档、计算、网络请求，批量操作等优先使用代码。"

    def pop_pending_feasible_report(
        self, *, for_code_agent: bool = False
    ) -> Optional[str]:
        report = self.pending_feasible_report
        if not report:
            return None
        # Feasible report injection is disabled; mark as used so it is not added
        # to the main agent or the code agent.
        self.feasible_report_used_by_worker = True
        self.feasible_report_used_by_code_agent = True
        return None

    def reset_scratchpad(self) -> None:
        self.scratchpad.clear()

    def reset_task_state(self) -> None:
        self.reset_execution_history()
        self.reset_scratchpad()
        self.set_pending_feasible_report(None)

    def reset_execution_history(self) -> None:
        self.execution_history = []
        self.initial_screenshot = None
        self.second_screenshot = None

    def record_execution_history(self, entry: Dict[str, Any]) -> None:
        self.execution_history.append(entry)

    def get_execution_history_text(self, max_entries: int = 50) -> str:
        if not self.execution_history:
            return "(no execution history)"
        entries = self.execution_history[-max_entries:]
        lines: List[str] = []
        for entry in entries:
            if isinstance(entry, dict):
                step = entry.get("step")
                plan = entry.get("plan")
                tool_name = entry.get("tool_name")
                tool_input = entry.get("tool_input")
                error = entry.get("error")
                header_parts: List[str] = []
                if step is not None:
                    header_parts.append(f"Step {step}")
                if tool_name:
                    header_parts.append(f"tool={tool_name}")
                if tool_input:
                    header_parts.append(f"input={tool_input}")
                header = " | ".join(header_parts) if header_parts else "Step"
                if plan:
                    header += f"\nPlan: {plan}"
                if error:
                    header += f"\nError: {error}"
                lines.append(header)
            else:
                lines.append(str(entry))
        return "\n\n".join(lines)

    # Resize from grounding model dim into OSWorld dim (1920 * 1080)
    def resize_coordinates(self, coordinates: List[int]) -> List[int]:
        grounding_width = self.engine_params_for_grounding["grounding_width"]
        grounding_height = self.engine_params_for_grounding["grounding_height"]

        return [
            round(coordinates[0] * self.width / grounding_width),
            round(coordinates[1] * self.height / grounding_height),
        ]
