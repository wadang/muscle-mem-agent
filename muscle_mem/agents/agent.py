import logging
import platform
from typing import Dict, List, Optional, Tuple

from muscle_mem.agents.grounding import ACI
from muscle_mem.agents.worker import Worker

logger = logging.getLogger("desktopenv.agent")


class UIAgent:
    """Base class for UI automation agents"""

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
    ):
        """Initialize UIAgent

        Args:
            worker_engine_params: Configuration parameters for the worker LLM agent
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (macos, linux, windows)
        """
        self.worker_engine_params = worker_engine_params
        self.grounding_agent = grounding_agent
        self.platform = platform

    def reset(self) -> None:
        """Reset agent state"""
        pass

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        """Generate next action prediction

        Args:
            instruction: Natural language instruction
            observation: Current UI state observation

        Returns:
            Tuple containing agent info dictionary and list of actions
        """
        pass


class AgentMm(UIAgent):
    """Agent that uses no hierarchy for less inference time"""

    def __init__(
        self,
        worker_engine_params: Dict,
        grounding_agent: ACI,
        platform: str = platform.system().lower(),
        max_trajectory_length: int = 5,  # 这个多寡，和模型能力相关之前默认是 8 ，本地运作改成 5
        enable_reflection: bool = True,
    ):
        """Initialize a minimalist AgentS2 without hierarchy

        Args:
            worker_engine_params: Configuration parameters for the worker agent.
            grounding_agent: Instance of ACI class for UI interaction
            platform: Operating system platform (darwin, linux, windows)
            max_trajectory_length: Maximum number of image turns to keep
            enable_reflection: Creates a reflection agent to assist the worker agent
        """

        super().__init__(worker_engine_params, grounding_agent, platform)
        self.max_trajectory_length = max_trajectory_length
        self.enable_reflection = enable_reflection

        self.reset()

    def reset(self) -> None:
        """Reset agent state and initialize components"""
        self.executor = Worker(
            worker_engine_params=self.worker_engine_params,
            grounding_agent=self.grounding_agent,
            platform=self.platform,
            max_trajectory_length=self.max_trajectory_length,
            enable_reflection=self.enable_reflection,
        )
        self._phase = "infeasible"
        self.last_infeasible_payload: Optional[Dict] = None
        infeasible_manager = getattr(
            self.grounding_agent, "infeasible_agent_manager", None
        )
        if infeasible_manager is not None and hasattr(infeasible_manager, "reset"):
            infeasible_manager.reset()

    def _run_infeasible_step(self, instruction: str, observation: Dict):
        manager = getattr(self.grounding_agent, "infeasible_agent_manager", None)
        if manager is None or not hasattr(manager, "generate_next_action"):
            return None
        try:
            return manager.generate_next_action(instruction, observation)
        except Exception as exc:
            logger.warning("Failed to run infeasible agent step: %s", exc)
            return None

    def _cache_feasible_report(self, payload: Optional[Dict]) -> None:
        if not payload or payload.get("feasible") is not True:
            return
        report = {
            "reason": (payload.get("reason") or "").strip(),
            "evidence": (payload.get("evidence") or "").strip(),
        }
        if hasattr(self.grounding_agent, "set_pending_feasible_report"):
            self.grounding_agent.set_pending_feasible_report(report)
            return
        setattr(self.grounding_agent, "pending_feasible_report", report)
        setattr(self.grounding_agent, "feasible_report_used_by_worker", False)
        setattr(self.grounding_agent, "feasible_report_used_by_code_agent", False)

    def predict(self, instruction: str, observation: Dict) -> Tuple[Dict, List[str]]:
        if self._phase == "done":
            info = {
                "plan": "INFEASIBLE",
                "plan_code": "",
                "exec_code": "FAIL",
                "reflection": None,
                "reflection_thoughts": None,
                "code_agent_output": None,
            }
            if self.last_infeasible_payload is not None:
                info["infeasible"] = self.last_infeasible_payload
            return info, ["FAIL"]
        if self._phase == "infeasible":
            result = self._run_infeasible_step(instruction, observation)
            if result is not None:
                payload = getattr(result, "payload", None)
                if payload is not None:
                    self.last_infeasible_payload = payload
                status = getattr(result, "status", None)
                if status == "ACTION":
                    info = getattr(result, "info", {}) or {}
                    if self.last_infeasible_payload is not None:
                        info["infeasible"] = self.last_infeasible_payload
                    return info, getattr(result, "actions", []) or []
                if status == "INFEASIBLE":
                    info = getattr(result, "info", {}) or {}
                    if self.last_infeasible_payload is not None:
                        info["infeasible"] = self.last_infeasible_payload
                    self._phase = "done"
                    return info, ["FAIL"]
                if status in {"FEASIBLE", "NO_DECISION", "NO_TASK"}:
                    if status == "FEASIBLE":
                        self._cache_feasible_report(self.last_infeasible_payload)
                    self._phase = "execute"
            else:
                self._phase = "execute"
                
        # # 测试代码 --- 直接返回 DONE
        
        # if self._phase == "execute":
        #     info = {
        #         "plan": "FEASIBLE",
        #         "plan_code": "",
        #         "exec_code": "DONE",
        #         "reflection": None,
        #         "reflection_thoughts": None,
        #         "code_agent_output": None,
        #     }

        #     return info, ["DONE"]
        
        # # ---

        # Initialize the three info dictionaries
        executor_info, actions = self.executor.generate_next_action(
            instruction=instruction, obs=observation
        )

        # concatenate the three info dictionaries
        info = {**{k: v for d in [executor_info or {}] for k, v in d.items()}}
        if self.last_infeasible_payload is not None:
            info["infeasible"] = self.last_infeasible_payload

        return info, actions
