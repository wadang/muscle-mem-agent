from typing import Any, List, Optional

from muscle_mem.agents.tools.registry import tool_action


class ScratchpadToolProvider:
    def __init__(
        self,
        scratchpad: Optional[List[str]] = None,
        owner: Optional[Any] = None,
        max_items: int = 100,
    ) -> None:
        self.scratchpad = scratchpad if scratchpad is not None else []
        self.owner = owner
        self.max_items = max_items

    def _normalize_entries(self, text: List[str]) -> List[str]:
        if not isinstance(text, list):
            raise ValueError("Text must be a list of strings")
        cleaned: List[str] = []
        for item in text:
            value = str(item or "").strip()
            if value:
                cleaned.append(value)
        if not cleaned:
            raise ValueError("No valid scratchpad items were provided")
        if len(cleaned) > self.max_items:
            raise ValueError(
                f"Scratchpad store is limited to {self.max_items} items"
            )
        return cleaned

    def _render(self, items: Optional[List[str]] = None) -> str:
        entries = items if items is not None else self.scratchpad
        if not entries:
            return "[ ] No scratchpad items yet"
        return "\n".join(f"{index + 1}. {entry}" for index, entry in enumerate(entries))

    def _format_output(self, view: str) -> str:
        task_instruction = None
        if self.owner is not None:
            task_instruction = getattr(self.owner, "current_task_instruction", None)
        output_parts = []
        if task_instruction:
            output_parts.append("**Task:**")
            output_parts.append(str(task_instruction))
        output_parts.append("**Scratchpad:**")
        output_parts.append(view)
        return "\n".join(output_parts).strip()

    @tool_action
    def save_scratchpad(self, text: List[str]):
        """Save reusable facts, elements, or text snippets into a task scratchpad to better equip yourself for the task.
        Args:
            text: List[str] entries to store in the scratchpad.
        """
        entries = self._normalize_entries(text)
        self.scratchpad = list(entries)
        view = self._render()
        return self._format_output(view)

    @tool_action
    def read_scratchpad(self, limit: Optional[int] = None):
        """Read the current scratchpad.
        Args:
            limit: Optional[int] number of most recent entries to return.
        """
        if limit is None:
            view = self._render()
        else:
            try:
                limit_value = int(limit)
            except (TypeError, ValueError):
                raise ValueError("limit must be an integer")
            if limit_value <= 0:
                raise ValueError("limit must be greater than 0")
            view = self._render(self.scratchpad[-limit_value:])
        return self._format_output(view)
