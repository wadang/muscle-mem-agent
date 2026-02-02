import json
from typing import Any, Dict, List, Tuple


def block_to_dict(block: Any) -> Dict[str, Any]:
    if isinstance(block, dict):
        return block
    result = {}
    for key in ("type", "text", "id", "name", "input", "citations"):
        if hasattr(block, key):
            result[key] = getattr(block, key)
    if not result and hasattr(block, "__dict__"):
        result = {k: v for k, v in vars(block).items() if not k.startswith("_")}
        if hasattr(block, "type"):
            result["type"] = getattr(block, "type")
    return result


def normalize_content_list(content: Any) -> List[Dict[str, Any]]:
    try:
        return [block_to_dict(item) for item in (content or [])]
    except Exception:
        return []


def extract_text_blocks(content: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text") or ""
            if text:
                texts.append(text)
    return texts


def extract_response_content(response: Any) -> Tuple[List[Dict[str, Any]], Any]:
    if isinstance(response, dict):
        content = response.get("content", [])
        stop_reason = response.get("stop_reason")
        return content, stop_reason
    content = getattr(response, "content", None)
    stop_reason = getattr(response, "stop_reason", None)
    if content is None:
        content = [{"type": "text", "text": str(response)}]
    return content, stop_reason


def summarize_tool_use(block: Dict[str, Any]) -> str:
    name = block.get("name") or "unknown"
    tool_input = block.get("input")
    if tool_input is None:
        return name
    try:
        payload = json.dumps(tool_input, ensure_ascii=False)
    except Exception:
        payload = str(tool_input)
    return f"{name}: {payload}"
