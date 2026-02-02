import inspect
from dataclasses import dataclass
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


def tool_action(func):
    func.is_tool_action = True
    return func


def _type_to_schema(annotation: Any) -> Dict[str, Any]:
    if annotation in (inspect._empty, Any):
        return {"type": "string"}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (list, List):
        item_schema = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Dict):
        value_schema = _type_to_schema(args[1]) if len(args) > 1 else {"type": "string"}
        return {"type": "object", "additionalProperties": value_schema}
    if origin in (tuple, Tuple):
        item_schema = _type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if origin is Union:
        non_null = [arg for arg in args if arg is not type(None)]
        null_allowed = len(non_null) != len(args)
        if len(non_null) == 1:
            schema = _type_to_schema(non_null[0])
            if "type" in schema:
                schema_type = schema["type"]
                if isinstance(schema_type, list):
                    schema["type"] = schema_type + ["null"]
                else:
                    schema["type"] = [schema_type, "null"]
            else:
                schema = {"anyOf": [schema, {"type": "null"}]}
            return schema
        if null_allowed:
            return {
                "anyOf": [_type_to_schema(arg) for arg in non_null] + [{"type": "null"}]
            }
        return {"anyOf": [_type_to_schema(arg) for arg in non_null]}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    return {"type": "string"}


def _describe_action(func: Callable[..., Any], name: str) -> str:
    if not func.__doc__:
        return name
    return func.__doc__.strip()


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    if not docstring:
        return {}
    lines = docstring.splitlines()
    header_indices = [
        idx
        for idx, line in enumerate(lines)
        if line.strip() in {"Args:", "Arguments:", "Parameters:"}
    ]
    if not header_indices:
        return {}
    start = header_indices[0] + 1
    params: Dict[str, str] = {}
    current_key = None
    current_desc: List[str] = []

    def flush_current():
        nonlocal current_key, current_desc
        if current_key:
            params[current_key] = " ".join(current_desc).strip()
        current_key = None
        current_desc = []

    for line in lines[start:]:
        if not line.strip():
            flush_current()
            continue
        if line.lstrip() == line:
            break
        bullet_match = re.match(r"^\s*-\s*([A-Za-z_]\w*)\b[^:]*:\s*(.+)$", line)
        arg_match = re.match(r"^\s*([A-Za-z_]\w*)\s*:\s*(.+)$", line)
        match = bullet_match or arg_match
        if match:
            flush_current()
            current_key = match.group(1)
            current_desc = [match.group(2).strip()]
        else:
            if current_key:
                current_desc.append(line.strip())
    flush_current()
    return params


def _build_tool_spec(func: Callable[..., Any], name: str) -> Dict[str, Any]:
    description = _describe_action(func, name)
    override_schema = getattr(func, "tool_input_schema", None)
    if override_schema is not None:
        return {
            "name": name,
            "description": description,
            "input_schema": override_schema,
        }

    try:
        type_hints = get_type_hints(func, include_extras=True)
    except TypeError:
        type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    param_descriptions = []
    # = _parse_docstring_params(
    #     inspect.getdoc(func) or ""
    # )
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        annotation = type_hints.get(param_name, param.annotation)
        schema = _type_to_schema(annotation)
        if param.default is not inspect._empty:
            schema["default"] = param.default
        else:
            is_optional = (
                isinstance(schema.get("type"), list) and "null" in schema["type"]
            )
            if not is_optional:
                required.append(param_name)
        if param_name in param_descriptions:
            schema["description"] = param_descriptions[param_name]
        properties[param_name] = schema

    input_schema = {"type": "object", "properties": properties}
    if required:
        input_schema["required"] = required

    return {"name": name, "description": description, "input_schema": input_schema}


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]

    def to_anthropic(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: List[ToolSpec] = []
        self._tool_map: Dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        if name in self._tool_map:
            raise ValueError(f"Tool '{name}' already registered")
        spec = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )
        self._tools.append(spec)
        self._tool_map[name] = spec

    def register_action_provider(self, provider: Any) -> None:
        provider_type = type(provider)
        for name, func in inspect.getmembers(provider_type, predicate=callable):
            if not getattr(func, "is_tool_action", False):
                continue
            spec = _build_tool_spec(func, name)
            handler = getattr(provider, name)
            self.register(
                name=spec["name"],
                description=spec["description"],
                input_schema=spec["input_schema"],
                handler=handler,
            )

    def build_tools(
        self, allow: Optional[List[str]] = None, deny: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        allow_set = None
        if allow and allow != ["*"] and allow != "*":
            allow_set = set(allow)
        deny_set = set(deny or [])
        tools: List[Dict[str, Any]] = []
        for spec in self._tools:
            if allow_set is not None and spec.name not in allow_set:
                continue
            if spec.name in deny_set:
                continue
            tools.append(spec.to_anthropic())
        return tools

    def dispatch(self, name: str, tool_input: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self._tool_map:
            raise ValueError(f"Unknown tool '{name}'")
        spec = self._tool_map[name]
        if tool_input is None:
            tool_input = {}
        if not isinstance(tool_input, dict):
            raise ValueError("Tool input must be an object")
        cleaned_input = {k: v for k, v in tool_input.items() if v is not None}
        return spec.handler(**cleaned_input)
