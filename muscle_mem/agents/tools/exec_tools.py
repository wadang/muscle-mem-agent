import json
import logging
import os
import re
import shlex
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Any, Dict, List, Optional

from muscle_mem.agents.tools.registry import tool_action
from muscle_mem.core.mllm import LMMAgent
from muscle_mem.utils.common_utils import call_llm_safe, parse_code_from_string


MAX_TOOL_RESULT_CHARS = 100_000
DEBUG_PAYLOAD_LIMIT = 4000
logger = logging.getLogger("desktopenv.agent")
TAVILY_API_URL = os.environ.get("TAVILY_API_URL", "https://api.tavily.com/search")
TAVILY_DEFAULT_MAX_RESULTS = 5
TAVILY_MAX_RESULTS_LIMIT = 15
JINA_API_URL = os.environ.get("JINA_API_URL", "https://r.jina.ai/")


def _load_api_key(env_name: str) -> str:
    value = str(os.environ.get(env_name, "") or "").strip()
    if value.startswith("your_default_") and value.endswith("_api_key_here"):
        return ""
    return value


def _load_tavily_timeout() -> float:
    raw = os.environ.get("TAVILY_TIMEOUT", "20")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 20.0


TAVILY_TIMEOUT = _load_tavily_timeout()


def _load_jina_timeout() -> float:
    raw = os.environ.get("JINA_TIMEOUT", "20")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 20.0


JINA_TIMEOUT = _load_jina_timeout()


def clamp_text(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    remaining = len(text) - limit
    return text[:limit] + f"\n\n...<truncated {remaining} chars>"


def debug_tool_io(tool_name: str, stage: str, payload: Any) -> None:
    try:
        body = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        body = str(payload)
    if len(body) > DEBUG_PAYLOAD_LIMIT:
        body = body[:DEBUG_PAYLOAD_LIMIT] + "\n...<truncated>"
    print(f"[{tool_name}] {stage}:\n{body}")


def format_controller_result(result: Dict[str, Any], tool_name: str) -> str:
    if not result:
        return f"{tool_name} result: (no output)"
    status = result.get("status", "unknown")
    return_code = result.get("returncode", result.get("return_code", ""))
    output = result.get("output", "")
    error = result.get("error", "")
    lines = [f"{tool_name} result:", f"Status: {status}"]
    if return_code != "":
        lines.append(f"Return Code: {return_code}")
    if output:
        lines.append(f"Output:\n{output}")
    if error:
        lines.append(f"Error:\n{error}")
    return clamp_text("\n".join(lines))


def _tavily_max_results(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return TAVILY_DEFAULT_MAX_RESULTS
    return max(1, min(TAVILY_MAX_RESULTS_LIMIT, parsed))


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def _extract_author_id(author: Dict[str, Any]) -> Optional[str]:
    for key in ("author_id", "scholar_id", "scholarid", "id"):
        value = author.get(key)
        if value:
            return str(value)
    return None


def _summarize_author(author: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "name",
        "affiliation",
        "email_domain",
        "interests",
        "citedby",
        "citedby5y",
        "hindex",
        "hindex5y",
        "i10index",
        "i10index5y",
        "url_picture",
        "homepage",
    )
    summary: Dict[str, Any] = {}
    author_id = _extract_author_id(author)
    if author_id:
        summary["author_id"] = author_id
    for field in fields:
        if field in author and author[field] is not None:
            summary[field] = author[field]
    return summary


def _summarize_publication(publication: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    bib = publication.get("bib")
    if isinstance(bib, dict):
        for field in (
            "title",
            "author",
            "year",
            "pub_year",
            "venue",
            "journal",
            "publisher",
        ):
            if field in bib and bib[field] is not None:
                summary[field] = bib[field]
        if "abstract" in bib and bib["abstract"] is not None:
            summary["abstract"] = bib["abstract"]

    for field in (
        "author_pub_id",
        "num_citations",
        "pub_url",
        "citation_url",
        "citedby_url",
        "eprint_url",
        "url_scholarbib",
    ):
        if field in publication and publication[field] is not None:
            summary[field] = publication[field]
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _match_publication_id(publication: Dict[str, Any], publication_id: str) -> bool:
    if not publication_id:
        return False
    for key in ("author_pub_id", "pub_id", "publication_id", "id"):
        value = publication.get(key)
        if value and str(value) == publication_id:
            return True
    for key in (
        "pub_url",
        "citation_url",
        "citedby_url",
        "eprint_url",
        "url_scholarbib",
    ):
        value = publication.get(key)
        if value and publication_id in str(value):
            return True
    return False


def _normalize_publication_id(author_id: str, publication_id: str) -> str:
    """Ensure publication_id includes the author prefix if it was omitted."""
    author_part = str(author_id or "").strip()
    publication_part = str(publication_id or "").strip()
    if not publication_part or ":" in publication_part or not author_part:
        return publication_part
    return f"{author_part}:{publication_part}"


def _fetch_html(url: str, timeout: float) -> str:
    req = urllib_request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; motor-mem-agent/0.1)",
            "Accept": "text/html,application/xhtml+xml",
        },
        method="GET",
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def _extract_html_by_class(html: str, class_name: str) -> str:
    # print(html)

    if not html or not class_name:
        return ""
    patterns = [
        rf'<div[^>]*class="[^"]*\b{re.escape(class_name)}\b[^"]*"[^>]*>(.*?)</div>',
        rf"<div[^>]*class='[^']*\b{re.escape(class_name)}\b[^']*'[^>]*>(.*?)</div>",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def _extract_json_like(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return None


def _parse_json_response(response: str) -> Optional[Any]:
    if response is None:
        return None
    response_text = str(response).strip()
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    fenced = parse_code_from_string(response_text)
    if fenced:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass
    candidate = _extract_json_like(response_text)
    if candidate and candidate != response_text and candidate != fenced:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None


class ExecutionToolProvider:
    def __init__(
        self, env_controller=None, engine_params: Optional[Dict[str, Any]] = None
    ) -> None:
        self.env_controller = env_controller
        self.engine_params = dict(engine_params or {})

    def set_env_controller(self, env_controller) -> None:
        self.env_controller = env_controller

    def set_engine_params(self, engine_params: Optional[Dict[str, Any]]) -> None:
        self.engine_params = dict(engine_params or {})

    @tool_action
    def bash(self, command: str, timeout_sec, description: Optional[str] = None):
        """/
        在项目环境中执行一次性的 bash 命令，支持可选的超时设置，并确保执行过程的安全与正确处理。
        **注意**：每次执行都是独立进程，不会保留上一次的环境变量、工作目录或会话状态。

        执行命令前，请遵循以下步骤：

        1. 目录验证：
        - 如果命令将创建新目录或文件，先使用 `ls` 验证父目录存在且位置正确
        - 例如，执行 "mkdir foo/bar" 之前，先用 `ls foo` 检查 "foo" 是否存在且是预期的父目录

        2. 命令执行：
        - 包含空格的文件路径必须用双引号括起来（如 cd "path with spaces/file.txt"）
        - 正确引用示例：
            - cd "/Users/name/My Documents"（正确）
            - cd /Users/name/My Documents（错误 - 会失败）
            - python3 "/path/with spaces/script.py"（正确）
            - python3 /path/with spaces/script.py（错误 - 会失败）
        - 确保正确引用后，执行命令
        - 捕获命令的输出

        使用说明：
        - command 参数是必需的。
        - 可以指定可选的超时时间(秒)，最长300秒/5分钟;如未指定，命令将在30秒后超时。
        - **强烈建议**你对执行的命令编写清晰、简练的描述（description）。对于简单命令，保持简短（5-10 个字）；对于复杂命令（如管道命令、生僻参数或一眼难以看懂的命令），请补充足够的上下文以阐明其作用。
        - 如果输出超过1000000个字符，返回前会被截断。
        """
        if self.env_controller is None:
            raise ValueError("env_controller is required for bash execution")
        command = str(command or "")
        if not command:
            raise ValueError("missing bash.command")
        # if any(token in command for token in ["rm -rf /", "shutdown", "reboot", "sudo "]):
        #     raise ValueError("blocked dangerous command")
        timeout_sec = float(timeout_sec or 30)
        timeout_sec = min(max(timeout_sec, 30.0), 300.0)
        description = str(description or "").strip()
        debug_payload = {"command": command, "timeout_sec": timeout_sec}
        if description:
            debug_payload["description"] = description
        debug_tool_io("run_bash", "input", debug_payload)
        quoted_command = shlex.quote(command)
        script_payload = (
            f"cmd={quoted_command}\n"
            "printf '%s\\n' \"$cmd\" >> ~/.bash_history\n"
            f"{command}"
        )
        result = self.env_controller.run_bash_script(
            script_payload, timeout=timeout_sec
        )
        debug_tool_io("run_bash", "output", result)
        return format_controller_result(result, "bash")

    # @tool_action
    def python(self, code: str):
        """Execute a Python script inside the project workspace."""
        if self.env_controller is None:
            raise ValueError("env_controller is required for python execution")
        code = str(code or "")
        if not code:
            raise ValueError("missing python.code")

        debug_tool_io("run_python", "input", {"code": code})
        result = self.env_controller.run_python_script(code)
        debug_tool_io("run_python", "output", result)
        return format_controller_result(result, "python")

    @tool_action
    def web_search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = TAVILY_DEFAULT_MAX_RESULTS,
        include_images: bool = False,
        include_answer: bool = True,
    ):
        """Search the public web for up-to-date information."""
        tavily_api_key = _load_api_key("TAVILY_API_KEY")
        if not tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY not set")

        query = str(query or "").strip()
        if not query:
            raise ValueError("web_search.query cannot be empty")

        search_depth = str(search_depth or "basic").lower()
        if search_depth not in {"basic", "advanced"}:
            search_depth = "basic"

        include_images = bool(include_images)
        include_answer = True if include_answer is None else bool(include_answer)
        max_results = _tavily_max_results(max_results)

        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_images": include_images,
            "include_answer": include_answer,
            "include_raw_content": False,
        }

        request_body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            TAVILY_API_URL,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=TAVILY_TIMEOUT) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Web search failed ({error.code}): {detail[:400]}"
            ) from error
        except urllib_error.URLError as error:
            reason = getattr(error, "reason", None) or str(error)
            raise RuntimeError(f"Web search connection error: {reason}") from error

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RuntimeError("Web search returned invalid JSON") from error

        lines: List[str] = []
        answer = str(data.get("answer") or "").strip()
        if answer:
            lines.append(f"Answer: {answer}")

        results = data.get("results") or []
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or f"Result {idx}")
            url = str(item.get("url") or "").strip()
            snippet = str(
                item.get("content")
                or item.get("snippet")
                or item.get("raw_content")
                or ""
            ).strip()
            result_lines = [f"[{idx}] {title}"]
            if url:
                result_lines.append(url)
            if snippet:
                result_lines.append(snippet)
            lines.append("\n".join(result_lines))

        if not lines:
            lines.append("No Tavily results were returned.")

        return clamp_text("\n\n".join(lines))

    @tool_action
    def scholarly_author(
        self,
        author_id: Optional[str] = None,
        fill_author: bool = True,
        max_publications: int = 5,
        fill_publications: bool = False,
    ):
        """Lookup a Google Scholar author by author_id and return profile details/publications."""
        from scholarly import scholarly

        author_id = str(author_id or "").strip()
        if not author_id:
            raise ValueError("scholarly_author requires author_id")

        max_publications = _clamp_int(max_publications, 5, 1, 50)

        selected_author = None

        try:
            selected_author = scholarly.search_author_id(author_id)
        except Exception as exc:
            return clamp_text(
                json.dumps(
                    {"error": f"search_author_id failed: {exc}"},
                    ensure_ascii=False,
                    indent=2,
                )
            )

        if selected_author is None:
            return "No author results found."

        result: Dict[str, Any] = {
            "author_id": _extract_author_id(selected_author),
        }

        if fill_author:
            try:
                selected_author = scholarly.fill(selected_author)
            except Exception as exc:
                result["error"] = f"fill author failed: {exc}"
                return clamp_text(
                    json.dumps(_json_safe(result), ensure_ascii=False, indent=2)
                )

        if isinstance(selected_author, dict):
            result["author"] = _summarize_author(selected_author)
            publications_raw = selected_author.get("publications") or []
        else:
            publications_raw = []

        publications: List[Dict[str, Any]] = []
        if isinstance(publications_raw, list):
            for publication in publications_raw[:max_publications]:
                if not isinstance(publication, dict):
                    continue
                publication_data = publication
                if fill_publications:
                    try:
                        publication_data = scholarly.fill(publication)
                    except Exception as exc:
                        publications.append(
                            {
                                "error": f"fill publication failed: {exc}",
                                "publication": _summarize_publication(publication),
                            }
                        )
                        continue
                publications.append(_summarize_publication(publication_data))

        result["publication_count"] = (
            len(publications_raw) if isinstance(publications_raw, list) else 0
        )
        result["publications"] = publications

        return clamp_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))

    @tool_action
    def scholarly_publication(
        self,
        author_id: Optional[str] = None,
        publication_id: Optional[str] = None,
        fill_publication: bool = True,
    ):
        """Query Google Scholar for publication info, including PDF address, via author_id + publication_id."""
        from scholarly import scholarly

        author_id = str(author_id or "").strip()
        publication_id = str(publication_id or "").strip()
        if not author_id or not publication_id:
            raise ValueError(
                "scholarly_publication requires author_id and publication_id"
            )
        normalized_publication_id = _normalize_publication_id(author_id, publication_id)

        try:
            author = scholarly.search_author_id(author_id)
        except Exception as exc:
            return clamp_text(
                json.dumps(
                    {"error": f"search_author_id failed: {exc}"},
                    ensure_ascii=False,
                    indent=2,
                )
            )

        if author is None:
            return "No author results found."

        publications_raw = (
            author.get("publications") if isinstance(author, dict) else []
        )
        if not publications_raw:
            try:
                author = scholarly.fill(author)
            except Exception as exc:
                return clamp_text(
                    json.dumps(
                        {"error": f"fill author failed: {exc}"},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            publications_raw = (
                author.get("publications") if isinstance(author, dict) else []
            )

        if not isinstance(publications_raw, list):
            return "No publication results found."

        matched_publication = None
        for publication in publications_raw:
            if not isinstance(publication, dict):
                continue
            if _match_publication_id(publication, normalized_publication_id):
                matched_publication = publication
                break

        if matched_publication is None:
            return "No publication results found."

        if fill_publication:
            try:
                matched_publication = scholarly.fill(matched_publication)
            except Exception as exc:
                return clamp_text(
                    json.dumps(
                        {"error": f"fill publication failed: {exc}"},
                        ensure_ascii=False,
                        indent=2,
                    )
                )

        result = {
            "author_id": author_id,
            "publication_id": normalized_publication_id,
            "publication": _summarize_publication(matched_publication),
        }
        citation_id = normalized_publication_id
        citation_suffix = (
            citation_id.split(":", 1)[1] if ":" in citation_id else citation_id
        )
        if not citation_id.startswith(f"{author_id}:"):
            citation_id = f"{author_id}:{citation_suffix}"
        scholar_url = (
            "https://scholar.google.com/citations?view_op=view_citation"
            f"&hl=en&user={author_id}&citation_for_view={citation_id}"
        )
        try:
            html = _fetch_html(scholar_url, timeout=JINA_TIMEOUT)
            result["publication_html_url"] = scholar_url
            result["publication_pdf_url_info"] = _extract_html_by_class(
                html, "gsc_oci_title_ggi"
            )
            if not result["publication_title_html"]:
                result["publication_title_html_error"] = "gsc_oci_title_ggi not found"
        except Exception as exc:
            result["publication_title_html_error"] = str(exc)
        return clamp_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))

    @tool_action
    def web_fetch(self, url: str, fields: Optional[List[str]] = None) -> str:
        """Fetch a webpage and optionally extract fields from the markdown; use scholarly_author/scholarly_publication for Google Scholar instead."""
        jina_api_key = _load_api_key("JINA_API_KEY")
        if not jina_api_key:
            logger.error("web_fetch missing JINA_API_KEY")
            raise RuntimeError("JINA_API_KEY not set")

        url = str(url or "").strip()
        if not url:
            logger.error("web_fetch missing url")
            raise ValueError("web_fetch.url cannot be empty")
        if not (url.startswith("http://") or url.startswith("https://")):
            logger.error("web_fetch invalid url: %s", url)
            raise ValueError("web_fetch.url must start with http:// or https://")

        normalized_fields: List[str] = []
        if fields is not None:
            if isinstance(fields, (list, tuple)):
                normalized_fields = [
                    str(item).strip() for item in fields if str(item).strip()
                ]
            else:
                field_text = str(fields).strip()
                if field_text:
                    normalized_fields = [field_text]

        logger.info("web_fetch start url=%s fields=%s", url, normalized_fields or None)

        base_url = JINA_API_URL
        if not base_url.endswith("/"):
            base_url += "/"
        request_url = f"{base_url}{url}"
        headers = {
            "Authorization": f"Bearer {jina_api_key}",
            "User-Agent": "curl/8.1.2",
        }
        masked_headers = dict(headers)
        if "Authorization" in masked_headers:
            masked_headers["Authorization"] = "Bearer ****"
        logger.debug(
            "web_fetch request_url=%s timeout=%s headers=%s",
            request_url,
            JINA_TIMEOUT,
            masked_headers,
        )
        req = urllib_request.Request(
            request_url,
            headers=headers,
            method="GET",
        )

        try:
            with urllib_request.urlopen(req, timeout=JINA_TIMEOUT) as response:
                markdown = response.read().decode("utf-8")
        except urllib_error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")
            logger.error(
                "web_fetch http_error url=%s code=%s detail=%s",
                url,
                error.code,
                detail[:200],
            )
            raise RuntimeError(
                f"Web fetch failed ({error.code}): {detail[:400]}"
            ) from error
        except urllib_error.URLError as error:
            reason = getattr(error, "reason", None) or str(error)
            logger.error("web_fetch url_error url=%s reason=%s", url, reason)
            raise RuntimeError(f"Web fetch connection error: {reason}") from error
        except Exception as error:
            logger.error("web_fetch unexpected_error url=%s error=%s", url, error)
            raise

        if not normalized_fields:
            logger.info("web_fetch success url=%s bytes=%d", url, len(markdown))
            return clamp_text(markdown)

        if not self.engine_params:
            logger.error("web_fetch missing engine_params for url=%s", url)
            raise RuntimeError(
                "engine_params is required for web_fetch field extraction"
            )

        extract_prompt = (
            "Extract the requested fields from the markdown content.\n"
            "Return a JSON object with keys exactly matching the requested fields.\n"
            "If a field is missing, return an empty string for that field.\n"
            "Return only JSON with no extra text."
        )
        markdown_excerpt = markdown[:20000]
        user_payload = {
            "url": url,
            "fields": normalized_fields,
            "content_markdown": markdown_excerpt,
        }
        messages = [
            {"role": "system", "content": [{"type": "text", "text": extract_prompt}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": json.dumps(user_payload)}],
            },
        ]

        agent = LMMAgent(engine_params=self.engine_params)
        response = call_llm_safe(agent, messages=messages, temperature=0.0)
        extracted = _parse_json_response(response)
        if extracted is None:
            extracted = {"error": "invalid_json", "raw": str(response)[:2000]}
        logger.info("web_fetch extracted url=%s fields=%d", url, len(normalized_fields))
        return clamp_text(json.dumps(extracted, ensure_ascii=False, indent=2))


ExecutionToolProvider.web_search.tool_input_schema = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search keywords or question."},
        "search_depth": {
            "type": "string",
            "enum": ["basic", "advanced"],
            "default": "basic",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": TAVILY_MAX_RESULTS_LIMIT,
            "default": TAVILY_DEFAULT_MAX_RESULTS,
        },
        "include_images": {"type": "boolean", "default": False},
        "include_answer": {"type": "boolean", "default": True},
    },
    "required": ["query"],
    "additionalProperties": False,
}

ExecutionToolProvider.web_fetch.tool_input_schema = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "Target webpage URL."},
        "fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fields to extract from the webpage.",
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}

ExecutionToolProvider.scholarly_author.tool_input_schema = {
    "type": "object",
    "properties": {
        "author_id": {
            "type": "string",
            "description": "Google Scholar author id if known.",
        },
        "fill_author": {
            "type": "boolean",
            "default": True,
            "description": "Whether to fill the selected author profile.",
        },
        "max_publications": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 5,
        },
        "fill_publications": {
            "type": "boolean",
            "default": False,
            "description": "Whether to fill each publication entry.",
        },
    },
    "required": ["author_id"],
    "additionalProperties": False,
}

ExecutionToolProvider.scholarly_publication.tool_input_schema = {
    "type": "object",
    "properties": {
        "author_id": {
            "type": "string",
            "description": "Google Scholar author id.",
        },
        "publication_id": {
            "type": "string",
            "description": "Publication id from the author's publication list.",
        },
        "fill_publication": {
            "type": "boolean",
            "default": True,
            "description": "Whether to fill the publication details.",
        },
    },
    "required": ["author_id", "publication_id"],
    "additionalProperties": False,
}
