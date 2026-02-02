import json
import re
import time
from copy import deepcopy
from io import BytesIO
from PIL import Image

from typing import Tuple, Dict, Any, List, Optional

from muscle_mem.memory.procedural_memory import PROCEDURAL_MEMORY

import logging

logger = logging.getLogger("desktopenv.agent")


BASE64_PLACEHOLDER = "<base64-omitted>"
BASE64_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
)


def _looks_like_base64(value: str) -> bool:
    """Heuristic check to avoid logging massive base64 blobs."""

    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < 64:
        return False
    sample = stripped[: min(len(stripped), 256)]
    return all(ch in BASE64_CHARS for ch in sample)


def _strip_base64(value: Any, key: Optional[str] = None) -> Any:
    """Recursively replace base64 payloads with a placeholder."""

    if isinstance(value, str):
        if "base64," in value:
            prefix = value.split("base64,", 1)[0]
            return f"{prefix}base64,{BASE64_PLACEHOLDER}"
        if (key == "data" or key == "image_data") and _looks_like_base64(value):
            return BASE64_PLACEHOLDER
        return value
    if isinstance(value, list):
        return [_strip_base64(item) for item in value]
    if isinstance(value, dict):
        return {k: _strip_base64(v, key=k) for k, v in value.items()}
    return value


def sanitize_messages_for_logging(messages: List[Dict]) -> List[Dict]:
    """Create a sanitized deep copy suitable for logging."""

    if messages is None:
        return []

    sanitized = []
    for message in messages:
        message_copy = {"role": message.get("role"), "content": []}
        for block in message.get("content", []):
            block_type = block.get("type")
            if block_type == "image_url":
                sanitized_block = {
                    "type": "image_url",
                    "image_url": {
                        "detail": block.get("image_url", {}).get("detail"),
                        "url": BASE64_PLACEHOLDER,
                    },
                }
            elif block_type == "image":
                sanitized_block = {
                    "type": "image",
                    "source": {
                        "type": block.get("source", {}).get("type"),
                        "media_type": block.get("source", {}).get("media_type"),
                        "data": BASE64_PLACEHOLDER,
                    },
                }
            else:
                sanitized_block = _strip_base64(deepcopy(block))
            message_copy["content"].append(sanitized_block)
        sanitized.append(message_copy)
    return sanitized


def sanitize_text_for_logging(text: Any) -> str:
    """Sanitize text or structured payloads before logging."""

    if isinstance(text, (dict, list)):
        try:
            sanitized = _strip_base64(deepcopy(text))
            return json.dumps(sanitized, ensure_ascii=False, indent=2)
        except TypeError:
            return str(text)
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return _strip_base64(text)


def format_messages_for_logging(messages: List[Dict]) -> str:
    try:
        return json.dumps(messages, ensure_ascii=False, indent=2)
    except TypeError:
        return str(messages)


def _normalize_messages_for_llm(messages: Optional[List[Dict]]) -> List[Dict]:
    """Drop empty text blocks and skip empty messages to satisfy strict LLM APIs."""

    if not messages:
        return []

    cleaned_messages: List[Dict] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content", [])
        normalized_content: List[Dict] = []
        for block in content or []:
            if not isinstance(block, dict):
                normalized_content.append(block)
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if text is None:
                    continue
                if isinstance(text, str) and not text.strip():
                    continue
            normalized_content.append(block)
        if not normalized_content:
            if role == "system":
                normalized_content = [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant.",
                    }
                ]
            else:
                continue
        cleaned_messages.append({"role": role, "content": normalized_content})
    return cleaned_messages


def get_model_name(agent) -> str:
    engine = getattr(agent, "engine", None)
    if engine is None:
        return "unknown"
    return getattr(engine, "model", engine.__class__.__name__)


def create_pyautogui_code(agent, code: str, obs: Dict) -> str:
    """
    Attempts to evaluate the code into a pyautogui code snippet with grounded actions using the observation screenshot.

    Args:
        agent (ACI): The grounding agent to use for evaluation.
        code (str): The code string to evaluate.
        obs (Dict): The current observation containing the screenshot.

    Returns:
        exec_code (str): The pyautogui code to execute the grounded action.

    Raises:
        Exception: If there is an error in evaluating the code.
    """
    agent.assign_screenshot(obs)  # Necessary for grounding
    exec_code = eval(code)
    return exec_code


def call_llm_safe(
    agent, temperature: float = 0.0, use_thinking: bool = False, **kwargs
) -> str:
    # Retry if fails
    max_retries = 10  # Set the maximum number of retries
    attempt = 0
    response = ""
    model_name = get_model_name(agent)
    while attempt < max_retries:
        attempt_number = attempt + 1
        context_messages = kwargs.get("messages")
        if context_messages is None:
            context_messages = getattr(agent, "messages", [])
        normalized_messages = _normalize_messages_for_llm(context_messages)
        kwargs["messages"] = normalized_messages
        sanitized_messages = sanitize_messages_for_logging(normalized_messages)
        sanitized_kwargs = sanitize_text_for_logging(
            {k: v for k, v in kwargs.items() if k != "messages"}
        )
        # logger.info(
        #     "LLM_CALL_REQUEST model=%s attempt=%d/%d temperature=%s use_thinking=%s kwargs=%s ",
        #     model_name,
        #     attempt_number,
        #     max_retries,
        #     temperature,
        #     use_thinking,
        #     sanitized_kwargs,
        # )
        logger.info(
            "LLM_CALL_MESSAGES model=%s attempt=%d/%d:\n%s",
            model_name,
            attempt_number,
            max_retries,
            format_messages_for_logging(sanitized_messages),
        )
        try:
            response = agent.get_response(
                temperature=temperature, use_thinking=use_thinking, **kwargs
            )
            assert response is not None, "Response from agent should not be None"
            safe_response = sanitize_text_for_logging(response)
            response_len = len(response) if isinstance(response, str) else "n/a"
            logger.info(
                "LLM_CALL_RESPONSE model=%s attempt=%d/%d length=%s:\n%s",
                model_name,
                attempt_number,
                max_retries,
                response_len,
                safe_response,
            )
            print("Response success!")
            break  # If successful, break out of the loop
        except Exception as e:
            attempt += 1
            logger.error(
                "LLM_CALL_EXCEPTION model=%s attempt=%d/%d error=%s",
                model_name,
                attempt_number,
                max_retries,
                sanitize_text_for_logging(str(e)),
            )
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)
    return response if response is not None else ""


def call_llm_safe_with_thinking(
    agent, temperature: float = 0.0, **kwargs
) -> Tuple[str, Optional[str]]:
    max_retries = 10
    attempt = 0
    response = ""
    thinking = None
    model_name = get_model_name(agent)
    while attempt < max_retries:
        attempt_number = attempt + 1
        context_messages = kwargs.get("messages")
        if context_messages is None:
            context_messages = getattr(agent, "messages", [])
        sanitized_messages = sanitize_messages_for_logging(context_messages)
        sanitized_kwargs = sanitize_text_for_logging(
            {k: v for k, v in kwargs.items() if k != "messages"}
        )
        logger.info(
            "LLM_CALL_REQUEST model=%s attempt=%d/%d temperature=%s use_thinking=%s kwargs=%s ",
            model_name,
            attempt_number,
            max_retries,
            temperature,
            True,
            sanitized_kwargs,
        )
        logger.info(
            "LLM_CALL_MESSAGES model=%s attempt=%d/%d:\n%s",
            model_name,
            attempt_number,
            max_retries,
            format_messages_for_logging(sanitized_messages),
        )
        try:
            response, thinking = agent.get_response_with_thinking(
                temperature=temperature, **kwargs
            )
            safe_response = sanitize_text_for_logging(response)
            response_len = len(response) if isinstance(response, str) else "n/a"
            logger.info(
                "LLM_CALL_RESPONSE model=%s attempt=%d/%d length=%s:\n%s",
                model_name,
                attempt_number,
                max_retries,
                response_len,
                safe_response,
            )
            print("Response success!")
            break
        except Exception as e:
            attempt += 1
            logger.error(
                "LLM_CALL_EXCEPTION model=%s attempt=%d/%d error=%s",
                model_name,
                attempt_number,
                max_retries,
                sanitize_text_for_logging(str(e)),
            )
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("Max retries reached. Handling failure.")
        time.sleep(1.0)
    return (response if response is not None else ""), thinking


def call_llm_formatted(generator, format_checkers, **kwargs):
    """
    Calls the generator agent's LLM and ensures correct formatting.

    Args:
        generator (ACI): The generator agent to call.
        obs (Dict): The current observation containing the screenshot.
        format_checkers (Callable): Functions that take the response and return a tuple of (success, feedback).
        **kwargs: Additional keyword arguments for the LLM call.

    Returns:
        response (str): The formatted response from the generator agent.
    """
    max_retries = 3  # Set the maximum number of retries
    attempt = 0
    response = ""
    if kwargs.get("messages") is None:
        messages = (
            generator.messages.copy()
        )  # Copy messages to avoid modifying the original
    else:
        messages = kwargs["messages"]
        del kwargs["messages"]  # Remove messages from kwargs to avoid passing it twice
    model_name = get_model_name(generator)
    while attempt < max_retries:
        response = call_llm_safe(generator, messages=messages, **kwargs)

        # Prepare feedback messages for incorrect formatting
        feedback_msgs = []
        for format_checker in format_checkers:
            success, feedback = format_checker(response)
            if not success:
                feedback_msgs.append(feedback)
        if not feedback_msgs:
            # logger.info(f"Response formatted correctly on attempt {attempt} for {generator.engine.model}")
            break
        safe_response = sanitize_text_for_logging(response)
        logger.error(
            "LLM_FORMAT_ERROR model=%s attempt=%d/%d feedback=%s\nResponse:\n%s",
            model_name,
            attempt + 1,
            max_retries,
            "; ".join(feedback_msgs),
            safe_response,
        )
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }
        )
        logger.info(
            "LLM_FORMAT_BAD_RESPONSE attempt=%d/%d\n%s",
            attempt + 1,
            max_retries,
            safe_response,
        )
        delimiter = "\n- "
        formatting_feedback = f"- {delimiter.join(feedback_msgs)}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROCEDURAL_MEMORY.FORMATTING_FEEDBACK_PROMPT.replace(
                            "FORMATTING_FEEDBACK", formatting_feedback
                        ),
                    }
                ],
            }
        )
        logger.info("Feedback:\n%s", formatting_feedback)

        attempt += 1
        if attempt == max_retries:
            logger.error(
                "LLM_FORMAT_MAX_RETRIES_REACHED model=%s attempts=%d",
                model_name,
                max_retries,
            )
        time.sleep(1.0)
    return response


def split_thinking_response(full_response: str) -> Tuple[str, str]:
    try:
        # Extract thoughts section
        thoughts = full_response.split("<thoughts>")[-1].split("</thoughts>")[0].strip()

        # Extract answer section
        answer = full_response.split("<answer>")[-1].split("</answer>")[0].strip()

        return answer, thoughts
    except Exception as e:
        return full_response, ""


def parse_code_from_string(input_string):
    """Parses a string to extract each line of code enclosed in triple backticks (```)

    Args:
        input_string (str): The input string containing code snippets.

    Returns:
        str: The last code snippet found in the input string, or an empty string if no code is found.
    """
    input_string = input_string.strip()

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"

    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)
    if len(matches) == 0:
        # return []
        return ""
    relevant_code = matches[
        -1
    ]  # We only care about the last match given it is the grounded action
    return relevant_code


def extract_agent_functions(code):
    """Extracts all agent function calls from the given code.

    Args:
        code (str): The code string to search for agent function calls.

    Returns:
        list: A list of all agent function calls found in the code.
    """
    pattern = r"(agent\.\w+\(\s*.*\))"  # Matches
    return re.findall(pattern, code)


def compress_image(image_bytes: bytes = None, image: Image = None) -> bytes:
    """Compresses an image represented as bytes.

    Compression involves resizing image into half its original size and saving to webp format.

    Args:
        image_bytes (bytes): The image data to compress.

    Returns:
        bytes: The compressed image data.
    """
    if not image:
        image = Image.open(BytesIO(image_bytes))
    output = BytesIO()
    image.save(output, format="WEBP")
    compressed_image_bytes = output.getvalue()
    return compressed_image_bytes
