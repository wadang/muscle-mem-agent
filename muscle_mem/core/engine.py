import os
from typing import Any, Dict

import backoff
from anthropic import Anthropic
import httpx
from openai import (
    AzureOpenAI,
    APIConnectionError,
    APIError,
    AzureOpenAI,
    OpenAI,
    RateLimitError,
)


class LMMEngine:
    pass


class LMMEngineOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        organization=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.organization = organization
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature  # Can force temperature to be the same (in the case of o3 requiring temperature to be 1)

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        organization = self.organization or os.getenv("OPENAI_ORG_ID")
        if not self.llm_client:
            if not self.base_url:
                self.llm_client = OpenAI(api_key=api_key, organization=organization)
            else:
                self.llm_client = OpenAI(
                    base_url=self.base_url, api_key=api_key, organization=organization
                )
        completion_message = (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                # max_completion_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=(
                    temperature if self.temperature is None else self.temperature
                ),
                **kwargs,
            )
            .choices[0]
            .message
        )
        return completion_message.content

    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENAI_API_KEY"
            )
        organization = self.organization or os.getenv("OPENAI_ORG_ID")
        if not self.llm_client:
            if not self.base_url:
                self.llm_client = OpenAI(api_key=api_key, organization=organization)
            else:
                self.llm_client = OpenAI(
                    base_url=self.base_url, api_key=api_key, organization=organization
                )
        completion_message = (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=(
                    temperature if self.temperature is None else self.temperature
                ),
                **kwargs,
            )
            .choices[0]
            .message
        )
        thinking = getattr(completion_message, "reasoning_content", None)
        return completion_message.content, thinking


class LMMEngineAnthropic(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        thinking=False,
        temperature=None,
        prompt_caching=True,
        prompt_cache_ttl=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None
        self.temperature = temperature
        self.prompt_caching = prompt_caching
        self.prompt_cache_ttl = prompt_cache_ttl

    def _normalize_tool_response(self, response):
        content_blocks = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block_type == "thinking":
                content_blocks.append({"type": "thinking", "thinking": block.thinking})
            else:
                content_blocks.append(
                    {"type": block_type or "unknown", "text": str(block)}
                )
        return {
            "content": content_blocks,
            "stop_reason": getattr(response, "stop_reason", None),
            "model": getattr(response, "model", None),
        }

    def _build_cache_control(self):
        cache_control = {"type": "ephemeral"}
        if self.prompt_cache_ttl:
            cache_control["ttl"] = self.prompt_cache_ttl
        return cache_control

    def _apply_prompt_caching(self, messages, tools):
        if not self.prompt_caching:
            return messages[0]["content"][0]["text"], messages[1:], tools, None

        system_content = messages[0].get("content", [])
        system_blocks = []
        if isinstance(system_content, list):
            for block in system_content:
                block_copy = dict(block)
                block_copy.setdefault("cache_control", self._build_cache_control())
                system_blocks.append(block_copy)
        else:
            system_blocks = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": self._build_cache_control(),
                }
            ]

        updated_messages = messages[1:]

        updated_tools = None
        if tools:
            updated_tools = [dict(tool) for tool in tools]
            updated_tool = dict(updated_tools[-1])
            updated_tool.setdefault("cache_control", self._build_cache_control())
            updated_tools[-1] = updated_tool

        extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        return system_blocks, updated_messages, updated_tools, extra_headers

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )

        print(f"Using Anthropic base_url: {self.base_url}, model: {self.model}")
        self.llm_client = Anthropic(
            base_url=self.base_url, api_key="", auth_token=api_key
        )
        # Use the instance temperature if not specified in the call
        temp = self.temperature if temperature is None else temperature
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=system,
                model=self.model,
                messages=message_payload,
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **request_kwargs,
            )
            if request_kwargs.get("tools"):
                return self._normalize_tool_response(full_response)
            return full_response.content[1].text
        response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **request_kwargs,
        )
        if request_kwargs.get("tools"):
            return self._normalize_tool_response(response)
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    # Compatible with Claude-3.7 Sonnet thinking mode
    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        """Generate the next message based on previous messages, and keeps the thinking tokens"""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        self.llm_client = Anthropic(api_key=api_key)
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        full_response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
            **request_kwargs,
        )

        if request_kwargs.get("tools"):
            return self._normalize_tool_response(full_response)

        thoughts = full_response.content[0].thinking
        answer = full_response.content[1].text
        full_response = (
            f"<thoughts>\n{thoughts}\n</thoughts>\n\n<answer>\n{answer}\n</answer>\n"
        )
        return full_response


class LMMEngineAnthropicLR(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        thinking=False,
        temperature=None,
        prompt_caching=False,
        prompt_cache_ttl=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.api_key = api_key
        self.llm_client = None
        self.temperature = temperature
        self.prompt_caching = prompt_caching
        self.prompt_cache_ttl = prompt_cache_ttl

    def _normalize_tool_response(self, response):
        content_blocks = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif block_type == "thinking":
                content_blocks.append({"type": "thinking", "thinking": block.thinking})
            else:
                content_blocks.append(
                    {"type": block_type or "unknown", "text": str(block)}
                )
        return {
            "content": content_blocks,
            "stop_reason": getattr(response, "stop_reason", None),
            "model": getattr(response, "model", None),
        }

    def _build_cache_control(self):
        cache_control = {"type": "ephemeral"}
        if self.prompt_cache_ttl:
            cache_control["ttl"] = self.prompt_cache_ttl
        return cache_control

    def _apply_prompt_caching(self, messages, tools):
        if not self.prompt_caching:
            return messages[0]["content"][0]["text"], messages[1:], tools, None

        system_content = messages[0].get("content", [])
        system_blocks = []
        if isinstance(system_content, list):
            for block in system_content:
                block_copy = dict(block)
                block_copy.setdefault("cache_control", self._build_cache_control())
                system_blocks.append(block_copy)
        else:
            system_blocks = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": self._build_cache_control(),
                }
            ]

        updated_messages = messages[1:]

        updated_tools = None
        if tools:
            updated_tools = [dict(tool) for tool in tools]
            updated_tool = dict(updated_tools[-1])
            updated_tool.setdefault("cache_control", self._build_cache_control())
            updated_tools[-1] = updated_tool

        extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
        return system_blocks, updated_messages, updated_tools, extra_headers

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )

        print(f"Using Anthropic base_url: {self.base_url}, model: {self.model}")

        disable_ssl = True
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": self.base_url}
        if disable_ssl:
            client_kwargs["http_client"] = httpx.Client(verify=False)
        self.llm_client = Anthropic(**client_kwargs)

        # Use the instance temperature if not specified in the call
        temp = self.temperature if temperature is None else temperature
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        if self.thinking:
            full_response = self.llm_client.messages.create(
                system=system,
                model=self.model,
                messages=message_payload,
                max_tokens=8192,
                thinking={"type": "enabled", "budget_tokens": 4096},
                **request_kwargs,
            )
            if request_kwargs.get("tools"):
                return self._normalize_tool_response(full_response)
            return full_response.content[1].text
        response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **request_kwargs,
        )
        if request_kwargs.get("tools"):
            return self._normalize_tool_response(response)
        return response.content[0].text

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    # Compatible with Claude-3.7 Sonnet thinking mode
    def generate_with_thinking(
        self, messages, temperature=0.0, max_new_tokens=None, **kwargs
    ):
        """Generate the next message based on previous messages, and keeps the thinking tokens"""
        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named ANTHROPIC_API_KEY"
            )
        self.llm_client = Anthropic(api_key=api_key)
        request_kwargs = dict(kwargs)
        tools = request_kwargs.get("tools")
        system = messages[0]["content"][0]["text"]
        message_payload = messages[1:]
        if self.prompt_caching:
            system, messages, cached_tools, extra_headers = self._apply_prompt_caching(
                messages, tools
            )
            if cached_tools is not None:
                request_kwargs["tools"] = cached_tools
            if extra_headers:
                existing_headers = request_kwargs.get("extra_headers")
                if isinstance(existing_headers, dict):
                    existing_headers.update(extra_headers)
                else:
                    request_kwargs["extra_headers"] = dict(extra_headers)
            message_payload = messages
        full_response = self.llm_client.messages.create(
            system=system,
            model=self.model,
            messages=message_payload,
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096},
            **request_kwargs,
        )

        if request_kwargs.get("tools"):
            return self._normalize_tool_response(full_response)

        thoughts = full_response.content[0].thinking
        answer = full_response.content[1].text
        full_response = (
            f"<thoughts>\n{thoughts}\n</thoughts>\n\n<answer>\n{answer}\n</answer>\n"
        )
        return full_response


class LMMEngineGemini(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named GEMINI_API_KEY"
            )
        base_url = self.base_url or os.getenv("GEMINI_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named GEMINI_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use the temperature passed to generate, otherwise use the instance's temperature, otherwise default to 0.0
        temp = self.temperature if temperature is None else temperature
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temp,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineOpenRouter(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named OPENROUTER_API_KEY"
            )
        base_url = self.base_url or os.getenv("OPEN_ROUTER_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named OPEN_ROUTER_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temp,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineAzureOpenAI(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        azure_endpoint=None,
        model=None,
        api_version=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_version = api_version
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.cost = 0.0
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "An API Key needs to be provided in either the api_key parameter or as an environment variable named AZURE_OPENAI_API_KEY"
            )
        api_version = self.api_version or os.getenv("OPENAI_API_VERSION")
        if api_version is None:
            raise ValueError(
                "api_version must be provided either as a parameter or as an environment variable named OPENAI_API_VERSION"
            )
        azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint is None:
            raise ValueError(
                "An Azure API endpoint needs to be provided in either the azure_endpoint parameter or as an environment variable named AZURE_OPENAI_ENDPOINT"
            )
        if not self.llm_client:
            self.llm_client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            **kwargs,
        )
        total_tokens = completion.usage.total_tokens
        self.cost += 0.02 * ((total_tokens + 500) / 1000)
        return completion.choices[0].message.content


class LMMEnginevLLM(LMMEngine):
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
        rate_limit=-1,
        temperature=None,
        **kwargs,
    ):
        assert model is not None, "model must be provided"
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None
        self.temperature = temperature

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(
        self,
        messages,
        temperature=0.0,
        top_p=0.8,
        repetition_penalty=1.05,
        max_new_tokens=512,
        **kwargs,
    ):
        api_key = self.api_key or os.getenv("vLLM_API_KEY")
        if api_key is None:
            raise ValueError(
                "A vLLM API key needs to be provided in either the api_key parameter or as an environment variable named vLLM_API_KEY"
            )
        base_url = self.base_url or os.getenv("vLLM_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "An endpoint URL needs to be provided in either the endpoint_url parameter or as an environment variable named vLLM_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        # Use self.temperature if set, otherwise use the temperature argument
        temp = self.temperature if self.temperature is not None else temperature
        completion = self.llm_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens if max_new_tokens else 4096,
            temperature=temp,
            top_p=top_p,
            extra_body={"repetition_penalty": repetition_penalty},
        )
        return completion.choices[0].message.content


class LMMEngineHuggingFace(LMMEngine):
    def __init__(self, base_url=None, api_key=None, rate_limit=-1, **kwargs):
        self.base_url = base_url
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("HF_TOKEN")
        if api_key is None:
            raise ValueError(
                "A HuggingFace token needs to be provided in either the api_key parameter or as an environment variable named HF_TOKEN"
            )
        base_url = self.base_url or os.getenv("HF_ENDPOINT_URL")
        if base_url is None:
            raise ValueError(
                "HuggingFace endpoint must be provided as base_url parameter or as an environment variable named HF_ENDPOINT_URL."
            )
        if not self.llm_client:
            self.llm_client = OpenAI(base_url=base_url, api_key=api_key)
        return (
            self.llm_client.chat.completions.create(
                model="tgi",
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )


class LMMEngineParasail(LMMEngine):
    def __init__(
        self, base_url=None, api_key=None, model=None, rate_limit=-1, **kwargs
    ):
        assert model is not None, "Parasail model id must be provided"
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.llm_client = None

    @backoff.on_exception(
        backoff.expo, (APIConnectionError, APIError, RateLimitError), max_time=60
    )
    def generate(self, messages, temperature=0.0, max_new_tokens=None, **kwargs):
        api_key = self.api_key or os.getenv("PARASAIL_API_KEY")
        if api_key is None:
            raise ValueError(
                "A Parasail API key needs to be provided in either the api_key parameter or as an environment variable named PARASAIL_API_KEY"
            )
        base_url = self.base_url
        if base_url is None:
            raise ValueError(
                "Parasail endpoint must be provided as base_url parameter or as an environment variable named PARASAIL_ENDPOINT_URL"
            )
        if not self.llm_client:
            self.llm_client = OpenAI(
                base_url=base_url if base_url else "https://api.parasail.io/v1",
                api_key=api_key,
            )
        return (
            self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_new_tokens if max_new_tokens else 4096,
                temperature=temperature,
                **kwargs,
            )
            .choices[0]
            .message.content
        )
