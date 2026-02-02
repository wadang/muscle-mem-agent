from typing import Optional, Tuple, Dict, Any

from muscle_mem.core.mllm import LMMAgent


def _normalize_engine_type(model_provider: str, model_url: Optional[str]) -> str:
    if model_provider == "anthropic" and model_url and "openrouter.ai" in model_url:
        return "anthropic"
    return model_provider


def _normalize_model_name(engine_type: str, model: str) -> str:
    if engine_type == "anthropic" and model.startswith("anthropic/"):
        return model.split("/", 1)[1]
    return model


def test_model_call(
    model_provider: str,
    model_url: Optional[str],
    model_api_key: Optional[str],
    model: str,
    model_temperature: Optional[float] = None,
    prompt: str = "Reply with 'pong' to confirm you can answer.",
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 64,
) -> Tuple[str, Dict[str, Any]]:
    """Run a minimal smoke test to verify a model call succeeds.

    Returns the model response and the resolved engine params.
    """
    engine_type = _normalize_engine_type(model_provider, model_url)
    model_name = _normalize_model_name(engine_type, model)

    print(
        f"Testing model call with engine_type: {engine_type}, model_name: {model_name}"
    )

    engine_params = {
        "engine_type": engine_type,
        "model": model_name,
        "base_url": model_url,
        "api_key": model_api_key,
        "temperature": model_temperature,
    }
    agent = LMMAgent(engine_params=engine_params, system_prompt=system_prompt)
    response = agent.get_response(
        user_message=prompt,
        temperature=model_temperature if model_temperature is not None else 0.0,
        max_new_tokens=max_new_tokens,
    )
    return response, engine_params


response, params = test_model_call(
    model_provider="anthropic",
    model_url="https://openrouter.ai/api",
    model_api_key="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",
    model="anthropic/claude-opus-4.5",
    model_temperature=1.0,
)
print("engine_params:", params)
print("response:", response)
