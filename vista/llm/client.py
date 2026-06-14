import asyncio
import json
import random

import litellm
from pydantic import BaseModel, ValidationError, create_model
from typing import Any, Dict

from vista.core.signature import Signature

import os
import time

from aiolimiter import AsyncLimiter

# Standard JSON mode — works on all providers (Groq, OpenRouter, OpenAI, etc.)
# Unlike passing a Pydantic model as response_format, this does NOT get
# translated into tool/function calls by litellm.
JSON_MODE = {"type": "json_object"}


class LLMClient:
    """Wrapper around litellm for structured completions."""

    _global_rate_limiter = None

    def __init__(self, model_name: str = "gpt-4o", api_base: str = None, rpm: float = None):
        self.model_name = model_name
        self.api_base = api_base

        # Configure the global rate limiter if specified or set in environment
        env_rpm = os.environ.get("VISTA_RPM")
        target_rpm = rpm
        if target_rpm is None and env_rpm:
            try:
                target_rpm = float(env_rpm)
            except ValueError:
                pass

        if target_rpm is not None and target_rpm > 0:
            if LLMClient._global_rate_limiter is None or LLMClient._global_rate_limiter.time_period != (60.0 / target_rpm):
                # Enforce a burst size of 1 with a delay between requests to prevent burst/concurrency limits.
                LLMClient._global_rate_limiter = AsyncLimiter(1, 60.0 / target_rpm)

    async def _call_llm(self, messages: list) -> str:
        """
        Core LLM call with unified error handling and exponential backoff for rate limits.
        Uses JSON mode for structured output — avoids litellm converting
        Pydantic models into tool calls on providers that don't support
        native JSON schema response_format (e.g., Groq).
        """

        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": JSON_MODE,
            "timeout": 90,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if LLMClient._global_rate_limiter:
                    await LLMClient._global_rate_limiter.acquire()
                response = await litellm.acompletion(**completion_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = (
                    "rate" in error_str
                    or "429" in error_str
                    or "resource_exhausted" in error_str
                    or "exhausted" in error_str
                    or "timeout" in error_str
                    or "time out" in error_str
                    or "500" in error_str
                    or "502" in error_str
                    or "503" in error_str
                    or "504" in error_str
                    or "internal" in error_str
                )
                if is_retryable and attempt < max_retries - 1:
                    wait_time = 15 * (2**attempt) + random.uniform(1, 5)
                    print(
                        f"LLM Client error/timeout occurred. Retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # Some providers return the generated text inside the error on schema violations
                    error_orig_str = str(e)
                    if "failed_generation" not in error_orig_str:
                        raise
                    try:
                        err_json = json.loads(error_orig_str[error_orig_str.find("{"):])
                        raw = err_json.get("error", {}).get("failed_generation")
                        if not raw:
                            raise e
                        return raw
                    except (json.JSONDecodeError, KeyError):
                        raise e

    def _inject_schema(
        self, messages: list, response_model: type[BaseModel]
    ) -> list:
        """Injects the JSON schema into the first system message so the LLM
        knows the exact output structure without relying on tool calls."""
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Explicitly list required top-level keys so the model can't invent its own
        required_keys = schema.get("required", list(schema.get("properties", {}).keys()))
        keys_hint = ", ".join(f'"{k}"' for k in required_keys)

        schema_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this exact schema:\n"
            f"```json\n{schema_str}\n```\n"
            f"Your response MUST be a JSON object with these top-level keys: {keys_hint}. "
            f"Do NOT add extra keys. Do NOT wrap in markdown. Output ONLY the JSON object."
        )

        messages = [m.copy() for m in messages]
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += schema_instruction
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})
        return messages

    async def predict(
        self, signature: Signature, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs the LLM based on the signature and parses structured output."""
        sys_prompt = signature.instructions or "You are a helpful AI assistant."
        sys_prompt += "\n\nGiven the following inputs, generate the requested outputs."

        user_content = ""
        for field in signature.inputs:
            val = inputs.get(field.name, "")
            user_content += f"--- {field.name.upper()} ---\n{val}\n\n"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        # Build a Pydantic model for the schema, but inject it into the
        # prompt instead of passing it as response_format
        output_fields = {f.name: (f.type_, ...) for f in signature.outputs}
        OutputModel = create_model("OutputModel", **output_fields)
        messages = self._inject_schema(messages, OutputModel)

        raw_output = await self._call_llm(messages)

        if raw_output is None:
            return {signature.outputs[0].name: ""}
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {signature.outputs[0].name: raw_output}

    async def generate_structured(
        self, messages: list, response_model: type[BaseModel], max_retries: int = 2
    ) -> BaseModel:
        """Generates a direct Pydantic model response with retry on validation failure."""
        messages = self._inject_schema(messages, response_model)

        last_error = None
        for attempt in range(max_retries + 1):
            raw = await self._call_llm(messages)

            if raw is None:
                raise ValueError("LLM returned empty response content")
            try:
                return response_model.model_validate_json(raw)
            except ValidationError as e:
                last_error = e
                if attempt < max_retries:
                    # Feed the error back so the model can self-correct
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your JSON was invalid. Pydantic error:\n{e}\n\n"
                            f"Fix the JSON and respond again with ONLY the corrected JSON object."
                        ),
                    })

        raise last_error
