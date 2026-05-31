import json

import litellm
from pydantic import BaseModel, create_model
from typing import Any, Dict

from vista.core.signature import Signature

# Standard JSON mode — works on all providers (Groq, OpenRouter, OpenAI, etc.)
# Unlike passing a Pydantic model as response_format, this does NOT get
# translated into tool/function calls by litellm.
JSON_MODE = {"type": "json_object"}


class LLMClient:
    """Wrapper around litellm for structured completions."""

    def __init__(self, model_name: str = "gpt-4o", api_base: str = None):
        self.model_name = model_name
        self.api_base = api_base

    async def _call_llm(self, messages: list) -> str:
        """
        Core LLM call with unified error handling.
        Uses JSON mode for structured output — avoids litellm converting
        Pydantic models into tool calls on providers that don't support
        native JSON schema response_format (e.g., Groq).
        """
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": JSON_MODE,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        try:
            response = await litellm.acompletion(**completion_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            # Some providers return the generated text inside the error on schema violations
            error_str = str(e)
            if "failed_generation" not in error_str:
                raise
            try:
                err_json = json.loads(error_str[error_str.find("{"):])
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
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        schema_instruction = (
            f"\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```"
        )

        # Append to existing system message or prepend a new one
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
        self, messages: list, response_model: type[BaseModel]
    ) -> BaseModel:
        """Generates a direct Pydantic model response (used by agents)."""
        messages = self._inject_schema(messages, response_model)
        raw = await self._call_llm(messages)

        if raw is None:
            raise ValueError("LLM returned empty response content")
        return response_model.model_validate_json(raw)
