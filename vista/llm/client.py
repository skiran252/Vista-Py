import json

import litellm
from pydantic import BaseModel, create_model
from typing import Any, Dict

from vista.core.signature import Signature


class LLMClient:
    """Wrapper around litellm for structured completions."""

    def __init__(self, model_name: str = "gpt-4o", api_base: str = None):
        self.model_name = model_name
        self.api_base = api_base

    async def _call_llm(self, messages: list, response_format=None) -> str:
        """
        Core LLM call with unified error handling.
        Returns the raw string content from the LLM response.
        """
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if response_format:
            completion_kwargs["response_format"] = response_format
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

        # Dynamic Pydantic model for enforcing output schema
        output_fields = {f.name: (f.type_, ...) for f in signature.outputs}
        OutputModel = create_model("OutputModel", **output_fields)

        raw_output = await self._call_llm(messages, response_format=OutputModel)

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
        raw = await self._call_llm(messages, response_format=response_model)

        if raw is None:
            raise ValueError("LLM returned empty response content")
        return response_model.model_validate_json(raw)
