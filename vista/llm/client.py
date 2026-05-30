import json
import litellm
from typing import Any, Dict
from pydantic import BaseModel, create_model

from vista.core.module import Signature


class LLMClient:
    """Wrapper around litellm for structured completions."""

    def __init__(self, model_name: str = "gpt-4o", api_base: str = None):
        self.model_name = model_name
        self.api_base = api_base

    async def predict(
        self, signature: Signature, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs the LLM based on the signature."""

        # 1. Format the system prompt
        sys_prompt = signature.instructions
        if not sys_prompt:
            sys_prompt = "You are a helpful AI assistant."

        sys_prompt += "\n\nGiven the following inputs, generate the requested outputs."

        # 2. Format the user input
        user_content = ""
        for field in signature.inputs:
            val = inputs.get(field.name, "")
            user_content += f"--- {field.name.upper()} ---\n{val}\n\n"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

        # 3. Create a Pydantic model for the expected outputs to enforce JSON schema
        output_fields = {}
        for field in signature.outputs:
            output_fields[field.name] = (field.type_, ...)

        OutputModel = create_model("OutputModel", **output_fields)

        # 4. Call LLM
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": OutputModel,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        response = await litellm.acompletion(**completion_kwargs)

        # 5. Parse output
        raw_output = response.choices[0].message.content
        if raw_output is None:
            return {signature.outputs[0].name: ""}
        try:
            parsed = json.loads(raw_output)
            return parsed
        except json.JSONDecodeError:
            return {signature.outputs[0].name: raw_output}

    async def generate_structured(
        self, messages: list, response_model: type[BaseModel]
    ) -> BaseModel:
        """Helper to generate a direct pydantic model (used by agents)."""
        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": response_model,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        response = await litellm.acompletion(**completion_kwargs)
        raw = response.choices[0].message.content
        if raw is None:
            raise ValueError("LLM returned empty response content")
        return response_model.model_validate_json(raw)
